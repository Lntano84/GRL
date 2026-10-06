"""The working graph model: a two-level Beta-Bernoulli edge model driven by the roster attribute.

Why a model is needed at all
----------------------------
A survey's value is not "how many edges does it reveal" but "would knowing them change the seed set
that gets chosen, and by how much".  Answering that requires a *distribution* over the unseen network,
because the unknown part is exactly what a survey is bought to reduce.  Any method in this comparison
that only leans on observed degree (``B1``) is a special case of doing without one.

The model
---------
Edges are independent Bernoulli variables.  The probability of a pair depends only on the pair's
attribute values (class groups) and on what surveys have already established:

* only pairs with **at least one surveyed endpoint** can carry information.  A pair surveyed from
  either side is *resolved*: an observed contact makes it an edge, its absence in the revealed
  neighbour list makes it a non-edge.  This is what "surveying a member reveals their neighbours"
  means, and it is why an unsurveyed pair is *unknown* rather than absent.  Treating unknown pairs as
  non-edges would let a method mistake ignorance for evidence.
* resolved pairs are held **fixed** in every sampled graph.  A sampled graph that contradicted an
  observation would be an internal inconsistency, and the seed solver would then be choosing seeds for
  a network the organisation already knows to be wrong.
* unresolved pairs are drawn with the posterior mean probability
  ``(n_ab + 1) / (d_ab + 2)``, a Krichevsky-Trofimov-smoothed edge rate over resolved pairs whose
  attribute values are ``(a, b)``, with an uninformative ``Beta(1, 1)`` prior.  So an attribute pair
  with no data falls back to the overall observed rate, and an attribute pair with data is pulled away
  from it only in proportion to the evidence.

The prior is fixed to ``Beta(1, 1)`` before any run.  It is not tuned, and no hyper-parameter of this
model is selected on outcomes: the experiment compares *decision rules given one model*, so the model
is part of the protocol rather than part of any arm.

Honest limitation: a two-level homophily model is a coarse picture of a contact network.  If the
look-ahead fails, one of the possible reasons is exactly that the model is too coarse -- which is a
different statement from "survey planning has no value", and the report keeps them apart.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

PRIOR_A = 1.0
PRIOR_B = 1.0


class HomophilyModel:
    """Posterior edge model over the roster, conditioned on the surveys performed so far."""

    def __init__(self, n: int, attr, base_rate: float = 0.0):
        self.n = int(n)
        self.attr = list(attr)
        self.base_rate = float(base_rate)
        self.groups = sorted(set(self.attr))
        self.gid = {g: i for i, g in enumerate(self.groups)}
        self.a_idx = np.asarray([self.gid[x] for x in self.attr], dtype=np.int64)
        self.resolved = np.zeros((self.n, self.n), dtype=bool)   # symmetric; diagonal unused
        self.observed = np.zeros((self.n, self.n), dtype=bool)
        self._flush()

    # ------------------------------------------------------------------ fitting
    def _flush(self) -> None:
        """Recompute per-attribute-pair resolved counts and their posterior means."""
        g = len(self.groups)
        ia = self.a_idx[:, None]
        ja = self.a_idx[None, :]
        flat = (ia * g + ja)
        res = self.resolved
        obs = self.observed
        upper = np.triu(np.ones((self.n, self.n), dtype=bool), 1)
        d = np.bincount(flat[upper & res].ravel(), minlength=g * g).reshape(g, g).astype(float)
        e = np.bincount(flat[upper & obs].ravel(), minlength=g * g).reshape(g, g).astype(float)
        d = d + d.T
        e = e + e.T
        self.d_ab = d
        self.e_ab = e
        self.p_ab = (e + PRIOR_A) / (d + PRIOR_A + PRIOR_B)
        self.pair_prob = self.p_ab[self.a_idx[:, None], self.a_idx[None, :]]

    def resolve(self, u: int, neighbours) -> int:
        """Record a survey of ``u``: every pair ``(u, w)`` becomes resolved, edge iff ``w`` in set.

        Returns the number of newly resolved pairs, so the caller can report survey yield.
        """
        nb = set(int(w) for w in neighbours)
        newly = 0
        for w in range(self.n):
            if w == u:
                continue
            if not self.resolved[u, w]:
                newly += 1
            self.resolved[u, w] = self.resolved[w, u] = True
            bit = w in nb
            self.observed[u, w] = self.observed[w, u] = bool(bit)
        self._flush()
        return newly

    def copy(self) -> "HomophilyModel":
        m = HomophilyModel.__new__(HomophilyModel)
        m.n = self.n
        m.attr = list(self.attr)
        m.base_rate = self.base_rate
        m.groups = list(self.groups)
        m.gid = dict(self.gid)
        m.a_idx = self.a_idx.copy()
        m.resolved = self.resolved.copy()
        m.observed = self.observed.copy()
        m._flush()
        return m

    # ------------------------------------------------------------------ sampling
    def edges_for_solver(self, rng):
        """One sampled graph, returned as ``(edges, probs)`` ready for :class:`ccim.eim.ic.RRSolver`.

        ``probs`` are the *generating* probabilities, so the solver sees the same model the sample came
        from.  Resolved pairs enter with probability 0 or 1 and are therefore never contradicted.
        """
        iu = np.triu_indices(self.n, 1)
        probs = self.pair_prob[iu]
        res = self.resolved[iu]
        obs = self.observed[iu]
        draw = rng.random(len(probs)) < np.where(res, np.where(obs, 1.0, 0.0), probs)
        keep = draw
        edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int64)
        return edges, probs[keep]

    def edge_list(self, rng):
        """Sampled adjacency as an ``(n, n)`` boolean matrix (convenience / tests)."""
        iu = np.triu_indices(self.n, 1)
        probs = self.pair_prob[iu]
        res = self.resolved[iu]
        obs = self.observed[iu]
        draw = rng.random(len(probs)) < np.where(res, np.where(obs, 1.0, 0.0), probs)
        adj = np.zeros((self.n, self.n), dtype=bool)
        adj[iu[0][draw], iu[1][draw]] = True
        return adj | adj.T

    def predict_edges(self, u: int, rng) -> list[int]:
        """Simulate surveying ``u``: draw its neighbour list from the model's *unresolved* pairs.

        Pairs already resolved keep their known value, so a simulated survey never contradicts an
        earlier one.
        """
        p = self.pair_prob[u].copy()
        res = self.resolved[u]
        obs = self.observed[u]
        p = np.where(res, np.where(obs, 1.0, 0.0), p)
        p[u] = 0.0
        return list(np.nonzero(rng.random(self.n) < p)[0])

    # ------------------------------------------------------------------ diagnostics
    def summary(self) -> dict:
        return {
            "n": self.n,
            "n_groups": len(self.groups),
            "resolved_pairs": int(self.resolved[np.triu_indices(self.n, 1)].sum()),
            "total_pairs": self.n * (self.n - 1) // 2,
            "observed_edges": int(self.observed[np.triu_indices(self.n, 1)].sum()),
            "resolved_edges": int(self.observed[np.triu_indices(self.n, 1)].sum()),
            "p_ab_min": float(self.p_ab.min()),
            "p_ab_max": float(self.p_ab.max()),
            "p_ab_diag_mean": float(np.mean(np.diag(self.p_ab))),
            "p_ab_offdiag_mean": float((self.p_ab.sum() - np.trace(self.p_ab)) /
                                       max(1.0, len(self.groups) ** 2 - len(self.groups))),
        }


def attribute_pair_table(model: HomophilyModel) -> dict:
    """Resolved counts per attribute pair, for the report's model-fit section."""
    out = defaultdict(dict)
    for i, gi in enumerate(model.groups):
        for j, gj in enumerate(model.groups):
            out[gi][gj] = {"resolved": float(model.d_ab[i, j]),
                           "edges": float(model.e_ab[i, j]),
                           "p": float(model.p_ab[i, j])}
    return dict(out)
