"""Independent Cascade influence estimation and the greedy seed selector, matching the official code.

Why this file exists
--------------------
The reward in exploratory influence maximization is *not* "how many nodes did I discover" -- it is the
influence that the chosen seed set achieves on the **complete** network.  Everything therefore rests on one
estimator, and it has to mean the same thing as the estimator the published results were produced with.

This is a faithful re-implementation of ``icm.py`` / ``utils.greedy`` / ``expts.influence`` from the
official repository (``kage08/graph_sample_rl``), not a lookalike:

* the live-edge formulation of IC is used -- a node is influenced iff it lies in the same live-edge
  connected component as at least one seed;
* the objective counts **every node in that component, seeds included**, which is what
  ``ws[i] = |target_nodes cap cc[i]|`` computes in ``icm.live_edge_to_adjlist``;
* the influence probability is ``p = 0.1`` on every edge;
* the seed set is chosen by **lazy greedy (CELF)** on the *discovered* graph, with ``k = 10``;
* the reward is that same seed set evaluated on the *complete* graph.

The original selects over ``list(range(len(graph)))`` where the environment's graph has all ``n`` nodes
added, so undiscovered nodes are technically candidates.  They are isolated in every live-edge sample and
so carry a marginal of 1; the behaviour is reproduced deliberately rather than "fixed", because changing
it would change the baseline numbers.

Numerical difference from the original
--------------------------------------
The original draws live edges with ``random.random() < p`` per edge.  This implementation draws the
*number* of live edges as ``Binomial(m, p)`` and then chooses which edges uniformly, which is
distributionally identical and much faster.  A test checks the two agree on component statistics.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PROP_PROBAB = 0.1      # icm/expts.influence: --prop-prob default
INFL_BUDGET = 10       # expts/influence.BUDGET and --infl-budget default
SAMPLES = 100          # expts/influence.SAMPLES and --samples default


class LiveEdgeObjective:
    """The IC objective as an expectation over sampled live-edge graphs.

    :meth:`greedy` reproduces ``utils.greedy``: standard greedy with lazy re-evaluation, exactly ``k``
    picks, candidates = every node index.  Laziness is omitted because the vectorised marginal is cheap
    enough to recompute for all candidates each round, and plain greedy is what laziness is an
    optimisation *of*.
    """

    def __init__(self, graph: nx.Graph, samples: int = SAMPLES, rng=None, p: float = PROP_PROBAB,
                 nodes=None):
        self.graph = graph
        self.p = p
        self.samples = int(samples)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.nodes = list(nodes) if nodes is not None else list(range(graph.number_of_nodes()))
        self.n = graph.number_of_nodes()
        self._index = {v: i for i, v in enumerate(sorted(graph.nodes()))}
        self._build()

    # ------------------------------------------------------------------ sampling
    def _build(self) -> None:
        g = self.graph
        edges = [(self._index[u], self._index[v]) for u, v in g.edges()]
        rows = np.array([e[0] for e in edges], dtype=np.int64) if edges else np.zeros(0, np.int64)
        cols = np.array([e[1] for e in edges], dtype=np.int64) if edges else np.zeros(0, np.int64)
        m = len(edges)
        lab = np.zeros((self.samples, self.n), dtype=np.int32)
        sizes = np.zeros((self.samples, self.n), dtype=np.int32)
        for s in range(self.samples):
            k = int(self.rng.binomial(m, self.p)) if m else 0
            keep = self.rng.choice(m, size=k, replace=False) if k else np.zeros(0, np.int64)
            data = np.ones(len(keep), dtype=np.int8)
            adj = coo_matrix((data, (rows[keep], cols[keep])), shape=(self.n, self.n))
            ncomp, labels = connected_components(adj, directed=False)
            counts = np.bincount(labels, minlength=ncomp).astype(np.int32)
            lab[s] = labels
            sizes[s] = counts[labels]
        self.labels = lab                 # (samples, n) component id of each node
        self.comp_size = sizes            # (samples, n) size of that node's component
        self._covered = np.zeros_like(lab, dtype=bool)

    # ------------------------------------------------------------------ evaluation
    def value(self, S) -> float:
        """Expected number of influenced nodes (seeds included), averaged over live-edge samples."""
        idx = [self._index[u] for u in S]
        if not idx:
            return 0.0
        total = 0.0
        for s in range(self.samples):
            seen = set()
            for i in idx:
                c = self.labels[s, i]
                if c not in seen:
                    seen.add(c)
                    total += self.comp_size[s, i]
        return total / self.samples

    def sample_values(self, S) -> np.ndarray:
        """Per-live-edge-graph reach of ``S``, so the Monte-Carlo error of the reward is reportable.

        The evaluation estimate is a mean over samples, so its standard error is ``sd/sqrt(samples)``.
        Reporting it matters: a policy difference smaller than that error cannot be read as a difference.
        """
        idx = [self._index[u] for u in S]
        out = np.zeros(self.samples)
        if not idx:
            return out
        for s in range(self.samples):
            seen = set()
            total = 0
            for i in idx:
                c = self.labels[s, i]
                if c not in seen:
                    seen.add(c)
                    total += self.comp_size[s, i]
            out[s] = total
        return out

    def value_with_error(self, S) -> tuple:
        """``(mean, standard_error)`` of the influence of ``S`` under this fixed sample set."""
        v = self.sample_values(S)
        if len(v) < 2:
            return (float(v.mean()) if len(v) else 0.0), 0.0
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))

    def marginal_all(self, covered) -> np.ndarray:
        """Node-indexed marginal gain of every candidate, given the per-sample covered-component mask.

        ``covered`` is indexed by **(sample, component id)**, while the result is indexed by **node**, so
        the mask has to be gathered through ``labels`` before it can be applied to ``comp_size``.
        Applying it positionally is a silent bug: the gain is then read off the wrong component.
        """
        alive = ~covered[np.arange(self.samples)[:, None], self.labels]
        return np.where(alive, self.comp_size, 0).sum(axis=0) / self.samples

    def greedy(self, k: int = INFL_BUDGET, candidates=None):
        """``utils.greedy``: ``k`` seeds maximising :meth:`value`, ties broken by node index."""
        cand_nodes = list(range(self.n)) if candidates is None else list(candidates)
        cand = np.asarray([self._index[u] for u in cand_nodes], dtype=np.int64)
        covered = np.zeros((self.samples, self.n), dtype=bool)   # per sample, per component id
        chosen: list[int] = []
        total = 0.0
        taken = np.zeros(self.n, dtype=bool)
        order = sorted(self._index, key=lambda v: self._index[v])
        for _ in range(k):
            gains = self.marginal_all(covered)
            gains[taken] = -1.0
            gi = int(cand[int(np.argmax(gains[cand]))])
            total += float(gains[gi])
            chosen.append(order[gi])
            taken[gi] = True
            for s in range(self.samples):
                covered[s, self.labels[s, gi]] = True
        return chosen, total


def influence_on(graph: nx.Graph, seeds, rng, samples: int = SAMPLES, p: float = PROP_PROBAB) -> float:
    """Evaluate a fixed seed set on a graph: the reward, exactly as the original computes it."""
    obj = LiveEdgeObjective(graph, samples=samples, rng=rng, p=p)
    return obj.value(seeds)
