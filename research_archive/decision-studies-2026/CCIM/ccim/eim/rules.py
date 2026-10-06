"""The cheap rules that the look-ahead has to beat, and the candidate pool it chooses from.

All five rules read **exactly** the same legal information as the look-ahead: the roster, the
pre-registered attribute, the resolved pairs, and the working model.  None of them can see the real
contact graph, and none of them is given a smaller action set.  They are the honest competition: an
expensive procedure that loses to a two-line rule has not earned its cost.

Naming note.  ``B4`` is *inspired by* IM-META's "survey nodes that matter for seeding" idea, but it is
expressed in this report's model and features.  It is an adaptation, not a reproduction of that paper,
and must not be cited as one.
"""

from __future__ import annotations

import numpy as np


def legal_nodes(model) -> np.ndarray:
    """Indices of roster members not yet surveyed."""
    return np.nonzero(~model.resolved.any(axis=1))[0]


def observed_degree(model) -> np.ndarray:
    """Confirmed degree: how many edges surveys have already established for each member."""
    return model.observed.sum(axis=1).astype(float)


def predicted_residual_degree(model) -> np.ndarray:
    """Expected number of *still unknown* neighbours per member, under the current model."""
    p = model.pair_prob.copy()
    known = model.resolved
    p = np.where(known, 0.0, p)
    np.fill_diagonal(p, 0.0)
    return p.sum(axis=1)


def coverage_probability(model, rng, theta: int, k: int, solver):
    """P( member is not reachable from the model's own seed set ), estimated by sampling.

    "Reachable" means reachable along live edges in a sampled graph, i.e. would already be influenced.
    A member who is usually *outside* the current seeds' reach is one whose hidden edges could matter.
    """
    seeds, _ = solver.solve(*model.edges_for_solver(rng))
    reach = np.zeros(model.n, dtype=float)
    for _ in range(int(theta)):
        lab = _live_labels(model, rng)
        for s in seeds:
            reach[lab[int(s)]] += 1.0
    return 1.0 - reach / float(theta)


def _live_labels(model, rng):
    """One live-edge draw; returns, for each node, the size of its live-edge component.

    Component *sizes* rather than component ids, because that is what the seed-coverage statistic needs
    and because sizes are invariant to how the labelling happens to be ordered.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges, _ = model.edges_for_solver(rng)
    if len(edges) == 0:
        return np.ones(model.n, dtype=np.int64)
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(model.n, model.n))
    _, lab = connected_components(adj, directed=False)
    return np.bincount(lab)[lab]


def seed_distance_weight(model, rng, k: int, samples: int, solver):
    """Per member: expected number of members it connects to that lie close to the model's seeds.

    This is the model-side analogue of "prefer information near the prospective seeds": a candidate
    scores high when its uncertain edges point into the region the seeds already matter for.  Closeness
    is "is influenced by the current seeds in this live-edge draw", which is the model's own notion of
    "already covered".
    """
    weight = np.zeros(model.n)
    for _ in range(int(samples)):
        seeds, _ = solver.solve(*model.edges_for_solver(rng))
        lab = _live_labels(model, rng)
        covered = np.zeros(model.n, dtype=bool)
        for s in seeds:
            covered |= (lab == lab[int(s)])
        close = covered.astype(float)
        residual = np.where(model.resolved, 0.0, model.pair_prob)
        np.fill_diagonal(residual, 0.0)
        weight += residual @ close
    return weight / float(samples)


def pick(score: np.ndarray, legal: np.ndarray, rng=None, uniform: bool = False) -> int:
    """Choose an index from ``legal``.  Ties are broken by smallest index unless ``uniform``."""
    if uniform:
        return int(rng.choice(legal))
    s = np.where(np.isin(np.arange(len(score)), legal), score, -np.inf)
    return int(np.argmax(s))


def candidate_pool(scores: dict, legal: np.ndarray, rng, size: int = 8) -> list[int]:
    """Deterministic-plus-random pool: every rule's first choice over the **full** legal action set,
    plus uniform draws from that same full set.

    The pool is small to keep the look-ahead affordable, but it is not "the pool the rules searched":
    each rule's headline action was computed on all legal members and is guaranteed a slot, and the
    random filler is drawn from all legal members too.  This removes the failure mode where a
    restricted action set is presented as if it were the full one.
    """
    pool: list[int] = []
    for name in sorted(scores):
        u = pick(scores[name], legal)
        if u not in pool:
            pool.append(u)
    while len(pool) < size:
        u = int(rng.choice(legal))
        if u not in pool:
            pool.append(u)
    return pool[:size]
