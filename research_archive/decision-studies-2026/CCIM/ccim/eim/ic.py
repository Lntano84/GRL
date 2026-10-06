"""Independent Cascade on an arbitrary edge-probability graph, plus a fixed seed solver.

Two estimators are needed and they must not be confused with each other:

``TrueInfluence``
    evaluates a seed set on the **real** contact graph with a single global probability ``p``, by
    Monte-Carlo over live-edge draws.  This is the confirmatory metric: the number that gets reported.

``RRSolver``
    selects ``k`` seeds on a **model** graph whose edge probabilities are heterogeneous (the
    Beta-Bernoulli posterior of :mod:`ccim.eim.gm`).  It is the fixed end-of-survey routine, used
    identically by every method.

Both are implemented with the reverse-reachable-set (RR) formulation of IC, using the *seeded* form:
pick a uniformly random node ``v``, collect every node that can reach ``v`` along live edges to form
the set ``Q``, and the expected influence of ``S`` is

    sigma(S) = n * E[ |Q| * 1{S cap Q != empty} ] / theta - 1

which is the influence *including the seeds*, i.e. the same quantity as the standard closed-neighbourhood
IC objective.  The ``- 1`` is not cosmetic: without it the estimator computes ``|N[S] \\ {v}|`` and a
seed solver would be rewarded for the open neighbourhood, systematically undervaluing seeds that fall
inside each other's reach.  Greedy coverage then maximises ``sum over hit sets of |Q|`` instead of the
hit count (the Ryan-Bailey marginal).

Determinism.  The solver breaks ties by the smallest candidate **index** (the roster order), never by
an attribute value and never randomly, so two methods that hand the solver the same model receive the
same seed set.  A method cannot gain by exploiting node naming, because naming only orders candidates
and every method gets the same roster.
"""

from __future__ import annotations

import numpy as np

RR_SCALE = None            # kept for readability: influence = n * weighted_hits / theta - 1


def sample_rr_sets(n, edges, probs, theta, rng):
    """Sample ``theta`` seeded RR sets and their sizes.

    Returns ``(sizes, sets)`` where ``sizes[i] = |Q_i|``.  Both are returned because the influence
    estimator is ``n * sum_{hit} |Q| / theta - 1``, i.e. it is the *weighted* coverage that matters.
    """
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    prb = np.concatenate([probs, probs])
    keep = rng.random(len(prb)) < prb
    src, dst = src[keep], dst[keep]
    targets = rng.integers(0, n, size=int(theta))
    if len(src) == 0:
        sets = [np.asarray([int(t)], dtype=np.int64) for t in targets]
        return np.ones(int(theta), dtype=np.float64), sets
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    starts = np.searchsorted(src, np.arange(n + 1))
    adj = [dst[starts[v]:starts[v + 1]] for v in range(n)]

    out = []
    for t in targets:
        stack = [int(t)]
        seen = {int(t)}
        while stack:
            v = stack.pop()
            for w in adj[v]:
                w = int(w)
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        out.append(np.fromiter(seen, dtype=np.int64, count=len(seen)))
    return np.asarray([len(s) for s in out], dtype=np.float64), out


def rr_members(n, rr_sets):
    """Invert RR sets into a ``node -> list of set ids`` membership table."""
    members = [[] for _ in range(n)]
    for sid, s in enumerate(rr_sets):
        for v in s:
            members[int(v)].append(sid)
    return members


def coverage_greedy(members, sizes, cand_idx, k):
    """Greedy coverage maximising ``sum of |Q| over newly hit sets`` (Ryan-Bailey), up to ``k`` picks.

    Ties are broken by smallest candidate index.  If fewer than ``k`` candidates add anything, the seed
    set is completed with the smallest remaining indices so that its size is always ``k``.
    """
    n_sets = len(sizes)
    hit = np.zeros(n_sets, dtype=bool)
    members_sets = [set(m) for m in members]
    counters = np.array([sum(sizes[sid] for sid in members[int(v)]) for v in cand_idx], dtype=np.float64)
    alive = np.ones(len(cand_idx), dtype=bool)
    chosen: list[int] = []
    for _ in range(int(k)):
        masked = np.where(alive, counters, -1.0)
        j = int(np.argmax(masked))       # argmax returns the first maximum -> smallest candidate index
        if masked[j] <= 0:
            rest = [int(c) for c, a in zip(cand_idx, alive) if a]
            chosen.extend(sorted(rest)[: int(k) - len(chosen)])
            break
        v = int(cand_idx[j])
        chosen.append(v)
        alive[j] = False
        newly = [sid for sid in members_sets[v] if not hit[sid]]
        if newly:
            hit[newly] = True
            fresh = set(newly)
            for jj, c in enumerate(cand_idx):
                if alive[jj]:
                    counters[jj] -= sum(sizes[sid] for sid in fresh if sid in members_sets[int(c)])
    return np.asarray(chosen, dtype=np.int64), float(counters_of(hit, sizes))


def counters_of(hit, sizes) -> float:
    return float(sizes[hit].sum())


class TrueInfluence:
    """Monte-Carlo IC value of a seed set on the real graph.

    A single :class:`TrueInfluence` instance is reused for every method inside one state, so all
    methods are scored on the *same* live-edge draws (common random numbers).  That is deliberate: the
    differences being tested are small, and independent draws per method would add variance that has
    nothing to do with the methods.
    """

    def __init__(self, n, edges, p: float, samples: int, rng):
        self.n = int(n)
        self.samples = int(samples)
        self.rng = rng
        probs = np.full(len(edges), float(p))
        self.sizes, self.rr = sample_rr_sets(self.n, edges, probs, self.samples, rng)
        self._members = None

    @property
    def members(self):
        if self._members is None:
            self._members = rr_members(self.n, self.rr)
        return self._members

    def _hits(self, seeds):
        hit = np.zeros(self.samples, dtype=bool)
        for v in seeds:
            for sid in self.members[int(v)]:
                hit[sid] = True
        return hit

    def value(self, seeds) -> float:
        """``n * E[|Q| 1{Q cap S != empty}] / theta - 1``, the closed-neighbourhood IC value."""
        if not seeds:
            return 0.0
        return influenced(self.n, self.sizes, self._hits(seeds))

    def sample_hits(self, seeds) -> np.ndarray:
        return self._hits(seeds)


def influenced(n: int, sizes: np.ndarray, hit: np.ndarray) -> float:
    """Shared estimator, so the solver's own predicted value and the grader use one formula."""
    return float(n) * float(sizes[hit].sum()) / float(len(sizes)) - 1.0


class RRSolver:
    """The fixed end-of-survey seed solver.  One instance per arm; :meth:`solve` is deterministic.

    ``theta`` is the RR-set budget.  It is part of the *protocol*, not of any method: every arm of the
    comparison calls the solver with the same budget on its own model, so a better seed set can only
    come from a better model.
    """

    def __init__(self, n, k: int, theta: int, rng):
        self.n = int(n)
        self.k = int(k)
        self.theta = int(theta)
        self.rng = rng

    def solve(self, edges, probs, candidates=None):
        sizes, rr = sample_rr_sets(self.n, edges, probs, self.theta, self.rng)
        members = rr_members(self.n, rr)
        cand = np.arange(self.n) if candidates is None else np.asarray(candidates, dtype=np.int64)
        idx, _ = coverage_greedy(members, sizes, cand, self.k)
        hit = np.zeros(len(rr), dtype=bool)
        for i in idx:
            for sid in members[int(i)]:
                hit[sid] = True
        return idx, influenced(self.n, sizes, hit)
