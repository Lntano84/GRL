"""M01 core solver: three-round stochastic matching with recourse, exact enumeration.

Frozen task semantics
---------------------
* undirected simple graph; no vertex failures, no added vertices;
* each edge has an independent success probability ``q_e`` and its state is persistent;
* each round selects a matching; non-maximal matchings and the empty matching are allowed;
* a successful edge contributes 2 matched vertices and both endpoints leave with their edges;
* a failed edge is deleted permanently while its endpoints remain;
* only the edges actually chosen are observed; unchosen edge states are never read;
* the main experiment has exactly THREE rounds, ``h`` = rounds remaining including the current one.

Value functions
---------------
``V_h^G``   : the greedy action rule (max expected weight) but evaluated with full lookahead:
              ``V_h^G(G) = E_S[2|S| + V_{h-1}^G(G_{M_G,S})]`` with ``M_G = argmax_M 2 sum q_e``.
``V_h^R``   : one-step lookahead evaluated with the greedy rule as continuation, re-solved every
              round: ``M_R(G) = argmax_M Qtilde_h(G, M)`` where
              ``Qtilde_h(G, M) = E_S[2|S| + V_{h-1}^G(G_{M,S})]``; ``V_h^R`` is the value of
              actually following that rule for ``h`` rounds, NOT ``max_M Qtilde_h``.
``V_h^*``   : the full finite-horizon optimum, ``V_h^*(G) = max_M E_S[2|S| + V_{h-1}^*(G_{M,S})]``.

Ties are broken by the lexicographically smallest sorted edge tuple.  No Monte Carlo is used.

Monotonicity note used for correctness
--------------------------------------
After a round, every edge of the chosen matching is gone (success removes it with its endpoints,
failure deletes it), and every edge incident to a matched vertex is gone.  Hence the remaining edge
set is always a SUBSET of the current one and the vertex set only shrinks.  The state is therefore
fully described by the surviving edge set, which makes the reachable state space a subset of the
``2^m`` edge subsets -- the basis of the memoisation below.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.setrecursionlimit(100_000)


# --------------------------------------------------------------------------- graph utilities

def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class Problem:
    """One frozen instance: ``n`` vertices, a lexicographically sorted edge list, probabilities."""

    def __init__(self, n: int, edges: list[tuple[int, int]], probs: list[float],
                 label: str = ""):
        assert len(edges) == len(probs)
        self.n = n
        self.label = label
        self.edges = [tuple(e) for e in edges]          # lexicographically sorted
        self.probs = [float(q) for q in probs]
        self.m = len(self.edges)
        self.full_mask = (1 << self.m) - 1
        self.incident = [0] * n                          # vertex -> edge mask
        for i, (u, v) in enumerate(self.edges):
            self.incident[u] |= 1 << i
            self.incident[v] |= 1 << i
        self.edge_covers_mask = [self.incident[u] | self.incident[v] for u, v in self.edges]

    # ---- matching machinery -------------------------------------------------
    def is_matching(self, emask: int) -> bool:
        used = 0
        for i in bits(emask):
            u, v = self.edges[i]
            if (used >> u) & 1 or (used >> v) & 1:
                return False
            used |= (1 << u) | (1 << v)
        return True

    def matching_tuple(self, emask: int) -> tuple:
        return tuple(self.edges[i] for i in bits(emask))

    def all_matchings(self, state: int) -> list[int]:
        """Every independent edge set inside ``state``, including non-maximal and empty ones."""
        avail = state
        out: list[int] = []

        def rec(remaining: int, used_v: int, chosen: int) -> None:
            out.append(chosen)
            r = remaining
            while r:
                low = r & -r
                i = low.bit_length() - 1
                r ^= low
                u, v = self.edges[i]
                if (used_v >> u) & 1 or (used_v >> v) & 1:
                    continue
                # only extend with edges whose index is above the last chosen one, to avoid repeats
                rest = r & ~((1 << (i + 1)) - 1)
                rec(rest, used_v | (1 << u) | (1 << v), chosen | low)

        rec(avail, 0, 0)
        return out

    def success_subsets(self, emask: int):
        """Yield ``(S, probability, matched_vertex_count)`` over the ``2^|M|`` outcome subsets."""
        idxs = list(bits(emask))
        k = len(idxs)
        for sub in range(1 << k):
            prob = 1.0
            s = 0
            cnt = 0
            for j in range(k):
                i = idxs[j]
                if (sub >> j) & 1:
                    prob *= self.probs[i]
                    s |= 1 << i
                    cnt += 1
                else:
                    prob *= 1.0 - self.probs[i]
            yield s, prob, 2 * cnt

    def transition(self, state: int, emask: int, S: int) -> int:
        """Feedback map: chosen edges leave, and every edge touching a matched vertex leaves."""
        new = state & ~emask
        for i in bits(S):
            new &= ~self.edge_covers_mask[i]
        return new

    # ---- decomposition ------------------------------------------------------
    def components(self, state: int) -> list[int]:
        """Edge masks of the connected components of the subgraph induced by ``state``.

        ``emask`` accumulates EDGES and ``vset`` accumulates VERTICES; the two masks use different
        index spaces and must not be conflated.  (An earlier version tested vertices against the edge
        mask, which reported every edge as its own component and inflated all values.)
        """
        remaining = state
        comps: list[int] = []
        while remaining:
            low = remaining & -remaining
            seed = low.bit_length() - 1
            u, v = self.edges[seed]
            emask = low
            vset = (1 << u) | (1 << v)
            frontier = [u, v]
            while frontier:
                x = frontier.pop()
                for j in bits(self.incident[x]):
                    if (emask >> j) & 1:
                        continue
                    emask |= 1 << j
                    a, b = self.edges[j]
                    for y in (a, b):
                        if not ((vset >> y) & 1):
                            vset |= 1 << y
                            frontier.append(y)
            comps.append(emask)
            remaining &= ~emask
        return comps


# --------------------------------------------------------------------------- the solvers

class Timeout(Exception):
    """Raised when a configuration exceeds its own CPU budget."""


class Solver:
    """Exact value functions with separate caches, so R can never read OPT's cache."""

    def __init__(self, prob: Problem, eps_scale: float | None = None, deadline=None):
        self.p = prob
        self.eps = 1e-10 * max(1, prob.n) if eps_scale is None else eps_scale
        self.deadline = deadline
        self._g: dict[tuple[int, int], float] = {}
        self._opt: dict[tuple[int, int], float] = {}
        self._r: dict[tuple[int, int], float] = {}
        self.transitions = 0
        self.matching_enumerations = 0
        self._match_cache: dict[int, list[int]] = {}
        self._action_cache: dict[tuple[str, int, int], int] = {}
        self._q3_cache: dict[tuple[int, int], float] = {}

    def _check(self) -> None:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise Timeout("configuration CPU budget exceeded")

    def matchings(self, state: int) -> list[int]:
        got = self._match_cache.get(state)
        if got is None:
            got = self.p.all_matchings(state)
            self.matching_enumerations += 1
            self._match_cache[state] = got
        return got

    def _better(self, new_v: float, new_M: int, best_v: float, best_M: int) -> bool:
        """True when ``(new_v, new_M)`` beats ``(best_v, best_M)`` under the frozen tie rule."""
        if new_v > best_v + self.eps:
            return True
        if abs(new_v - best_v) <= self.eps:
            return self.p.matching_tuple(new_M) < self.p.matching_tuple(best_M)
        return False

    def _argmax(self, scored) -> tuple[int, float]:
        best_v = None
        best_M = 0
        for M, v in scored:
            if best_v is None or self._better(v, M, best_v, best_M):
                best_v, best_M = v, M
        return best_M, best_v

    def q_of_matching(self, state: int, emask: int, continuation) -> float:
        """``E_S[2|S| + continuation(transition(state, emask, S))]`` for a given continuation."""
        total = 0.0
        for S, prob, cnt in self.p.success_subsets(emask):
            if prob == 0.0:
                continue
            self.transitions += 1
            total += prob * (cnt + continuation(self.p.transition(state, emask, S)))
        return total

    # ---- G -----------------------------------------------------------------
    def greedy_action(self, state: int) -> int:
        """``argmax_M 2 sum_{e in M} q_e``; ties by lexicographically smallest edge tuple."""
        key = ("G", state, 0)
        got = self._action_cache.get(key)
        if got is not None:
            return got
        scored = [(M, 2.0 * sum(self.p.probs[i] for i in bits(M)))
                  for M in self.matchings(state)]
        best_M, _ = self._argmax(scored)
        self._action_cache[key] = best_M
        return best_M

    def V_G(self, state: int, h: int) -> float:
        if state == 0 or h <= 0:
            return 0.0
        self._check()
        key = (state, h)
        got = self._g.get(key)
        if got is not None:
            return got
        comps = self.p.components(state)
        if len(comps) > 1:
            val = sum(self.V_G(c, h) for c in comps)
            self._g[key] = val
            return val
        M = self.greedy_action(state)
        val = self.q_of_matching(state, M, lambda s: self.V_G(s, h - 1))
        self._g[key] = val
        return val

    # ---- OPT ---------------------------------------------------------------
    def V_OPT(self, state: int, h: int) -> float:
        if state == 0 or h <= 0:
            return 0.0
        self._check()
        key = (state, h)
        got = self._opt.get(key)
        if got is not None:
            return got
        comps = self.p.components(state)
        if len(comps) > 1:
            val = sum(self.V_OPT(c, h) for c in comps)
            self._opt[key] = val
            return val
        scored = [(M, self.q_of_matching(state, M, lambda s: self.V_OPT(s, h - 1)))
                  for M in self.matchings(state)]
        _, best = self._argmax(scored)
        self._opt[key] = best
        return best

    def OPT_action(self, state: int, h: int = 3) -> tuple[int, float]:
        """The optimal first action and its value; used for reporting, not by R."""
        scored = [(M, self.q_of_matching(state, M, lambda s: self.V_OPT(s, h - 1)))
                  for M in self.matchings(state)]
        return self._argmax(scored)

    # ---- R -----------------------------------------------------------------
    def Qtilde(self, state: int, M: int, h: int) -> float:
        """One-step lookahead with the full greedy policy as the continuation estimate."""
        return self.q_of_matching(state, M, lambda s: self.V_G(s, h - 1))

    def lookahead_scores(self, state: int, h: int) -> list[tuple[int, float]]:
        return [(M, self.Qtilde(state, M, h)) for M in self.matchings(state)]

    def R_action(self, state: int, h: int) -> tuple[int, list[int], float]:
        """The R rule's action: argmax of Qtilde, ties by lexicographically smallest edge tuple.

        Returns ``(best_mask, argmax_set, best_value)`` where ``argmax_set`` holds EVERY action
        within ``eps`` of the maximum -- needed for the robust main metric.
        """
        scored = self.lookahead_scores(state, h)
        best = max(v for _, v in scored)
        argmax = [M for M, v in scored if v >= best - self.eps]
        argmax.sort(key=lambda M: self.p.matching_tuple(M))
        return argmax[0], argmax, best

    def V_R(self, state: int, h: int) -> float:
        """Value of actually FOLLOWING the R rule for ``h`` rounds (not max_M Qtilde)."""
        if state == 0 or h <= 0:
            return 0.0
        self._check()
        key = (state, h)
        got = self._r.get(key)
        if got is not None:
            return got
        comps = self.p.components(state)
        if len(comps) > 1:
            val = sum(self.V_R(c, h) for c in comps)
            self._r[key] = val
            return val
        M, _, _ = self.R_action(state, h)
        val = self.q_of_matching(state, M, lambda s: self.V_R(s, h - 1))
        self._r[key] = val
        return val

    def Q3_star(self, state: int, M: int) -> float:
        """``E[2|S| + V_2^*(G_{M,S})]`` -- the true three-round value of taking ``M`` now."""
        key = (state, M)
        got = self._q3_cache.get(key)
        if got is not None:
            return got
        val = self.q_of_matching(state, M, lambda s: self.V_OPT(s, 2))
        self._q3_cache[key] = val
        return val

    # ---- diagnostics -------------------------------------------------------
    def cache_sizes(self) -> dict:
        return {"G_states": len(self._g), "OPT_states": len(self._opt), "R_states": len(self._r),
                "matching_enumerations": self.matching_enumerations,
                "transitions": self.transitions}

    def clear_all(self) -> None:
        self._g.clear()
        self._opt.clear()
        self._r.clear()
        self._match_cache.clear()
        self._action_cache.clear()
        self._q3_cache.clear()
        self.transitions = 0
        self.matching_enumerations = 0
