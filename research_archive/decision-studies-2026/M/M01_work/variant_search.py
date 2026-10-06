"""Which dynamics reproduce the frozen path example?

Frozen target, path a-b-c-d with q = (0.4, 0.9, 0.4) on edges (a,b), (b,c), (c,d), TWO rounds:
    G   first action = the middle edge (b,c), value 1.96
    R   first action = an outer edge,      value 2.248
    OPT first action = an outer edge,      value 2.248

My first model (full adaptive matching each round) gives 3.4 for both G and OPT, so one of the
dynamics rules is different.  This script enumerates small, clearly-stated variants of the feedback
rule and reports which reproduce the targets.  Each variant also has to keep the single-edge
requirement V = 2q for every h, otherwise it is not the intended model.
"""
from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


def matchings(n, edges, avail_mask, max_size=None):
    """All independent subsets of the available edges (by index), including non-maximal and empty."""
    out = []

    def rec(i, used_v, chosen):
        if i == len(edges):
            if max_size is None or len(chosen) <= max_size:
                out.append(tuple(chosen))
            return
        # skip edge i
        rec(i + 1, used_v, chosen)
        if (avail_mask >> i) & 1:
            u, v = edges[i]
            if not ((used_v >> u) & 1 or (used_v >> v) & 1):
                rec(i + 1, used_v | (1 << u) | (1 << v), chosen + [i])

    rec(0, 0, [])
    return out


def make_variant(name, *, remove_vertex_on_failure, keep_failed_edge, max_edges_per_round):
    def transition(n, edges, avail, M, S):
        """Return the new availability mask after choosing M and observing success set S."""
        new = avail
        matched_v = 0
        for i in M:
            u, v = edges[i]
            new &= ~(1 << i)                      # chosen edges always leave
            if i in S:
                matched_v |= (1 << u) | (1 << v)
            elif remove_vertex_on_failure:
                matched_v |= (1 << u) | (1 << v)
        for i in range(len(edges)):
            if not ((new >> i) & 1):
                continue
            u, v = edges[i]
            if ((matched_v >> u) & 1) or ((matched_v >> v) & 1):
                new &= ~(1 << i)                  # incident edges leave with their vertex
        return new

    def subsets(M, probs):
        k = len(M)
        for sub in range(1 << k):
            p = 1.0
            S = set()
            for j in range(k):
                i = M[j]
                if (sub >> j) & 1:
                    p *= probs[i]
                    S.add(i)
                else:
                    p *= 1.0 - probs[i]
            yield frozenset(S), p

    def greedy_action(n, edges, avail, probs):
        best = None
        for M in matchings(n, edges, avail, max_edges_per_round):
            v = 2.0 * sum(probs[i] for i in M)
            key = (tuple(edges[i] for i in M),)
            if best is None or v > best[0] + 1e-12 or (
                    abs(v - best[0]) <= 1e-12 and key < best[2]):
                best = (v, M, key)
        return best[1]

    def build(n, edges, probs):
        @lru_cache(maxsize=None)
        def V(avail, h, which):
            if avail == 0 or h <= 0:
                return 0.0
            if which == "G":
                M = greedy_action(n, edges, avail, probs)
                return sum(p * (2 * len(S) + V(transition(n, edges, avail, M, S), h - 1, "G"))
                           for S, p in subsets(M, probs))
            best = None
            for M in matchings(n, edges, avail, max_edges_per_round):
                if which == "OPT":
                    cont = lambda a: V(a, h - 1, "OPT")
                else:  # R: one-step lookahead with the greedy policy as continuation
                    cont = lambda a: V(a, h - 1, "G")
                v = sum(p * (2 * len(S) + cont(transition(n, edges, avail, M, S)))
                        for S, p in subsets(M, probs))
                if best is None or v > best[0] + 1e-12:
                    best = (v, M)
            return best[0]

        def V_of(avail, h, which):
            return V(avail, h, which)

        def action(avail, h, which):
            if which == "G":
                return greedy_action(n, edges, avail, probs)
            cont = (lambda a: V(a, h - 1, "OPT")) if which == "OPT" else (lambda a: V(a, h - 1, "G"))
            best = None
            for M in matchings(n, edges, avail, max_edges_per_round):
                v = sum(p * (2 * len(S) + cont(transition(n, edges, avail, M, S)))
                        for S, p in subsets(M, probs))
                key = tuple(edges[i] for i in M)
                if best is None or v > best[0] + 1e-12 or (
                        abs(v - best[0]) <= 1e-12 and key < best[2]):
                    best = (v, M, key)
            return best[1]

        return V_of, action

    return {"name": name, "build": build, "transition": transition,
            "remove_vertex_on_failure": remove_vertex_on_failure,
            "keep_failed_edge": keep_failed_edge,
            "max_edges_per_round": max_edges_per_round}


def main() -> int:
    n = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    probs = [0.4, 0.9, 0.4]
    full = (1 << len(edges)) - 1
    target_g, target_opt = 1.96, 2.248

    print("=" * 100)
    print("  VARIANT SEARCH for the frozen path example")
    print(f"  target: G value {target_g}, R/OPT value {target_opt}, G action = middle edge")
    print("=" * 100)

    variants = []
    for rvf in (False, True):
        for me in (None, 1, 2):
            variants.append(make_variant(f"remove_vertex_on_failure={rvf}, max_edges={me}",
                                         remove_vertex_on_failure=rvf,
                                         keep_failed_edge=False,
                                         max_edges_per_round=me))

    for v in variants:
        V_of, action = v["build"](n, edges, probs)
        g_act = tuple(edges[i] for i in action(full, 2, "G"))
        o_act = tuple(edges[i] for i in action(full, 2, "OPT"))
        r_act = tuple(edges[i] for i in action(full, 2, "R"))
        g, o, r = V_of(full, 2, "G"), V_of(full, 2, "OPT"), V_of(full, 2, "R")
        # single-edge requirement
        V1, a1 = v["build"](2, [(0, 1)], [0.4])
        single = V1(1, 3, "OPT")
        hit = abs(g - target_g) < 1e-9 and abs(o - target_opt) < 1e-9
        print(f"\n  {v['name']}")
        print(f"    G   action {g_act}  value {g:.6f}")
        print(f"    R   action {r_act}  value {r:.6f}")
        print(f"    OPT action {o_act}  value {o:.6f}")
        print(f"    single edge q=0.4, h=3: {single:.6f} (must be 0.8)")
        print(f"    MATCHES FROZEN TARGET: {'YES' if hit else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
