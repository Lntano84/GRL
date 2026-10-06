"""Diagnostic that must run BEFORE the full N01 experiment: is the reversal test vacuous?

The concern
-----------
The N01 quantity ``X_c - X_d`` collapses to
``|R(A) n R(D)| + |R(B) n R(C)| - (|R(A) n R(C)| + |R(B) n R(D)|)``
where ``A`` is the live-edge reachable set of candidate ``a`` and so on.  The structural condition
constrains the *deterministic* two-hop regions ``R2``, not the live-edge reachable sets ``R``, so it is
not obvious that ``X_c = X_d``.  If it DID hold identically, the comparison would be vacuous: real
diffusion would satisfy the same identity as the proxy, and the proxy would lose no expressiveness.
This script measures how often the two coincide, and how often the cross-overlap is non-empty.
"""
from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import (  # noqa: E402
    Graph, build_live_adjacency, draw_live_arcs, parse_ca_grqc, within_hops,
)


def reach_set(adj, src):
    seen = bytearray(len(adj))
    seen[src] = 1
    q = deque([src])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if not seen[u]:
                seen[u] = 1
                q.append(u)
    return seen


def main() -> int:
    arcs, undirected, node_ids, _ = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    cases = json.loads((HERE / "N01_cases.json").read_text(encoding="utf-8"))["cases"]

    n_worlds = 64
    rng = random.Random(777)
    which = rng.sample(range(len(cases)), 8)

    print("=" * 104)
    print("  VACUITY DIAGNOSTIC -- does the structural condition force X_c == X_d in the live-edge world?")
    print("=" * 104)
    print(f"  {len(which)} comparisons x {n_worlds} worlds (worlds 0..{n_worlds-1})")

    tot = 0
    equal_exact = 0
    nonzero_ident = 0
    overlap_ad = 0
    overlap_bc = 0
    diffs = []
    per_case_rows = []

    for ci in which:
        cs = cases[ci]
        a, b, c, d = cs["a"], cs["b"], cs["c"], cs["d"]
        eq = nz = oad = obc = 0
        cdiffs = []
        for wid in range(n_worlds):
            live = draw_live_arcs(graph, wid)
            adj = build_live_adjacency(graph, live)
            R = {k: reach_set(adj, v) for k, v in (("a", a), ("b", b), ("c", c), ("d", d))}

            def inter(x, y):
                return sum(1 for i in range(len(x)) if x[i] and y[i])

            ad, bc, ac, bd = inter(R["a"], R["d"]), inter(R["b"], R["c"]), \
                inter(R["a"], R["c"]), inter(R["b"], R["d"])

            def spread(seeds):
                seen = bytearray(graph.n)
                q = deque()
                for s in seeds:
                    if not seen[s]:
                        seen[s] = 1
                        q.append(s)
                n = len(q)
                while q:
                    v = q.popleft()
                    for u in adj[v]:
                        if not seen[u]:
                            seen[u] = 1
                            n += 1
                            q.append(u)
                return n

            Xc = spread(cs["S_ac"]) - spread(cs["S_bc"])
            Xd = spread(cs["S_ad"]) - spread(cs["S_bd"])
            # identity check
            ident = (ad - ac) - (bd - bc)
            if Xc == Xd:
                eq += 1
            if ident != 0:
                nz += 1
            if ad > 0:
                oad += 1
            if bc > 0:
                obc += 1
            cdiffs.append(Xc - Xd)
            diffs.append(Xc - Xd)
            tot += 1
        equal_exact += eq
        nonzero_ident += nz
        overlap_ad += oad
        overlap_bc += obc
        per_case_rows.append({
            "case": ci, "group": cs["group"], "a": a, "b": b, "c": c, "d": d,
            "worlds": n_worlds,
            "X_c_equals_X_d_exactly": eq,
            "identity_term_nonzero": nz,
            "worlds_with_Ra_n_Rd_nonempty": oad,
            "worlds_with_Rb_n_Rc_nonempty": obc,
            "mean_X_c_minus_X_d": sum(cdiffs) / len(cdiffs),
            "max_abs_X_c_minus_X_d": max(abs(x) for x in cdiffs),
        })
        print(f"  case {ci:>2} ({cs['group']:<10}) X_c==X_d in {eq:>3}/{n_worlds} worlds | "
              f"R(a)nR(d) nonempty {oad:>3} | R(b)nR(c) nonempty {obc:>3} | "
              f"max|X_c-X_d| {max(abs(x) for x in cdiffs)}")

    print(f"\n  TOTAL: X_c == X_d exactly in {equal_exact}/{tot} (world, comparison) pairs "
          f"= {equal_exact/tot:.2%}")
    print(f"         cross-overlap (R(a)nR(d)) or (R(b)nR(c)) non-empty in "
          f"{(overlap_ad+overlap_bc)}/{2*tot} single-side checks")
    print(f"         max |X_c - X_d| over all pairs: {max(abs(x) for x in diffs)}")
    print(f"         mean |X_c - X_d| over all pairs: "
          f"{sum(abs(x) for x in diffs)/len(diffs):.4f}")

    out = {
        "purpose": "check whether the N01 reversal test is vacuous under the frozen structural condition",
        "n_comparisons_sampled": len(which), "n_worlds": n_worlds,
        "world_seed_range": [0, n_worlds - 1],
        "total_pairs": tot,
        "X_c_equals_X_d_exactly": equal_exact,
        "fraction_identical": equal_exact / tot,
        "worlds_with_Ra_n_Rd_nonempty": overlap_ad,
        "worlds_with_Rb_n_Rc_nonempty": overlap_bc,
        "max_abs_X_c_minus_X_d": max(abs(x) for x in diffs),
        "mean_abs_X_c_minus_X_d": sum(abs(x) for x in diffs) / len(diffs),
        "per_case": per_case_rows,
    }
    (HERE / "vacuity_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'vacuity_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
