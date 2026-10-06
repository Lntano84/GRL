"""Compare the solver's action tables with the independent tables, state by state, then trace."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import advance, build_tables, independent_sets  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    configs = generate_configs()
    cfg = {c["config_id"]: c for c in configs}["n6_m6_s1_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)

    print("  (A) independent_sets vs solver's matchings, per state")
    bad = 0
    for st in range(1 << len(edges)):
        ind = sorted(tuple(sorted(edges[i] for i in M)) for M in independent_sets(n, edges, st))
        sol = sorted(p.matching_tuple(M) for M in p.all_matchings(st))
        if ind != sol:
            bad += 1
            if bad <= 5:
                print(f"    state {bin(st)}: ind={ind} sol={sol}")
    print(f"    mismatching states: {bad}")

    print("\n  (B) transition agreement on every (state, matching, subset)")
    bad_t = 0
    for st in range(1 << len(edges)):
        for M in p.all_matchings(st):
            for S, prob, cnt in p.success_subsets(M):
                Midx = tuple(i for i in range(len(edges)) if (M >> i) & 1)
                SetS = frozenset(i for i in range(len(edges)) if (S >> i) & 1)
                a = p.transition(st, M, S)
                b = advance(n, edges, st, Midx, SetS)
                if a != b:
                    bad_t += 1
                    if bad_t <= 5:
                        print(f"    state {bin(st)} M={Midx} S={sorted(SetS)}: "
                              f"solver={bin(a)} ind={bin(b)}")
    print(f"    mismatching transitions: {bad_t}")

    print("\n  (C) action tables: independent vs solver")
    for h in (1, 2, 3):
        mism = 0
        for st in range(1 << len(edges)):
            if st == 0:
                continue
            a_ind = action(st, h, "OPT")
            a_sol, _ = s.OPT_action(st, h)
            if tuple(sorted(edges[i] for i in a_ind)) != p.matching_tuple(a_sol):
                mism += 1
                if mism <= 4:
                    print(f"    h={h} state {bin(st)}: ind={tuple(edges[i] for i in a_ind)} "
                          f"sol={p.matching_tuple(a_sol)}")
        print(f"    h={h}: mismatching states {mism}")

    print("\n  (D) independent value table for OPT vs solver V_OPT")
    for h in (1, 2, 3):
        mism = 0
        for st in range(1 << len(edges)):
            if st == 0:
                continue
            v_ind = value(st, h, "OPT")
            v_sol = s.V_OPT(st, h)
            if abs(v_ind - v_sol) > 1e-9:
                mism += 1
                if mism <= 4:
                    print(f"    h={h} state {bin(st)}: ind={v_ind:.10f} sol={v_sol:.10f}")
        print(f"    h={h}: mismatching states {mism}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
