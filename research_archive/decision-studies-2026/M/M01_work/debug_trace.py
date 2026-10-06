"""Trace the R simulation against the solver's policy on n6_m6_s0_HOM to localise the residual gap."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import advance, build_tables  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    configs = generate_configs()
    cfg = {c["config_id"]: c for c in configs}["n6_m6_s0_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)

    print("  R action comparison at the root and along the way")
    for h in (3, 2, 1):
        a_ind = tuple(edges[i] for i in action(p.full_mask, h, "R"))
        a_sol = p.matching_tuple(s.R_action(p.full_mask, h)[0])
        print(f"    h={h} root: ind={a_ind} sol={a_sol} same={tuple(sorted(a_ind))==a_sol}")

    print("\n  All R states where the independent and solver actions differ")
    diff = 0
    for st in range(1 << len(edges)):
        if st == 0:
            continue
        for h in (1, 2, 3):
            a_ind = tuple(sorted(edges[i] for i in action(st, h, "R")))
            a_sol = p.matching_tuple(s.R_action(st, h)[0])
            if a_ind != a_sol:
                diff += 1
                if diff <= 10:
                    print(f"    h={h} state={bin(st)}: ind={a_ind} sol={a_sol}")
    print(f"    total differing (state,h): {diff}")

    print("\n  Root R lookahead scores: independent vs solver")
    for M in p.all_matchings(p.full_mask):
        Midx = tuple(i for i in range(len(edges)) if (M >> i) & 1)
        # independent lookahead score
        v_ind = 0.0
        import itertools
        for r in range(len(Midx) + 1):
            for succ in itertools.combinations(Midx, r):
                prob = 1.0
                for i in Midx:
                    prob *= probs[i] if i in succ else (1.0 - probs[i])
                v_ind += prob * (2 * len(succ) + value(advance(n, edges, p.full_mask, Midx,
                                                               frozenset(succ)), 2, "G"))
        v_sol = s.Qtilde(p.full_mask, M, 3)
        flag = "" if abs(v_ind - v_sol) < 1e-9 else "   <-- DIFFERS"
        print(f"    {str(p.matching_tuple(M)):<28} ind={v_ind:.6f} sol={v_sol:.6f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
