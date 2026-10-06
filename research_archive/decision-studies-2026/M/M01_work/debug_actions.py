"""Compare solver actions vs the independent tables on n6_m6_s1_HOM, method by method."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import build_tables  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    cfg = {c["config_id"]: c for c in generate_configs()}["n6_m6_s1_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)

    for method, get in (("G", lambda st, h: s.greedy_action(st)),
                        ("R", lambda st, h: s.R_action(st, h)[0]),
                        ("OPT", lambda st, h: s.OPT_action(st, h)[0])):
        diffs = []
        for st in range(1 << len(edges)):
            if st == 0:
                continue
            for h in (1, 2, 3):
                ai = tuple(sorted(edges[i] for i in action(st, h, method)))
                asol = p.matching_tuple(get(st, h))
                if ai != asol:
                    diffs.append((h, st, ai, asol))
        print(f"  {method}: {len(diffs)} action differences")
        for d in diffs[:8]:
            print(f"    h={d[0]} state={bin(d[1])} ind={d[2]} sol={d[3]}")
            # what does each think the two candidates are worth?
            M_ind = tuple(i for i, e in enumerate(edges) if e in d[2])
            M_sol = tuple(i for i, e in enumerate(edges) if e in d[3])
            print(f"        ind value of its own choice  = {value(d[1], d[0], method):.10f}")
            print(f"        solver V_{method}(state,h)     = "
                  f"{ {'G': s.V_G, 'R': s.V_R, 'OPT': s.V_OPT}[method](d[1], d[0]):.10f}")
    print()
    print("  V values: solver vs table, all states")
    for method, f in (("G", s.V_G), ("R", s.V_R), ("OPT", s.V_OPT)):
        bad = 0
        for st in range(1 << len(edges)):
            for h in (1, 2, 3):
                if abs(f(st, h) - value(st, h, method)) > 1e-9:
                    bad += 1
                    if bad <= 4:
                        print(f"    {method} h={h} state={bin(st)}: solver={f(st,h)} "
                              f"table={value(st,h,method)}")
        print(f"    {method}: {bad} value differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
