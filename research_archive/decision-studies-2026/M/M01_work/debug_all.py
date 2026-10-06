"""Full table/action/transition diagnostic for every s=0,1 two-model instance."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import advance, build_tables, independent_sets  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def diag(cfg):
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)
    root = p.full_mask

    problems = []
    for st in range(1 << len(edges)):
        ind = sorted(tuple(sorted(edges[i] for i in M)) for M in independent_sets(n, edges, st))
        sol = sorted(p.matching_tuple(M) for M in p.all_matchings(st))
        if ind != sol:
            problems.append(f"matchings differ at {bin(st)}")
        for M in p.all_matchings(st):
            Midx = tuple(i for i in range(len(edges)) if (M >> i) & 1)
            for S, prob, cnt in p.success_subsets(M):
                SetS = frozenset(i for i in range(len(edges)) if (S >> i) & 1)
                if p.transition(st, M, S) != advance(n, edges, st, Midx, SetS):
                    problems.append(f"transition differs at {bin(st)} M={Midx}")
        for h in (1, 2, 3):
            for method in ("G", "R", "OPT"):
                ai = tuple(sorted(edges[i] for i in action(st, h, method)))
                if method == "G":
                    asol = p.matching_tuple(s.greedy_action(st))
                elif method == "R":
                    asol = p.matching_tuple(s.R_action(st, h)[0])
                else:
                    asol = p.matching_tuple(s.OPT_action(st, h)[0])
                if ai != asol:
                    problems.append(f"action differs {method} h={h} state={bin(st)} "
                                    f"ind={ai} sol={asol}")
                vi = value(st, h, method)
                vsol = {"G": s.V_G, "R": s.V_R, "OPT": s.V_OPT}[method](st, h)
                if abs(vi - vsol) > 1e-9:
                    problems.append(f"value differs {method} h={h} state={bin(st)} "
                                    f"ind={vi} sol={vsol}")
    return problems, value, s, p, root


def main() -> int:
    configs = generate_configs()
    by_id = {c["config_id"]: (i, c) for i, c in enumerate(configs)}
    for cid in ("n6_m6_s0_HOM", "n6_m6_s0_HET", "n6_m6_s1_HOM", "n6_m6_s1_HET"):
        idx, cfg = by_id[cid]
        problems, value, s, p, root = diag(cfg)
        print(f"\n  {cid}: {len(problems)} table problems")
        for pr in problems[:6]:
            print(f"    {pr}")
        print(f"    V_OPT root: ind {value(root,3,'OPT'):.10f} sol {s.V_OPT(root,3):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
