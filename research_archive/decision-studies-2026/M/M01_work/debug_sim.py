"""Compare the independent simulation against the solver's own policy evaluation, step by step."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import advance, build_tables, simulate  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    cfg = {c["config_id"]: c for c in generate_configs()}["n6_m6_s1_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)
    root = p.full_mask
    print(f"edges={edges} q={probs}")

    print("\n  solver's own evaluation of the OPT and R rules, decomposed by first action")
    for method, act_fn, cont_fn in (
            ("OPT", lambda st, h: s.OPT_action(st, h)[0], lambda st, h: s.V_OPT(st, h)),
            ("R", lambda st, h: s.R_action(st, h)[0], lambda st, h: s.V_R(st, h))):
        M = act_fn(root, 3)
        print(f"    {method}: first action {p.matching_tuple(M)}")
        tot = 0.0
        for S, prob, cnt in p.success_subsets(M):
            nxt = p.transition(root, M, S)
            v = cont_fn(nxt, 2)
            tot += prob * (cnt + v)
            print(f"      S={p.matching_tuple(S)} p={prob:.4f} rew={cnt} next={bin(nxt)} "
                  f"cont={v:.6f}")
        print(f"      weighted total = {tot:.10f}   (solver V_{method}(root,3)="
              f"{cont_fn(root,3):.10f})")

    print("\n  independent simulation vs solver V, per method")
    for method, vsol in (("G", s.V_G(root, 3)), ("R", s.V_R(root, 3)), ("OPT", s.V_OPT(root, 3))):
        ve = value(root, 3, method)
        # explicit expectation using the independent tables, expanded by hand
        M = action(root, 3, method)
        tot = 0.0
        Midx = tuple(i for i in range(len(edges)) if (M >> i) & 1) if isinstance(M, int) else M
        for r in range(len(Midx) + 1):
            for succ in itertools.combinations(Midx, r):
                prob = 1.0
                for i in Midx:
                    prob *= probs[i] if i in succ else (1.0 - probs[i])
                nxt = advance(n, edges, root, Midx, frozenset(succ))
                tot += prob * (2 * len(succ) + value(nxt, 2, method))
        print(f"    {method}: root action(ind)={tuple(edges[i] for i in Midx)} "
              f"expanded={tot:.10f} value()={ve:.10f} solver={vsol:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
