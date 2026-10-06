"""Localise the disagreement between the recursive solver and the independent enumeration."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    configs = generate_configs()
    cfg = {c["config_id"]: c for c in configs}["n6_m6_s1_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    print(f"config {cfg['config_id']}: n={n} edges={edges} q={probs}")

    p = Problem(n, edges, probs)
    s = Solver(p)

    # every edge subset is a reachable state
    states = [m for m in range(1 << len(edges))]
    mismatches = []
    for h in (1, 2, 3):
        for st in states:
            if st == 0:
                continue
            # solver's action
            if h == 1:
                a_solver = s.greedy_action(st)
            else:
                a_solver, _, _ = s.R_action(st, h)
            # brute-force action with the solver's own value functions (no independent code)
            best = None
            for M in p.all_matchings(st):
                v = s.Qtilde(st, M, h)
                key = p.matching_tuple(M)
                if best is None or v > best[0] + s.eps or (
                        abs(v - best[0]) <= s.eps and key < best[2]):
                    best = (v, M, key)
            a_bf = best[1]
            if p.matching_tuple(a_solver) != p.matching_tuple(a_bf):
                mismatches.append((h, st, p.matching_tuple(a_solver), p.matching_tuple(a_bf)))
    print(f"\n  G/R action mismatches between solver's R_action and its own brute force: "
          f"{len(mismatches)}")
    for m in mismatches[:10]:
        print(f"    h={m[0]} state={bin(m[1])} solver={m[2]} bruteforce={m[3]}")

    # now compare the solver's V_OPT against an explicit policy evaluation of OPT_action
    print("\n  V_OPT check: recursion vs explicit evaluation of the OPT policy")
    for st in states:
        if st == 0:
            continue
        a_opt, v_opt = s.OPT_action(st, 3)
        # explicit: sum over outcomes, recurse by re-deriving OPT actions
        total = 0.0
        for S, prob, cnt in p.success_subsets(a_opt):
            nxt = p.transition(st, a_opt, S)
            if nxt == 0:
                total += prob * (cnt + 0.0)
            else:
                _, v2 = s.OPT_action(nxt, 2)
                total += prob * (cnt + v2)
        if abs(total - v_opt) > 1e-9:
            print(f"    MISMATCH state={bin(st)} opts={p.matching_tuple(a_opt)} "
                  f"explicit={total:.10f} recursion={v_opt:.10f}")
    print("    (no output above means every state agrees)")

    # and the same for the root, printing the full breakdown
    root = p.full_mask
    print("\n  ROOT breakdown (OPT, h=3):")
    a_opt, v_opt = s.OPT_action(root, 3)
    print(f"    OPT first action {p.matching_tuple(a_opt)}, V_OPT={v_opt:.10f}")
    for S, prob, cnt in p.success_subsets(a_opt):
        nxt = p.transition(root, a_opt, S)
        v2 = s.V_OPT(nxt, 2)
        print(f"      S={p.matching_tuple(S)} p={prob:.4f} reward={cnt} next={bin(nxt)} "
              f"V2*={v2:.6f}")
    print("\n  ROOT breakdown (R, h=3):")
    a_r, argmax_r, best_r = s.R_action(root, 3)
    print(f"    R first action {p.matching_tuple(a_r)}, lookahead={best_r:.10f}")
    for S, prob, cnt in p.success_subsets(a_r):
        nxt = p.transition(root, a_r, S)
        print(f"      S={p.matching_tuple(S)} p={prob:.4f} reward={cnt} next={bin(nxt)} "
              f"V_G(next,2)={s.V_G(nxt,2):.6f}  V_R(next,2)={s.V_R(nxt,2):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
