"""M01 validation gate: the four required checks plus the frozen path example.

Required:
  1. a single edge with success probability q has value 2q for every h >= 1 (failure cannot be retried)
  2. one round: V_1^G = V_1^R = V_1^*
  3. two rounds: V_2^R = V_2^*
  4. three rounds: V_3^G <= V_3^R <= V_3^*

Frozen path example (implementation check, NOT a main-experiment result):
  path a-b-c-d with q = (0.4, 0.9, 0.4) on edges (a,b), (b,c), (c,d), two rounds:
  G picks the middle edge first and is worth 1.96 matched vertices;
  R and OPT pick an outer edge first and are worth 2.248.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m01_solver import Problem, Solver  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print("=" * 100)
    print("  M01 VALIDATION GATE")
    print("=" * 100)

    # ---------------------------------------------------------------- check 1
    print("\n  CHECK 1 -- single edge, value must be 2q for every h (no retry after failure)")
    for q in (0.0, 0.25, 0.4, 0.5, 0.9, 1.0):
        p = Problem(2, [(0, 1)], [q])
        s = Solver(p)
        vals = {h: (s.V_G(p.full_mask, h), s.V_R(p.full_mask, h), s.V_OPT(p.full_mask, h))
                for h in (1, 2, 3, 5)}
        ok = all(abs(v[0] - 2 * q) < 1e-12 and abs(v[1] - 2 * q) < 1e-12
                 and abs(v[2] - 2 * q) < 1e-12 for v in vals.values())
        detail = (f"q={q}: 2q={2*q}, h=1 {vals[1]}, h=2 {vals[2]}, h=3 {vals[3]}")
        check(f"single edge q={q}", ok, detail)

    # ---------------------------------------------------------------- checks 2-4
    print("\n  CHECK 2-4 -- ordering across horizons on a set of small instances")
    instances = [
        ("P4 path q=.4/.9/.4", 4, [(0, 1), (1, 2), (2, 3)], [0.4, 0.9, 0.4]),
        ("triangle q=.5", 3, [(0, 1), (0, 2), (1, 2)], [0.5, 0.5, 0.5]),
        ("disjoint edges", 4, [(0, 1), (2, 3)], [0.3, 0.8]),
        ("triangle+pendant", 4, [(0, 1), (0, 2), (1, 2), (2, 3)], [0.5, 0.6, 0.4, 0.7]),
        ("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ("star4", 4, [(0, 1), (0, 2), (0, 3)], [0.9, 0.5, 0.2]),
        ("two triangles", 6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
         [0.4, 0.6, 0.5, 0.5, 0.5, 0.9]),
    ]
    for name, n, edges, probs in instances:
        p = Problem(n, sorted(edges), probs)
        s = Solver(p)
        g1, r1, o1 = s.V_G(p.full_mask, 1), s.V_R(p.full_mask, 1), s.V_OPT(p.full_mask, 1)
        s2 = Solver(p)
        g2, r2, o2 = s2.V_G(p.full_mask, 2), s2.V_R(p.full_mask, 2), s2.V_OPT(p.full_mask, 2)
        s3 = Solver(p)
        g3, r3, o3 = s3.V_G(p.full_mask, 3), s3.V_R(p.full_mask, 3), s3.V_OPT(p.full_mask, 3)
        eps = 1e-9
        check(f"{name} | h=1 G=R=OPT", abs(g1 - r1) < eps and abs(r1 - o1) < eps,
              f"G={g1:.12f} R={r1:.12f} OPT={o1:.12f}")
        check(f"{name} | h=2 R=OPT", abs(r2 - o2) < eps,
              f"R={r2:.12f} OPT={o2:.12f} G={g2:.12f}")
        check(f"{name} | h=3 G<=R<=OPT",
              g3 <= r3 + eps and r3 <= o3 + eps,
              f"G={g3:.12f} R={r3:.12f} OPT={o3:.12f}")

    # ---------------------------------------------------------------- frozen path
    print("\n  FROZEN PATH EXAMPLE -- a-b-c-d, q=(0.4, 0.9, 0.4), TWO rounds")
    p = Problem(4, [(0, 1), (1, 2), (2, 3)], [0.4, 0.9, 0.4])
    s = Solver(p)
    root = p.full_mask
    g_act = p.matching_tuple(s.greedy_action(root))
    M_r, argmax_r, best_q = s.R_action(root, 2)
    r_act = p.matching_tuple(M_r)
    M_o, o_val = s.OPT_action(root, 2)
    o_act = p.matching_tuple(M_o)
    vg, vr, vo = s.V_G(root, 2), s.V_R(root, 2), s.V_OPT(root, 2)
    print(f"    G   first action {g_act}   V_2^G = {vg:.6f}")
    print(f"    R   first action {r_act}   V_2^R = {vr:.6f}")
    print(f"    OPT first action {o_act}   V_2^* = {vo:.6f}")
    print(f"    R argmax set (within eps): "
          f"{[p.matching_tuple(M) for M in argmax_r]}")
    check("path: G picks the middle edge", g_act == ((1, 2),), f"{g_act}")
    check("path: G value is 1.96", abs(vg - 1.96) < 1e-9, f"{vg:.12f}")
    # R's optimum is the matching of BOTH outer edges; either outer edge alone also counts as
    # "an outer-edge action", and the middle edge must not be used.
    check("path: R picks outer edge(s), never the middle edge",
          len(r_act) >= 1 and all(e in ((0, 1), (2, 3)) for e in r_act),
          f"{r_act}")
    check("path: R value is 2.248", abs(vr - 2.248) < 1e-9, f"{vr:.12f}")
    check("path: OPT value is 2.248", abs(vo - 2.248) < 1e-9, f"{vo:.12f}")
    check("path: R = OPT at h=2", abs(vr - vo) < 1e-9, f"{vr:.12f} vs {vo:.12f}")

    print("\n" + "=" * 100)
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"    - {f}")
        return 1
    print("  ALL VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
