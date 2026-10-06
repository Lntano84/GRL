"""Localise the residual gap: compare the table value with the value of ACTUALLY following the policy."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from independent_check import advance, build_tables  # noqa: E402
from m01_solver import Problem, Solver  # noqa: E402
from m01_worker import generate_configs  # noqa: E402


def main() -> int:
    cfg = {c["config_id"]: c for c in generate_configs()}["n6_m6_s1_HOM"]
    n, edges, probs = cfg["n"], cfg["edges"], cfg["probs"]
    p = Problem(n, edges, probs)
    s = Solver(p)
    action, value, cache = build_tables(n, edges, probs)
    root = (1 << len(edges)) - 1

    def policy_value_from(avail0, h0, method):
        """Expectation of ACTUALLY following the policy table from ``avail0`` for ``h0`` rounds."""
        idxs = [i for i in range(len(edges)) if (avail0 >> i) & 1]
        tot = 0.0
        for bits in range(1 << len(idxs)):
            w = 0
            pr = 1.0
            for j, i in enumerate(idxs):
                if (bits >> j) & 1:
                    w |= 1 << i
                    pr *= probs[i]
                else:
                    pr *= 1.0 - probs[i]
            avail, acc = avail0, 0
            for h in range(h0, 0, -1):
                M = action(avail, h, method)
                if not M:
                    continue          # a pass consumes the round; it does not end the process
                succ = frozenset(i for i in M if (w >> i) & 1)
                acc += 2 * len(succ)
                avail = advance(n, edges, avail, M, succ)
            tot += pr * acc
        return tot

    Mroot = action(root, 3, "OPT")
    print(f"root action: {Mroot} -> {[edges[i] for i in Mroot]}")
    print(f"solver OPT root action: {p.matching_tuple(s.OPT_action(root, 3)[0])}")
    print()
    print("  state            table value   solver value   actual-policy value")
    for r in range(len(Mroot) + 1):
        for succ in itertools.combinations(Mroot, r):
            nxt = advance(n, edges, root, Mroot, frozenset(succ))
            tv = value(nxt, 2, "OPT")
            sv = s.V_OPT(nxt, 2)
            pv = policy_value_from(nxt, 2, "OPT")
            flag = "" if abs(tv - pv) < 1e-9 else "   <-- table != actual policy"
            print(f"  {bin(nxt):<14} {tv:>12.6f} {sv:>14.6f} {pv:>20.6f}{flag}")

    print()
    print("  same check at the root itself:")
    for method in ("G", "R", "OPT"):
        tv = value(root, 3, method)
        pv = policy_value_from(root, 3, method)
        sv = {"G": s.V_G, "R": s.V_R, "OPT": s.V_OPT}[method](root, 3)
        print(f"    {method:<4} table={tv:.6f} solver={sv:.6f} actual-policy={pv:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
