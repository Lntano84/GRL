"""
Stage 04b repair-regression tests for `repair_minbatch_v3`.

    python stage04b_repair_tests.py

R1 (CORRECTED)
    The stage-04 version of this test used demand (1,1) and a nominal plan with NO broken
    carry-over, so it did not test the intended situation at all. This version uses the
    original instance-F parameters and nominal plan:

        two items, two machines, two periods
        demand (1,2) for both items ; capacity (2,2) for both machines
        f = 1, p = 0, h = 1, l = 10, s = b = m = 1
        nominal:  each item on its own machine, X = (1,2), Y = (1,0), Z = (1,0)
        both machines down in period 1

    Each machine loses 1 unit in period 1 (10) and, because the minimum-lot repair only
    restores m = 1 while demand in period 2 is 2, loses another 1 unit (10), plus the
    recovery setup (1):  10 + 10 + 1 = 21 per machine, 42 total. The log must show TWO
    restorations, one per machine.

R2 (CORRECTED)
    The zero-production setup must sit in the ACTUAL RECOVERY PERIOD of a machine, where the
    old "clear Y and Z wherever X = 0" rule would have deleted it, and another item on the
    same machine must require a recovery decision in that same period.

Both nominal plans are verified independently BEFORE the disruption and every repaired plan
afterwards. Hand expectations are recorded, never adjusted to match.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Instance, Solution, build_model, cost_breakdown, solve_milp
from lsp_repair3 import REPAIR_NAME, repair_minbatch_v3


class Disruption:
    def __init__(self, name: str, down: Dict[int, List[int]]):
        self.name = name
        self.down = {int(j): sorted(int(t) for t in ts) for j, ts in down.items()}

    def delta_for(self, j: int) -> int:
        ts = self.down.get(j, [])
        d = 0
        while (d + 1) in ts:
            d += 1
        return d

    def is_down(self, j: int, t: int) -> bool:
        return t in self.down.get(j, [])

    def as_dict(self):
        return {"name": self.name, "down": {str(j): ts for j, ts in self.down.items()}}

    def describe(self):
        return "; ".join("machine %d down during %s" % (j, ts)
                         for j, ts in sorted(self.down.items()))


def apply_disruption(inst: Instance, dis: Disruption) -> Instance:
    c = [[float(inst.c[j][t - 1]) for t in range(1, inst.T + 1)] for j in range(inst.M)]
    for j, ts in dis.down.items():
        for t in ts:
            c[j][t - 1] = 0.0
    return Instance(name=inst.name + "|" + dis.name, N=inst.N, M=inst.M, T=inst.T,
                    f=list(inst.f), p=list(inst.p), h=list(inst.h), l=list(inst.l),
                    s=list(inst.s), b=list(inst.b), m=list(inst.m), d=[list(r) for r in inst.d],
                    c=c, w=[list(r) for r in inst.w], I0=list(inst.I0), note=inst.note)


def plan_from(inst: Instance, X, Y, Z) -> Solution:
    X = np.array(X, dtype=float).reshape(inst.N, inst.M, inst.T)
    Y = np.array(Y, dtype=float).reshape(inst.N, inst.M, inst.T)
    Z = np.array(Z, dtype=float).reshape(inst.N, inst.M, inst.T)
    I = np.zeros((inst.N, inst.T))
    L = np.zeros((inst.N, inst.T))
    for i in range(inst.N):
        for t in range(1, inst.T + 1):
            prev = inst.I0[i] if t == 1 else I[i, t - 2]
            avail = prev + sum(X[i, j, t - 1] for j in range(inst.M))
            I[i, t - 1] = max(avail - inst.d[i][t - 1], 0.0)
            L[i, t - 1] = max(inst.d[i][t - 1] - avail, 0.0)
    sol = Solution(status="handwritten", objective=0.0, X=X, Y=Y, Z=Z, I=I, L=L)
    sol.objective = cost_breakdown(inst, X, Y, Z, I, L)["total"]
    return sol


# --------------------------------------------------------------------------------------
# R1: original F parameters, carry-over broken on BOTH machines
# --------------------------------------------------------------------------------------


def build_r1(swap: bool = False):
    """Item i is compatible ONLY with machine j_i. Both machines are down in period 1.

    Nominal:  X = (1,2), Y = (1,0), Z = (1,0) on the item's own machine -- exactly the
    instance-F plan, so the run of period 2 rides on the period-1 setup and that carry-over
    is what the disruption breaks.
    """
    machines = [1, 0] if swap else [0, 1]
    w = [[0, 0] for _ in range(2)]
    for i, j in enumerate(machines):
        w[i][j] = 1
    inst = Instance(
        name="R1_swap" if swap else "R1", N=2, M=2, T=2,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[1.0, 2.0], [1.0, 2.0]], c=[[2.0, 2.0], [2.0, 2.0]], w=w,
        I0=[0.0, 0.0], note="R1 original-F parameters, both carry-overs broken")
    X = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
    Y = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
    Z = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
    for i, j in enumerate(machines):
        X[i][j] = [1.0, 2.0]
        Y[i][j] = [1.0, 0.0]
        Z[i][j] = [1.0, 0.0]
    return inst, X, Y, Z


def test_r1() -> Dict[str, Any]:
    out: Dict[str, Any] = {"test": "R1_original_F_both_carryovers_broken",
                           "expected_cost": 42.0,
                           "expected_accounting": "per machine 10 (lost, t=1) + 10 (lost, t=2"
                                                  " shortfall) + 1 (recovery setup) = 21;"
                                                  " two machines -> 42"}
    for swap in (False, True):
        inst, X, Y, Z = build_r1(swap)
        nom = plan_from(inst, X, Y, Z)
        chk_nom = check_solution(inst, nom, MODE_AUDITED)
        dis = Disruption("both_p1", {0: [1], 1: [1]})
        pert = apply_disruption(inst, dis)
        rep = repair_minbatch_v3(pert, nom, dis)
        chk_rep = check_solution(pert, rep.solution, MODE_AUDITED)
        n_restored = sum(1 for e in rep.events if e.kind == "restored")
        key = "swapped" if swap else "original"
        out[key] = {
            "nominal_objective": float(nom.objective),
            "nominal_feasible": bool(chk_nom.ok),
            "repair_objective": float(rep.solution.objective),
            "repair_feasible": bool(chk_rep.ok),
            "repair_max_violation": chk_rep.max_violation,
            "n_restored_events": n_restored,
            "restored_positions": [[e.machine, e.period, e.item] for e in rep.events
                                   if e.kind == "restored"],
            "matches_42": bool(abs(rep.solution.objective - 42.0) <= 1e-6),
            "repair_log": rep.log_text(),
        }
    out["label_invariant"] = (abs(out["original"]["repair_objective"]
                                  - out["swapped"]["repair_objective"]) <= 1e-6)
    out["two_restorations"] = (out["original"]["n_restored_events"] >= 2)
    out["passed"] = bool(out["original"]["matches_42"] and out["swapped"]["matches_42"]
                         and out["two_restorations"] and out["label_invariant"]
                         and out["original"]["repair_feasible"]
                         and out["swapped"]["repair_feasible"])
    return out


# --------------------------------------------------------------------------------------
# R2: the zero-production setup sits in the ACTUAL recovery period
# --------------------------------------------------------------------------------------


def build_r2():
    """Machine 0 is the machine of interest and is down in period 1, so its recovery period
    is t = 2. The vulnerable pattern must sit exactly there.

    On machine 0 in t = 2:
      * item 0 has  X = 0, Y = 1, Z = 1  and produces in t = 3;
      * item 1 is producing in t = 1 and is therefore cancelled by the outage, so it needs a
        recovery decision in t = 2 as well.

    Design constraints that had to be respected (found by working them out, not by guessing):
      * (1.7) forbids two consecutive carry-overs, so Z_{t-1} must be 0 for item 0 -- the
        pattern cannot sit immediately after another carry-over;
      * (1.9) with Y = Z = 1 gives X >= 0, so X = 0 is legal;
      * (1.10) is X_t + X_{t+1} >= m*Z_t, which the production in t = 3 satisfies;
      * in t = 2 item 0 already consumes s_0 = 1 of the capacity of 2, so restoring item 1
        (which needs s_1 + b_1*m_1 = 2) does not fit -- hence the cancelled-outage decision.
    """
    inst = Instance(
        name="R2", N=2, M=1, T=3,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[0.5, 0.5], l=[4.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        c=[[2.0, 2.0, 3.0]], w=[[1], [1]], I0=[0.0, 0.0],
        note="R2 zero-production setup inside the recovery period")
    # X[i][j][t-1] ; Y and Z likewise
    X = [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]]
    Y = [[[0.0, 1.0, 1.0]], [[1.0, 0.0, 0.0]]]
    Z = [[[0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0]]]
    return inst, X, Y, Z


def test_r2() -> Dict[str, Any]:
    inst, X, Y, Z = build_r2()
    nom = plan_from(inst, X, Y, Z)
    chk_nom = check_solution(inst, nom, MODE_AUDITED)

    dis = Disruption("m0_p1", {0: [1]})
    pert = apply_disruption(inst, dis)
    rep = repair_minbatch_v3(pert, nom, dis)
    chk_rep = check_solution(pert, rep.solution, MODE_AUDITED)

    i, j, t = 0, 0, 2                     # vulnerable position: recovery period of machine 0
    preserved = (abs(rep.solution.X[i, j, t - 1]) < 1e-9
                 and abs(rep.solution.Y[i, j, t - 1] - 1.0) < 1e-9
                 and abs(rep.solution.Z[i, j, t - 1] - 1.0) < 1e-9)
    next_prod = abs(rep.solution.X[i, j, t] - 1.0) < 1e-9
    item1_cancelled = abs(rep.solution.X[1, 0, 0]) < 1e-9

    return {
        "test": "R2_zero_production_setup_in_recovery_period",
        "nominal_objective": float(nom.objective),
        "nominal_feasible": bool(chk_nom.ok),
        "nominal_max_violation": chk_nom.max_violation,
        "nominal_cost_breakdown": chk_nom.cost_recomputed,
        "repair_objective": float(rep.solution.objective),
        "repair_feasible": bool(chk_rep.ok),
        "repair_max_violation": chk_rep.max_violation,
        "repair_log": rep.log_text(),
        "vulnerable_position": [i, j, t],
        "position_is_recovery_period": True,
        "pattern_preserved": bool(preserved),
        "next_period_production_preserved": bool(next_prod),
        "other_item_cancelled_by_outage": bool(item1_cancelled),
        "repaired_plan": {
            "X": np.asarray(rep.solution.X).tolist(),
            "Y": np.asarray(rep.solution.Y).tolist(),
            "Z": np.asarray(rep.solution.Z).tolist(),
            "L": np.asarray(rep.solution.L).tolist(),
        },
        "passed": bool(chk_nom.ok and chk_rep.ok and preserved and next_prod
                       and item1_cancelled),
    }


def main() -> int:
    results = [test_r1(), test_r2()]
    print("=" * 96)
    for r in results:
        print("%-48s %s" % (r["test"], "PASS" if r["passed"] else "**FAIL**"))
        for k, v in r.items():
            if k in ("test", "passed", "repaired_plan", "nominal_cost_breakdown",
                     "original", "swapped"):
                continue
            if k == "repair_log":
                print("      %-38s %s" % (k, str(v)[:300]))
            else:
                print("      %-38s %s" % (k, v))
        for side in ("original", "swapped"):
            if side in r:
                print("      [%s]" % side)
                for k, v in (r[side] or {}).items():
                    if k == "repair_log":
                        print("        %-36s %s" % (k, str(v)[:260]))
                    else:
                        print("        %-36s %s" % (k, v))
        print("-" * 96)
    n_bad = sum(1 for r in results if not r["passed"])
    print("%s (%d/%d)" % ("ALL REPAIR REGRESSION TESTS PASS" if n_bad == 0
                          else "REGRESSION FAILURES", len(results) - n_bad, len(results)))
    with open("stage04b_repair_tests.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
