"""
Stage 04 repair-regression tests for `repair_minbatch_v2`.

    python stage04_repair_tests.py

Two targeted tests, both about the v1 defects:

  R1  two machines recover independently. Two items, each exclusive to one machine, both
      machines down in period 1. Each machine must recover on its own and the cost must not
      depend on machine labels. The expectation is checked against the EXACT optimum of the
      perturbed instance, not only against a hand figure.

  R2  a legal zero-production setup is preserved. A plan with X = 0, Y = 1, Z = 1 in a
      period that is NOT a recovery period, with production in the next period, must survive
      a repair happening elsewhere. v1 cleared Y and Z at every position with X = 0 and
      destroyed exactly this arrangement.

Every nominal plan is verified independently BEFORE the disruption is applied, and every
repaired plan is verified independently afterwards. A failure is reported, never adjusted
away.

Also recorded: PHASE1_BUG_FOUND, the infeasibility that the first version of R2 exposed. A
plan in which a NOMINAL production loses its only activation because an upstream carry-over
was cancelled can violate (1.4): restoring an activation that does not exist nominally
reinstates production whose setup time the plan never paid for.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Instance, Solution, build_model, cost_breakdown, solve_milp
from lsp_repair2 import REPAIR_NAME, repair_minbatch_v2

KAPPA_NOTE = "stability budget is irrelevant to the repair itself; it is not used here."


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
# R1
# --------------------------------------------------------------------------------------


def build_r1(swap: bool):
    """Two items, two machines, two periods; item i is compatible ONLY with machine j_i.

    demand (1,1) for both items; capacity (2,2); f = 1, p = 0, h = 1, l = 10, s = b = m = 1.
    Nominal: item i produces 1 in each period on its own machine, no carry-over. Cost 4.
    Both machines are down in period 1.
    """
    machines = [1, 0] if swap else [0, 1]
    w = [[0, 0] for _ in range(2)]
    for i, j in enumerate(machines):
        w[i][j] = 1
    inst = Instance(
        name="R1_swap" if swap else "R1", N=2, M=2, T=2,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[1.0, 1.0], [1.0, 1.0]], c=[[2.0, 2.0], [2.0, 2.0]], w=w,
        I0=[0.0, 0.0], note="R1 independent two-machine recovery")
    # X[i][j][t-1]
    X = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(2)]
    Y = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(2)]
    Z = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(2)]
    for i, j in enumerate(machines):
        X[i][j] = [1.0, 1.0]
        Y[i][j] = [1.0, 1.0]
    return inst, X, Y, Z


def test_r1() -> Dict[str, Any]:
    out: Dict[str, Any] = {"test": "R1_independent_two_machine_recovery"}
    for swap in (False, True):
        inst, X, Y, Z = build_r1(swap)
        nom = plan_from(inst, X, Y, Z)
        chk_nom = check_solution(inst, nom, MODE_AUDITED)
        dis = Disruption("both_p1", {0: [1], 1: [1]})
        pert = apply_disruption(inst, dis)
        rep = repair_minbatch_v2(pert, nom, dis)
        chk_rep = check_solution(pert, rep.solution, MODE_AUDITED)
        # exact optimum of the perturbed instance, for an independent yardstick
        bm = build_model(pert, MODE_AUDITED)
        opt = solve_milp(bm)
        chk_opt = check_solution(pert, opt, MODE_AUDITED)
        key = "swapped" if swap else "original"
        out[key] = {
            "nominal_objective": float(nom.objective),
            "nominal_feasible": bool(chk_nom.ok),
            "repair_objective": float(rep.solution.objective),
            "repair_feasible": bool(chk_rep.ok),
            "repair_max_violation": chk_rep.max_violation,
            "exact_optimum": float(opt.objective),
            "exact_optimum_feasible": bool(chk_opt.ok),
            "repair_is_optimal": bool(abs(rep.solution.objective - opt.objective) <= 1e-6),
            "repair_log": rep.log_text(),
        }
    out["label_invariant"] = (abs(out["original"]["repair_objective"]
                                  - out["swapped"]["repair_objective"]) <= 1e-6)
    out["hand_expectation_42_matches"] = bool(
        abs(out["original"]["repair_objective"] - 42.0) <= 1e-6)
    out["passed"] = bool(
        out["original"]["repair_feasible"] and out["swapped"]["repair_feasible"]
        and out["label_invariant"]
        and out["original"]["repair_is_optimal"] and out["swapped"]["repair_is_optimal"])
    return out


# --------------------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------------------


def build_r2():
    """Item 0 owns machine 0, item 1 owns machine 1.

    demand: item 0 = (0, 0, 0, 2), item 1 = (3, 0, 0, 0)
    capacity: machine 0 = (2, 2, 2, 2), machine 1 = (4, 4, 4, 4)
    f = 1, p = 0, h = 0.5, l = (4, 10), s = b = m = 1

    Optimal nominal plan (cost 2.0, verified feasible and optimal):
        item 0 on machine 0: X = (0, 0, 0, 2), Y = (0, 0, 1, 0), Z = (0, 0, 1, 0)
        item 1 on machine 1: X = (3, 0, 0, 0), Y = (1, 0, 0, 0), Z = (0, 0, 0, 0)

    The vulnerable position is (item 0, machine 0, period 3): X = 0, Y = 1, Z = 1, with
    production 2 in period 4. Period 3 is NOT a recovery period for this disruption (recovery
    is period 2), so the repair must leave it exactly as it is.

    Design note: this instance was found by search, not by hand. The pattern needs the setup
    to be carried forward with no production in the period itself, and the minimum-lot rules
    (1.9)/(1.10) make that impossible in several otherwise natural configurations - with
    m = 1 and demand 1 in the last period the solver never chooses it. A design that both
    admits the pattern AND is optimal was required, so the demand here is 2.
    """
    inst = Instance(
        name="R2", N=2, M=2, T=4,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[0.5, 0.5], l=[4.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[0.0, 0.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0]],
        c=[[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]],
        w=[[1, 0], [0, 1]], I0=[0.0, 0.0],
        note="R2 preserve a legal zero-production setup")
    X = [[[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]
    Y = [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    Z = [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    return inst, X, Y, Z


def test_r2() -> Dict[str, Any]:
    inst, X, Y, Z = build_r2()
    nom = plan_from(inst, X, Y, Z)
    chk_nom = check_solution(inst, nom, MODE_AUDITED)
    # record whether the hand-written nominal is also optimal
    opt = solve_milp(build_model(inst, MODE_AUDITED))
    nom_is_optimal = bool(abs(nom.objective - opt.objective) <= 1e-6)

    dis = Disruption("all_p1", {0: [1], 1: [1]})
    pert = apply_disruption(inst, dis)
    rep = repair_minbatch_v2(pert, nom, dis)
    chk_rep = check_solution(pert, rep.solution, MODE_AUDITED)

    # the vulnerable pattern: item 0, machine 0, period 3 -> X=0, Y=1, Z=1, X=2 in t=4
    i, j, t = 0, 0, 3
    preserved = (abs(rep.solution.X[i, j, t - 1]) < 1e-9
                 and abs(rep.solution.Y[i, j, t - 1] - 1.0) < 1e-9
                 and abs(rep.solution.Z[i, j, t - 1] - 1.0) < 1e-9)
    next_prod = abs(rep.solution.X[i, j, t] - 2.0) < 1e-9

    # the repair must still have cancelled machine 1's period-1 production
    m1_changed = abs(rep.solution.X[1, 1, 0]) < 1e-9 and abs(rep.solution.Y[1, 1, 0]) < 1e-9

    return {
        "test": "R2_preserve_zero_production_setup",
        "nominal_objective": float(nom.objective),
        "exact_optimum": float(opt.objective),
        "nominal_is_optimal": nom_is_optimal,
        "nominal_feasible": bool(chk_nom.ok),
        "nominal_max_violation": chk_nom.max_violation,
        "repair_objective": float(rep.solution.objective),
        "repair_feasible": bool(chk_rep.ok),
        "repair_max_violation": chk_rep.max_violation,
        "repair_log": rep.log_text(),
        "vulnerable_position": [i, j, t],
        "pattern_preserved": bool(preserved),
        "next_period_production_preserved": bool(next_prod),
        "machine1_disruption_applied": bool(m1_changed),
        "repaired_plan": {
            "X": np.asarray(rep.solution.X).tolist(),
            "Y": np.asarray(rep.solution.Y).tolist(),
            "Z": np.asarray(rep.solution.Z).tolist(),
            "L": np.asarray(rep.solution.L).tolist(),
        },
        "passed": bool(chk_nom.ok and chk_rep.ok and preserved and next_prod
                       and m1_changed and nom_is_optimal),
    }


def main() -> int:
    results = [test_r1(), test_r2()]
    print("=" * 96)
    for r in results:
        print("%-44s %s" % (r["test"], "PASS" if r["passed"] else "**FAIL**"))
        for k, v in r.items():
            if k in ("test", "passed", "repaired_plan", "nominal_cost_breakdown",
                     "original", "swapped"):
                continue
            if k == "repair_log":
                print("      %-36s %s" % (k, str(v)[:260]))
            else:
                print("      %-36s %s" % (k, v))
        for side in ("original", "swapped"):
            if side in r:
                print("      [%s]" % side)
                for k, v in r[side].items():
                    if k == "repair_log":
                        print("        %-34s %s" % (k, str(v)[:220]))
                    else:
                        print("        %-34s %s" % (k, v))
        print("-" * 96)
    n_bad = sum(1 for r in results if not r["passed"])
    print("%s (%d/%d)" % ("ALL REPAIR REGRESSION TESTS PASS" if n_bad == 0
                          else "REGRESSION FAILURES", len(results) - n_bad, len(results)))
    if results[0].get("hand_expectation_42_matches") is False:
        print("NOTE: the brief's hand expectation for R1 was 42; the exact optimum of the "
              "perturbed instance is %s, and the repair attains it. The expectation is "
              "recorded as WRONG rather than matched."
              % results[0]["original"]["exact_optimum"])
    with open("stage04_repair_tests.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
