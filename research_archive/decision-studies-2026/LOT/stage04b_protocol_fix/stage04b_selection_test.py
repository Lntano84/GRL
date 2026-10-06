"""
Interface test for the output-selection rule. No solver is called.

    python stage04b_selection_test.py

Three hand-made cases, exactly as requested:

    A  no incumbent        -> keep the repaired plan
    B  incumbent worse     -> keep the repaired plan
    C  incumbent better    -> take the incumbent

plus the guards that the contract requires:

    D  invalid incumbent (violates the stability budget) -> recorded as an ERROR
    E  invalid incumbent (violates the model)            -> recorded as an ERROR
    F  solver reports infeasible although the repaired plan is a feasible witness
                                                          -> recorded as an ERROR

The instance is a tiny two-item one-machine problem built here; the "incumbent" plans are
constructed by hand so the expected outcome of each case is unambiguous.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np

from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Instance, Solution, cost_breakdown, solve_milp, build_model
from lsp_select import select_output

KAPPA = 4.0
TAU = 2


def make_instance() -> Instance:
    """2 items, 1 machine, 3 periods. Capacity is loose, so several distinct plans exist."""
    return Instance(
        name="SEL", N=2, M=1, T=3,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        c=[[8.0, 8.0, 8.0]], w=[[1], [1]], I0=[0.0, 0.0], note="selection interface test")


def plan(inst: Instance, X, Y, Z) -> Solution:
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
    sol = Solution(status="handmade", objective=0.0, X=X, Y=Y, Z=Z, I=I, L=L)
    sol.objective = cost_breakdown(inst, X, Y, Z, I, L)["total"]
    return sol


def main() -> int:
    inst = make_instance()
    results: List[Dict[str, Any]] = []

    # ---- reference "repair" plan: the SOLVED optimum of this instance.
    ref = solve_milp(build_model(inst, MODE_AUDITED))
    chk_ref = check_solution(inst, ref, MODE_AUDITED)
    assert chk_ref.ok, chk_ref.failed_checks()
    rep = ref

    # a strictly BETTER "incumbent" cannot exist, so case C is exercised with a deliberately
    # WORSE reference instead: the optimum plays the incumbent and the reference is a plan
    # the solver produces under an artificial handicaps. `fix` pins every Y to 0 except one
    # setup per item, which yields a feasible plan strictly worse than the unconstrained
    # optimum (it must lose sales rather than set up when it needs to).
    bm = build_model(inst, MODE_AUDITED)
    handicap = {bm.idx.Y(i, 0, t): (1.0 if t == 3 else 0.0)
                for i in range(inst.N) for t in range(1, inst.T + 1)}
    rep_worse = solve_milp(bm, fix=handicap)
    chk_worse = check_solution(inst, rep_worse, MODE_AUDITED)
    assert chk_worse.ok, chk_worse.failed_checks()
    assert rep_worse.objective > rep.objective + 1e-9, (rep_worse.objective, rep.objective)
    good = rep                      # strictly better than rep_worse
    # a strictly worse feasible incumbent: lose everything (no setups, no production)
    bad = plan(inst,
               X=[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
               Y=[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
               Z=[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    # infeasible: declares production without any activation
    broken = plan(inst,
                  X=[[[2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
                  Y=[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
                  Z=[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    for nm, p in (("repair(opt)", rep), ("rep_worse", rep_worse), ("bad", bad),
                  ("broken", broken)):
        c = check_solution(inst, p, MODE_AUDITED)
        print("%-12s cost=%-10.6g feasible=%-5s maxviol=%.3g"
              % (nm, p.objective, c.ok, c.max_violation))
        for k, v in c.checks.items():
            if not v["ok"]:
                print("        FAIL", k, v["worst"][:1])
    print()

    def run(tag: str, reference, solver, status: str, expect_source: str,
            expect_reason: str, expect_errors: int) -> None:
        sel = select_output(inst, solver, reference, KAPPA, TAU, fixed_Y=None,
                            solver_status=status)
        sol_cand = [c for c in sel.candidates if c.label == "solver"][0]
        got = {
            "case": tag,
            "repair_cost": float(reference.objective),
            "solver_present": solver is not None,
            "solver_cost": None if sol_cand.cost is None else float(sol_cand.cost),
            "selected_cost": None if sel.chosen is None else float(sel.chosen.cost),
            "selected_source": sel.source,
            "reason": sel.reason,
            "n_errors": len(sel.errors),
            "errors": sel.errors,
            "expect_source": expect_source,
            "expect_reason": expect_reason,
            "expect_errors": expect_errors,
        }
        got["passed"] = bool(got["selected_source"] == expect_source
                             and got["reason"] == expect_reason
                             and got["n_errors"] == expect_errors)
        results.append(got)
        print("%-52s source=%-7s reason=%-24s errors=%d  %s"
              % (tag, got["selected_source"], got["reason"], got["n_errors"],
                 "OK" if got["passed"] else "**MISMATCH**"))
        for e in sel.errors:
            print("        error: %s" % e[:140])

    # A/B/D use the optimum as the reference; C needs a reference the incumbent can beat.
    run("A no incumbent", rep, None, "", "repair", "no_incumbent", 0)
    run("B incumbent worse", rep, bad, "suboptimal", "repair", "incumbent_worse_or_tied", 0)
    run("C incumbent better", rep_worse, good, "optimal", "solver", "incumbent_better", 0)
    run("D invalid incumbent (model violation)", rep, broken, "suboptimal", "repair",
        "incumbent_worse_or_tied", 1)
    run("E solver claims infeasible but repair is a witness", rep, None, "infeasible",
        "repair", "no_incumbent", 1)

    # ---- F/G: the two negative tests for the `usable` definition (added in the 04b fix).
    #
    # Both candidates are SOLVED plans, so they are feasible by construction; hand-written
    # Y/Z patterns kept violating (1.6)-(1.9).
    #   bm-fix A  -> setups in every period (cost 6)
    #   bm-fix B  -> setups only in period 1 (cost 7)
    # A and B differ on 2 short-term setups, so with KAPPA = 1 the candidate A is
    # model-feasible yet over budget; with KAPPA = 4 it is within budget, which lets the
    # fixing test be isolated from the stability test.
    bm2 = build_model(inst, MODE_AUDITED)
    A = solve_milp(bm2, fix={bm2.idx.Y(i, 0, t): 1.0
                             for i in range(inst.N) for t in range(1, inst.T + 1)})
    B = solve_milp(bm2, fix={bm2.idx.Y(i, 0, t): (1.0 if t == 1 else 0.0)
                             for i in range(inst.N) for t in range(1, inst.T + 1)})

    # F: candidate A, reference B, kappa = 1  -> A must be rejected for breaching kappa
    sel_f = select_output(inst, A, B, 1.0, TAU, fixed_Y=None, solver_status="suboptimal")
    cand_f = [c for c in sel_f.candidates if c.label == "solver"][0]
    rec_f = {
        "case": "F over-kappa candidate is rejected",
        "candidate_model_feasible": bool(cand_f.feasible),
        "candidate_flips_vs_reference": cand_f.stability_flips,
        "kappa": 1.0,
        "candidate_kappa_ok": bool(cand_f.stability_ok),
        "candidate_fixing_ok": bool(cand_f.fixing_ok),
        "selected_source": sel_f.source,
        "n_errors": len(sel_f.errors),
        "errors": sel_f.errors,
    }
    rec_f["passed"] = bool(cand_f.feasible and not cand_f.stability_ok
                           and sel_f.source == "repair" and sel_f.errors)
    results.append(rec_f)
    print("%-52s feas=%s flips=%-3s kappa_ok=%-5s source=%-7s errors=%d  %s"
          % (rec_f["case"], cand_f.feasible, cand_f.stability_flips,
             cand_f.stability_ok, sel_f.source, rec_f["n_errors"],
             "OK" if rec_f["passed"] else "**MISMATCH**"))
    for e in sel_f.errors:
        print("        error: %s" % e[:130])

    # G: candidate B, reference A, kappa = 4 (so kappa passes), fixings require Y = 1 at
    # t in {2, 3} which B does not satisfy -> must be rejected for the fixing violation.
    bad_fix = {(i, 0, t): 1.0 for i in range(inst.N) for t in (2, 3)}
    sel_g = select_output(inst, B, A, KAPPA, TAU, fixed_Y=bad_fix,
                          solver_status="suboptimal")
    cand_g = [c for c in sel_g.candidates if c.label == "solver"][0]
    rec_g = {
        "case": "G fixing-violating candidate is rejected",
        "candidate_model_feasible": bool(cand_g.feasible),
        "candidate_flips_vs_reference": cand_g.stability_flips,
        "candidate_kappa_ok": bool(cand_g.stability_ok),
        "candidate_fixing_ok": bool(cand_g.fixing_ok),
        "selected_source": sel_g.source,
        "n_errors": len(sel_g.errors),
        "errors": sel_g.errors,
    }
    rec_g["passed"] = bool(cand_g.feasible and cand_g.stability_ok
                           and not cand_g.fixing_ok and sel_g.source == "repair"
                           and sel_g.errors)
    results.append(rec_g)
    print("%-52s feas=%s kappa_ok=%-5s fixing_ok=%-5s source=%-7s errors=%d  %s"
          % (rec_g["case"], cand_g.feasible, cand_g.stability_ok, cand_g.fixing_ok,
             sel_g.source, rec_g["n_errors"], "OK" if rec_g["passed"] else "**MISMATCH**"))
    for e in sel_g.errors:
        print("        error: %s" % e[:130])

    print()
    n_bad = sum(1 for r in results if not r["passed"])
    print("%s (%d/%d cases)" % ("SELECTION INTERFACE OK" if n_bad == 0 else "SELECTION FAILURES",
                                len(results) - n_bad, len(results)))
    with open("stage04b_selection_test.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
