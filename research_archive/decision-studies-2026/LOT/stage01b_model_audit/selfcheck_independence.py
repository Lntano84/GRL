"""Falsification tests for the independence claims made in stage 01b.

An independence claim is only meaningful if you can show what would happen when it fails.
Each test below corrupts one thing and checks WHO notices. Two outcomes are informative:

  * the checker/path notices           -> it is genuinely independent of the corruption
  * the checker/path does NOT notice   -> it is NOT independent, and the claim is false

Run:  python selfcheck_independence.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np

import lsp_checker
import lsp_independent
import lsp_model
from lsp_checker import check_solution
from lsp_independent import (check_binary_logic, evaluate_from_model_solution,
                            independent_cost_breakdown)
from lsp_instances import hand_instances, micro_instances
from lsp_model import (MODE_AUDITED, MODE_PAPER, assignment_of_solution, build_model,
                       solve_lp_with_binaries_fixed, solve_milp)

RESULTS: List[Dict[str, Any]] = []


def record(name: str, expectation: str, observed: str, ok: bool) -> None:
    RESULTS.append({"name": name, "expected": expectation, "observed": observed, "ok": ok})
    print("%-58s %s" % (name, "OK" if ok else "*** CLAIM FALSIFIED ***"))
    print("      expectation: %s" % expectation)
    print("      observed   : %s" % observed)


def main() -> int:
    inst = hand_instances()["A_ample"]
    mode = MODE_AUDITED
    bm = build_model(inst, mode)
    sol = solve_milp(bm)
    print("reference: solver obj=%s\n" % sol.objective)

    # ==================================================================================
    # 1. Is the checker's cost function its own, or does it call lsp_model's?
    #    Corrupt lsp_model.cost_breakdown and COUNT the calls.
    # ==================================================================================
    print("--- 1. checker cost independence ---")
    calls = {"n": 0}
    orig_cb = lsp_model.cost_breakdown

    def counting_cb(*a, **k):
        calls["n"] += 1
        out = dict(orig_cb(*a, **k))
        out["total"] += 7.0
        out["setup"] += 7.0
        return out

    lsp_model.cost_breakdown = counting_cb
    try:
        rep = check_solution(inst, sol, mode)
        ck = lsp_checker._independent_cost_breakdown(inst, sol.X, sol.Y, sol.Z, sol.I, sol.L)
    finally:
        lsp_model.cost_breakdown = orig_cb

    ok = (calls["n"] == 0) and abs(ck["total"] - sol.objective) < 1e-9
    record("checker does not route through lsp_model.cost_breakdown",
           "0 calls to the corrupted model helper, and the checker's own total still "
           "equals the solver objective (%s)" % sol.objective,
           "%d call(s); checker total=%s" % (calls["n"], ck["total"]), ok)

    # ==================================================================================
    # 2. Does the checker recompute U, or read it from the model?
    #    Give a model a LARGER U than the formulation allows; the checker must reject the
    #    resulting plan against its own, correct U.
    #    The instance must be one where (1.8) is actually BINDING; on instance A the
    #    minimum in U is attained by the demand term, so a scaled U is still non-binding
    #    there and nothing can be detected.
    # ==================================================================================
    print("\n--- 2. checker U independence ---")
    inst_f = hand_instances()["F_carryover"]
    f_paper = solve_milp(build_model(inst_f, MODE_PAPER))
    orig_ub = lsp_model.activation_upper_bounds

    def inflated_ub(inst_, mode_):
        upper, notes = orig_ub(inst_, mode_)
        return {k: v * 3.0 for k, v in upper.items()}, notes

    lsp_model.activation_upper_bounds = inflated_ub
    try:
        bm_bad = build_model(inst_f, MODE_PAPER)   # allows X up to 3*U = 6
        sol_bad = solve_milp(bm_bad)
        rep_bad = check_solution(inst_f, sol_bad, MODE_PAPER)  # checker uses the true U
    finally:
        lsp_model.activation_upper_bounds = orig_ub

    act = rep_bad.checks["activation"]
    ok = (sol_bad.objective < f_paper.objective - 1e-9) and (act["ok"] is False)
    record("checker rejects a plan built with an inflated U_ijt",
           "on paper_literal F the true U=1 is binding (obj 11); with U inflated to 3 the "
           "model returns a cheaper but illegal plan, and the checker flags an activation "
           "violation against its own U",
           "true obj=%s, inflated obj=%s, checker activation ok=%s (max viol %.3g)"
           % (f_paper.objective, sol_bad.objective, act["ok"], act["max_violation"]), ok)

    # ==================================================================================
    # 3. Is the independent LP really independent of bm.A?
    #    Corrupt a coefficient in bm.A. The shared-matrix path must reproduce the corrupted
    #    model; the independent LP (which never reads bm.A) must still return the true value.
    # ==================================================================================
    print("\n--- 3. independent LP vs shared matrix ---")
    bm3 = build_model(inst, mode)
    A = bm3.A.tolil()
    r = bm3.row_index("capacity[j0,t1]")
    A[r, bm3.idx.X(0, 0, 1)] += 0.5                 # c_jt effectively becomes 5.5
    bm3.A = A.tocsr()
    sol3 = solve_milp(bm3)
    shared_status, shared_obj, _ = solve_lp_with_binaries_fixed(
        bm3, assignment_of_solution(sol3, inst, mode))
    ev3 = evaluate_from_model_solution(inst, mode, sol3)

    shared_follows = abs(shared_obj - sol3.objective) < 1e-6
    indep_true = ev3["status"] == "optimal" and abs(ev3["objective"] - sol.objective) < 1e-6
    solver_differs = abs(sol3.objective - sol.objective) > 1e-9
    ok = shared_follows and indep_true and solver_differs
    record("independent LP ignores a corrupted bm.A; shared path does not",
           "corrupted model gives a different objective (%s vs %s); the shared-matrix LP "
           "reproduces the corrupted value; the independent LP still returns %s"
           % (sol3.objective, sol.objective, sol.objective),
           "corrupted MILP obj=%s (differs=%s); shared LP obj=%s (follows=%s); "
           "independent LP obj=%s (true=%s)"
           % (sol3.objective, solver_differs, shared_obj, shared_follows,
              ev3["objective"], indep_true), ok)
    print("      => this is exactly why the shared-matrix path is labelled a "
          "'shared-matrix enumeration check', not an independent one.")

    # ==================================================================================
    # 4. Does the independent path check the binary logic itself?
    #    Feed it a pattern with Z=1 while Y=0 (violating (1.6)) and see it refuse.
    #    This is run under paper_literal: under audited_v1 any Z at t = T is itself
    #    infeasible, so the test would pass for the wrong reason.
    # ==================================================================================
    print("\n--- 4. independent binary-rule check ---")
    Y_ok = [[[1.0]]]
    Y_bad = [[[0.0]]]
    Z_one = [[[1.0]]]
    st_bad, _, _, viol_bad = lsp_independent.solve_independent_lp(inst, MODE_PAPER,
                                                                  Y_bad, Z_one)
    st_ok, obj_ok, _, viol_ok = lsp_independent.solve_independent_lp(inst, MODE_PAPER,
                                                                     Y_ok, Z_one)
    ok = (st_bad == "binary_infeasible") and (st_ok == "optimal")
    record("independent path rejects Z=1 with Y=0, accepts Z=1 with Y=1",
           "(1.6) violated -> binary_infeasible naming the rule; (1.6) respected -> optimal",
           "Y=0,Z=1 -> %s (%s); Y=1,Z=1 -> %s obj=%s"
           % (st_bad, (viol_bad or ["-"])[0], st_ok, obj_ok), ok)

    # ==================================================================================
    # 5. Full independent enumeration on every micro instance must agree with the MILP.
    # ==================================================================================
    print("\n--- 5. independent enumeration vs MILP on all micro instances ---")
    bad = []
    for name, mi in micro_instances().items():
        bmi = build_model(mi, mode)
        soli = solve_milp(bmi)
        evi = evaluate_from_model_solution(mi, mode, soli)
        d = abs(evi["objective"] - soli.objective) if evi["status"] == "optimal" else np.inf
        if d > 1e-6 * (1 + abs(soli.objective)):
            bad.append((name, soli.objective, evi["objective"]))
    record("independent re-evaluation of MILP binaries, 8/8 micro instances",
           "every MILP plan is binary-feasible and re-optimises to the same objective",
           "mismatches: %s" % (bad or "none"), not bad)

    n_bad = sum(1 for r in RESULTS if not r["ok"])
    print("\n%s  (%d/%d claims hold)" % (
        "ALL INDEPENDENCE CLAIMS SUPPORTED" if n_bad == 0 else "CLAIMS FALSIFIED",
        len(RESULTS) - n_bad, len(RESULTS)))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
