"""
Stage 05 pre-flight checks. All three must pass before the 128-run batch.

    .venv-hs/Scripts/python.exe stage05_preflight.py

1. MODEL TRANSFER. Compare the HiGHS model built by `lsp_highs.build_highs` against the
   audited SciPy system, on small instances, INCLUDING an instance with fixed/released
   variables: variable order, objective, bounds, row bounds, integrality and the matrix must
   all agree. Then solve both ways and compare the objective.

2. REPAIR VECTOR FEASIBILITY. The full repaired vector must be feasible for EVERY method's
   subproblem (its fixings + the stability row). Checked by re-running the independent
   checker on the repaired plan and by confirming each fixed Y matches the repair value.

3. MIP START ADOPTION. For one FULL and one restricted subproblem, confirm from the HiGHS log
   that the submitted start was accepted (`MIP start solution is feasible ...`) and that the
   warm run's returned incumbent is not worse than the submitted one.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp as scipy_milp
from scipy.sparse import coo_matrix, vstack as sp_vstack

sys.path.insert(0, ".")

from lsp_checker import check_solution
from lsp_gen import all_instances, apply_disruption, disruptions_for
from lsp_highs import build_highs, repair_vector, solve_highs
from lsp_model import (MODE_AUDITED, Bounds as _B, Instance, LinearConstraint as _L,
                       build_model, solve_milp, unpack)
from lsp_repair3 import binary_change_count
from lsp_select import is_usable, select_output

KAPPA = 4.0
BUDGET = 5.0
SOURCE_STATES = "../stage04_runtime_calibration/stage04_states.json"


def scipy_equivalent(bm, fixed, stab_row):
    """The audited SciPy system, assembled the same way, for a direct comparison."""
    n = bm.idx.n
    A = bm.A.tocsr()
    rl, ru = bm.row_lb.copy(), bm.row_ub.copy()
    lb, ub = bm.var_lb.copy(), bm.var_ub.copy()
    coef, rhs = stab_row
    cols = list(coef.keys())
    A = sp_vstack([A, coo_matrix(([coef[c] for c in cols], ([0] * len(cols), cols)),
                                 shape=(1, n)).tocsr()], format="csr")
    rl = np.concatenate([rl, [-np.inf]])
    ru = np.concatenate([ru, [rhs]])
    if fixed:
        k = list(fixed.keys())
        A = sp_vstack([A, coo_matrix((np.ones(len(k)), (np.arange(len(k)), np.array(k))),
                                     shape=(len(k), n)).tocsr()], format="csr")
        v = np.array([fixed[c] for c in k], dtype=float)
        rl = np.concatenate([rl, v])
        ru = np.concatenate([ru, v])
        for c in k:
            lb[c], ub[c] = fixed[c], fixed[c]
    return A, rl, ru, lb, ub


def check1_model_transfer() -> Dict[str, Any]:
    print("=" * 100)
    print("1. MODEL TRANSFER")
    out: Dict[str, Any] = {"cases": []}
    # a tiny hand instance with a fixed/release pattern, plus one real state
    from lsp_gen import all_instances as alli
    inst0, meta0 = alli()[0]
    dis = disruptions_for(inst0)[0]
    cases = [("small instance, no fixing", inst0, {}),
             ("small instance, fixed/release Y", inst0, "partial")]
    for tag, inst, mode in cases:
        bm = build_model(inst, MODE_AUDITED)
        if mode == "partial":
            # fix the first half of the short-term Y to 0, release the rest
            short = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
                     for t in range(1, min(4, inst.T + 1))]
            half = short[: max(1, len(short) // 2)]
            fixed = {bm.idx.Y(i, j, t): 0.0 for (i, j, t) in half}
        else:
            fixed = {}
        stab = ({}, 0.0)
        h, lp, n_cols, n_rows = build_highs(bm, fixed, stab)
        A, rl, ru, lb, ub = scipy_equivalent(bm, fixed, stab)

        # compare every transferred array
        ok_n = (n_cols == bm.idx.n)
        ok_rows = (n_rows == A.shape[0])
        ok_cost = np.allclose(np.asarray(lp.col_cost_), bm.c)
        ok_lb = np.allclose(np.asarray(lp.col_lower_), lb)
        ok_ub = np.allclose(np.asarray(lp.col_upper_), ub)
        ok_rl = np.allclose(np.asarray(lp.row_lower_), rl)
        ok_ru = np.allclose(np.asarray(lp.row_upper_), ru)
        # matrix: reconstruct the rowwise arrays into a dense form and compare
        H = np.zeros((n_rows, n_cols))
        st = np.asarray(lp.a_matrix_.start_)
        ix = np.asarray(lp.a_matrix_.index_)
        vl = np.asarray(lp.a_matrix_.value_)
        for r in range(n_rows):
            for k in range(st[r], st[r + 1]):
                H[r, ix[k]] = vl[k]
        ok_A = np.allclose(H, A.toarray())
        integ = np.array([1 if v == __import__("highspy").HighsVarType.kInteger else 0
                          for v in lp.integrality_])
        ok_int = np.array_equal(integ, bm.integrality)

        # solve both ways and compare
        # NOTE: at a short time limit the two solvers legitimately stop at DIFFERENT
        # incumbents, so an objective mismatch there says nothing about the transfer. The
        # transfer itself is verified by the ten array comparisons above; here each returned
        # plan is additionally put through the INDEPENDENT checker, and the objectives are
        # compared only when both runs proved optimality.
        s_sol = scipy_milp(c=bm.c, constraints=LinearConstraint(A, lb=rl, ub=ru),
                           integrality=bm.integrality, bounds=Bounds(lb=lb, ub=ub),
                           options={"time_limit": BUDGET})
        run = solve_highs(bm, fixed, stab, BUDGET, seed=0)

        arr_ok = all([ok_n, ok_rows, ok_cost, ok_lb, ok_ub, ok_rl, ok_ru, ok_A, ok_int])
        scipy_feasible = None
        highs_feasible = None
        if s_sol.x is not None:
            Xs, Ys, Zs, Is, Ls = unpack(s_sol.x, bm.idx)
            from lsp_model import Solution as _S
            scipy_feasible = check_solution(
                inst, _S(status="s", objective=float(s_sol.fun), X=Xs, Y=Ys, Z=Zs, I=Is, L=Ls,
                         mode=MODE_AUDITED), MODE_AUDITED).ok
        if run.solution is not None:
            highs_feasible = check_solution(inst, run.solution, MODE_AUDITED).ok
        both_optimal = (s_sol.status == 0 and run.status == "Optimal")
        obj_gap = None
        if both_optimal and run.objective is not None:
            obj_gap = abs(run.objective - float(s_sol.fun))
        rec = {
            "case": tag, "n_cols_match": bool(ok_n), "n_rows_match": bool(ok_rows),
            "objective_vector_match": bool(ok_cost), "col_lower_match": bool(ok_lb),
            "col_upper_match": bool(ok_ub), "row_lower_match": bool(ok_rl),
            "row_upper_match": bool(ok_ru), "matrix_match": bool(ok_A),
            "integrality_match": bool(ok_int),
            "scipy_objective": (float(s_sol.fun) if s_sol.x is not None else None),
            "scipy_status": int(s_sol.status),
            "scipy_solution_passes_checker": scipy_feasible,
            "highs_objective": run.objective,
            "highs_status": run.status,
            "highs_solution_passes_checker": highs_feasible,
            "both_proved_optimal": bool(both_optimal),
            "objective_abs_diff_if_both_optimal": obj_gap,
            "n_fixed": len(fixed),
        }
        rec["passed"] = bool(arr_ok and highs_feasible and scipy_feasible
                             and (obj_gap is None
                                  or obj_gap <= 1e-6 * (1 + abs(float(s_sol.fun)))))
        out["cases"].append(rec)
        print("   %-32s arrays=%s highs_feasible=%s scipy_feasible=%s both_opt=%s "
              "diff=%s  %s"
              % (tag, arr_ok, highs_feasible, scipy_feasible, both_optimal,
                 ("%.3g" % obj_gap) if obj_gap is not None else "n/a",
                 "OK" if rec["passed"] else "**FAIL**"))
    out["passed"] = all(c["passed"] for c in out["cases"])
    return out


def load_states():
    with open(SOURCE_STATES, encoding="utf-8") as fh:
        return json.load(fh)


def solution_from_record(inst, rec):
    from lsp_model import Solution
    p = rec["plan"]
    return Solution(status="stored", objective=float(rec["objective"]),
                    X=np.array(p["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def method_fixing(bm, repair, tau, method, kappa):
    """Same rules as stage 03/04/04b."""
    inst = bm.inst
    def possible(u):
        i, j, t = u
        return inst.w[i][j] != 0 and inst.s[i] <= inst.c[j][t - 1] + 1e-9
    short = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
             for t in range(1, tau + 1)]
    feas = [u for u in short if possible(u)]
    if method == "FULL":
        release = feas
    elif method == "EMPTY":
        release = []
    else:
        k = int(method.split("-")[1])
        scores = {}
        for (i, j, t) in feas:
            shortage = sum(float(repair.L[i, r - 1]) for r in range(t, inst.T + 1))
            scores[(i, j, t)] = (inst.l[i] * shortage) / (inst.s[i] + inst.b[i] * inst.m[i])
        release = sorted(feas, key=lambda u: (-scores[u], u))[:k]
    rel = set(release)
    fixed, fixed_Y = {}, {}
    for (i, j, t) in short:
        if (i, j, t) not in rel:
            val = float(repair.Y[i, j, t - 1])
            fixed[bm.idx.Y(i, j, t)] = val
            fixed_Y[(i, j, t)] = val
    return fixed, fixed_Y, release


def stability_row(bm, repair, tau, kappa):
    coef, n_one = {}, 0
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, tau + 1):
                yref = float(repair.Y[i, j, t - 1])
                if yref > 0.5:
                    n_one += 1
                    coef[bm.idx.Y(i, j, t)] = -1.0
                else:
                    coef[bm.idx.Y(i, j, t)] = 1.0
    return coef, kappa - float(n_one)


def check2_repair_feasible(states) -> Dict[str, Any]:
    print("=" * 100)
    print("2. REPAIR VECTOR FEASIBILITY IN EVERY SUBPROBLEM")
    out: Dict[str, Any] = {"cases": []}
    insts = {m["name"]: (i, m) for i, m in all_instances()}
    for state, rec in sorted(states.items()):
        name, dis_name = state.split("|")
        inst0, meta = insts[name]
        dis = next(d for d in disruptions_for(inst0) if d.name == dis_name)
        pert = apply_disruption(inst0, dis)
        repair = solution_from_record(pert, rec["repair"])
        chk = check_solution(pert, repair, MODE_AUDITED)
        bm = build_model(pert, MODE_AUDITED)
        stab = stability_row(bm, repair, meta["tau"], KAPPA)
        x0 = repair_vector(bm, repair)
        row = {"state": state, "repair_model_feasible": bool(chk.ok),
               "repair_max_violation": chk.max_violation, "methods": {}}
        for method in ("FULL", "EMPTY", "SHORTAGE-12", "SHORTAGE-24"):
            fixed, fixed_Y, release = method_fixing(bm, repair, meta["tau"], method, KAPPA)
            # the repair's Y must equal every fixing value, and must satisfy the stability row
            fix_ok = all(abs(float(repair.Y[i, j, t - 1]) - v) < 1e-9
                         for (i, j, t), v in fixed_Y.items())
            lhs = sum(c * x0[col] for col, c in stab[0].items())
            stab_ok = lhs <= stab[1] + 1e-9
            # and the vector must be feasible for the subproblem (rows + fixings)
            A, rl, ru, lb, ub = scipy_equivalent(bm, fixed, stab)
            ax = A @ x0
            feas = bool(np.all(ax <= ru + 1e-9) and np.all(ax >= rl - 1e-9)
                        and np.all(x0 <= ub + 1e-9) and np.all(x0 >= lb - 1e-9))
            row["methods"][method] = {"n_fixed": len(fixed), "n_released": len(release),
                                      "fixings_match_repair": bool(fix_ok),
                                      "stability_lhs": float(lhs),
                                      "stability_rhs": float(stab[1]),
                                      "stability_ok": bool(stab_ok),
                                      "repair_vector_subproblem_feasible": feas}
            ok = chk.ok and fix_ok and stab_ok and feas
            if not ok:
                row.setdefault("failures", []).append(method)
        row["passed"] = bool(row["repair_model_feasible"]
                             and not row.get("failures"))
        out["cases"].append(row)
        print("   %-28s repair_feasible=%-5s methods ok=%s"
              % (state, row["repair_model_feasible"],
                 "ALL" if row["passed"] else row.get("failures")))
    out["passed"] = all(c["passed"] for c in out["cases"])
    return out


def check3_adoption(states) -> Dict[str, Any]:
    print("=" * 100)
    print("3. MIP START ADOPTION (one FULL + one restricted subproblem)")
    out: Dict[str, Any] = {"cases": []}
    insts = {m["name"]: (i, m) for i, m in all_instances()}
    state = "large_rho1.10_s0|D1_m0_2p"
    if state not in states:
        state = sorted(states)[-1]
    name, dis_name = state.split("|")
    inst0, meta = insts[name]
    dis = next(d for d in disruptions_for(inst0) if d.name == dis_name)
    pert = apply_disruption(inst0, dis)
    repair = solution_from_record(pert, states[state]["repair"])
    bm = build_model(pert, MODE_AUDITED)
    stab = stability_row(bm, repair, meta["tau"], KAPPA)
    x0 = repair_vector(bm, repair)
    for method in ("FULL", "SHORTAGE-12"):
        fixed, fixed_Y, release = method_fixing(bm, repair, meta["tau"], method, KAPPA)
        cold = solve_highs(bm, fixed, stab, BUDGET, seed=0, start_values=None)
        warm = solve_highs(bm, fixed, stab, BUDGET, seed=0, start_values=x0)
        rec = {
            "state": state, "method": method,
            "cold_status": cold.status, "cold_objective": cold.objective,
            "warm_status": warm.status, "warm_objective": warm.objective,
            "set_solution_status": warm.set_solution_status,
            "start_adopted": warm.start_adopted,
            "start_reported_objective": warm.start_reported_objective,
            "start_message": warm.start_message,
            "repair_cost": float(repair.objective),
        }
        if warm.solution is not None:
            sel = select_output(pert, warm.solution, repair, KAPPA, meta["tau"],
                                fixed_Y=fixed_Y, solver_status=warm.status)
            rec["warm_selected_cost"] = (None if sel.chosen is None else sel.chosen.cost)
            rec["warm_selected_source"] = sel.source
            rec["warm_select_errors"] = sel.errors
        # adoption must be confirmed AND the warm incumbent must not be worse than the start
        ok = bool(warm.start_adopted)
        if rec["start_reported_objective"] is not None and warm.objective is not None:
            ok = ok and warm.objective <= rec["start_reported_objective"] + 1e-6 * (
                1 + abs(rec["start_reported_objective"]))
        rec["passed"] = bool(ok)
        out["cases"].append(rec)
        print("   %-12s cold=%-12s warm=%-12s adopted=%-5s reported=%-12s %s"
              % (method, cold.objective, warm.objective, warm.start_adopted,
                 rec["start_reported_objective"], "OK" if rec["passed"] else "**FAIL**"))
        if rec["start_message"]:
            print("        log: %s" % rec["start_message"])
    out["passed"] = all(c["passed"] for c in out["cases"])
    return out


def main() -> int:
    t0 = time.perf_counter()
    states = load_states()
    r1 = check1_model_transfer()
    r2 = check2_repair_feasible(states)
    r3 = check3_adoption(states)
    res = {"check1_model_transfer": r1, "check2_repair_feasible": r2,
           "check3_mip_start_adoption": r3,
           "all_passed": bool(r1["passed"] and r2["passed"] and r3["passed"]),
           "wall_s": time.perf_counter() - t0,
           "highspy_version": __import__("importlib.metadata", fromlist=["version"]).version("highspy")}
    with open("stage05_preflight.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("=" * 100)
    print("PREFLIGHT %s  (model transfer=%s  repair feasibility=%s  adoption=%s)"
          % ("PASSED" if res["all_passed"] else "FAILED", r1["passed"], r2["passed"],
             r3["passed"]))
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
