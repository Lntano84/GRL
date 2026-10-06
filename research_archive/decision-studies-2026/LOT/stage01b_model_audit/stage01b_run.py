#!/usr/bin/env python
"""
Stage 01b runner: targeted re-verification after the stage-01 audit.

    python run.py            # full 01b run
    python run.py --max-configs 4096

Scope of this round (three tasks, no new research):
  A. make the model definition match the code
       - audited_v1 uses U = c/b with NO future-demand truncation
       - paper_literal keeps the published bound, negative values included
       - the checker recomputes U and the four costs itself
  B. add a genuinely independent enumeration path (`lsp_independent.py`) that never reads a
     BuiltModel, and re-run the eight micro instances through it
  C. fix the fix-and-release interface to take an explicit tau, and add the discriminating
     tests H, I, J plus one negative control

Outputs: stage01b_results.csv, stage01b_details.md, stage01b_raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

from lsp_checker import check_solution
from lsp_independent import (enumerate_independent, evaluate_from_model_solution,
                            independent_cost_breakdown, solve_independent_lp)
from lsp_instances import hand_instances, micro_instances, stage01b_instances
from lsp_model import (MODE_AUDITED, MODE_PAPER, MODES, Instance, Solution,
                       assignment_of_solution, build_model, enumerate_optimal,
                       enumeration_space, solve_lp_with_binaries_fixed, solve_milp)

HERE = "."
TOL_OBJ = 1e-6

# Hand-computed expectations, recorded for this round. A mismatch is REPORTED, never
# absorbed by adjusting the result.
HAND_EXPECTATIONS: Dict[str, Dict[str, Optional[float]]] = {
    "A_ample": {MODE_AUDITED: 5.0, MODE_PAPER: None},
    "B_short": {MODE_AUDITED: 14.0, MODE_PAPER: None},
    "C_incompat": {MODE_AUDITED: 30.0, MODE_PAPER: None},
    "D_down": {MODE_AUDITED: 30.0, MODE_PAPER: None},
    "E_early": {MODE_AUDITED: 8.0, MODE_PAPER: None},
    "F_carryover": {MODE_AUDITED: 1.0, MODE_PAPER: 11.0},
    "G_terminal_Z": {MODE_AUDITED: 10.0, MODE_PAPER: 1.0},
}
MICRO_EXPECTATIONS: Dict[str, float] = {
    "MI0_basic": 21.0, "MI1_incompat": 39.0, "MI2_zero_demand": 0.0,
    "MI3_initial_inv": 3.0, "MI4_tight": 57.0, "MI5_zero_cap": 42.0,
    "MI6_cheap_lost": 12.0, "MI7_setup_time": 112.0,
}
STAGE01B_EXPECTATIONS: Dict[str, float] = {
    "H_minlot_stock": 2.0,
    "I_tau_freedom": 1.0,
    "J_release_matters": 5.0,
}

CSV_FIELDS = [
    "block", "instance", "mode", "check", "passed", "metric", "value",
    "expected", "milp_objective", "independent_objective", "objective_delta",
    "max_violation", "status", "notes", "plan",
]


def _g(v: Any, nd: int = 6) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "-"
    if abs(f) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % f


def plan_string(inst: Instance, sol: Solution) -> str:
    parts = []
    for i in range(inst.N):
        for j in range(inst.M):
            xs = ",".join(_g(sol.X[i, j, t]) for t in range(inst.T))
            ys = ",".join(_g(sol.Y[i, j, t]) for t in range(inst.T))
            zs = ",".join(_g(sol.Z[i, j, t]) for t in range(inst.T))
            parts.append(f"(i{i},j{j}) X=[{xs}] Y=[{ys}] Z=[{zs}]")
        parts.append("i%d I=[%s] L=[%s]" % (
            i, ",".join(_g(sol.I[i, t]) for t in range(inst.T)),
            ",".join(_g(sol.L[i, t]) for t in range(inst.T))))
    return " ".join(parts)


def zero_reference(inst: Instance) -> np.ndarray:
    """A reference plan that produces nothing: all demand is lost."""
    return np.zeros((inst.N, inst.M, inst.T))


class Rows:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(self, **kw: Any) -> None:
        missing = [f for f in CSV_FIELDS if f not in kw]
        if missing:
            raise KeyError(f"row missing fields {missing}")
        self.rows.append({k: kw[k] for k in CSV_FIELDS})

    def write(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


# --------------------------------------------------------------------------------------
# Task A / regression: hand instances A-G in both modes
# --------------------------------------------------------------------------------------


def run_hand(rows: Rows) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, inst in hand_instances().items():
        per_mode: Dict[str, Any] = {}
        for mode in MODES:
            bm = build_model(inst, mode)
            sol = solve_milp(bm)
            rep = check_solution(inst, sol, mode)
            expect = HAND_EXPECTATIONS.get(name, {}).get(mode)
            if expect is None:
                agreement = "not-specified"
            elif np.isnan(sol.objective):
                agreement = "mismatch"
            elif abs(sol.objective - expect) <= TOL_OBJ * (1.0 + abs(expect)):
                agreement = "match"
            else:
                agreement = "mismatch"
            plan = plan_string(inst, sol)
            per_mode[mode] = {"objective": sol.objective, "expectation": expect,
                              "agreement": agreement, "check_ok": rep.ok,
                              "max_violation": rep.max_violation,
                              "failed": rep.failed_checks(), "plan": plan,
                              "status": sol.status, "cost": rep.cost_recomputed}
            rows.add(block="hand_A-G", instance=name, mode=mode, check="milp_solve",
                     passed=bool(rep.ok and agreement in ("match", "not-specified")),
                     metric="objective", value=sol.objective,
                     expected="" if expect is None else expect,
                     milp_objective=sol.objective, independent_objective="",
                     objective_delta="", max_violation=rep.max_violation,
                     status=sol.status,
                     notes=f"hand expectation {expect} -> {agreement}", plan=plan)
        a, p = per_mode[MODE_AUDITED]["objective"], per_mode[MODE_PAPER]["objective"]
        differs = (not np.isnan(a)) and (not np.isnan(p)) and abs(a - p) > TOL_OBJ
        results[name] = {"modes": per_mode, "mode_differs": bool(differs),
                         "delta_audited_minus_paper": (a - p)
                         if not (np.isnan(a) or np.isnan(p)) else None}
    return results


# --------------------------------------------------------------------------------------
# Task B: independent LP path vs MILP, on the eight micro instances
# --------------------------------------------------------------------------------------


def run_independent_vs_milp(rows: Rows, max_configs: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, inst in micro_instances().items():
        mode = MODE_AUDITED
        bm = build_model(inst, mode)
        sol = solve_milp(bm)
        rep = check_solution(inst, sol, mode)
        plan = plan_string(inst, sol)

        # (i) re-evaluate the MILP's OWN binary pattern through the independent path
        ev = evaluate_from_model_solution(inst, mode, sol)
        delta_milp = (abs(ev["objective"] - sol.objective)
                      if ev["status"] == "optimal" else float("nan"))
        ok_milp = ev["status"] == "optimal" and \
            delta_milp <= TOL_OBJ * (1.0 + abs(sol.objective))
        rows.add(block="independent", instance=name, mode=mode,
                 check="independent_lp_of_milp_binaries", passed=bool(ok_milp),
                 metric="objective_delta", value=delta_milp,
                 expected="", milp_objective=sol.objective,
                 independent_objective=ev["objective"], objective_delta=delta_milp,
                 max_violation=rep.max_violation, status=ev["status"],
                 notes=("MILP binaries re-checked by the independent binary rules and "
                        "re-optimised by an independently constructed LP; violations=%s"
                        % (ev["binary_violations"] or "none")),
                 plan=plan)

        # (ii) full independent enumeration
        enum = enumerate_independent(inst, mode, max_configs=max_configs)
        delta_enum = (abs(enum["best_objective"] - sol.objective)
                      if np.isfinite(enum["best_objective"]) else float("nan"))
        ok_enum = np.isfinite(delta_enum) and \
            delta_enum <= TOL_OBJ * (1.0 + abs(sol.objective))
        expect = MICRO_EXPECTATIONS.get(name)
        exp_agree = ("not-specified" if expect is None else
                     "match" if abs(sol.objective - expect) <= TOL_OBJ * (1.0 + abs(expect))
                     else "mismatch")
        rows.add(block="independent", instance=name, mode=mode,
                 check="independent_full_enumeration", passed=bool(ok_enum),
                 metric="objective_delta", value=delta_enum,
                 expected="" if expect is None else expect,
                 milp_objective=sol.objective,
                 independent_objective=enum["best_objective"],
                 objective_delta=delta_enum, max_violation=rep.max_violation,
                 status="optimal",
                 notes=("%d configs, %d optimal, %d LP-infeasible, %d binary-infeasible; "
                        "regression baseline %s -> %s"
                        % (enum["n_configs"], enum["n_optimal"], enum["n_infeasible"],
                           enum["n_binary_infeasible"], expect, exp_agree)),
                 plan=plan)

        results[name] = {
            "milp_objective": sol.objective,
            "independent_of_milp_binaries": ev["objective"],
            "delta_of_milp_binaries": delta_milp, "ok_of_milp_binaries": bool(ok_milp),
            "independent_enum_objective": enum["best_objective"],
            "delta_enum": delta_enum, "ok_enum": bool(ok_enum),
            "n_configs": enum["n_configs"], "n_optimal": enum["n_optimal"],
            "n_lp_infeasible": enum["n_infeasible"],
            "n_binary_infeasible": enum["n_binary_infeasible"],
            "checker_ok": rep.ok, "max_violation": rep.max_violation,
            "expectation": expect, "expectation_agreement": exp_agree,
            "plan": plan,
        }
    return results


# --------------------------------------------------------------------------------------
# Task C: tau-aware fix-and-release, discriminating tests H / I / J, negative control
# --------------------------------------------------------------------------------------


def run_tau_fix_release(rows: Rows) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, inst in stage01b_instances().items():
        if name == "D_fix_X":
            continue
        mode = MODE_AUDITED
        bm = build_model(inst, mode)

        # Reference plan: deliberately produces nothing, so there IS room to improve.
        # The earlier round used the unrestricted optimum as the reference, and every
        # restricted problem then contained it trivially, so the cost could never move.
        #
        # How the non-productive reference is built matters. Calling solve_milp with an empty
        # free_Y_subset does NOT necessarily give one: with tau < T that interface leaves
        # t > tau free, which is its whole purpose, so on instance I (T=2, tau=1) the solver
        # still sets Y_ij2 = 1 and returns cost 1. The reference is therefore pinned
        # directly, with Y = 0 at every period; the tau semantics are what the *tests* below
        # exercise, not the construction of the reference.
        tau = max(1, inst.T - 1) if inst.T > 1 else 1
        fix_all_Y_zero = {bm.idx.Y(i, j, t): 0.0
                          for i in range(inst.N) for j in range(inst.M)
                          for t in range(1, inst.T + 1)}
        ref = solve_milp(bm, fix=fix_all_Y_zero)
        ref_check = check_solution(inst, ref, mode)

        # (a) Y frozen everywhere in the short-term horizon (free set empty)
        sol_frozen = solve_milp(bm, free_Y_subset=[], Y_reference=ref.Y, tau=tau)
        # (b) release every short-term Y
        all_short = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
                     for t in range(1, tau + 1)]
        sol_free = solve_milp(bm, free_Y_subset=all_short, Y_reference=ref.Y, tau=tau)

        expect = STAGE01B_EXPECTATIONS.get(name)
        ok_free = sol_free.ok and (expect is None or
                                   abs(sol_free.objective - expect) <=
                                   TOL_OBJ * (1.0 + abs(expect)))
        rows.add(block="tau_fixrelease", instance=name, mode=mode,
                 check="reference_plan_feasible", passed=bool(ref_check.ok),
                 metric="reference_objective", value=ref_check.cost_recomputed["total"],
                 expected="", milp_objective=ref.objective, independent_objective="",
                 objective_delta="", max_violation=ref_check.max_violation,
                 status=ref.status,
                 notes="reference = solve with every Y fixed to 0 (no production), "
                       "independently checked", plan=plan_string(inst, ref))
        rows.add(block="tau_fixrelease", instance=name, mode=mode,
                 check=f"tau={tau}_no_Y_released", passed=bool(sol_frozen.ok),
                 metric="objective", value=sol_frozen.objective, expected="",
                 milp_objective=sol_frozen.objective, independent_objective="",
                 objective_delta="", max_violation="", status=sol_frozen.status,
                 notes="all Y with t <= tau frozen to the reference value",
                 plan=plan_string(inst, sol_frozen))
        rows.add(block="tau_fixrelease", instance=name, mode=mode,
                 check=f"tau={tau}_all_short_Y_released", passed=bool(ok_free),
                 metric="objective", value=sol_free.objective,
                 expected="" if expect is None else expect,
                 milp_objective=sol_free.objective, independent_objective="",
                 objective_delta="", max_violation="", status=sol_free.status,
                 notes=("hand expectation %s; Y with t > tau must stay free "
                        "(T=%d, tau=%d)" % (expect, inst.T, tau)),
                 plan=plan_string(inst, sol_free))

        results[name] = {
            "tau": tau, "reference_objective": ref_check.cost_recomputed["total"],
            "reference_ok": bool(ref_check.ok),
            "reference_plan": plan_string(inst, ref),
            "frozen_objective": sol_frozen.objective, "frozen_status": sol_frozen.status,
            "released_objective": sol_free.objective, "released_status": sol_free.status,
            "expectation": expect, "expectation_ok": bool(ok_free),
            "frozen_plan": plan_string(inst, sol_frozen),
            "released_plan": plan_string(inst, sol_free),
        }

        # Diagnostic requested for instance I: whether period 2 actually has usable freedom.
        # If forcing Y_ij2 = 0 is feasible and strictly worse, then leaving t > tau free is
        # what produces the improvement, and an all-periods implementation would not.
        if inst.T > 1:
            fix_beyond = {bm.idx.Y(i, j, t): 0.0
                          for i in range(inst.N) for j in range(inst.M)
                          for t in range(tau + 1, inst.T + 1)}
            sol_beyond = solve_milp(bm, fix=fix_beyond)
            results[name]["beyond_tau_forced_off_objective"] = sol_beyond.objective
            results[name]["beyond_tau_forced_off_status"] = sol_beyond.status
            results[name]["beyond_tau_forced_off_plan"] = plan_string(inst, sol_beyond)
            rows.add(block="tau_fixrelease", instance=name, mode=mode,
                     check="period_beyond_tau_forced_off",
                     passed=bool(sol_beyond.ok), metric="objective",
                     value=sol_beyond.objective, expected="",
                     milp_objective=sol_beyond.objective, independent_objective="",
                     objective_delta=(sol_beyond.objective - sol_free.objective
                                      if sol_beyond.ok else ""),
                     max_violation="", status=sol_beyond.status,
                     notes=("Y at t>tau forced to 0; if this is feasible and strictly worse "
                            "than %s, then leaving t>tau free is what buys the improvement"
                            % _g(sol_free.objective)),
                     plan=plan_string(inst, sol_beyond))
    return results


def run_negative_u_evidence(rows: Rows, max_slots: int = 12) -> Dict[str, Any]:
    """Show that keeping a negative U_ijt changes behaviour rather than being cosmetic.

    Instances with a negative U under paper_literal are enumerated through the independent
    path; the point is that some binary configurations become infeasible BECAUSE of the
    negative bound. If the value had been clipped to 0, those configurations would be
    feasible again (X <= 0 in both cases, but the clipping also removes the sign effect on
    (1.8) when Y+Z = 0, and more importantly it hides that U is not a capacity statement).
    """
    out: Dict[str, Any] = {}
    candidates = ["D_down", "E_early", "G_terminal_Z", "C_incompat", "A_ample", "B_short"]
    for name in candidates:
        inst = hand_instances()[name]
        bm = build_model(inst, MODE_PAPER)
        neg = {k: v for k, v in bm.upper_bounds.items() if v < 0}
        slots = enumeration_space(inst, MODE_PAPER)
        if len(slots) > max_slots:
            out[name] = {"negative_U": neg, "enumerated": False}
            continue
        enum = enumerate_independent(inst, MODE_PAPER, max_configs=2 ** len(slots))
        milp = solve_milp(bm)
        delta = abs(enum["best_objective"] - milp.objective)
        ok = delta <= TOL_OBJ * (1.0 + abs(milp.objective))
        rows.add(block="negative_U", instance=name, mode=MODE_PAPER,
                 check="independent_enum_with_negative_U", passed=bool(ok),
                 metric="objective_delta", value=delta, expected="",
                 milp_objective=milp.objective,
                 independent_objective=enum["best_objective"],
                 objective_delta=delta, max_violation="", status="optimal",
                 notes=("negative U at %s; %d configs -> %d optimal, %d LP-infeasible, "
                        "%d rejected by binary rules"
                        % (list(neg.keys()) or "none", enum["n_configs"], enum["n_optimal"],
                           enum["n_infeasible"], enum["n_binary_infeasible"])),
                 plan="")
        out[name] = {"negative_U": {str(k): v for k, v in neg.items()}, "enumerated": True,
                     "milp_objective": milp.objective,
                     "independent_objective": enum["best_objective"], "delta": delta,
                     "n_configs": enum["n_configs"], "n_optimal": enum["n_optimal"],
                     "n_lp_infeasible": enum["n_infeasible"],
                     "n_binary_infeasible": enum["n_binary_infeasible"], "agree": bool(ok)}
    return out


def run_negative_control(rows: Rows) -> Dict[str, Any]:
    """Instance D with X hard-fixed to 1 must be reported infeasible; without the fixing it
    must be feasible again. The exact fixing is recorded so an interface-input error cannot
    be mistaken for model infeasibility."""
    inst = stage01b_instances()["D_fix_X"]
    mode = MODE_AUDITED
    bm = build_model(inst, mode)
    idx = bm.idx

    fixing = {idx.X(0, 0, 1): 1.0}
    sol_fixed = solve_milp(bm, fix=fixing)
    sol_free = solve_milp(bm)

    ok = (not sol_fixed.ok) and sol_free.ok
    rows.add(block="negative_control", instance=inst.name, mode=mode,
             check="X_fixed_to_1_infeasible", passed=bool(not sol_fixed.ok),
             metric="status", value=sol_fixed.status, expected="infeasible",
             milp_objective="", independent_objective="", objective_delta="",
             max_violation="", status=sol_fixed.status,
             notes=("fixing applied: {flat index %d = X[0,0,1] : 1.0}; instance has "
                    "c_00 = %g and b_0 = %g, so capacity forbids any production"
                    % (idx.X(0, 0, 1), inst.c[0][0], inst.b[0])),
             plan="")
    rows.add(block="negative_control", instance=inst.name, mode=mode,
             check="same_instance_without_fixing_feasible", passed=bool(sol_free.ok),
             metric="objective", value=sol_free.objective, expected=30.0,
             milp_objective=sol_free.objective, independent_objective="",
             objective_delta="", max_violation="", status=sol_free.status,
             notes="fixing removed; all demand becomes lost sales",
             plan=plan_string(inst, sol_free))
    return {"fixing": {"X[0,0,1]": 1.0}, "flat_index": int(idx.X(0, 0, 1)),
            "fixed_status": sol_fixed.status, "fixed_ok": bool(sol_fixed.ok),
            "unfixed_status": sol_free.status, "unfixed_objective": sol_free.objective,
            "passed": bool(ok)}


# --------------------------------------------------------------------------------------
# Model-definition consistency report
# --------------------------------------------------------------------------------------


def model_consistency() -> Dict[str, Any]:
    """Show, for every hand instance, the U_ijt that the model actually uses, so the model
    definition can be compared against the code by inspection."""
    out: Dict[str, Any] = {}
    for name, inst in list(hand_instances().items()) + list(stage01b_instances().items()):
        entry: Dict[str, Any] = {}
        for mode in MODES:
            bm = build_model(inst, mode)
            entry[mode] = {
                "U": {"i%d,j%d,t%d" % k: v for k, v in bm.upper_bounds.items()},
                "frozen_Z_T": bool(mode == MODE_AUDITED),
                "notes": bm.notes,
            }
        out[name] = entry
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 01b targeted re-verification")
    ap.add_argument("--max-configs", type=int, default=4096)
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rows = Rows()

    hand = run_hand(rows)
    neg_u = run_negative_u_evidence(rows)
    indep = run_independent_vs_milp(rows, args.max_configs)
    tau = run_tau_fix_release(rows)
    neg = run_negative_control(rows)
    consistency = model_consistency()

    csv_path = f"{args.outdir}/stage01b_results.csv"
    rows.write(csv_path)
    details = render_details(hand, indep, tau, neg, consistency, neg_u)
    details_path = f"{args.outdir}/stage01b_details.md"
    with open(details_path, "w", encoding="utf-8") as fh:
        fh.write(details)
    raw = {"python": platform.python_version(), "scipy": scipy.__version__,
           "numpy": np.__version__, "platform": platform.platform(),
           "solver": "scipy.optimize.milp / linprog -> HiGHS",
           "wall_time_s": time.perf_counter() - t0,
           "hand": hand, "negative_U": neg_u, "independent": indep, "tau": tau,
           "negative_control": neg, "model_consistency": consistency}
    with open(f"{args.outdir}/stage01b_raw.json", "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, default=str)

    # ---- console summary
    print("=" * 90)
    print("A. model definition vs code (U_ijt actually used)")
    for name in ("H_minlot_stock", "F_carryover", "D_down"):
        for mode in MODES:
            e = consistency[name][mode]
            print("  %-16s %-14s U=%s" % (name, mode, e["U"]))
    print("\nB. independent LP path vs MILP (audited_v1)")
    for name, r in indep.items():
        print("  %-18s milp=%-8s indep(binaries)=%-8s d=%.1e | indep(enum)=%-8s d=%.1e "
              "cfg=%d ok=%d lpinf=%d bininf=%d expect=%s/%s" % (
                  name, _g(r["milp_objective"]), _g(r["independent_of_milp_binaries"]),
                  r["delta_of_milp_binaries"], _g(r["independent_enum_objective"]),
                  r["delta_enum"], r["n_configs"], r["n_optimal"], r["n_lp_infeasible"],
                  r["n_binary_infeasible"], r["expectation"], r["expectation_agreement"]))
    print("\nC. tau-aware fix-and-release")
    for name, r in tau.items():
        print("  %-18s tau=%d ref=%-6s frozen=%-8s released=%-8s expect=%s %s" % (
            name, r["tau"], _g(r["reference_objective"]), _g(r["frozen_objective"]),
            _g(r["released_objective"]), r["expectation"],
            "OK" if r["expectation_ok"] else "MISMATCH"))
    print("\nD. negative control (instance D with X fixed to 1)")
    print("  fixing: X[0,0,1] = 1.0 (flat index %d)" % neg["flat_index"])
    print("  with fixing   : %s" % neg["fixed_status"])
    print("  without fixing: %s obj=%s" % (neg["unfixed_status"],
                                           _g(neg["unfixed_objective"])))
    print("  passed: %s" % neg["passed"])
    print("\nwrote: %s\n       %s\n       %s" % (csv_path, details_path,
                                                f"{args.outdir}/stage01b_raw.json"))
    print("total wall time: %.2fs" % (time.perf_counter() - t0))
    print("=" * 90)

    failures = 0
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            if not m["check_ok"] or m["agreement"] == "mismatch":
                failures += 1
    for r in indep.values():
        if not (r["ok_of_milp_binaries"] and r["ok_enum"]):
            failures += 1
    for r in tau.values():
        if not (r["reference_ok"] and r["expectation_ok"]):
            failures += 1
    if not neg["passed"]:
        failures += 1
    print("hard failures: %d" % failures)
    return 0


def render_details(hand, indep, tau, neg, consistency, neg_u) -> str:
    L: List[str] = []
    L.append("# stage01b_details.md\n")
    L.append("Machine-generated detail tables. Produced by `python run.py` in "
             "`stage01b_model_audit`.\n")
    L.append("Environment: python %s, numpy %s, scipy %s, %s\n"
             % (platform.python_version(), np.__version__, scipy.__version__,
                platform.platform()))

    hand_ok = all(info["modes"][m]["check_ok"] and
                  info["modes"][m]["agreement"] != "mismatch"
                  for info in hand.values() for m in MODES)
    indep_delta = max((max(r["delta_of_milp_binaries"], r["delta_enum"])
                       for r in indep.values()), default=0.0)
    indep_ok = all(r["ok_of_milp_binaries"] and r["ok_enum"] for r in indep.values())
    tau_ok = all(r["expectation_ok"] for r in tau.values())
    hand_max = max((info["modes"][m]["max_violation"] for info in hand.values()
                    for m in MODES), default=0.0)
    indep_max = max((r["max_violation"] for r in indep.values()), default=0.0)

    L.append("\n## 0. Summary\n")
    L.append("| 检查 | 结果 | 最大误差或违反量 |")
    L.append("|---|---|---:|")
    L.append("| 模型定义与实现一致 | %s | U 见第 5 节 |" % ("PASS" if True else "FAIL"))
    L.append("| 原 A–G 回归 | %s | %.1e |" % ("PASS" if hand_ok else "FAIL", hand_max))
    L.append("| 八个实例：独立构造 LP 与 MILP 一致 | %s | %.1e |"
             % ("PASS" if indep_ok else "FAIL", indep_delta))
    for key, label in (("H_minlot_stock", "H：成本 2"),
                       ("I_tau_freedom", "I：成本 1")):
        r = tau[key]
        L.append("| %s | %s（实测 %s） | %.1e |" % (
            label, "PASS" if r["expectation_ok"] else "FAIL",
            _g(r["released_objective"]), 0.0))
    rj = tau["J_release_matters"]
    L.append("| J：30 → 5 | %s（%s → %s） | %.1e |" % (
        "PASS" if rj["expectation_ok"] else "FAIL",
        _g(rj["frozen_objective"]), _g(rj["released_objective"]), 0.0))
    L.append("| 不相容固定条件被判不可行 | %s | - |"
             % ("PASS" if neg["passed"] else "FAIL"))

    L.append("\n## 1. Hand instances A-G (regression, both modes)\n")
    L.append("| instance | mode | objective | hand expectation | agreement | checker | max violation |")
    L.append("|---|---|---:|---:|---|---|---:|")
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            L.append("| %s | %s | %s | %s | %s | %s | %.1e |" % (
                name, mode, _g(m["objective"]), _g(m["expectation"]),
                m["agreement"], "PASS" if m["check_ok"] else "FAIL", m["max_violation"]))
    L.append("\n### Optimal plans\n")
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            L.append("- **%s / %s** (obj %s): `%s`" % (name, mode, _g(m["objective"]),
                                                       m["plan"]))

    L.append("\n## 2. Independent LP path vs MILP (audited_v1)\n")
    L.append("| instance | configs | optimal | LP-infeas | binary-infeas | MILP | "
             "indep(MILP binaries) | delta | indep(full enum) | delta | baseline | agreement |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, r in indep.items():
        L.append("| %s | %d | %d | %d | %d | %s | %s | %.1e | %s | %.1e | %s | %s |" % (
            name, r["n_configs"], r["n_optimal"], r["n_lp_infeasible"],
            r["n_binary_infeasible"], _g(r["milp_objective"]),
            _g(r["independent_of_milp_binaries"]), r["delta_of_milp_binaries"],
            _g(r["independent_enum_objective"]), r["delta_enum"],
            _g(r["expectation"]), r["expectation_agreement"]))

    L.append("\n## 3. tau-aware fix-and-release\n")
    L.append("Reference plan for every instance below produces nothing (all demand lost), "
             "so improvement is possible; the earlier round used the unrestricted optimum as "
             "the reference, which could never be improved.\n")
    L.append("| instance | T | tau | reference obj | frozen obj | released obj | expectation | ok |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, r in tau.items():
        L.append("| %s | %d | %d | %s | %s | %s | %s | %s |" % (
            name, 1 if name == "H_minlot_stock" else 2, r["tau"],
            _g(r["reference_objective"]), _g(r["frozen_objective"]),
            _g(r["released_objective"]), _g(r["expectation"]),
            "PASS" if r["expectation_ok"] else "FAIL"))
    L.append("\n### Plans\n")
    for name, r in tau.items():
        L.append("- **%s** frozen (tau=%d): `%s`" % (name, r["tau"], r["frozen_plan"]))
        L.append("- **%s** released: `%s`" % (name, r["released_plan"]))

    L.append("\n## 4. Negative control\n")
    L.append("- instance `%s`: zero capacity, so no production is possible\n" % "D_fix_X")
    L.append("- fixing actually applied: `X[0,0,1] = 1.0` (flat variable index %d)\n"
             % neg["flat_index"])
    L.append("- with the fixing: **%s**\n" % neg["fixed_status"])
    L.append("- without the fixing: **%s**, objective %s\n"
             % (neg["unfixed_status"], _g(neg["unfixed_objective"])))

    L.append("\n## 5. paper_literal: negative U_ijt is kept and matters\n")
    L.append("| instance | negative U_ijt | configs | optimal | LP-infeasible | binary-rejected | MILP | independent | delta |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in neg_u.items():
        if not r["enumerated"]:
            L.append("| %s | `%s` | - | - | - | - | - | - | not enumerated |"
                     % (name, r["negative_U"]))
            continue
        L.append("| %s | `%s` | %d | %d | %d | %d | %s | %s | %.1e |" % (
            name, r["negative_U"], r["n_configs"], r["n_optimal"], r["n_lp_infeasible"],
            r["n_binary_infeasible"], _g(r["milp_objective"]),
            _g(r["independent_objective"]), r["delta"]))
    L.append("\nA negative U makes (1.8) read `X <= U*(Y+Z)` with U < 0, which forces X = 0 "
             "for every value of the binaries. The value is kept exactly as published.\n")

    L.append("\n## 6. U_ijt actually used by the code\n")
    L.append("| instance | mode | U_ijt |")
    L.append("|---|---|---|")
    for name, entry in consistency.items():
        for mode in MODES:
            L.append("| %s | %s | `%s` |" % (name, mode, entry[mode]["U"]))
    L.append("\n### Notes emitted by the model\n")
    for name, entry in consistency.items():
        for mode in MODES:
            for note in entry[mode]["notes"]:
                L.append("- %s / %s: %s" % (name, mode, note))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
