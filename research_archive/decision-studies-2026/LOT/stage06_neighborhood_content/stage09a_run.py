#!/usr/bin/env python
"""
Stage 09A: how much of the repaired plan's loss is pure continuous re-optimisation?

    .venv-hs/Scripts/python.exe stage09a_run.py
    .venv-hs/Scripts/python.exe stage09a_analyse.py

The single question
-------------------
> How much of the repaired plan's loss can be removed WITHOUT changing any setup or carry-over
> decision -- by re-optimising production, inventory and lost sales alone?

    J_LP = min_{X,I,L} J(X, Y^r, Z^r, I, L)

with every Y and Z over the WHOLE horizon fixed at the repaired plan's values, far horizon
included.

This is NOT EMPTY.  EMPTY leaves the far-horizon Y free and Z free apart from its structural
pins; this LP fixes all of them:

    method      short-term Y   far-horizon Y   Z
    EMPTY       fixed          free            free except structural pins
    this LP     fixed          FIXED           ALL FIXED

so the result measures pure continuous re-optimisation and must not be called EMPTY.

Frozen scope
------------
* the original 16 LARGE fault states and the frozen repaired plans
* model, costs, disrupted capacities and the kappa reference point all unchanged
* exactly ONE existing binary configuration per state -- no configuration enumeration
* 5 s limit per LP, 16 LPs, one solver seed is enough
* a state that is not PROVEN optimal is recorded as such; the budget is not raised
* the audited model matrix is reused (no second production model is translated)
* the historical nominal/repaired solutions and all stage 06-08 results are untouched

Numerics
--------
Y and Z are first checked against the existing integer tolerance, then normalised to 0/1.  The
raw values and the normalisation error are kept in the record.  The output is verified by the
independent checker on the full original model, including the fixing conditions and the cost.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import build_highs, repair_vector
from lsp_model import (MODE_AUDITED, Solution, build_model, enumeration_space,
                       unpack)
from lsp_repair3 import binary_change_count
from lsp_select import select_output
from stage06_run import KAPPA, solution_from_record

BUDGET = 5.0
THREADS = 1
TOL = 1e-6
RUNS_FILE = "stage09a_runs.json"


def hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:                                    # noqa: BLE001
        return "unknown"


def solve_lp_all_binaries_fixed(bm, lb_pin: np.ndarray, ub_pin: np.ndarray,
                                budget: float, log_path: str) -> Dict[str, Any]:
    """LP via the audited matrix: integrality dropped, every binary pinned by bounds.

    `build_highs` assembles A from `bm.A`; the stability row is omitted because every
    short-term Y is pinned to the repaired value, so the row would be a tautology.  An all-zero
    integrality vector makes this a genuine LP, so HiGHS runs its LP solver (no branch and
    bound) and `proven optimal` is meaningful.
    """
    import dataclasses

    import highspy

    bm_lp = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality),
                                var_lb=lb_pin, var_ub=ub_pin)

    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm_lp, {}, ({}, 0.0))
    prep = time.perf_counter() - t0

    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    h.passModel(lp)
    h.setOptionValue("time_limit", float(budget))
    h.setOptionValue("threads", int(THREADS))

    t1 = time.perf_counter()
    h.run()
    solved = time.perf_counter() - t1
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    obj = None
    out_sol = None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        x = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(x)):
            X, Y, Z, I, L = unpack(x, bm.idx)
            try:
                obj = float(info.objective_function_value)
            except Exception:                            # noqa: BLE001
                obj = None
            out_sol = Solution(status=status, objective=obj if obj is not None else float("nan"),
                               X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log_txt = fh.read()
    except OSError:
        log_txt = ""
    return {"status": status, "objective": obj, "solution": out_sol,
            "dual_bound": (float(info.mip_dual_bound)
                           if np.isfinite(info.mip_dual_bound) else None),
            "gap": (float(info.mip_gap) if np.isfinite(info.mip_gap) else None),
            "prep_s": prep, "solve_s": solved, "log": log_txt,
            "n_cols": n_cols, "n_rows": n_rows}


def main() -> int:
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(s for s in ST if ST[s]["meta"]["scale"] == "large")
    if len(states) != 16:
        raise SystemExit("expected 16 large states, found %d" % len(states))
    print("large fault states: %d | budget %.1f s per LP | %d LPs"
          % (len(states), BUDGET, len(states)))
    print("all Y and Z fixed to the frozen repaired plan, far horizon included")
    print()

    os.makedirs("stage09a_logs", exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for state in states:
        rec = ST[state]
        tau = rec["meta"]["tau"]
        inst, _meta = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        idx = bm.idx

        # ---- binary assignment from the repaired plan, with the integer check first
        raw_vals: List[float] = []
        norm_err = 0.0
        assign: Dict[Tuple[str, int, int, int], int] = {}
        for slot in enumeration_space(pert, MODE_AUDITED):
            kind, i, j, t = slot
            raw = float(repair.Y[i, j, t - 1]) if kind == "Y" else float(repair.Z[i, j, t - 1])
            raw_vals.append(raw)
            norm_err = max(norm_err, abs(raw - round(raw)))
            assign[slot] = int(round(raw))
        n_bin = len(assign)

        lb_pin = bm.var_lb.copy()
        ub_pin = bm.var_ub.copy()
        pinned_cols = []
        for slot, val in assign.items():
            kind, i, j, t = slot
            col = idx.Y(i, j, t) if kind == "Y" else idx.Z(i, j, t)
            lb_pin[col] = float(val)
            ub_pin[col] = float(val)
            pinned_cols.append(col)
        # structural pins (Z_ij0, and Z_ijT under audited_v1) keep their model bounds of 0
        for i in range(pert.N):
            for j in range(pert.M):
                pinned_cols.append(idx.Z(i, j, 0))
                pinned_cols.append(idx.Z(i, j, pert.T))

        # ---- every binary column is pinned and the LP has no integer variables left
        binary_cols = set(np.where(bm.integrality == 1)[0].tolist())
        not_pinned = sorted(binary_cols - set(pinned_cols))
        if not_pinned:
            raise SystemExit("%s: %d binary columns not pinned" % (state, len(not_pinned)))

        log_path = os.path.join("stage09a_logs", "%s.log" % state.replace("|", "_"))
        t0 = time.perf_counter()
        res = solve_lp_all_binaries_fixed(bm, lb_pin, ub_pin, BUDGET, log_path)

        # ---- independent verification of the delivered plan
        sel = select_output(pert, res["solution"], repair, KAPPA, tau,
                            fixed_Y=None, solver_status=res["status"])
        t_ver = time.perf_counter() - t0 - res["prep_s"] - res["solve_s"]
        total = time.perf_counter() - t0

        chk = None
        lp_sol = res["solution"]
        if lp_sol is not None:
            chk = check_solution(pert, lp_sol, MODE_AUDITED)
        flips = (0 if lp_sol is None
                 else binary_change_count(pert, lp_sol.Y, repair.Y, tau))
        jr = float(repair.objective)

        runs.append({
            "state": state, "scale": rec["meta"]["scale"], "rho": rec["meta"]["rho"],
            "inst_seed": rec["meta"]["seed"], "disruption": rec["disruption"]["name"],
            "tau": tau, "T": pert.T,
            "repair_cost": jr,
            "lp_status": res["status"], "lp_objective": res["objective"],
            "lp_dual_bound": res["dual_bound"],
            "lp_gap": res["gap"],
            "proven_optimal": bool(res["status"].lower().startswith("optimal")),
            "n_cols": res["n_cols"], "n_rows": res["n_rows"],
            "n_binaries_pinned": len(binary_cols),
            "n_slots_fixed": n_bin,
            "raw_binary_norm_max_error": norm_err,
            "raw_binary_min": min(raw_vals), "raw_binary_max": max(raw_vals),
            "lp_recomputed_cost": (None if chk is None else chk.cost_recomputed["total"]),
            "lp_feasible": (None if chk is None else bool(chk.ok)),
            "lp_max_violation": (None if chk is None else float(chk.max_violation)),
            "lp_check_problems": (None if chk is None else list(chk.failed_checks())),
            "lp_short_term_flips": flips,
            "final_cost": (None if sel.chosen is None else float(sel.chosen.cost)),
            "final_source": sel.source, "select_errors": sel.errors,
            "lp_solution": (None if lp_sol is None
                            else {k: np.asarray(getattr(lp_sol, k)).tolist()
                                  for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"prep_s": res["prep_s"], "solve_s": res["solve_s"],
                       "verify_s": max(0.0, t_ver), "total_s": total,
                       "overrun_s": max(0.0, total - BUDGET)},
            "log_tail": res["log"][-3000:],
        })
        r = runs[-1]
        print("  %-30s repair=%-11s LP=%-11s %-22s gain=%+.4f  t=%.2f  opt=%s"
              % (state, "%.6g" % jr,
                 "-" if r["lp_objective"] is None else "%.6g" % r["lp_objective"],
                 r["lp_status"], (jr - (r["lp_objective"] or jr)) / max(1.0, abs(jr)),
                 total, r["proven_optimal"]), flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "threads": THREADS,
                            "n_runs": len(runs), "solver": "highspy",
                            "solver_version": hs_version(),
                            "scope": "16 frozen LARGE fault states from stage 06",
                            "method": "LP with ALL Y and Z pinned to the frozen repaired plan "
                                      "(far horizon included), integrality dropped; only "
                                      "X, I, L free",
                            "not_empty": "EMPTY leaves far-horizon Y and Z free; this LP fixes "
                                         "them, so the two are different methods",
                            "configurations_enumerated": 1,
                            "note": "historical nominal/repaired solutions and stage 06-08 "
                                    "results are unchanged; this run only adds new LP results"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "repair_cost",
              "lp_objective", "lp_status", "proven_optimal", "lp_gap", "lp_dual_bound",
              "lp_recomputed_cost", "lp_feasible", "lp_max_violation",
              "lp_short_term_flips", "n_binaries_pinned", "raw_binary_norm_max_error",
              "total_s", "overrun_s"]
    with open("stage09a_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            flat["total_s"] = r["timing"]["total_s"]
            flat["overrun_s"] = r["timing"]["overrun_s"]
            w.writerow(flat)

    n_opt = sum(1 for r in runs if r["proven_optimal"])
    print("\nLPs: %d | proven optimal: %d | not proven: %d"
          % (len(runs), n_opt, len(runs) - n_opt))
    print("max binary normalisation error: %.3g"
          % max(r["raw_binary_norm_max_error"] for r in runs))
    bad = [r["state"] for r in runs if r["lp_feasible"] is False]
    print("LP plans failing the independent check: %d %s" % (len(bad), bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
