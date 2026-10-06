#!/usr/bin/env python
"""
Stage 11 mechanism diagnostics: actual reduction, LP relaxation, plan transferability.

    .venv-hs/Scripts/python.exe stage11_diagnostics.py

Three diagnostics, all OFFLINE.  None of them selects a configuration and none of them gives any
method extra information.

(1) ACTUAL REDUCTION, not just the released count.
    Records the binary count before and after fixing, and whatever the solver's normal log
    reliably exposes about the presolve sizes.  This matters because the short-term Y is already
    fixed, so some slots may be IMPLICITLY fixed by the constraints: matching the raw slot count
    does NOT guarantee matching the real search dimension.  If AB's advantage comes with stronger
    variable elimination, that is a mechanism lead and the reduction explanation cannot be
    declared settled.  Any field the interface does not expose is recorded as MISSING rather
    than reconstructed through a new solver interface.

(2) LP RELAXATION, one per (state, configuration).
    Apply the configuration's fixing conditions and relax the REMAINING binaries to [0, 1].
    Record the relaxation objective, whether it proved optimal, and the time.  Cap 5 s each.
    These are lower bounds for DIFFERENT restricted problems, NOT a common bound for FULL.
    A higher relaxation objective can simply mean a stronger restriction, so it must not be read
    as "this algorithm is better"; combined with the final feasible solution it shows whether the
    distance between the feasible solution and the relaxation bound changes appreciably.

(3) PLAN TRANSFER.
    Check directly, without re-solving, whether the plan AB found satisfies each matched-random
    configuration's fixing conditions.  If a matched configuration ADMITS the AB plan but did not
    find equal quality within the budget, then the gap cannot be attributed to that random
    subproblem excluding the plan.  If the transfer fails, only record which fixing conditions
    were violated -- that says nothing about the random configuration's optimum.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import stage11_run as S
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_checker import check_solution
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_release import (LP_BUDGET, build_state_context, pinned_bounds, solve_lp_fixed,
                         stage11_columns)
from stage06_run import solution_from_record
from lsp_highs import build_highs

RELAX_BUDGET = 5.0
OUT_FILE = "stage11_diagnostics.json"


def solve_relaxation(C, method: str, log_path: str) -> Dict[str, Any]:
    """Apply the fixing conditions, relax the remaining binaries to [0,1], solve the LP."""
    import dataclasses

    import highspy

    bm = C["bm"]
    rel, fixed, _fz = stage11_columns(C, method)
    lb = bm.var_lb.copy()
    ub = bm.var_ub.copy()
    for slot, val in C["assign"].items():
        kind, i, j, t = slot
        col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
        lb[col] = float(val)
        ub[col] = float(val)
    # relax the RELEASED binaries to [0, 1]; continuous columns keep their own bounds
    for (kind, i, j, t) in rel:
        col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
        lb[col] = 0.0
        ub[col] = 1.0

    bm_lp = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality),
                                var_lb=lb, var_ub=ub)
    h, lp, n_cols, n_rows = build_highs(bm_lp, {}, ({}, 0.0))
    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("time_limit", RELAX_BUDGET)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    wall = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    obj = None
    try:
        v = float(info.objective_function_value)
        if np.isfinite(v):
            obj = v
    except Exception:                                    # noqa: BLE001
        obj = None
    prov = False
    try:
        prov = "optimal" in status.lower()
    except Exception:                                    # noqa: BLE001
        prov = False
    return {"status": status, "objective": obj, "proven_optimal": prov,
            "wall_s": wall, "n_released": len(rel)}


def main() -> int:
    with open("stage06_states.json", encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open("stage11_matched_random.json", encoding="utf-8") as fh:
        MATCH = json.load(fh)
    with open("stage11_runs.json", encoding="utf-8") as fh:
        RUNS = json.load(fh)["runs"]
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    idx = {(r["state"], r["method"], int(r["seed"])): r for r in RUNS}

    out: Dict[str, Any] = {"reduction": {}, "relaxation": {}, "transfer": {}}
    os.makedirs("stage11_logs", exist_ok=True)

    print("=" * 116)
    print("(1) ACTUAL REDUCTION: released slots vs binary columns left free")
    print("%-30s %-8s %8s %10s %10s %12s" %
          ("state", "method", "released", "binary_in", "bin_after", "presolve"))
    print("-" * 116)
    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "inst": pert, "repair": repair})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in S.MR_NAMES}
        n_bin_in = len(C["assign"])
        out["reduction"][st] = {}
        for method in S.METHODS:
            rel, fixed, fz = stage11_columns(C, method)
            n_bin_after = n_bin_in - len(rel)
            # what the normal log reliably exposes after a presolve pass
            presolve = None
            try:
                import dataclasses

                import highspy
                lb = bm.var_lb.copy()
                ub = bm.var_ub.copy()
                for slot, val in C["assign"].items():
                    kind, i, j, t = slot
                    col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
                    lb[col] = float(val)
                    ub[col] = float(val)
                bm2 = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality),
                                          var_lb=lb, var_ub=ub)
                h, lp, n_cols, n_rows = build_highs(bm2, fixed, ({}, 0.0))
                h.setOptionValue("output_flag", False)
                h.setOptionValue("presolve", "on")
                h.passModel(lp)
                h.run()
                presolve = {"num_col": int(h.getNumCol()), "num_row": int(h.getNumRow())}
            except Exception as exc:                     # noqa: BLE001
                presolve = {"error": repr(exc)[:120]}
            out["reduction"][st][method] = {
                "released": len(rel), "binary_before": n_bin_in,
                "binary_after": n_bin_after, "fixed_columns": len(fixed),
                "presolve": presolve,
            }
            if method in ("AB", "ABC", "MR-Z-1"):
                print("%-30s %-8s %8d %10d %10d %12s"
                      % (st, method, len(rel), n_bin_in, n_bin_after,
                         "MISSING" if presolve is None or "error" in (presolve or {})
                         else "%d/%d" % (presolve["num_col"], presolve["num_row"])))
    print()

    # ---------------------------------------------------------------- (2) relaxation
    print("=" * 116)
    print("(2) LP RELAXATION with the released binaries relaxed to [0,1]  (cap %.0f s each)"
          % RELAX_BUDGET)
    print("%-30s %-8s %14s %-10s %8s" % ("state", "method", "relax objective", "status", "t(s)"))
    print("-" * 116)
    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"]})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in S.MR_NAMES}
        out["relaxation"][st] = {}
        for method in S.METHODS:
            res = solve_relaxation(C, method, os.path.join("stage11_logs", "relax.log"))
            out["relaxation"][st][method] = res
            if method in ("AB", "ABC", "MR-Z-1"):
                print("%-30s %-8s %14s %-10s %8.2f"
                      % (st, method, "MISSING" if res["objective"] is None
                         else "%.6g" % res["objective"], res["status"][:10], res["wall_s"]))
    print()
    print("  NOTE: these are lower bounds for DIFFERENT restricted problems, not a common bound")
    print("        for FULL.  A higher value can simply mean a stronger restriction.")
    print()

    # ---------------------------------------------------------------- (3) transfer
    print("=" * 116)
    print("(3) PLAN TRANSFER: does the AB plan satisfy each matched-random configuration?")
    print("-" * 116)
    n_admit = 0
    n_checked = 0
    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"]})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in S.MR_NAMES}
        out["transfer"][st] = {}
        for seed in S.SEEDS:
            ab = idx.get((st, "AB", seed))
            if ab is None or ab["final_solution"] is None:
                continue
            sol = Solution(status="ab", objective=float(ab["final_cost"]),
                           X=np.array(ab["final_solution"]["X"], float).reshape(pert.N, pert.M, pert.T),
                           Y=np.array(ab["final_solution"]["Y"], float).reshape(pert.N, pert.M, pert.T),
                           Z=np.array(ab["final_solution"]["Z"], float).reshape(pert.N, pert.M, pert.T),
                           I=np.array(ab["final_solution"]["I"], float).reshape(pert.N, pert.T),
                           L=np.array(ab["final_solution"]["L"], float).reshape(pert.N, pert.T),
                           mode=MODE_AUDITED)
            for name in S.MR_NAMES:
                n_checked += 1
                fixedZ = C["mr_sets"][name]
                viol = []
                for (i, j, t) in fixedZ:
                    if abs(float(sol.Z[i, j, t - 1]) - float(repair.Z[i, j, t - 1])) > 0.5:
                        viol.append((i, j, t))
                # the short-term Y is fixed in BOTH, so it can never be the obstacle
                shortY_viol = sum(
                    1 for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)
                    if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5)
                admits = (not viol) and shortY_viol == 0
                n_admit += int(admits)
                mr = idx.get((st, name, seed))
                out["transfer"][st]["%s|s%d" % (name, seed)] = {
                    "admits_AB_plan": bool(admits), "n_z_violations": len(viol),
                    "short_Y_violations": int(shortY_viol),
                    "AB_cost": float(ab["final_cost"]),
                    "MR_cost": None if mr is None else mr["final_cost"],
                    "violations_sample": [list(u) for u in viol[:5]],
                }
                if seed == 0:
                    print("  %-30s %-8s admits=%-5s Z_viol=%-3d  AB=%-11s MR=%s"
                          % (st, name, admits, len(viol),
                             "%.6g" % ab["final_cost"],
                             "n/a" if mr is None else "%.6g" % (mr["final_cost"] or float("nan"))))
    print()
    print("  AB plans checked against matched configurations: %d | admitted: %d"
          % (n_checked, n_admit))
    if n_admit and n_admit < n_checked:
        print("  For the admitted cases a matched configuration COULD have used the AB plan, so a")
        print("  quality gap there cannot be attributed to the subproblem excluding that plan.")
    if n_admit == 0:
        print("  No matched configuration admits the AB plan, so this diagnostic gives no")
        print("  'the subproblem excluded the good plan' statement -- only which fixing")
        print("  conditions were violated.  It says nothing about their optima.")
    print("  NOTE: a violation here does NOT imply the matched configuration's optimum is worse.")

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nwrote %s" % OUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
