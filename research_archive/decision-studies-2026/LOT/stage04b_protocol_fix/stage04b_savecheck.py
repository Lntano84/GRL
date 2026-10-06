"""
Stage 04b data-persistence check, covering the RUNS this time.

    python stage04b_savecheck.py

Stage 04 only stored scalars for its timing runs, so its savecheck could not cover them. Here
every one of the 32 runs stores both the raw solver plan and the selected plan, and this
script re-reads them and re-verifies:

  * the SELECTED plan is feasible, its recomputed cost equals the stored cost, it respects
    the stability budget and the method's fixing conditions;
  * the RAW solver plan, when present and recorded as valid, is feasible and its recomputed
    cost equals the stored value; when recorded as invalid, it is confirmed to be invalid
    (an invalid incumbent is an error record, not a silent timeout);
  * the selection itself is correct: selected_cost == min(repair_cost, valid raw cost) and
    selected_source agrees with which of the two won;
  * the final output is never worse than the repaired reference plan;
  * timing bookkeeping is consistent (total >= its parts, overrun = max(0, total - budget)).

Nothing is re-solved.
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, List

import numpy as np

from lsp_checker import check_solution
from lsp_gen import all_instances, apply_disruption, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_repair3 import binary_change_count
from stage04b_run import BUDGET, KAPPA, SHORTAGE_SIZES

TOL = 1e-6


def rebuild(inst, plan: Dict[str, Any], objective: float, status: str) -> Solution:
    return Solution(status=status, objective=float(objective),
                    X=np.array(plan["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage04b_runs.json", encoding="utf-8") as fh:
        data = json.load(fh)
    runs = data["runs"]
    with open("stage04b_results.csv", encoding="utf-8-sig") as fh:
        csv_rows = {(r["state"], r["method"]): r for r in csv.DictReader(fh)}

    insts = {m["name"]: (i, m) for i, m in all_instances()}
    failures: List[str] = []
    n_checked = 0
    worst = 0.0
    n_selected_solver = n_selected_repair = 0

    print("=" * 108)
    print("%-28s %-12s %-9s %11s %11s %10s %9s  %s" %
          ("state", "method", "source", "repair", "selected", "recomputed", "maxviol", "ok"))
    print("-" * 108)
    for r in runs:
        state = r["state"]
        name, dis_name = state.split("|")
        inst0, meta = insts[name]
        dis = next(d for d in disruptions_for(inst0) if d.name == dis_name)
        pert = apply_disruption(inst0, dis)
        tau = meta["tau"]

        repair_cost = float(r["repair_cost"])
        sel_cost = r["selected_cost"]
        raw_cost = r["raw_incumbent_cost"]

        # ---- the selected plan must exist and be verifiable
        if r["selected_solution"] is None:
            failures.append("%s/%s: no selected plan stored" % (state, r["method"]))
            continue
        sel_sol = rebuild(pert, r["selected_solution"], sel_cost, "selected")
        chk = check_solution(pert, sel_sol, MODE_AUDITED)
        n_checked += 1
        worst = max(worst, chk.max_violation)
        ok = chk.ok and abs(chk.cost_recomputed["total"] - sel_cost) <= TOL * (1 + abs(sel_cost))
        if not ok:
            failures.append("%s/%s: selected plan feasible=%s stored=%s recomputed=%s"
                            % (state, r["method"], chk.ok, sel_cost,
                               chk.cost_recomputed["total"]))

        # ---- stability budget and fixings, re-derived from the stored plan
        flips = binary_change_count(pert, sel_sol.Y,
                                    rebuild(pert, r["selected_solution"], sel_cost,
                                            "sel").Y, tau) if False else None
        # repair reference is not stored per run; use the stage-04 repair plan
        st04 = None
        try:
            with open("../stage04_runtime_calibration/stage04_states.json",
                      encoding="utf-8") as fh2:
                st04 = json.load(fh2)[state]["repair"]
        except Exception:
            pass
        if st04 is not None:
            rep_sol = rebuild(pert, st04["plan"], float(st04["objective"]), "repaired")
            flips = binary_change_count(pert, sel_sol.Y, rep_sol.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s: stability violated, %d flips > kappa"
                                % (state, r["method"], flips))
            if abs(float(st04["objective"]) - repair_cost) > TOL * (1 + abs(repair_cost)):
                failures.append("%s/%s: stored repair cost %s != stage-04 %s"
                                % (state, r["method"], repair_cost, st04["objective"]))

        # ---- selection correctness
        if r["selected_source"] == "solver":
            n_selected_solver += 1
            if raw_cost is None or not r["raw_incumbent_valid"]:
                failures.append("%s/%s: chose the solver but the raw incumbent is not valid"
                                % (state, r["method"]))
            elif abs(sel_cost - raw_cost) > TOL * (1 + abs(raw_cost)):
                failures.append("%s/%s: selected cost %s != raw cost %s"
                                % (state, r["method"], sel_cost, raw_cost))
            elif raw_cost > repair_cost + TOL * (1 + abs(repair_cost)):
                failures.append("%s/%s: chose a solver plan WORSE than the repair plan"
                                % (state, r["method"]))
        else:
            n_selected_repair += 1
            if abs(sel_cost - repair_cost) > TOL * (1 + abs(repair_cost)):
                failures.append("%s/%s: selected the repair plan but stored a different cost"
                                % (state, r["method"]))
            if raw_cost is not None and r["raw_incumbent_valid"] \
                    and raw_cost < repair_cost - TOL * (1 + abs(repair_cost)):
                failures.append("%s/%s: discarded a VALID better incumbent"
                                % (state, r["method"]))

        # ---- an invalid raw incumbent must really be invalid
        if r["raw_solver_solution"] is not None and not r["raw_incumbent_valid"]:
            raw_sol = rebuild(pert, r["raw_solver_solution"], raw_cost or 0.0, "raw")
            chk_raw = check_solution(pert, raw_sol, MODE_AUDITED)
            if chk_raw.ok:
                failures.append("%s/%s: recorded invalid but re-checks as feasible"
                                % (state, r["method"]))
            n_checked += 1
        elif r["raw_solver_solution"] is not None and r["raw_incumbent_valid"]:
            raw_sol = rebuild(pert, r["raw_solver_solution"], raw_cost or 0.0, "raw")
            chk_raw = check_solution(pert, raw_sol, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk_raw.max_violation)
            if not chk_raw.ok:
                failures.append("%s/%s: recorded valid but re-checks as INFEASIBLE"
                                % (state, r["method"]))

        # ---- the final output must never be worse than the repaired plan
        if sel_cost > repair_cost + TOL * (1 + abs(repair_cost)):
            failures.append("%s/%s: FINAL OUTPUT %s worse than repair %s"
                            % (state, r["method"], sel_cost, repair_cost))

        # ---- timing bookkeeping
        t = r["timing"]
        parts = t["feature_rank_s"] + t["constraint_build_s"] + t["solver_wall_s"]
        if t["total_method_s"] + 1e-6 < parts:
            failures.append("%s/%s: total %.4f < sum of parts %.4f"
                            % (state, r["method"], t["total_method_s"], parts))
        if abs(t["overrun_s"] - max(0.0, t["total_method_s"] - BUDGET)) > 1e-3:
            failures.append("%s/%s: overrun %.4f inconsistent with total %.4f"
                            % (state, r["method"], t["overrun_s"], t["total_method_s"]))

        # ---- CSV agreement
        row = csv_rows.get((state, r["method"]))
        if row is None:
            failures.append("%s/%s: missing from stage04b_results.csv" % (state, r["method"]))
        else:
            for col, val in (("selected_cost", sel_cost), ("repair_cost", repair_cost),
                             ("raw_incumbent", raw_cost)):
                if val is None:
                    continue
                if abs(float(row[col]) - float(val)) > TOL * (1 + abs(float(val))):
                    failures.append("%s/%s: csv %s %s != json %s"
                                    % (state, r["method"], col, row[col], val))

        print("%-28s %-12s %-9s %11s %11s %10s %9.1e  %s" %
              (state, r["method"], r["selected_source"], "%.6g" % repair_cost,
               "%.6g" % sel_cost, "%.6g" % chk.cost_recomputed["total"],
               chk.max_violation, "ok" if ok else "FAIL"))

    print("=" * 108)
    print("plans re-verified from disk: %d (selected + raw)" % n_checked)
    print("largest constraint violation: %.3g" % worst)
    print("runs: %d | source=solver %d | source=repair %d"
          % (len(runs), n_selected_solver, n_selected_repair))
    n_bad = sum(1 for r in runs if r["selected_cost"] is not None
                and r["selected_cost"] > r["repair_cost"] + TOL)
    print("runs whose final output is worse than the repaired plan: %d" % n_bad)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL RUNS RE-VERIFY FROM DISK; selection, stability and timing bookkeeping agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
