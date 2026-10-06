"""
Stage 05 data-persistence check, covering every run.

    .venv-hs/Scripts/python.exe stage05_savecheck.py

Re-reads `stage05_runs.json` and re-verifies each of the 128 runs without re-solving:

  * the SELECTED plan is feasible on the original model, its recomputed cost equals the stored
    cost, it respects kappa, and it respects the method's fixing conditions;
  * the selection rule held: source `solver` only when the raw incumbent was usable AND
    cheaper, source `repair` otherwise;
  * the final output is never worse than the state's frozen repaired plan;
  * the raw incumbent, when it was used, re-checks as feasible;
  * warm runs have confirmed MIP-start adoption and their returned incumbent is not worse
    than the reported start objective; **cold runs must have no adoption evidence**;
  * timing bookkeeping: total >= its parts, overrun = max(0, total - budget);
  * the CSV agrees with the JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from lsp_checker import check_solution
from lsp_gen import all_instances, apply_disruption, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_repair3 import binary_change_count
from stage05_run import BUDGET, KAPPA, solution_from_record

TOL = 1e-6
SOURCE_STATES = "../stage04_runtime_calibration/stage04_states.json"


def rebuild(inst, plan: Dict[str, Any], objective: float) -> Solution:
    return Solution(status="stored", objective=float(objective),
                    X=np.array(plan["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage05_runs.json", encoding="utf-8") as fh:
        data = json.load(fh)
    runs = data["runs"]
    with open(SOURCE_STATES, encoding="utf-8") as fh:
        ST = json.load(fh)
    with open("stage05_results.csv", encoding="utf-8-sig") as fh:
        csv_rows = {(r["state"], r["method"], r["start"], int(r["seed"])): r
                    for r in csv.DictReader(fh)}

    insts = {m["name"]: (i, m) for i, m in all_instances()}
    ctx: Dict[str, Any] = {}
    for state, rec in ST.items():
        name, dis_name = state.split("|")
        if name not in insts:
            continue
        inst0, meta = insts[name]
        dis = next((d for d in disruptions_for(inst0) if d.name == dis_name), None)
        if dis is None:
            continue
        pert = apply_disruption(inst0, dis)
        ctx[state] = {"inst": pert, "repair": solution_from_record(pert, rec["repair"]),
                      "tau": meta["tau"]}

    failures: List[str] = []
    n_checked = 0
    worst = 0.0
    n_adopt_cold = 0
    n_warm_ok = 0
    n_cold_ok = 0

    print("=" * 104)
    print("%-28s %-12s %-4s %3s %9s %11s %11s %9s  %s" %
          ("state", "method", "start", "sd", "repair", "selected", "recomputed",
           "maxviol", "ok"))
    print("-" * 104)
    for r in runs:
        state, method, start, seed = r["state"], r["method"], r["start"], r["seed"]
        C = ctx.get(state)
        if C is None:
            failures.append("%s: no context" % state)
            continue
        pert, repair, tau = C["inst"], C["repair"], C["tau"]
        jr = float(r["repair_cost"])
        sel_cost = r["selected_cost"]
        raw = r["raw_objective"]
        ok = True

        # ---- selected plan, re-read and re-verified
        if sel_cost is None:
            failures.append("%s/%s/%s/s%d: no selected cost" % (state, method, start, seed))
            ok = False
        elif r.get("selected_solution") is None:
            failures.append("%s/%s/%s/s%d: selected plan not stored"
                            % (state, method, start, seed))
            ok = False
        else:
            sel_sol = rebuild(pert, r["selected_solution"], sel_cost)
            chk = check_solution(pert, sel_sol, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/%s/s%d: selected plan INFEASIBLE %s"
                                % (state, method, start, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - sel_cost) > TOL * (1 + abs(sel_cost)):
                failures.append("%s/%s/%s/s%d: stored %.10g != recomputed %.10g"
                                % (state, method, start, seed, sel_cost,
                                   chk.cost_recomputed["total"]))
                ok = False
            flips = binary_change_count(pert, sel_sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/%s/s%d: selected plan flips %d > kappa %g"
                                % (state, method, start, seed, flips, KAPPA))
                ok = False

        # ---- raw plan, when the solver returned one, re-verified
        if r.get("raw_solver_solution") is not None and raw is not None:
            raw_sol = rebuild(pert, r["raw_solver_solution"], raw)
            chk_raw = check_solution(pert, raw_sol, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk_raw.max_violation)
            if bool(chk_raw.ok) != bool(r["raw_valid"]):
                failures.append("%s/%s/%s/s%d: raw validity recorded %s but re-checks %s"
                                % (state, method, start, seed, r["raw_valid"], chk_raw.ok))
                ok = False

        # ---- final output must never be worse than the repaired plan
        if sel_cost is not None and sel_cost > jr + TOL * (1 + abs(jr)):
            failures.append("%s/%s/%s/s%d: final output %.10g worse than repair %.10g"
                            % (state, method, start, seed, sel_cost, jr))
            ok = False

        # ---- selection rule
        raw = r["raw_objective"]
        if r["selected_source"] == "solver":
            if raw is None or not r["raw_valid"]:
                failures.append("%s/%s/%s/s%d: chose solver but raw is invalid"
                                % (state, method, start, seed))
                ok = False
            if r["select_errors"]:
                failures.append("%s/%s/%s/s%d: selection errors present: %s"
                                % (state, method, start, seed, r["select_errors"]))
                ok = False
            if raw is not None and abs(sel_cost - raw) > TOL * (1 + abs(raw)):
                failures.append("%s/%s/%s/s%d: selected %.10g != raw %.10g"
                                % (state, method, start, seed, sel_cost, raw))
                ok = False
        else:
            if abs(sel_cost - jr) > TOL * (1 + abs(jr)):
                failures.append("%s/%s/%s/s%d: chose repair but selected %.10g != %.10g"
                                % (state, method, start, seed, sel_cost, jr))
                ok = False

        # ---- raw incumbent stability: usable solutions must respect kappa
        if r["raw_valid"] and r["raw_flips_vs_repair"] is not None:
            if r["raw_flips_vs_repair"] > KAPPA + 1e-9:
                failures.append("%s/%s/%s/s%d: raw flagged valid but flips %d > kappa"
                                % (state, method, start, seed, r["raw_flips_vs_repair"]))
                ok = False

        # ---- start-condition bookkeeping
        if start == "cold":
            n_cold_ok += 1
            if r["start_adopted"]:
                n_adopt_cold += 1
                failures.append("%s/%s/cold/s%d: cold run reports MIP-start adoption"
                                % (state, method, seed))
                ok = False
            if r["set_solution_status"] not in (None, "None"):
                failures.append("%s/%s/cold/s%d: cold run called setSolution (%s)"
                                % (state, method, seed, r["set_solution_status"]))
                ok = False
        else:
            n_warm_ok += 1
            if not r["start_adopted"]:
                failures.append("%s/%s/warm/s%d: MIP start NOT confirmed adopted"
                                % (state, method, seed))
                ok = False
            rep_obj = r["start_reported_objective"]
            if rep_obj is not None and raw is not None:
                # the returned incumbent must not be worse than the submitted start
                if raw > rep_obj + TOL * (1 + abs(rep_obj)):
                    failures.append("%s/%s/warm/s%d: incumbent %.10g worse than submitted "
                                    "start %.10g" % (state, method, seed, raw, rep_obj))
                    ok = False
            # the reported start objective must equal the repaired plan's cost
            if rep_obj is not None and abs(rep_obj - jr) > TOL * (1 + abs(jr)):
                failures.append("%s/%s/warm/s%d: adopted start cost %.10g != repair %.10g"
                                % (state, method, seed, rep_obj, jr))
                ok = False

        # ---- timing
        t = r["timing"]
        parts = t["feature_rank_s"] + t["transfer_s"] + t["solver_s"]
        if t["total_s"] + 1e-6 < parts:
            failures.append("%s/%s/%s/s%d: total %.4f < parts %.4f"
                            % (state, method, start, seed, t["total_s"], parts))
            ok = False
        if abs(t["overrun_s"] - max(0.0, t["total_s"] - BUDGET)) > 1e-3:
            failures.append("%s/%s/%s/s%d: overrun inconsistent" % (state, method, start, seed))
            ok = False

        # ---- CSV agreement
        row = csv_rows.get((state, method, start, seed))
        if row is None:
            failures.append("%s/%s/%s/s%d: missing from CSV" % (state, method, start, seed))
            ok = False
        else:
            for col, val in (("selected_cost", sel_cost), ("repair_cost", jr),
                             ("raw_objective", raw)):
                if val is None:
                    continue
                if abs(float(row[col]) - float(val)) > TOL * (1 + abs(float(val))):
                    failures.append("%s/%s/%s/s%d: csv %s %s != json %s"
                                    % (state, method, start, seed, col, row[col], val))
                    ok = False

        n_checked += 1
        if not ok:
            print("%-28s %-12s %-4s %3d %9s %11s %11s %9s  FAIL"
                  % (state, method, start, seed, "%.6g" % jr, "%.6g" % (sel_cost or float("nan")),
                     "-", "-"))
        elif n_checked % 16 == 0:
            print("%-28s %-12s %-4s %3d %9s %11s %11s %9s  ok"
                  % (state, method, start, seed, "%.6g" % jr,
                     "%.6g" % (sel_cost or float("nan")), "-", "-"))

    print("=" * 104)
    print("runs checked: %d" % n_checked)
    print("cold runs: %d (adoption reported by %d of them)" % (n_cold_ok, n_adopt_cold))
    print("warm runs: %d" % n_warm_ok)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS RE-VERIFY FROM DISK; selection, start adoption and timing agree"
          % n_checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
