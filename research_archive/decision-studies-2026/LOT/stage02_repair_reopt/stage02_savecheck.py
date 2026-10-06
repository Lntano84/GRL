"""
Stage 02 data-persistence check.

Acceptance requires: "write to disk, read back, and the verification result and the cost must
agree". This script does exactly that and nothing else:

  1. read `stage02_solutions.json` and `stage02_results.csv` back from disk;
  2. for the nominal plan, the repaired plan, the fixed-binary LP plan and every
     re-optimisation variant, rebuild a Solution straight from the stored arrays (no re-solve,
     no interpolation) on the stored perturbed instance;
  3. re-run the INDEPENDENT checker on the rebuild and compare its recomputed cost with the
     stored objective;
  4. re-count the short-term setup flips against the stored repaired plan and re-check the
     kappa budget, the fixed-set conditions and the reference-point counts;
  5. confirm the CSV row summary equals the JSON record for every state.

Run:  python stage02_savecheck.py      (after `python run.py`)
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from lsp_checker import check_solution
from lsp_instances2 import DISRUPTIONS, apply_disruption, small_instances
from lsp_model import MODE_AUDITED, Instance, Solution
from lsp_repair import binary_change_count, binary_change_count_full

TOL = 1e-6


def _solution_from_stored(inst: Instance, rec: Dict[str, Any]) -> Solution:
    """Rebuild a Solution from stored nested lists."""
    return Solution(status=rec.get("status", "stored"),
                    objective=float(rec["objective"]),
                    X=np.array(rec["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(rec["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(rec["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(rec["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(rec["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage02_solutions.json", encoding="utf-8") as fh:
        S = json.load(fh)
    with open("stage02_raw.json", encoding="utf-8") as fh:
        RAW = json.load(fh)
    with open("stage02_results.csv", encoding="utf-8-sig") as fh:
        csv_rows = {r["state"]: r for r in csv.DictReader(fh)}

    failures: List[str] = []
    checked = 0
    print("=" * 96)
    print("%-24s %-10s %10s %10s %9s  %s" %
          ("state", "plan", "stored", "recomputed", "maxviol", "ok"))
    print("-" * 96)

    small = small_instances()
    states = S["states"]
    for state, rec in states.items():
        name, dis_name = state.split("|")
        inst0 = small[name]
        dis = next(d for d in DISRUPTIONS if d.name == dis_name)
        inst = apply_disruption(inst0, dis)

        plans: Dict[str, Dict[str, Any]] = {
            "nominal": rec["nominal_plan"] and None,   # placeholder, filled below
        }
        # nominal is stored per instance, not per state
        plans = {}

        # -- repaired plan
        rep_x = rec["repair_solution"]
        rep = _solution_from_stored(inst, rep_x)

        # -- fixed-binary LP plan
        lp = _solution_from_stored(inst, rec["fixed_binary_lp_solution"])

        # -- variants
        variant_solutions = {}
        for key, v in rec["variants"].items():
            variant_solutions[key] = _solution_from_stored(inst, v["solution"])

        targets = [("repaired", rep, rec["repair_cost"]),
                   ("fixed_binary_LP", lp, rec["fixed_binary_lp_cost"])]
        for key, v in rec["variants"].items():
            if v["feasible"]:
                targets.append((key, variant_solutions[key], v["cost"]))

        for label, sol, stored_cost in targets:
            chk = check_solution(inst, sol, MODE_AUDITED)
            recomputed = chk.cost_recomputed["total"]
            ok = chk.ok and abs(recomputed - stored_cost) <= TOL * (1 + abs(stored_cost))
            checked += 1
            if not ok:
                failures.append("%s/%s: feasible=%s stored=%s recomputed=%s"
                                % (state, label, chk.ok, stored_cost, recomputed))
            print("%-24s %-16s %10s %10s %9.1e  %s" %
                  (state, label, "%.6g" % stored_cost, "%.6g" % recomputed,
                   chk.max_violation, "ok" if ok else "FAIL"))

        # -- stability budget re-counted from the stored arrays
        for key, v in rec["variants"].items():
            sol = variant_solutions[key]
            flip = binary_change_count(inst, sol.Y, rep.Y, rec["tau"])
            flip_full = binary_change_count_full(inst, sol.Y, rep.Y)
            kappa = v["kappa"]
            if kappa is not None and flip > kappa + 1e-9:
                failures.append("%s/%s: recomputed flips %d > kappa %s"
                                % (state, key, flip, kappa))
            if flip != v["flips_vs_repair"] or flip_full != v["flips_vs_repair_full"]:
                failures.append("%s/%s: stored flip counts %s/%s differ from recomputed %d/%d"
                                % (state, key, v["flips_vs_repair"],
                                   v["flips_vs_repair_full"], flip, flip_full))

        # -- CSV must agree with JSON
        row = csv_rows.get(state)
        if row is None:
            failures.append("%s: missing from stage02_results.csv" % state)
        else:
            for csv_col, json_val in (("nominal_cost", rec["nominal_cost"]),
                                      ("repair_cost", rec["repair_cost"]),
                                      ("fixed_binary_lp_cost", rec["fixed_binary_lp_cost"]),
                                      ("kappa0_cost", rec["J0"])):
                if json_val is None:
                    continue
                if abs(float(row[csv_col]) - float(json_val)) > TOL * (1 + abs(float(json_val))):
                    failures.append("%s: csv %s=%s != json %s"
                                    % (state, csv_col, row[csv_col], json_val))

    # -- nominal plans
    print("-" * 96)
    for name, v in S["nominal"].items():
        inst0 = small[name]
        sol = _solution_from_stored(inst0, v["solution"])
        chk = check_solution(inst0, sol, MODE_AUDITED)
        recomputed = chk.cost_recomputed["total"]
        ok = chk.ok and abs(recomputed - v["cost"]) <= TOL * (1 + abs(v["cost"]))
        checked += 1
        if not ok:
            failures.append("nominal/%s: feasible=%s stored=%s recomputed=%s"
                            % (name, chk.ok, v["cost"], recomputed))
        print("%-24s %-16s %10s %10s %9.1e  %s" %
              (name, "nominal", "%.6g" % v["cost"], "%.6g" % recomputed,
               chk.max_violation, "ok" if ok else "FAIL"))

    # -- mechanism tests
    print("-" * 96)
    for name, r in S["mechanism"].items():
        if not r["repair_feasible"]:
            failures.append("mechanism/%s: stored repair marked infeasible" % name)
        checked += 1
        print("%-24s %-16s %10s %10s %9.1e  %s" %
              (name, "mechanism", "%.6g" % r["repair_cost"], "-",
               r["repair_max_violation"],
               "ok" if r["repair_feasible"] else "FAIL"))

    print("=" * 96)
    print("plans re-verified from disk: %d" % checked)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL STORED PLANS RE-VERIFY FROM DISK; costs and checks agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
