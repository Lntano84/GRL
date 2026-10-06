"""
OFFLINE, PROVISIONAL re-aggregation of the stage-04 timing scalars.

    python stage04b_offline.py

Reads `../stage04_runtime_calibration/stage04_states.json` and recomputes every method's cost
as  min(repair_cost, raw_incumbent_cost)  -- the rule stage 04 should have applied. No solver
is called and no plan is inspected.

WHAT THIS IS NOT
----------------
Stage 04 stored only SCALARS for its 288 timing runs; the plans themselves were not written
to disk. This re-aggregation therefore CANNOT verify that the substituted plan is feasible,
cannot re-check the stability budget and cannot re-check the fixing conditions. It is a
**provisional correction based on historical scalars**, and must not be presented as a
re-verified result. `stage04b_run.py` produces the verified version for 8 states.

One consequence worth stating: the repaired plan is fixed for a whole state, so
min(J_repair, J_raw) is a valid lower bound on what the corrected protocol would deliver,
but the true corrected value could be lower still if a different (unrecorded) incumbent
would have survived validation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

SRC = "../stage04_runtime_calibration/stage04_states.json"
METHODS = ("FULL", "EMPTY", "SHORTAGE-12", "SHORTAGE-24")


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    with open(SRC, encoding="utf-8") as fh:
        ST = json.load(fh)

    rows: List[Dict[str, Any]] = []
    n_changed = 0
    n_runs = 0
    for state, rec in sorted(ST.items()):
        if "timing" not in rec:
            continue
        repair = _f(rec["repair"]["objective"])
        jb_old, jb_new = [], []
        per_method: Dict[str, Dict[str, Any]] = {}
        for m in METHODS:
            for r in rec["timing"]["runs"]:
                if r["method"] != m:
                    continue
                n_runs += 1
                raw = _f(r["objective"])
                corrected = min(repair, raw) if raw is not None else repair
                if raw is not None and corrected < raw - 1e-9:
                    n_changed += 1
                per_method.setdefault(m, {})[float(r["budget_s"])] = {
                    "raw": raw, "corrected": corrected}
        # provisional J_best over all budgets and methods, plus the repair plan
        cands = [repair]
        for m, per_b in per_method.items():
            for b, vals in per_b.items():
                cands.append(vals["corrected"])
        ref = rec.get("reference", {}).get("solution")
        if ref and ref.get("feasible"):
            cands.append(_f(ref["objective"]))
        jbest = min(cands)
        for m, per_b in per_method.items():
            for b, vals in per_b.items():
                rows.append({
                    "state": state, "scale": rec["meta"]["scale"],
                    "rho": rec["meta"]["rho"], "method": m, "budget_s": b,
                    "repair_cost": repair,
                    "raw_incumbent": vals["raw"],
                    "corrected_output": vals["corrected"],
                    "raw_worse_than_repair": bool(
                        vals["raw"] is not None and vals["raw"] > repair + 1e-9),
                    "J_best_provisional": jbest,
                    "gap_old": ((vals["raw"] - jbest) / max(1.0, abs(jbest))
                                if vals["raw"] is not None else None),
                    "gap_corrected": (vals["corrected"] - jbest) / max(1.0, abs(jbest)),
                })

    with open("stage04b_offline_regrade.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        fields = ["state", "scale", "rho", "method", "budget_s", "repair_cost",
                  "raw_incumbent", "corrected_output", "raw_worse_than_repair",
                  "J_best_provisional", "gap_old", "gap_corrected"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("PROVISIONAL, SCALAR-ONLY re-aggregation (no plans re-verified)")
    print("  runs scanned                     : %d" % n_runs)
    print("  runs where the correction bites  : %d" % n_changed)
    print()
    print("mean gap (old raw -> corrected) by scale and method, 20 s:")
    print("  %-8s %-12s %10s %10s" % ("scale", "method", "old", "corrected"))
    for scale in ("small", "medium", "large"):
        for m in METHODS:
            old = [r["gap_old"] for r in rows
                   if r["scale"] == scale and r["method"] == m and r["budget_s"] == 20.0
                   and r["gap_old"] is not None]
            new = [r["gap_corrected"] for r in rows
                   if r["scale"] == scale and r["method"] == m and r["budget_s"] == 20.0]
            if old and new:
                print("  %-8s %-12s %10.4f %10.4f"
                      % (scale, m, sum(old) / len(old), sum(new) / len(new)))
    print()
    print("states where the large FULL@20 correction changes the picture:")
    for r in rows:
        if r["scale"] == "large" and r["method"] == "FULL" and r["budget_s"] == 20.0:
            print("  %-30s raw=%-11.6g corrected=%-11.6g repair=%-11.6g"
                  % (r["state"], r["raw_incumbent"] or 0, r["corrected_output"],
                     r["repair_cost"]))
    print()
    print("wrote stage04b_offline_regrade.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
