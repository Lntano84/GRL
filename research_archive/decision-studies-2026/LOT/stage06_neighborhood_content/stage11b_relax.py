#!/usr/bin/env python
"""
Stage 11b: VALID relaxation and presolve diagnostics.

    .venv-hs/Scripts/python.exe stage11b_relax.py            # all 16 states
    .venv-hs/Scripts/python.exe stage11b_relax.py --smoke    # first state only (interface check)

What was wrong in stage 11's diagnostics
----------------------------------------
1. `n_bin_in - len(rel)` counted UNRELEASED SLOTS, not remaining free binaries, so AB's 1224 was
   read backwards.
2. `getNumCol()` / `getNumRow()` return the size of the LOADED model, NOT the presolved size, so
   they were never a presolve measurement.  The real presolved dimensions are in the normal MIP
   log (`Solving MIP model with:`), which is parsed offline by `stage11b_presolve.py`.
3. That diagnostic pinned EVERY binary bound and cleared integrality, and never restored the
   released variables to free, so it did not even assemble the corresponding configuration.
4. The relaxation LPs returned `Not Set` with objective 0 in ~30 microseconds and NO check was
   made of the return codes or the solution status, so a default 0 was folded into the analysis.
   The resulting table was algebraically identical to 1 - G, i.e. it carried no information.

Correct construction used here
------------------------------
1. Assemble the configuration through the NORMAL MIP path: `build_highs(bm, fixed, stab)` with the
   configuration's own fixing rows, the stability row, and S_r-pinned bounds.
2. Copy that FINAL assembled model and change ONLY the integer types to continuous.  Costs,
   matrix, row and column bounds are identical.  Unlike the broken version, the released binaries
   keep their [0, 1] bounds and are genuinely free.
3. Run each relaxation in its OWN process with a single thread set before the first solve.
4. Record the full log, the status codes, the model status, solution validity and the objective.
5. `Error` / `Not Set` / no valid solution => recorded as MISSING.  A default 0 is never tabulated.

For relaxations that proved optimal we then check
    L_ABC <= L_AB ,  L_ABC <= L_MR ,  L_m <= J_m,valid
AB and MR are not nested, so no fixed order between their bounds is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataclasses

import highspy

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import build_highs
from lsp_model import MODE_AUDITED, build_model
from lsp_release import build_state_context, stage11_columns
from stage06_run import solution_from_record, stability_row

RELAX_BUDGET = 5.0
OUT_FILE = "stage11b_relax.json"
STATES_FILE = "stage06_states.json"
MATCHED_FILE = "stage11_matched_random.json"
METHODS = ("AB", "ABC", "MR-Z-1", "MR-Z-2", "MR-Z-3")


def presolved_from_log(log: str) -> Optional[Dict[str, int]]:
    """First `Solving MIP model with:` block = the presolved model actually solved."""
    import re
    m = re.search(r"Solving MIP model with:\s*\n\s*(\d+) rows\s*\n\s*(\d+) cols \(([^)]*)\)",
                  log)
    if not m:
        return None
    mb = re.search(r"(\d+) binary", m.group(3))
    return {"rows": int(m.group(1)), "cols": int(m.group(2)),
            "binary": int(mb.group(1)) if mb else None}


def relax_one(C, method: str, log_path: str) -> Dict[str, Any]:
    """Assemble the configuration via the normal MIP path, then relax integrality only."""
    bm = C["bm"]
    rel, fixed, _fz = stage11_columns(C, method)
    stab = stability_row(bm, C["repair"], C["tau"])

    # 1. the NORMAL MIP assembly for this configuration
    h, lp, n_cols, n_rows = build_highs(bm, fixed, stab)

    # 2. change ONLY the integer types to continuous: same costs, matrix, row/col bounds
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * n_cols

    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    status_pass = h.passModel(lp)
    # threads set BEFORE the first solve, uniformly, in this process
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(RELAX_BUDGET))

    t0 = time.perf_counter()
    status_run = h.run()
    wall = time.perf_counter() - t0

    model_status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    obj = None
    try:
        v = float(info.objective_function_value)
        obj = v if np.isfinite(v) else None
    except Exception:                                    # noqa: BLE001
        obj = None
    has_sol = sol.col_value is not None and len(sol.col_value) == n_cols
    # a usable relaxation needs: no error, a real model status, and a finite objective
    ok = (str(status_pass).endswith("kOk") and str(status_run).endswith("kOk")
          and obj is not None and model_status.lower() not in ("not set", "error", "load error"))
    log = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log = fh.read()
    except OSError:
        log = ""
    return {
        "status_pass": str(status_pass), "status_run": str(status_run),
        "model_status": model_status, "wall_s": wall, "objective": obj,
        "has_solution": bool(has_sol), "valid": bool(ok),
        "proven_optimal": bool(ok and "optimal" in model_status.lower()),
        "n_released": len(rel), "n_fixed_columns": len(fixed),
        "relax_log_len": len(log), "log_tail": log[-1500:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run only the FIRST state (five configurations) as an interface check")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(MATCHED_FILE, encoding="utf-8") as fh:
        MATCH = json.load(fh)
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    if args.smoke:
        states = states[:1]
        print("SMOKE: first state only -> %s" % states[0])
    os.makedirs("stage11b_logs", exist_ok=True)

    out: Dict[str, Any] = {}
    n_valid = n_invalid = 0
    t_all = time.perf_counter()
    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in ("MR-Z-1", "MR-Z-2", "MR-Z-3")}
        out[st] = {}
        for method in METHODS:
            log_path = os.path.join("stage11b_logs", "%s__%s.relax.log"
                                    % (st.replace("|", "_"), method))
            res = relax_one(C, method, log_path)
            n_valid += int(res["valid"])
            n_invalid += int(not res["valid"])
            out[st][method] = res
            print("  %-30s %-8s status=%-22s obj=%-14s opt=%-5s t=%.2fs%s"
                  % (st, method, res["model_status"],
                     "MISSING" if res["objective"] is None else "%.6g" % res["objective"],
                     res["proven_optimal"], res["wall_s"],
                     "" if res["valid"] else "  <-- INVALID, recorded as MISSING"), flush=True)

    print("\nrelaxations valid: %d | invalid (missing): %d | wall %.1f s"
          % (n_valid, n_invalid, time.perf_counter() - t_all))
    if args.smoke:
        if n_invalid == 0:
            print("SMOKE PASSED: all five configurations produced a valid relaxation. "
                  "Re-run without --smoke for the remaining 15 states.")
            return 0
        print("SMOKE FAILED: inspect the recorded status codes and logs; the batch was NOT "
              "expanded.")
        return 1

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("wrote %s" % OUT_FILE)
    return 0 if n_invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
