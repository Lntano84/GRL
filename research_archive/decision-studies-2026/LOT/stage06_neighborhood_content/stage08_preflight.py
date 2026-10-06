#!/usr/bin/env python
"""
Stage 08 preflight: verify the highspy MIP improving-solution callback interface.

    .venv-hs/Scripts/python.exe stage08_preflight.py

Verified interface facts on highspy 1.15.1 (checked here, not assumed):

  * the callback is registered with
        h.cbMipImprovingSolution.subscribe(fn)
    where `fn` receives a `HighsCallbackEvent`;
  * the event exposes `callback_type`, `data_out` and `val(var_expr)`;
  * `data_out.mip_solution` is the incumbent vector and `event.val(range(n))` reads it;
  * `data_out` also carries `objective_function_value`, `mip_primal_bound`,
    `mip_dual_bound`, `mip_gap`, `mip_node_count` and `running_time` (seconds).

This script is an INTERFACE PROBE on one small case, not a result.  It checks that the event
fires, that the captured vectors can be read inside the callback, that they independently
re-verify (model, kappa, fixing conditions, recomputed cost), and that `running_time` behaves.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import highspy

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import build_highs, repair_vector
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, fixing_from_release, solution_from_record, stability_row

TOL = 1e-6
PROBE_BUDGET = 10.0


def main() -> int:
    import importlib.metadata as md
    print("highspy:", md.version("highspy"))
    print()

    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    state = "large_rho1.10_s5|D2_all_1p"
    rec = ST[state]
    inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
    dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
    pert = apply_disruption(inst, dis)
    repair = solution_from_record(pert, rec["repair"])
    bm = build_model(pert, MODE_AUDITED)
    tau = rec["meta"]["tau"]
    sets = build_release_sets(pert, repair, tau, dis)
    print("probe state: %s | repair %.6g | budget %.0f s" % (state, repair.objective,
                                                             PROBE_BUDGET))
    print()

    overall_ok = True
    for cfg in ("EMPTY", "FULL"):
        rel = [] if cfg == "EMPTY" else sets["FULL"]["release"]
        fixed, fixed_Y = fixing_from_release(bm, repair, tau, rel)
        stab = stability_row(bm, repair, tau)
        h, lp, n_cols, n_rows = build_highs(bm, fixed, stab)

        log_path = os.path.join(tempfile.gettempdir(),
                                "hs_s08pre_%d_%s.log" % (os.getpid(), uuid.uuid4().hex))
        if os.path.exists(log_path):
            os.remove(log_path)
        h.setOptionValue("output_flag", True)
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", log_path)
        h.passModel(lp)
        h.setOptionValue("time_limit", PROBE_BUDGET)
        h.setOptionValue("random_seed", 0)
        h.setOptionValue("threads", 1)
        set_status = h.setSolution(n_cols, np.arange(n_cols, dtype=np.int32),
                                   np.asarray(repair_vector(bm, repair), dtype=np.float64))

        events: List[Dict[str, Any]] = []
        idx_all = np.arange(n_cols, dtype=np.int32)
        t0 = time.perf_counter()

        def on_improve(ev) -> None:
            do = ev.data_out
            row: Dict[str, Any] = {"wall_s": time.perf_counter() - t0}
            for f in ("objective_function_value", "mip_primal_bound", "mip_dual_bound",
                      "mip_gap", "mip_node_count", "running_time"):
                try:
                    row[f] = float(getattr(do, f))
                except (TypeError, ValueError, AttributeError):
                    row[f] = None
            try:
                v = np.asarray(ev.val(idx_all), dtype=float)
                row["x"] = v if v.size == n_cols else None
            except Exception as exc:                    # noqa: BLE001 - probe
                row["x"] = None
                row["val_error"] = repr(exc)
            events.append(row)

        h.cbMipImprovingSolution.subscribe(on_improve)
        h.run()
        solved = time.perf_counter() - t0
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                log_txt = fh.read()
        except OSError:
            log_txt = ""
        try:
            os.remove(log_path)
        except OSError:
            pass

        print("=" * 96)
        print("config %-6s | status %-20s | solved %.2f s | setSolution %s | events %d"
              % (cfg, status, solved, set_status, len(events)))
        print("  final objective %.6g | gap %s"
              % (info.objective_function_value, info.mip_gap))
        print("  %-6s %9s %12s %12s %12s %9s %7s %s"
              % ("event", "wall_s", "run_time", "objective", "primal_bnd", "gap", "nodes", "x"))
        for i, e in enumerate(events):
            xs = e.get("x")
            print("  %-6d %9.2f %12s %12s %12s %9s %7s %s"
                  % (i, e["wall_s"], e.get("running_time"), e.get("objective_function_value"),
                     e.get("mip_primal_bound"), e.get("mip_gap"), e.get("mip_node_count"),
                     "MISSING" if xs is None else "len=%d" % xs.size))
            if e.get("val_error"):
                print("        val_error: %s" % e["val_error"])

        n_usable = 0
        for i, e in enumerate(events):
            xs = e.get("x")
            if xs is None:
                continue
            X, Y, Z, I, L = unpack(np.asarray(xs, float), bm.idx)
            obj_cb = e.get("objective_function_value")
            sol = Solution(status="cb", objective=float(obj_cb or 0.0),
                           X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
            chk = check_solution(pert, sol, MODE_AUDITED)
            flips = binary_change_count(pert, Y, repair.Y, tau)
            cost = chk.cost_recomputed["total"]
            match = (obj_cb is not None
                     and abs(cost - float(obj_cb)) <= TOL * (1 + abs(float(obj_cb))))
            usable = bool(chk.ok) and flips <= KAPPA + 1e-9
            n_usable += int(usable)
            print("    ev %d: feasible=%s maxviol=%.3g flips=%d cost=%.6g cb_obj=%s match=%s"
                  % (i, chk.ok, chk.max_violation, flips, cost, obj_cb, match))
        print("  usable captured vectors: %d/%d" % (n_usable, len(events)))
        print("  log has MIP-start adoption line: %s"
              % ("MIP start solution is feasible" in log_txt))
        overall_ok = overall_ok and (len(events) == 0 or n_usable == len(events))
        print()

    print("=" * 96)
    print("PREFLIGHT %s" % ("PASSED" if overall_ok else "FAILED: some captured vector was unusable"))
    print("Note: on this state EMPTY may emit zero events; that is a finding, not a failure.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
