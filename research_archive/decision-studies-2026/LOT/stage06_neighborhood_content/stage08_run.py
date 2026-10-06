#!/usr/bin/env python
"""
Stage 08: continuous-search trajectory diagnostic.

    .venv-hs/Scripts/python.exe stage08_run.py --resume
    .venv-hs/Scripts/python.exe stage08_savecheck.py
    .venv-hs/Scripts/python.exe stage08_analyse.py

Questions
---------
  1. When does EMPTY actually produce a usable improvement?
  2. Does FULL produce EARLIER improvements, and can those solutions be transferred legally to
     EMPTY?  Because every restricted method fixes the short-term Y outside its release set
     against the SAME reference S_r, we have F_EMPTY subset F_FULL; an EMPTY solution is
     therefore always admissible for FULL, but a FULL solution need NOT satisfy EMPTY's fixing
     conditions.  Redefining the reference in order to accept a wide-neighbourhood solution would
     change the protocol, so it is not done here.

Design (frozen)
---------------
* 16 frozen LARGE fault states from stage 06 (the same set stage 07 used)
* methods EMPTY and FULL, solver seeds 0 and 1, 16 x 2 x 2 = 64 runs
* ONE continuous 40 s solve per run -- no interruption, no restart, no time slicing
* start, kappa and the fixing reference are the frozen repaired plan S_r
* model, solver, threads identical to stage 07

Method
------
HiGHS' MIP improving-solution callback records every incumbent improvement with its time, the
reported objective and the FULL variable vector.  Inside the callback only cheap work is done
(copy time, scalars, vector); every recorded vector is independently re-verified afterwards.

Trajectories are then read as prefixes: the best valid solution already found before 5 / 10 / 20
/ 40 s.  These are PREFIXES OF ONE 40 s RUN, not independent runs at those time limits, and they
are reported as such.  Verification is offline, so a checkpoint value is "the quality of the
solution found by that time", not a strict real-time delivery result; both the end-to-end wall
clock and the solver clock are recorded, and neither model preparation, start submission nor
callback overhead is excluded from the wall clock.

40 s is a DIAGNOSTIC WINDOW.  It does not mean the deployment budget has changed from 20 s.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import build_highs, repair_vector
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from lsp_select import select_output
from stage06_run import KAPPA, fixing_from_release, solution_from_record, stability_row

BUDGET = 40.0
KAPPA_LOCAL = 4.0
SEEDS = (0, 1)
CONFIGS = ("EMPTY", "FULL")
THREADS = 1
SHUFFLE_SEED = 20260929
METHODS = ("EMPTY", "FULL")
RUNS_FILE = "stage08_runs.json"
CKPT_FILE = "stage08_runs.partial.json"


def _g(v, nd=6):
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(x):
        return "-"
    return "0" if abs(x) < 10 ** (-nd) else ("%." + str(nd) + "g") % x


def run_trace(bm, pert, repair, tau, rel, stab, x0, budget: float, seed: int,
              log_path: str) -> Dict[str, Any]:
    """One continuous solve with the improving-solution callback attached."""
    import highspy

    fixed, fixed_Y = fixing_from_release(bm, repair, tau, rel)
    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm, fixed, stab)

    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    h.passModel(lp)
    h.setOptionValue("time_limit", float(budget))
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("threads", int(THREADS))
    set_status = h.setSolution(n_cols, np.arange(n_cols, dtype=np.int32),
                               np.asarray(x0, dtype=np.float64))
    t_submit = time.perf_counter()

    # rel as a set of (i, j, t) for the cheap kappa-relevant flip count inside the callback
    y_slots = [(i, j, t) for t in range(1, tau + 1)
               for j in range(bm.inst.M) for i in range(bm.inst.N)]
    y_cols = np.array([bm.idx.Y(i, j, t) for (i, j, t) in y_slots], dtype=np.int32)
    ref_y = np.array([1.0 if float(repair.Y[i, j, t - 1]) > 0.5 else 0.0
                      for (i, j, t) in y_slots])

    events: List[Dict[str, Any]] = []
    idx_all = np.arange(n_cols, dtype=np.int32)

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
            if v.size != n_cols or not np.all(np.isfinite(v)):
                row["x"] = None
            else:
                row["x"] = v
                # cheap structural facts only; full verification happens after the solve
                yv = np.array([1.0 if v[c] > 0.5 else 0.0 for c in y_cols])
                row["short_term_flips"] = int(np.sum(np.abs(yv - ref_y)))
        except Exception as exc:                        # noqa: BLE001
            row["x"] = None
            row["val_error"] = repr(exc)
        events.append(row)

    h.cbMipImprovingSolution.subscribe(on_improve)
    t_run0 = time.perf_counter()
    h.run()
    t_run1 = time.perf_counter()

    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log_txt = fh.read()
    except OSError:
        log_txt = ""

    final_sol = None
    obj = None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        x = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(x)):
            X, Y, Z, I, L = unpack(x, bm.idx)
            try:
                obj = float(info.objective_function_value)
            except Exception:                            # noqa: BLE001
                obj = None
            final_sol = Solution(status=status, objective=obj if obj is not None else float("nan"),
                                 X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
    return {
        "events": events, "final_solution": final_sol, "final_objective": obj,
        "status": status, "set_solution_status": str(set_status),
        "mip_gap": (float(info.mip_gap) if np.isfinite(info.mip_gap) else None),
        "dual_bound": (float(info.mip_dual_bound) if np.isfinite(info.mip_dual_bound) else None),
        "n_cols": n_cols, "n_rows": n_rows,
        "log": log_txt,
        "wall_total_s": time.perf_counter() - t0,
        "wall_transfer_s": t_run0 - t0,
        "wall_solve_s": t_run1 - t_run0,
        "fixed_Y": fixed_Y,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(s for s in ST if ST[s]["meta"]["scale"] == "large")
    print("large fault states: %d (frozen from stage 06)" % len(states))
    if len(states) != 16:
        raise SystemExit("expected 16 large states, found %d" % len(states))

    plan = [(st, cfg, sd) for st in states for cfg in CONFIGS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("runs planned: %d (one continuous %.0f s solve each, shuffled with seed %d)"
          % (len(plan), BUDGET, SHUFFLE_SEED))

    runs: List[Dict[str, Any]] = []
    done: set = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {"%s|%s|%d" % (r["state"], r["config"], r["seed"]) for r in runs}
            print("resuming: %d runs already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA_LOCAL,
                                "configs": list(CONFIGS), "seeds": list(SEEDS),
                                "shuffle_seed": SHUFFLE_SEED, "threads": THREADS,
                                "n_runs": len(runs), "complete": len(runs) == len(plan)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    cache: Dict[str, Any] = {}
    for st in states:
        rec = ST[st]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        tau = rec["meta"]["tau"]
        sets = build_release_sets(pert, repair, tau, dis)
        cache[st] = {"meta": rec["meta"], "inst": pert, "repair": repair, "bm": bm, "tau": tau,
                     "stab": stability_row(bm, repair, tau), "x0": repair_vector(bm, repair),
                     "legal": sets["legal"], "FULL": sets["FULL"]["release"], "EMPTY": [],
                     "jr": float(repair.objective)}

    os.makedirs("stage08_logs", exist_ok=True)
    t_batch = time.perf_counter()
    for (state, config, seed) in plan:
        if "%s|%s|%d" % (state, config, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        rel = C[config]
        log_path = os.path.join("stage08_logs",
                                "%s__%s__s%d.log"
                                % (state.replace("|", "_"), config, seed))
        tr = run_trace(bm, pert, repair, tau, rel, C["stab"], C["x0"], BUDGET, seed, log_path)

        # ---- final delivered plan, selected against S_r with the stage 04b protocol
        sel = select_output(pert, tr["final_solution"], repair, KAPPA_LOCAL, tau,
                            fixed_Y=tr["fixed_Y"], solver_status=tr["status"])
        final_cost = None if sel.chosen is None else float(sel.chosen.cost)
        final_sol = None if (sel.chosen is None or sel.chosen.solution is None) \
            else sel.chosen.solution

        # ---- per-event independent verification, kept small: one record per event
        ev_recs: List[Dict[str, Any]] = []
        n_usable = 0
        for k, e in enumerate(tr["events"]):
            rec_i: Dict[str, Any] = {kk: vv for kk, vv in e.items() if kk != "x"}
            xs = e.get("x")
            if xs is None:
                rec_i["usable"] = False
                rec_i["reason"] = "no vector captured"
                ev_recs.append(rec_i)
                continue
            X, Y, Z, I, L = unpack(np.asarray(xs, float), bm.idx)
            obj_cb = e.get("objective_function_value")
            sol = Solution(status="cb", objective=float(obj_cb or 0.0),
                           X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
            chk = check_solution(pert, sol, MODE_AUDITED)
            flips = binary_change_count(pert, Y, repair.Y, tau)
            # fixing conditions of THIS config's release set
            relset = set(rel)
            viol = 0
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, tau + 1):
                        if (i, j, t) in relset:
                            continue
                        if abs(float(Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                            viol += 1
            cost = float(chk.cost_recomputed["total"])
            obj_match = (obj_cb is not None
                         and abs(cost - float(obj_cb)) <= 1e-6 * (1 + abs(float(obj_cb))))
            usable = bool(chk.ok) and flips <= KAPPA_LOCAL + 1e-9 and viol == 0 and obj_match
            n_usable += int(usable)
            rec_i.update({
                "feasible": bool(chk.ok), "max_violation": float(chk.max_violation),
                "cost_recomputed": cost, "objective_matches": bool(obj_match),
                "flips": int(flips), "fixing_violations": int(viol),
                "usable": bool(usable),
                "improved_over_repair": bool(cost < C["jr"] - 1e-6 * (1 + abs(C["jr"]))),
                "sol": {kk: np.asarray(getattr(sol, kk)).tolist() for kk in ("X", "Y", "Z", "I", "L")},
            })
            ev_recs.append(rec_i)

        runs.append({
            "state": state, "config": config, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"],
            "disruption": ST[state]["disruption"]["name"],
            "budget_s": BUDGET, "repair_cost": C["jr"],
            "n_released": len(rel), "n_legal": len(C["legal"]),
            "status": tr["status"], "set_solution_status": tr["set_solution_status"],
            "mip_gap": tr["mip_gap"], "dual_bound": tr["dual_bound"],
            "start_adopted": "MIP start solution is feasible" in tr["log"],
            "final_objective_raw": tr["final_objective"],
            "final_cost": final_cost, "final_source": sel.source,
            "select_errors": sel.errors,
            "final_solution": (None if final_sol is None
                               else {kk: np.asarray(getattr(final_sol, kk)).tolist()
                                     for kk in ("X", "Y", "Z", "I", "L")}),
            "n_events": len(tr["events"]), "n_usable_events": n_usable,
            "events": ev_recs,
            "timing": {"wall_total_s": tr["wall_total_s"],
                       "wall_transfer_s": tr["wall_transfer_s"],
                       "wall_solve_s": tr["wall_solve_s"]},
            "log_tail": tr["log"][-4000:],
        })
        write_ckpt()
        last = runs[-1]
        first_imp = next((e["wall_s"] for e in ev_recs
                          if e.get("usable") and e.get("improved_over_repair")), None)
        print("  [%2d/%2d] %-30s %-6s s%d events=%-3d usable=%-3d final=%-12s first_imp=%s t=%.1f"
              % (len(runs), len(plan), state, config, seed, last["n_events"],
                 last["n_usable_events"], _g(final_cost),
                 "-" if first_imp is None else "%.2f" % first_imp, tr["wall_total_s"]),
              flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA_LOCAL,
                            "configs": list(CONFIGS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "threads": THREADS,
                            "scope": "16 frozen LARGE fault states from stage 06",
                            "solver": "highspy", "solver_version": _hs_version(),
                            "protocol": "one continuous solve per run; no interruption, no "
                                        "restart, no time slicing; start, kappa and fixing "
                                        "reference all S_r; DEP not used",
                            "checkpoints_s": [5.0, 10.0, 20.0, 40.0],
                            "note": "checkpoints are PREFIXES of the same 40 s run, not "
                                    "independent runs; verification is offline, so a checkpoint "
                                    "value is the quality found BY that time"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "config", "seed",
              "repair_cost", "final_cost", "final_source", "n_released", "n_legal",
              "status", "mip_gap", "dual_bound", "start_adopted", "n_events",
              "n_usable_events"]
    with open("stage08_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)

    print("\nruns: %d | wall %.1f s" % (len(runs), time.perf_counter() - t_batch))
    print("usable improvement events: %d | total events: %d"
          % (sum(r["n_usable_events"] for r in runs), sum(r["n_events"] for r in runs)))
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    print("selection errors: %d | outputs worse than repair: %d"
          % (sum(1 for r in runs if r["select_errors"]), worse))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:                                    # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
