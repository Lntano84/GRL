#!/usr/bin/env python
"""
Stage 14: does group-releasing the carry-overs at each machine's RECOVERY period keep most of
the benefit of releasing every short-term Z?

    .venv-hs/Scripts/python.exe stage14_run.py --resume
    .venv-hs/Scripts/python.exe stage14_analyse.py

The rule must use only information known AT FAULT TIME:

    r_j = the first period in which machine j has positive capacity again after its outage

For disruption D2 (all machines down in period 1) r_j = 2; for D1 (machine 0 down in periods
1-2) r_0 = 3.  Writing the rule as "release t = 2" would NOT be deployable; writing it as
"release Z_{i,j,r_j} for every product i on the affected machines" is.

Configurations (frozen)
-----------------------
    A         far-horizon Y released, ALL short-term Z fixed, far-horizon Z fixed
    AB        far-horizon Y released, ALL short-term Z released, far-horizon Z fixed
    RECOVERY  far-horizon Y released, only { Z_{i,j,r_j} } released, far-horizon Z fixed
    END       far-horizon Y released, only { Z_{i,j,tau} } released, far-horizon Z fixed

RECOVERY and END release the SAME NUMBER of machine-period groups per state; what differs is
the position: the recovery period versus the frozen-window boundary.  Group release only
PERMITS both sides of a carry-over swap to move together -- it does not force a swap and does
not forbid a new or cancelled carry-over.

Scope
-----
* all 16 development states (not only the certificate states)
* 2 solver seeds, 4 configurations -> 128 MIPs, 20 s each including initialisation
* the start is each run's OWN continuous LP solution; no stage 12/13 pool, no Q, no
  change-position guidance
* full plans, actual timings and presolve sizes are saved
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import repair_vector
from lsp_model import MODE_AUDITED, build_model
from lsp_neighborhood import build_release_sets
from lsp_release import (LP_BUDGET, build_state_context, pinned_bounds, solve_lp_fixed,
                         solve_mip, stage14_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record, stability_row

BUDGET = 20.0
SEEDS = (0, 1)
SHUFFLE_SEED = 20261004
METHODS = ("A", "AB", "RECOVERY", "END")
STATES_FILE = "stage06_states.json"
RUNS_FILE = "stage14_runs.json"
CKPT_FILE = "stage14_runs.partial.json"


def recovery_period(dis, j: int, T: int) -> int:
    """First period with positive capacity again for machine j (per lsp_neighborhood)."""
    delta = dis.delta_for(j)
    if delta == 0:
        return T + 1            # this machine was never disrupted
    if delta >= T:
        return T + 1
    return delta + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    print("states: %d | methods %s | seeds %s | MIPs %d"
          % (len(states), list(METHODS), list(SEEDS), len(states) * len(METHODS) * len(SEEDS)))

    plan = [(st, m, sd) for st in states for m in METHODS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    runs: List[Dict[str, Any]] = []
    done: Set[str] = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {"%s|%s|%d" % (r["state"], r["method"], r["seed"]) for r in runs}
            print("resuming: %d already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHODS),
                                "seeds": list(SEEDS), "n_runs": len(runs)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    cache: Dict[str, Any] = {}
    for st in states:
        rec = ALL[st]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        tau = rec["meta"]["tau"]
        C = build_state_context(inst, pert, repair, bm, tau)
        # S_r's own short-term-Z fixing set for the AB-style handling, via stage11_columns
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert,
                  "dis": dis, "mr_sets": {}})
        C["mr_sets"] = {}                      # not used here; required only by stage11_columns
        cache[st] = C

    def build_method(C, method: str):
        return stage14_columns(C, method)

    os.makedirs("stage14_logs", exist_ok=True)
    t_all = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = float(repair.objective)
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), method, seed)

        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage14_logs", tag + "__lp.log"))
        rel, fixed_cols, meta = build_method(C, method)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else C["x0"])
        mip = solve_mip(bm, fixed_cols, stability_row(bm, repair, tau), remaining, seed, x_lp,
                        os.path.join("stage14_logs", tag + ".mip.log"))
        wall = time.perf_counter() - t_run0

        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", pert, mip["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", pert, lp["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        cands.append(evaluate_candidate("repair", pert, repair, repair, KAPPA, tau, None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        best = min(usable, key=lambda c: c.cost) if usable else None
        errors = [("INVALID %s: %s" % (c.label, c.problems[:2]))
                  for c in cands if c.present and not is_usable(c)]
        pre = _presolve(mip["log"])

        runs.append({
            "state": state, "method": method, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ALL[state]["disruption"]["name"],
            "repair_cost": jr, "budget_s": BUDGET,
            "n_rel_Y": meta["n_rel_Y"], "n_rel_Z": meta["n_rel_Z"],
            "rel_size": meta["rel_size"], "recovery_periods": meta["recovery_periods"],
            "lp_objective": lp["objective"],
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "presolve": pre,
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": ("none" if best is None else best.label),
            "strictly_below_lp": bool(best is not None and lp["objective"] is not None
                                      and float(best.cost) < lp["objective"]
                                      - 1e-9 * (1 + abs(lp["objective"]))),
            "select_errors": errors,
            "final_solution": (None if best is None or best.solution is None
                               else {k: np.asarray(getattr(best.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"], "total_s": wall,
                       "budget_given_s": remaining, "overrun_s": max(0.0, wall - BUDGET)},
            "log_tail": mip["log"][-1200:],
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-30s %-9s s%d relZ=%-5d final=%-11s (%s) G=%+.4f t=%.1f"
              % (len(runs), len(plan), state, method, seed, meta["n_rel_Z"],
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)), wall),
              flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHODS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "scope": "16 development states, stage 11 instances",
                            "rule": "RECOVERY releases Z_{i,j,r_j} with r_j the first period "
                                    "with positive capacity after the outage (known at fault "
                                    "time); END releases Z_{i,j,tau}",
                            "note": "no stage 12/13 plan pool, no Q, no change-position "
                                    "guidance; every run starts from its own LP solution"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "method", "seed",
              "repair_cost", "lp_objective", "mip_status", "mip_objective", "mip_dual_bound",
              "n_rel_Y", "n_rel_Z", "rel_size", "final_cost", "final_source",
              "strictly_below_lp", "total_s", "overrun_s"]
    with open("stage14_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            flat["total_s"] = r["timing"]["total_s"]
            flat["overrun_s"] = r["timing"]["overrun_s"]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    print("\nruns %d | wall %.1f s | select errors %d | worse than repair %d"
          % (len(runs), time.perf_counter() - t_all, bad, worse))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if (bad == 0 and worse == 0) else 1


def _presolve(log: str) -> Dict[str, Any]:
    import re
    m = re.search(r"Solving MIP model with:\s*\n\s*(\d+) rows\s*\n\s*(\d+) cols \(([^)]*)\)", log)
    if not m:
        return {"missing": True}
    mb = re.search(r"(\d+) binary", m.group(3))
    return {"rows": int(m.group(1)), "cols": int(m.group(2)),
            "binary": int(mb.group(1)) if mb else None}


if __name__ == "__main__":
    sys.exit(main())
