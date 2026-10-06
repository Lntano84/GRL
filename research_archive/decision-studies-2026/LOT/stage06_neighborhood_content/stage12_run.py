#!/usr/bin/env python
"""
Stage 12: quality-retention diagnostic for the restricted problems.

    .venv-hs/Scripts/python.exe stage12_run.py --smoke     # first state only
    .venv-hs/Scripts/python.exe stage12_run.py --resume
    .venv-hs/Scripts/python.exe stage12_analyse.py

The only two questions
----------------------
> How much of AB's advantage over the matched random fixing is CONFIRMED by differences in the
> restricted problems' optimal values?  And how much quality may AB itself be losing by fixing C?

Frozen scope
------------
* the original 16 development states; instances, repair plans and matched random sets unchanged
* only AB, MR-Z-1 and ABC -- MR-Z-1 chosen BY LABEL, not because it is best or worst
* ONE solver seed, fixed at 0
* at most 60 s per run; a run that proves optimal stops early
* 16 x 3 x 1 = 48 MIPs, solve budget cap 48 minutes, no parameter sweeps, no automatic extension

This is an OFFLINE QUALITY DIAGNOSTIC.  It does NOT change the 20 s deployment budget and is NOT
a deployment comparison between the three methods.

Common start pool
-----------------
For each state, the plans already saved by stage 11 form a shared pool.  For each configuration:
  1. keep the pool plans that satisfy ALL of that configuration's fixing conditions;
  2. submit the cheapest of those;
  3. record the origin of the start, its verification, and the adoption log line.

Fixing values and the stability budget remain relative to the ORIGINAL frozen S_r -- they are
never re-based onto the new start.  ABC admits plans produced under AB or a matched random
configuration because those subproblems are nested inside it; between AB and MR eligibility must
be checked plan by plan.

Why the start pool matters: it spends the diagnostic budget on TIGHTENING the quality interval
rather than re-discovering plans that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_release import (build_state_context, pinned_bounds, solve_lp_fixed, solve_mip,
                         stage11_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record, stability_row

BUDGET = 60.0
SEED = 0
LP_BUDGET = 5.0
METHODS = ("AB", "MR-Z-1", "ABC")
STATES_FILE = "stage06_states.json"
MATCHED_FILE = "stage11_matched_random.json"
PREV_RUNS = "stage11_runs.json"
RUNS_FILE = "stage12_runs.json"
CKPT_FILE = "stage12_runs.partial.json"


def eligibility(C, plan: Dict[str, Any], method: str) -> Tuple[bool, int]:
    """Count fixing-condition violations of `plan` for `method`.  Short-term Y is fixed in all
    three configurations, so it is checked too.  Uses the SAME frozen S_r values."""
    bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
    rel, _fixed, _fz = stage11_columns(C, method)
    rel_set = set(rel)
    Y = np.array(plan["Y"], float).reshape(pert.N, pert.M, pert.T)
    Z = np.array(plan["Z"], float).reshape(pert.N, pert.M, pert.T)
    viol = 0
    for i in range(pert.N):
        for j in range(pert.M):
            for t in range(1, tau + 1):
                if ("Y", i, j, t) in rel_set:
                    continue
                if abs(float(Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                    viol += 1
    for (kind, i, j, t) in C["all_slots"]:
        if kind != "Z" or (kind, i, j, t) in rel_set:
            continue
        if abs(float(Z[i, j, t - 1]) - float(repair.Z[i, j, t - 1])) > 0.5:
            viol += 1
    return viol == 0, viol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(MATCHED_FILE, encoding="utf-8") as fh:
        MATCH = json.load(fh)
    with open(PREV_RUNS, encoding="utf-8") as fh:
        PREV = json.load(fh)["runs"]
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    if args.smoke:
        states = states[:1]
        print("SMOKE: first state only -> %s" % states[0])
    print("states %d | methods %s | seed %d | budget %.0f s"
          % (len(states), list(METHODS), SEED, BUDGET))

    plans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in PREV:
        if r.get("final_solution") is not None and r["final_cost"] is not None:
            plans[r["state"]].append({"method": r["method"], "seed": int(r["seed"]),
                                      "cost": float(r["final_cost"]),
                                      "plan": r["final_solution"]})

    plan_list = [(st, m) for st in states for m in METHODS]
    runs: List[Dict[str, Any]] = []
    done = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {"%s|%s" % (r["state"], r["method"]) for r in runs}
            print("resuming: %d already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHODS),
                                "seed": SEED, "n_runs": len(runs)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    os.makedirs("stage12_logs", exist_ok=True)
    t_all = time.perf_counter()
    for (state, method) in plan_list:
        if "%s|%s" % (state, method) in done:
            continue
        rec = ALL[state]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[state]["sets"][n]["fixed_slots"]}
                        for n in ("MR-Z-1", "MR-Z-2", "MR-Z-3")}
        jr = float(repair.objective)
        tag = "%s__%s" % (state.replace("|", "_"), method)

        # ---- eligible pool plans, cheapest first
        elig = []
        for p in plans.get(state, []):
            ok, viol = eligibility(C, p["plan"], method)
            if ok:
                # cost of the plan as re-verified on this state's model
                sol = Solution(status="pool", objective=p["cost"],
                               X=np.array(p["plan"]["X"], float).reshape(pert.N, pert.M, pert.T),
                               Y=np.array(p["plan"]["Y"], float).reshape(pert.N, pert.M, pert.T),
                               Z=np.array(p["plan"]["Z"], float).reshape(pert.N, pert.M, pert.T),
                               I=np.array(p["plan"]["I"], float).reshape(pert.N, pert.T),
                               L=np.array(p["plan"]["L"], float).reshape(pert.N, pert.T),
                               mode=MODE_AUDITED)
                chk = check_solution(pert, sol, MODE_AUDITED)
                flips = binary_change_count(pert, sol.Y, repair.Y, tau)
                if chk.ok and flips <= KAPPA + 1e-9:
                    elig.append({"origin": "%s/s%d" % (p["method"], p["seed"]),
                                 "cost": float(chk.cost_recomputed["total"]),
                                 "reported": p["cost"], "plan": p["plan"], "solution": sol})
        elig.sort(key=lambda e: e["cost"])
        chosen = elig[0] if elig else None

        # ---- the common continuous LP is still the fallback start
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage12_logs", tag + "__lp.log"))
        if chosen is not None and lp["objective"] is not None \
                and lp["objective"] < chosen["cost"]:
            start_sol, start_src = lp["solution"], "S_LP"
        elif chosen is not None:
            start_sol, start_src = chosen["solution"], "pool:" + chosen["origin"]
        else:
            start_sol, start_src = lp["solution"], "S_LP(no eligible pool plan)"
        x_start = vector_of(start_sol, bm) if start_sol is not None else C["x0"]

        rel, fixed_cols, _fz = stage11_columns(C, method)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}
        t0 = time.perf_counter()
        mip = solve_mip(bm, fixed_cols, stability_row(bm, repair, tau), BUDGET, SEED, x_start,
                        os.path.join("stage12_logs", tag + ".mip.log"))
        wall = time.perf_counter() - t0

        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", pert, mip["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", pert, lp["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        for e in elig:
            cands.append(evaluate_candidate("pool", pert, e["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        cands.append(evaluate_candidate("repair", pert, repair, repair, KAPPA, tau, None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        best = min(usable, key=lambda c: c.cost) if usable else None
        errors = [("INVALID %s: %s" % (c.label, c.problems[:2]))
                  for c in cands if c.present and not is_usable(c)]

        runs.append({
            "state": state, "method": method, "seed": SEED, "budget_s": BUDGET,
            "repair_cost": jr,
            "n_pool_plans": len(plans.get(state, [])), "n_eligible_pool": len(elig),
            "pool_origins": [e["origin"] for e in elig],
            "start_source": start_src,
            "start_cost": (None if start_sol is None else float(start_sol.objective)),
            "lp_objective": lp["objective"],
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": ("none" if best is None else best.label),
            "final_stability_flips": (None if best is None else best.stability_flips),
            "select_errors": errors,
            "wall_s": wall,
            "log_tail": mip["log"][-1500:],
        })
        write_ckpt()
        r = runs[-1]
        print("  %-30s %-8s pool=%-2d elig=%-2d start=%-18s final=%-11s (%s) %s opt=%s t=%.1f"
              % (state, method, r["n_pool_plans"], len(elig), start_src,
                 "%.6g" % r["final_cost"] if r["final_cost"] else "-", r["final_source"],
                 mip["status"][:12], r["mip_proven_optimal"], wall), flush=True)

    if args.smoke:
        print("\nSMOKE: %d runs" % len(runs))
        return 0 if all(not r["select_errors"] for r in runs) else 1

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "methods": list(METHODS),
                            "n_runs": len(runs),
                            "scope": "16 development states, stage 11 instances and sets",
                            "role": "OFFLINE QUALITY DIAGNOSTIC; not a deployment comparison "
                                    "and it does not change the 20 s budget",
                            "start_pool": "stage 11 saved plans eligible for the configuration",
                            "note": "fixing values and kappa stay relative to the original S_r"},
                   "runs": runs}, fh, indent=2, default=str)

    n_opt = sum(1 for r in runs if r["mip_proven_optimal"])
    bad = sum(1 for r in runs if r["select_errors"])
    print("\nruns %d | proven optimal %d | select errors %d | wall %.1f s"
          % (len(runs), n_opt, bad, time.perf_counter() - t_all))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
