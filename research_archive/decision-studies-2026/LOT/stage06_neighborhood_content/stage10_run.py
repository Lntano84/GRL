#!/usr/bin/env python
"""
Stage 10: independent confirmation of AB on NEW instances.

    .venv-hs/Scripts/python.exe stage10_run.py --resume
    .venv-hs/Scripts/python.exe stage10_savecheck.py
    .venv-hs/Scripts/python.exe stage10_analyse.py

Research question
-----------------
On new large instances that took no part in choosing the configuration, does AB robustly beat a
reasonably warm-started full solve, and keep its advantage over ABC?

PRIMARY comparison (fixed BEFORE seeing the new results): AB vs FULL.

    G_n = (1/4) * sum_{d in {D1,D2}} sum_{s in {0,1}}
              ( J_FULL,n,d,s - J_AB,n,d,s ) / max(1, |J_r,n,d|)

Secondary (pre-specified): AB vs ABC, AB vs A, AB vs BC.  The primary comparison is not replaced
because a secondary one looks better, and no per-instance best-configuration is reported as a
strategy.

Configurations and why both ABC and FULL are needed
---------------------------------------------------
    AB     the frozen candidate under test
    A      is the short-term Z release still worth anything?
    ABC    stage-09B reference: short-term Y ALL fixed
    FULL   the original task: short-term Y may also change inside the ORIGINAL kappa budget
    BC     the cheap alternative (proves optimal in under a second in stage 09B)

Beating ABC alone would not show AB is competitive for the ORIGINAL re-optimisation task, so FULL
returns here.  ABC and FULL differ in exactly one respect: whether the short-term Y is released.

Protocol
--------
* 16 new nominal instances (seeds 6-13), 32 fault states, D1/D2, frozen generation
* model, kappa=4, tau=6, solver version, 1 thread unchanged
* every method first runs the SAME continuous LP, then starts its MIP from that solution
* every unreleased binary fixed to S_r's normalised value
* fixing conditions and the kappa reference remain the original frozen repair plan
* 20 s TOTAL budget per run, including initialisation, preparation and verification
* solver seeds 0 and 1; no slicing, no restart, no parameter changes
* delivered output = best valid among {solver incumbent, S_LP, S_r} -- never just the worse S_r
* if a MIP proves optimal early it simply stops; the remaining budget is NOT consumed on purpose,
  and its running time is itself a result

32 states x 5 configurations x 2 seeds = 320 MIPs.
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

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import repair_vector
from lsp_model import MODE_AUDITED, build_model
from lsp_release import (CONFIGS, LP_BUDGET, build_state_context, fixed_columns,
                         flips_by_group, pinned_bounds, solve_lp_fixed, solve_mip, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record, stability_row

BUDGET = 20.0
SEEDS = (0, 1)
SHUFFLE_SEED = 20261001
METHODS = ("AB", "A", "ABC", "FULL", "BC")
STATES_FILE = "stage10_states.json"
RUNS_FILE = "stage10_runs.json"
CKPT_FILE = "stage10_runs.partial.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(ST)
    print("states: %d (from %s)" % (len(states), STATES_FILE))
    if len(states) != 32:
        raise SystemExit("expected 32 states, found %d" % len(states))
    insts = sorted({s.split("|")[0] for s in states})
    if len(insts) != 16:
        raise SystemExit("expected 16 nominal instances, found %d" % len(insts))
    print("nominal instances: %d | methods %s | seeds %s"
          % (len(insts), list(METHODS), list(SEEDS)))

    plan = [(st, m, sd) for st in states for m in METHODS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("MIPs planned: %d" % len(plan))

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
                                "seeds": list(SEEDS), "n_runs": len(runs),
                                "complete": len(runs) == len(plan)},
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
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"meta": rec["meta"], "inst": pert, "repair": repair, "tau": tau,
                  "bm": bm, "stab": stability_row(bm, repair, tau),
                  "x0": repair_vector(bm, repair), "jr": float(repair.objective)})
        cache[st] = C

    os.makedirs("stage10_logs", exist_ok=True)
    t_batch = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = C["jr"]
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), method, seed)

        # ---- phase 0: the SAME continuous LP for every method, re-run per run
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage10_logs", tag + "__lp.log"))

        # ---- phase 1: MIP on the remaining budget
        rel, fixed_cols = fixed_columns(C, method)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_lp = vector_of(lp["solution"], bm) if lp["solution"] is not None else C["x0"]
        mip = solve_mip(bm, fixed_cols, C["stab"], remaining, seed, x_lp,
                        os.path.join("stage10_logs", tag + ".mip.log"))

        # ---- output: best VALID among {solver, S_LP, S_r}
        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", pert, mip["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", pert, lp["solution"], repair, KAPPA, tau,
                                            fixed_Y))
        cands.append(evaluate_candidate("repair", pert, repair, repair, KAPPA, tau, None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        errors: List[str] = []
        for c in cands:
            if c.label in ("solver", "lp") and c.present and not is_usable(c):
                errors.append("INVALID %s candidate: %s" % (c.label, c.problems[:2]))
        if not usable:
            best, source = None, "none"
        else:
            best = min(usable, key=lambda c: c.cost)
            source = best.label
        total = time.perf_counter() - t_run0
        final_sol = None if best is None else best.solution
        fg = ({"A": 0, "B": 0, "C": 0, "short_Y": 0} if final_sol is None
              else flips_by_group(pert, final_sol, repair, tau))
        lp_obj = lp["objective"]
        final_cost = None if best is None else float(best.cost)

        runs.append({
            "state": state, "method": method, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ST[state]["disruption"]["name"],
            "budget_s": BUDGET, "repair_cost": jr,
            "n_released": len(rel), "released_groups": list(CONFIGS[method][0]),
            "releases_short_term_Y": bool(CONFIGS[method][1]),
            "raw_binary_norm_max_error": C["norm_err"],
            "lp_status": lp["status"], "lp_objective": lp_obj,
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "final_cost": final_cost, "final_source": source,
            "strictly_below_lp": bool(final_cost is not None and lp_obj is not None
                                      and final_cost < lp_obj - 1e-9 * (1 + abs(lp_obj))),
            "final_max_violation": (None if best is None else best.max_violation),
            "final_stability_flips": (None if best is None else best.stability_flips),
            "final_flips_by_group": fg,
            "candidate_costs": {c.label: c.cost for c in cands if c.cost is not None},
            "select_errors": errors,
            "final_solution": (None if final_sol is None
                               else {k: np.asarray(getattr(final_sol, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "mip_prep_s": mip["prep_s"], "mip_solve_s": mip["solve_s"],
                       "budget_given_s": remaining, "total_s": total,
                       "overrun_s": max(0.0, total - BUDGET)},
            "log_tail": mip["log"][-2000:],
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-30s %-4s s%d rel=%-5d final=%-11s (%s) G_vs_repair=%+.4f opt=%-5s t=%.1f"
              % (len(runs), len(plan), state, method, seed, len(rel),
                 "-" if final_cost is None else "%.6g" % final_cost, source,
                 (jr - (final_cost or jr)) / max(1.0, abs(jr)),
                 r["mip_proven_optimal"], total), flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "lp_budget_s": LP_BUDGET, "kappa": KAPPA,
                            "methods": list(METHODS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "scope": "16 NEW large nominal instances, seeds 6-13; 32 states",
                            "primary_comparison": "AB vs FULL",
                            "secondary_comparisons": ["AB vs ABC", "AB vs A", "AB vs BC"],
                            "new_instance_seeds": [6, 7, 8, 9, 10, 11, 12, 13],
                            "solver": "highspy", "solver_version": _hs_version(),
                            "protocol": "same LP for every method, then MIP from that solution; "
                                        "unreleased binaries fixed to S_r; kappa and fixings "
                                        "relative to S_r; 20 s total including the LP",
                            "output_rule": "best valid among {solver incumbent, S_LP, S_r}",
                            "note": "instances were NOT redrawn on nominal gap, repair cost or "
                                    "algorithm performance"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "method", "seed",
              "repair_cost", "lp_objective", "mip_objective", "mip_status",
              "mip_proven_optimal", "mip_dual_bound", "mip_gap", "mip_start_adopted",
              "n_released", "final_cost", "final_source", "strictly_below_lp",
              "final_stability_flips", "total_s", "overrun_s"]
    with open("stage10_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            flat["total_s"] = r["timing"]["total_s"]
            flat["overrun_s"] = r["timing"]["overrun_s"]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    not_adopted = sum(1 for r in runs if not r["mip_start_adopted"])
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    print("\nMIPs: %d | wall %.1f s" % (len(runs), time.perf_counter() - t_batch))
    print("selection errors: %d | MIP start not adopted: %d | output worse than repair: %d"
          % (bad, not_adopted, worse))
    print("proven optimal early: %s"
          % {m: "%d/%d" % (sum(1 for r in runs if r["method"] == m and r["mip_proven_optimal"]),
                           sum(1 for r in runs if r["method"] == m)) for m in METHODS})
    print("final source counts: %s"
          % {s: sum(1 for r in runs if r["final_source"] == s)
             for s in ("solver", "lp", "repair", "none")})
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if (bad == 0 and not_adopted == 0 and worse == 0) else 1


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:                                # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
