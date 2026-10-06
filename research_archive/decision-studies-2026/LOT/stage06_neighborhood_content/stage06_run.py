#!/usr/bin/env python
"""
Stage 06 phase 2: the fixed-size neighbourhood CONTENT experiment.

    .venv-hs/Scripts/python.exe stage06_run.py --resume
    .venv-hs/Scripts/python.exe stage06_savecheck.py
    .venv-hs/Scripts/python.exe stage06_analyse.py

Research question
-----------------
On INDEPENDENT new nominal instances, does picking the same 24 short-term Y variables using
fault / carry-over / capacity-competition information beat a count- and position-matched random
set, and does it improve on the simple FULL / EMPTY strategy?

Frozen design
-------------
states        32 (16 new nominal instances x {D1, D2})
configs        8 : FULL, EMPTY, SHORTAGE-24, WINDOW-24, DEPENDENCY-24, MATCHED-RANDOM x3
solver seeds   0, 1        budget 20 s per run, all WARM with the SAME submitted plan
total        32 x 8 x 2 = 512 solves

All eight configurations keep the far-horizon Y and every Z free; only the short-term Y release
set changes. The two new rules and the matched-random construction are frozen in
`lsp_neighborhood.py` and are not tuned here.

Statistics are computed per NOMINAL INSTANCE (the two disruptions and the two solver seeds are
averaged inside an instance first), so 64 runs are never treated as 64 independent samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance
from lsp_highs import repair_vector, solve_highs
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_neighborhood import CONFIGS, build_release_sets
from lsp_repair3 import binary_change_count
from lsp_select import select_output

BUDGET = 20.0
KAPPA = 4.0
SEEDS = (0, 1)
SHUFFLE_SEED = 20260927
THREADS = 1
NEW_SEEDS = (2, 3, 4, 5)
RUNS_FILE = "stage06_runs.json"
CKPT_FILE = "stage06_runs.partial.json"


def _f(v):
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


def _g(v, nd=6):
    x = _f(v)
    if x is None or not np.isfinite(x):
        return "-"
    if abs(x) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % x


def solution_from_record(inst, rec) -> Solution:
    p = rec["plan"]
    return Solution(status="stored", objective=float(rec["objective"]),
                    X=np.array(p["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def stability_row(bm, repair, tau):
    coef, n_one = {}, 0
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, tau + 1):
                yref = float(repair.Y[i, j, t - 1])
                if yref > 0.5:
                    n_one += 1
                    coef[bm.idx.Y(i, j, t)] = -1.0
                else:
                    coef[bm.idx.Y(i, j, t)] = 1.0
    return coef, float(KAPPA) - float(n_one)


def fixing_from_release(bm, repair, tau, release):
    rel = set(release)
    fixed, fixed_Y = {}, {}
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, tau + 1):
                if (i, j, t) not in rel:
                    val = float(repair.Y[i, j, t - 1])
                    fixed[bm.idx.Y(i, j, t)] = val
                    fixed_Y[(i, j, t)] = val
    return fixed, fixed_Y


def plan_key(r) -> str:
    return "%s|%s|%d" % (r["state"], r["config"], r["seed"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(ST)
    print("states: %d" % len(states))

    plan = [(st, cfg, sd) for st in states for cfg in CONFIGS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("runs planned: %d (shuffled with seed %d)" % (len(plan), SHUFFLE_SEED))

    runs: List[Dict[str, Any]] = []
    done: set = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {plan_key(r) for r in runs}
            print("resuming: %d runs already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "seeds": list(SEEDS),
                                "configs": list(CONFIGS), "shuffle_seed": SHUFFLE_SEED,
                                "threads": THREADS, "n_runs": len(runs),
                                "complete": len(runs) == len(plan)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    # ---- per-state context: instance, repair, neighbourhood sets, submitted start
    cache: Dict[str, Any] = {}
    for st in states:
        rec = ST[st]
        inst, _meta = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
        dis = next(d for d in __import__("lsp_gen", fromlist=["disruptions_for"])
                   .disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        sets = build_release_sets(pert, repair, rec["meta"]["tau"], dis)
        cache[st] = {
            "meta": rec["meta"], "inst": pert, "repair": repair, "bm": bm,
            "tau": rec["meta"]["tau"], "stab": stability_row(bm, repair, rec["meta"]["tau"]),
            "x0": repair_vector(bm, repair), "sets": sets,
            "repair_chk": check_solution(pert, repair, MODE_AUDITED),
        }

    t_batch = time.perf_counter()
    for (state, config, seed) in plan:
        if "%s|%s|%d" % (state, config, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        t0 = time.perf_counter()

        t_feat0 = time.perf_counter()
        rel = C["sets"][config]["release"]
        fixed, fixed_Y = fixing_from_release(bm, repair, tau, rel)
        t_feat = time.perf_counter() - t_feat0

        deadline = t0 + BUDGET
        remaining = max(0.1, deadline - time.perf_counter())
        run = solve_highs(bm, fixed, C["stab"], remaining, seed=seed,
                          start_values=C["x0"], threads=THREADS)

        t_ver0 = time.perf_counter()
        sel = select_output(pert, run.solution, repair, KAPPA, tau,
                            fixed_Y=fixed_Y, solver_status=run.status)
        t_ver = time.perf_counter() - t_ver0
        total = time.perf_counter() - t0
        raw_cand = [c for c in sel.candidates if c.label == "solver"][0]

        runs.append({
            "state": state, "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ST[state]["disruption"]["name"],
            "config": config, "seed": seed, "budget_s": BUDGET,
            "repair_cost": float(repair.objective),
            "n_released": len(rel), "n_legal": C["sets"]["n_legal"],
            "n_slots_short": C["sets"]["n_slots"],
            "release": [list(u) for u in rel],
            "match_info": C["sets"][config].get("match"),
            "highs_status": run.status,
            "raw_objective": run.objective, "raw_valid": bool(raw_cand.feasible),
            "raw_problems": raw_cand.problems,
            "raw_flips_vs_repair": raw_cand.stability_flips,
            "mip_gap": run.mip_gap, "dual_bound": run.dual_bound,
            "start_adopted": run.start_adopted,
            "start_reported_objective": run.start_reported_objective,
            "selected_cost": (None if sel.chosen is None else float(sel.chosen.cost)),
            "selected_source": sel.source, "select_reason": sel.reason,
            "select_errors": sel.errors,
            "selected_max_violation": (None if sel.chosen is None
                                       else sel.chosen.max_violation),
            "selected_stability_flips": (None if sel.chosen is None
                                         else sel.chosen.stability_flips),
            "improved_over_repair": bool(
                sel.chosen is not None
                and sel.chosen.cost < float(repair.objective) - 1e-6 * (1 + abs(repair.objective))),
            "raw_solver_solution": (None if run.solution is None else
                                    {k: np.asarray(getattr(run.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "selected_solution": (None if sel.chosen is None or sel.chosen.solution is None
                                  else {k: np.asarray(getattr(sel.chosen.solution, k)).tolist()
                                        for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"feature_rank_s": t_feat, "transfer_s": run.transfer_wall_s,
                       "solver_s": run.solved_wall_s, "verify_select_s": t_ver,
                       "total_s": total, "overrun_s": max(0.0, total - BUDGET)},
        })
        write_ckpt()
        if len(runs) % 8 == 0 or len(runs) == len(plan):
            print("  [%3d/%3d] %-30s %-16s s%d sel=%-12s src=%-7s t=%5.2f"
                  % (len(runs), len(plan), state, config, seed,
                     _g(runs[-1]["selected_cost"]), sel.source, total), flush=True)

    # ---- final write
    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "seeds": list(SEEDS),
                            "configs": list(CONFIGS), "shuffle_seed": SHUFFLE_SEED,
                            "threads": THREADS, "n_runs": len(runs),
                            "new_instance_seeds": list(NEW_SEEDS),
                            "solver": "highspy", "solver_version": _hs_version(),
                            "note": "all runs warm with the SAME submitted repaired plan; "
                                    "only the short-term Y release set differs"},
               "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "config", "seed",
              "repair_cost", "raw_objective", "raw_valid", "selected_cost",
              "selected_source", "improved_over_repair", "n_released", "n_legal",
              "highs_status", "mip_gap", "dual_bound", "start_adopted",
              "selected_stability_flips", "total_s", "overrun_s"]
    with open("stage06_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)

    bad_sel = sum(1 for r in runs if r["select_errors"])
    not_adopted = sum(1 for r in runs if not r["start_adopted"])
    worse = sum(1 for r in runs if r["selected_cost"] is not None
                and r["selected_cost"] > r["repair_cost"] + 1e-6)
    improved = sum(1 for r in runs if r["improved_over_repair"])
    print("\nruns: %d | wall %.1f s" % (len(runs), time.perf_counter() - t_batch))
    print("selection errors: %d | outputs worse than repair: %d | warm without adoption: %d"
          % (bad_sel, worse, not_adopted))
    print("runs that improved on the repaired plan: %d / %d" % (improved, len(runs)))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if (bad_sel == 0 and worse == 0 and not_adopted == 0) else 1


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
