#!/usr/bin/env python
"""
Stage 07: find a better start inside the narrow (EMPTY) subproblem, then widen the release set.

    .venv-hs/Scripts/python.exe stage07_run.py --resume
    .venv-hs/Scripts/python.exe stage07_savecheck.py
    .venv-hs/Scripts/python.exe stage07_analyse.py

Research question
-----------------
Spend the first 5 s solving the EMPTY subproblem, then hand the remaining 15 s to DEPENDENCY-24
or FULL.  Does that beat running EMPTY continuously (E20) or restarting EMPTY (E5->E15)?

Why this is worth asking
------------------------
`stage06_migration_certificate.py` certifies 10 (state, seed) pairs on large where EMPTY's
delivered plan is strictly cheaper than DEPENDENCY-24's, yet that EMPTY plan satisfies
DEPENDENCY-24's fixing conditions with 0 short-term flips -- i.e. it is a member of
F_DEPENDENCY and kappa-feasible against the SAME reference S_r.  DEPENDENCY-24 therefore did not
miss that plan because its release set excluded it; it failed to FIND it.  Stage 07 tests whether
supplying such a plan as a MIP start lets the wider release set continue from it.

Methods (all on the SAME 16 frozen large fault states, same solver seeds, same kappa, same S_r)
----------------------------------------------------------------------------------------------
    E20       EMPTY for the whole 20 s                                   deployment baseline
    E5->E15   EMPTY 5 s, then a NEW EMPTY solver with the phase-1 plan   restart control
    E5->D15   EMPTY 5 s, then DEPENDENCY-24 with the phase-1 plan        widen to DEPENDENCY
    E5->F15   EMPTY 5 s, then FULL with the phase-1 plan                 widen to FULL

Frozen constraints
------------------
* kappa and the fixing conditions are ALWAYS relative to the original frozen repaired plan S_r.
  The reference point is never redefined in phase 2.
* DEPENDENCY-24's release set is the STAGE 06 set, read from `stage06_runs.json`.  It is NOT
  recomputed from the phase-1 result, so only one factor changes between E5->E15 and E5->D15.
* Phase 2 receives the best VERIFIED phase-1 plan; if phase 1 did not improve on S_r, S_r itself
  is submitted.
* Every method is charged its full wall-clock time.  Each staged run computes its OWN 5 s phase-1
  prefix, so phase 2 receives exactly the remainder.  (An earlier version shared one prefix
  between the three staged methods and then granted phase 2 a fresh budget, giving those runs
  ~25 s of solver time; that version was discarded and every staged run was re-run.)
* No solution is borrowed from any other run.

16 states x 4 methods x 2 seeds = 128 runs, ~43 min of solve budget.
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
from lsp_highs import repair_vector, solve_highs
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from lsp_select import select_output
from stage06_run import fixing_from_release, solution_from_record, stability_row

BUDGET = 20.0
PHASE1_BUDGET = 5.0
KAPPA = 4.0
SEEDS = (0, 1)
THREADS = 1
SHUFFLE_SEED = 20260928
METHODS = ("E20", "E5->E15", "E5->D15", "E5->F15")
PHASE2_CONFIG = {"E20": None, "E5->E15": "EMPTY", "E5->D15": "DEPENDENCY-24", "E5->F15": "FULL"}
STAGED = ("E5->E15", "E5->D15", "E5->F15")
STAGE06_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage06_runs.json")
RUNS_FILE = "stage07_runs.json"
CKPT_FILE = "stage07_runs.partial.json"


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


def vector_of(sol: Solution, bm) -> np.ndarray:
    """Full variable vector in the model's own column order (same layout as repair_vector)."""
    x = np.zeros(bm.idx.n)
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, bm.inst.T + 1):
                x[bm.idx.X(i, j, t)] = float(sol.X[i, j, t - 1])
                x[bm.idx.Y(i, j, t)] = float(sol.Y[i, j, t - 1])
                x[bm.idx.Z(i, j, t)] = float(sol.Z[i, j, t - 1])
            x[bm.idx.Z(i, j, 0)] = 0.0
        for t in range(1, bm.inst.T + 1):
            x[bm.idx.I(i, t)] = float(sol.I[i, t - 1])
            x[bm.idx.L(i, t)] = float(sol.L[i, t - 1])
    return x


def plan_of(sol: Optional[Solution]) -> Optional[Dict[str, Any]]:
    if sol is None:
        return None
    return {k: np.asarray(getattr(sol, k)).tolist() for k in ("X", "Y", "Z", "I", "L")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    # ---- frozen scope: the 16 LARGE fault states, no cherry-picking
    states = sorted(s for s in ST if ST[s]["meta"]["scale"] == "large")
    print("large fault states: %d (frozen from stage 06; no state is selected on outcome)"
          % len(states))

    # ---- DEPENDENCY-24's release set is the STAGE 06 set, never recomputed
    with open(STAGE06_RUNS, encoding="utf-8") as fh:
        s6 = json.load(fh)["runs"]
    dep_release: Dict[str, List[Tuple[int, int, int]]] = {}
    for r in s6:
        if r["config"] == "DEPENDENCY-24":
            dep_release.setdefault(r["state"], [tuple(u) for u in r["release"]])
    missing = [s for s in states if s not in dep_release]
    if missing:
        raise SystemExit("stage 06 DEPENDENCY-24 release set missing for: %s" % missing)
    print("DEPENDENCY-24 release sets taken from stage06_runs.json for all %d states"
          % len(states))

    plan = [(st, m, sd) for st in states for m in METHODS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("runs planned: %d (shuffled with seed %d)" % (len(plan), SHUFFLE_SEED))

    runs: List[Dict[str, Any]] = []
    done: set = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {"%s|%s|%d" % (r["state"], r["method"], r["seed"]) for r in runs}
            print("resuming: %d runs already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "phase1_budget_s": PHASE1_BUDGET,
                                "kappa": KAPPA, "methods": list(METHODS),
                                "seeds": list(SEEDS), "n_runs": len(runs),
                                "complete": len(runs) == len(plan)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    # ---- per-state context
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
        cache[st] = {
            "meta": rec["meta"], "inst": pert, "repair": repair, "bm": bm, "tau": tau,
            "stab": stability_row(bm, repair, tau), "x0": repair_vector(bm, repair),
            "legal": sets["legal"],
            # EMPTY and the stage-06 DEPENDENCY-24 set (frozen, not recomputed)
            "rel": {"EMPTY": [], "DEPENDENCY-24": dep_release[st], "FULL": sets["FULL"]["release"]},
            "jr": float(repair.objective),
        }
        for name, rel in cache[st]["rel"].items():
            if name != "EMPTY":
                assert all(tuple(u) in {tuple(v) for v in sets["legal"]} for u in rel), \
                    "%s release contains an illegal candidate" % name

    # ---- phase-1 context: EMPTY for PHASE1_BUDGET.
    # Each staged run computes its OWN phase 1.  An earlier version shared one phase-1 prefix
    # between the three staged methods of a (state, seed); that version then handed phase 2 a
    # FRESH full budget instead of the remainder, so those runs received ~25 s of solver time
    # instead of 20 s.  Recomputing per run removes that failure mode entirely and makes the
    # budget accounting unambiguous, at the cost of ~5 s per run.
    stage06_pair = {}
    for st in states:
        for sd in SEEDS:
            e20 = next(r for r in s6 if r["state"] == st and r["config"] == "EMPTY"
                       and int(r["seed"]) == sd)
            stage06_pair[(st, sd)] = float(e20["selected_cost"])

    def phase1(state: str, seed: int, deadline: float) -> Dict[str, Any]:
        """EMPTY for at most PHASE1_BUDGET, measured from `deadline - BUDGET`."""
        C = cache[state]
        bm, repair, tau = C["bm"], C["repair"], C["tau"]
        t_phase1_start = time.perf_counter()
        fixed, fixed_Y = fixing_from_release(bm, repair, tau, [])
        run = solve_highs(bm, fixed, C["stab"], PHASE1_BUDGET, seed=seed,
                          start_values=C["x0"], threads=THREADS)
        sel = select_output(C["inst"], run.solution, repair, KAPPA, tau,
                            fixed_Y=fixed_Y, solver_status=run.status)
        elapsed = time.perf_counter() - t_phase1_start   # full charge: prep+transfer+verify+select
        best = sel.chosen.solution if (sel.chosen is not None and sel.chosen.solution is not None) \
            else repair
        cost = float(sel.chosen.cost) if sel.chosen is not None else C["jr"]
        return {"run": run, "best": best, "cost": cost, "elapsed": elapsed,
                "adopted": bool(run.start_adopted), "source": sel.source,
                "phase1_sel": sel, "fixed_Y": fixed_Y}

    t_batch = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        deadline = time.perf_counter() + BUDGET
        rec: Dict[str, Any] = {"state": state, "method": method, "seed": seed,
                               "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
                               "inst_seed": C["meta"]["seed"],
                               "disruption": ST[state]["disruption"]["name"],
                               "budget_s": BUDGET, "repair_cost": C["jr"]}

        if method == "E20":
            # ---- reference arm: EMPTY continuously for the whole budget
            fixed, fixed_Y = fixing_from_release(bm, repair, tau, [])
            run = solve_highs(bm, fixed, C["stab"], BUDGET, seed=seed,
                              start_values=C["x0"], threads=THREADS)
            sel = select_output(pert, run.solution, repair, KAPPA, tau,
                                fixed_Y=fixed_Y, solver_status=run.status)
            rec.update({
                "phase1": None, "phase2": {
                    "config": "EMPTY", "release_size": 0,
                    "raw_objective": run.objective, "highs_status": run.status,
                    "mip_gap": run.mip_gap, "dual_bound": run.dual_bound,
                    "start_adopted": bool(run.start_adopted),
                    "start_reported_objective": run.start_reported_objective,
                    "solver_s": run.solved_wall_s, "transfer_s": run.transfer_wall_s,
                },
                "phase2_raw_valid": bool(
                    [c for c in sel.candidates if c.label == "solver"][0].feasible),
                "phase2_raw_flips": [c for c in sel.candidates
                                     if c.label == "solver"][0].stability_flips,
                "final_cost": (None if sel.chosen is None else float(sel.chosen.cost)),
                "final_source": sel.source, "select_errors": sel.errors,
                "final_solution": (None if sel.chosen is None or sel.chosen.solution is None
                                   else plan_of(sel.chosen.solution)),
                "phase2_solution": plan_of(run.solution),
                "phase2_improved_over_phase1": None,
            })
        else:
            # ---- staged arms: each computes its OWN phase 1, then gets only the remainder
            p1 = phase1(state, seed, deadline)
            remaining = max(0.1, deadline - time.perf_counter())
            p2cfg = PHASE2_CONFIG[method]
            rel = C["rel"][p2cfg]
            fixed, fixed_Y = fixing_from_release(bm, repair, tau, rel)
            x_start = vector_of(p1["best"], bm)
            run2 = solve_highs(bm, fixed, C["stab"], remaining, seed=seed,
                               start_values=x_start, threads=THREADS)
            sel2 = select_output(pert, run2.solution, repair, KAPPA, tau,
                                 fixed_Y=fixed_Y, solver_status=run2.status)
            # best candidate across phase 1 and phase 2, all validated against the SAME S_r
            cands = [c for c in sel2.candidates if c.label == "solver"]
            raw2 = cands[0] if cands else None
            final_cost = None if sel2.chosen is None else float(sel2.chosen.cost)
            final_sol = None if (sel2.chosen is None or sel2.chosen.solution is None) \
                else sel2.chosen.solution
            rec.update({
                "phase1": {
                    "config": "EMPTY", "budget_s": PHASE1_BUDGET,
                    "raw_objective": p1["run"].objective, "highs_status": p1["run"].status,
                    "mip_gap": p1["run"].mip_gap,
                    "start_adopted": bool(p1["run"].start_adopted),
                    "cost_after_select": float(p1["cost"]),
                    "improved_over_repair": bool(p1["cost"] < C["jr"] - 1e-6 * (1 + abs(C["jr"]))),
                    "wall_s": float(p1["elapsed"]),
                },
                "phase2": {
                    "config": p2cfg, "release_size": len(rel),
                    "raw_objective": run2.objective, "highs_status": run2.status,
                    "mip_gap": run2.mip_gap, "dual_bound": run2.dual_bound,
                    "start_adopted": bool(run2.start_adopted),
                    "start_reported_objective": run2.start_reported_objective,
                    "solver_s": run2.solved_wall_s, "transfer_s": run2.transfer_wall_s,
                    "budget_given_s": float(remaining),
                },
                "phase2_raw_valid": (None if raw2 is None else bool(raw2.feasible)),
                "phase2_raw_flips": (None if raw2 is None else raw2.stability_flips),
                "final_cost": final_cost, "final_source": sel2.source,
                "select_errors": sel2.errors,
                "final_solution": plan_of(final_sol),
                "phase2_solution": plan_of(run2.solution),
                "phase2_improved_over_phase1": bool(
                    final_cost is not None
                    and final_cost < float(p1["cost"]) - 1e-6 * (1 + abs(float(p1["cost"])))),
            })

        total = time.perf_counter() - (deadline - BUDGET)
        rec["timing"] = {"total_s": total, "overrun_s": max(0.0, total - BUDGET)}
        runs.append(rec)
        write_ckpt()
        if len(runs) % 4 == 0 or len(runs) == len(plan):
            print("  [%3d/%3d] %-28s %-8s s%d final=%-12s (%s) t=%5.1f"
                  % (len(runs), len(plan), state, method, seed,
                     _g(rec["final_cost"]), rec["final_source"], total), flush=True)

    # ---- final write
    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "phase1_budget_s": PHASE1_BUDGET,
                            "kappa": KAPPA, "methods": list(METHODS),
                            "seeds": list(SEEDS), "n_runs": len(runs),
                            "shuffle_seed": SHUFFLE_SEED, "threads": THREADS,
                            "scope": "16 frozen LARGE fault states from stage 06",
                            "phase2_release": PHASE2_CONFIG,
                            "dep_release_source": "stage06_runs.json (frozen, not recomputed)",
                            "solver": "highspy", "solver_version": _hs_version(),
                            "note": "kappa and fixing conditions are always relative to the "
                                    "original frozen repaired plan S_r; phase 2 never "
                                    "redefines the reference point"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "method", "seed",
              "repair_cost", "final_cost", "final_source"]
    with open("stage07_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)

    bad = sum(1 for r in runs if r["select_errors"])
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    print("\nruns: %d | wall %.1f s" % (len(runs), time.perf_counter() - t_batch))
    print("selection errors: %d | outputs worse than repair: %d" % (bad, worse))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if (bad == 0 and worse == 0) else 1


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
