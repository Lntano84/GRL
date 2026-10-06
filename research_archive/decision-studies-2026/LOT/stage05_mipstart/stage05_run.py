#!/usr/bin/env python
"""
Stage 05: does restricting the variable set still pay once the FULL solver receives the SAME
MIP start?

    .venv-hs/Scripts/python.exe stage05_run.py
    .venv-hs/Scripts/python.exe stage05_savecheck.py

Design (frozen)
---------------
states      : the 8 stage-04b states (medium/large x rho in {0.75,1.10} x seed 0 x {D1,D2})
nominal and repaired plans : the SAME stored plans, unchanged
model       : identical matrices, kappa, tau, fixing rules and Y/Z freedom as stage 04/04b
methods     : FULL, EMPTY, SHORTAGE-12, SHORTAGE-24
start       : cold (no start submitted), warm (the repaired plan submitted via setSolution)
solver seed : 0, 1 (paired across the two start conditions)
budget      : 20 s per run
execution   : single thread, serial, run order shuffled with a fixed seed

    8 x 4 x 2 x 2 = 128 runs, about 43 min of solve budget.

Both start conditions use the SAME highspy version. A cold result from the old SciPy path is
never compared against a warm result from highspy, because that would mix solver/interface
changes into the comparison.

Metrics (J_r is the state's frozen repair cost, so these are relative cost differences and
NOT optimality gaps):
    delta_start     = (J_cold - J_warm) / max(1, |J_r|)
    delta_restrict  = (J_warm,FULL - J_warm,a) / max(1, |J_r|)

Outputs: stage05_results.csv, stage05_runs.json, stage05_details.md
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
from lsp_gen import all_instances, apply_disruption, disruptions_for
from lsp_highs import repair_vector, solve_highs
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_repair3 import binary_change_count
from lsp_select import is_usable, select_output

BUDGET = 20.0
KAPPA = 4.0
METHODS = ("FULL", "EMPTY", "SHORTAGE-12", "SHORTAGE-24")
SEEDS = (0, 1)
STARTS = ("cold", "warm")
SHUFFLE_SEED = 20260926
THREADS = 1
SOURCE_STATES = "../stage04_runtime_calibration/stage04_states.json"
SELECTED = [(s, r) for s in ("medium", "large") for r in (0.75, 1.10)]
INST_SEED = 0


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


def _g(v: Any, nd: int = 6) -> str:
    x = _f(v)
    if x is None or not np.isfinite(x):
        return "-"
    if abs(x) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % x


def solution_from_record(inst, rec: Dict[str, Any]) -> Solution:
    p = rec["plan"]
    return Solution(status="stored", objective=float(rec["objective"]),
                    X=np.array(p["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def method_fixing(bm, repair: Solution, tau: int, method: str):
    """Identical release rules to stage 03/04/04b: the shortage score
    a(i,j,t) = l_i * sum_{r>=t} L^r_ir / (s_i + b_i m_i), dictionary tie-breaking, and
    candidates that compatibility or capacity rule out are excluded."""
    inst = bm.inst

    def possible(u):
        i, j, t = u
        return inst.w[i][j] != 0 and inst.s[i] <= inst.c[j][t - 1] + 1e-9

    short = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
             for t in range(1, tau + 1)]
    feas = [u for u in short if possible(u)]
    if method == "FULL":
        release = list(feas)
        note = "all legal short-term Y"
    elif method == "EMPTY":
        release = []
        note = "empty"
    else:
        k = int(method.split("-")[1])
        scores = {}
        for (i, j, t) in feas:
            shortage = sum(float(repair.L[i, r - 1]) for r in range(t, inst.T + 1))
            scores[(i, j, t)] = (inst.l[i] * shortage) / (inst.s[i] + inst.b[i] * inst.m[i])
        release = sorted(feas, key=lambda u: (-scores[u], u))[:k]
        note = "top %d by shortage score" % k
        if len(feas) < k:
            note += " (only %d legal candidates; all released)" % len(feas)
    rel = set(release)
    fixed, fixed_Y = {}, {}
    for (i, j, t) in short:
        if (i, j, t) not in rel:
            val = float(repair.Y[i, j, t - 1])
            fixed[bm.idx.Y(i, j, t)] = val
            fixed_Y[(i, j, t)] = val
    return fixed, fixed_Y, release, note


def stability_row(bm, repair: Solution, tau: int):
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


def count_free(bm, fixed, tau):
    """Slots, structurally pinned and genuinely free, as separate numbers."""
    inst = bm.inst
    free_Y = free_Z = struct_Z = method_Y = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                cY, cZ = bm.idx.Y(i, j, t), bm.idx.Z(i, j, t)
                if bm.var_lb[cY] < bm.var_ub[cY] - 1e-12:
                    if cY in fixed:
                        method_Y += 1
                    else:
                        free_Y += 1
                if bm.var_lb[cZ] < bm.var_ub[cZ] - 1e-12:
                    free_Z += 1
                else:
                    struct_Z += 1
    return {"Y_slots": inst.N * inst.M * inst.T, "Z_slots": inst.N * inst.M * inst.T,
            "Z_structurally_pinned": struct_Z, "Y_method_pinned": method_Y,
            "free_Y_slots": free_Y, "free_Z_slots": free_Z}


def build_todo(states):
    insts = {m["name"]: (i, m) for i, m in all_instances()}
    todo = []
    for name, (inst0, meta) in sorted(insts.items()):
        if meta["scale"] not in ("medium", "large") or meta["seed"] != INST_SEED \
                or meta["rho"] not in (0.75, 1.10):
            continue
        for dis in disruptions_for(inst0):
            state = "%s|%s" % (name, dis.name)
            if state in states:
                todo.append((state, meta, inst0, dis))
    return sorted(todo)


def plan_key(r: Dict[str, Any]) -> str:
    return "%s|%s|%s|%d" % (r["state"], r["method"], r["start"], r["seed"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs already present in stage05_runs.partial.json")
    args = ap.parse_args(argv)

    with open(SOURCE_STATES, encoding="utf-8") as fh:
        ST = json.load(fh)
    todo = build_todo(ST)
    print("states: %d" % len(todo))

    plan: List[Tuple[str, str, str, int]] = [
        (state, method, start, seed)
        for (state, _, _, _) in todo
        for method in METHODS for start in STARTS for seed in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("runs planned: %d (shuffled with seed %d)" % (len(plan), SHUFFLE_SEED))

    runs: List[Dict[str, Any]] = []
    done: set = set()
    # Checkpointing. The first version of this runner wrote only at the very end, so an
    # interrupted session lost every completed run. Results are now written after EVERY run
    # and can be resumed with --resume.
    ckpt = f"{args.outdir}/stage05_runs.partial.json"
    if args.resume and os.path.exists(ckpt):
        try:
            with open(ckpt, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {plan_key(r) for r in runs}
            print("resuming: %d runs already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("could not read checkpoint (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_checkpoint() -> None:
        tmp = ckpt + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "seeds": list(SEEDS),
                                "starts": list(STARTS), "methods": list(METHODS),
                                "shuffle_seed": SHUFFLE_SEED, "threads": THREADS,
                                "source_states": SOURCE_STATES, "solver": "highspy",
                                "solver_version": _hs_version(),
                                "complete": len(runs) == len(plan), "n_runs": len(runs)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, ckpt)

    t_batch = time.perf_counter()
    cache: Dict[str, Any] = {}
    for state, meta, inst0, dis in todo:
        pert = apply_disruption(inst0, dis)
        repair = solution_from_record(pert, ST[state]["repair"])
        bm = build_model(pert, MODE_AUDITED)
        cache[state] = {
            "meta": meta, "inst": pert, "repair": repair, "bm": bm,
            "tau": meta["tau"], "stab": stability_row(bm, repair, meta["tau"]),
            "x0": repair_vector(bm, repair),
            "repair_chk": check_solution(pert, repair, MODE_AUDITED),
        }

    for (state, method, start, seed) in plan:
        if plan_key({"state": state, "method": method, "start": start, "seed": seed}) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        t0 = time.perf_counter()

        # features + ranking + fixing construction, INSIDE the budget
        t_feat0 = time.perf_counter()
        fixed, fixed_Y, release, note = method_fixing(bm, repair, tau, method)
        free = count_free(bm, fixed, tau)
        t_feat = time.perf_counter() - t_feat0

        deadline = t0 + BUDGET
        remaining = max(0.1, deadline - time.perf_counter())
        run = solve_highs(bm, fixed, C["stab"], remaining, seed=seed,
                          start_values=(C["x0"] if start == "warm" else None),
                          threads=THREADS)

        # verification + output selection, also counted
        t_ver0 = time.perf_counter()
        sel = select_output(pert, run.solution, repair, KAPPA, tau,
                            fixed_Y=fixed_Y, solver_status=run.status)
        t_ver = time.perf_counter() - t_ver0
        total = time.perf_counter() - t0

        raw_cand = [c for c in sel.candidates if c.label == "solver"][0]
        runs.append({
            "state": state, "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "method": method, "start": start, "seed": seed, "budget_s": BUDGET,
            "repair_cost": float(repair.objective),
            "repair_feasible": bool(C["repair_chk"].ok),
            "release_note": note, "n_fixed": len(fixed), "n_released": len(release),
            "free": free,
            "highs_status": run.status,
            "raw_objective": run.objective,
            "raw_valid": bool(raw_cand.feasible),
            "raw_problems": raw_cand.problems,
            "raw_flips_vs_repair": raw_cand.stability_flips,
            "mip_gap": run.mip_gap, "dual_bound": run.dual_bound,
            "set_solution_status": run.set_solution_status,
            "start_adopted": run.start_adopted,
            "start_reported_objective": run.start_reported_objective,
            "start_message": run.start_message,
            "selected_cost": (None if sel.chosen is None else float(sel.chosen.cost)),
            "selected_source": sel.source, "select_reason": sel.reason,
            "select_errors": sel.errors,
            "selected_max_violation": (None if sel.chosen is None
                                       else sel.chosen.max_violation),
            "selected_stability_flips": (None if sel.chosen is None
                                         else sel.chosen.stability_flips),
            # BOTH plans are stored so the savecheck can re-read and re-verify them
            "raw_solver_solution": (None if run.solution is None else
                                    {k: np.asarray(getattr(run.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "selected_solution": (None if sel.chosen is None or sel.chosen.solution is None
                                  else {k: np.asarray(getattr(sel.chosen.solution, k)).tolist()
                                        for k in ("X", "Y", "Z", "I", "L")}),
            "final_output_worse_than_repair": bool(
                sel.chosen is not None
                and sel.chosen.cost > float(repair.objective) + 1e-6 * (1 + abs(repair.objective))),
            "timing": {"feature_rank_s": t_feat, "transfer_s": run.transfer_wall_s,
                       "solver_s": run.solved_wall_s, "verify_select_s": t_ver,
                       "total_s": total, "overrun_s": max(0.0, total - BUDGET)},
            "n_cols": run.n_cols, "n_rows": run.n_rows,
        })
        print("  %-28s %-12s %-4s seed=%d raw=%-12s sel=%-12s %-7s adopted=%-5s t=%5.2f"
              % (state, method, start, seed, _g(run.objective), _g(runs[-1]["selected_cost"]),
                 sel.source, run.start_adopted, total), flush=True)
        write_checkpoint()          # survives an interrupted session

    # ---- metrics
    idx = {(r["state"], r["method"], r["start"], r["seed"]): r for r in runs}
    for r in runs:
        jr = r["repair_cost"]
        d = max(1.0, abs(jr))
        cold = idx.get((r["state"], r["method"], "cold", r["seed"]))
        warm = idx.get((r["state"], r["method"], "warm", r["seed"]))
        if r["start"] == "cold" and warm is not None and r["selected_cost"] is not None \
                and warm["selected_cost"] is not None:
            r["delta_start"] = (r["selected_cost"] - warm["selected_cost"]) / d
        else:
            r["delta_start"] = None
        full_warm = idx.get((r["state"], "FULL", "warm", r["seed"]))
        if r["start"] == "warm" and full_warm is not None \
                and r["selected_cost"] is not None and full_warm["selected_cost"] is not None:
            r["delta_restrict"] = (full_warm["selected_cost"] - r["selected_cost"]) / d
        else:
            r["delta_restrict"] = None

    with open(f"{args.outdir}/stage05_runs.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "seeds": list(SEEDS),
                            "starts": list(STARTS), "methods": list(METHODS),
                            "shuffle_seed": SHUFFLE_SEED, "threads": THREADS,
                            "source_states": SOURCE_STATES,
                            "solver": "highspy", "solver_version": _hs_version(),
                            "note": "cold and warm both use this highspy version; the "
                                    "repaired plan is the same for every method and both "
                                    "start conditions",
                            "n_runs": len(runs)},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "method", "start", "seed", "repair_cost",
              "raw_objective", "raw_valid", "selected_cost", "selected_source",
              "highs_status", "mip_gap", "dual_bound", "start_adopted",
              "start_reported_objective", "set_solution_status", "delta_start",
              "delta_restrict", "total_s", "overrun_s", "free_Y_slots", "free_Z_slots",
              "Z_structurally_pinned", "n_fixed", "n_released",
              "final_output_worse_than_repair", "select_errors"]
    with open(f"{args.outdir}/stage05_results.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)

    with open(f"{args.outdir}/stage05_details.md", "w", encoding="utf-8") as fh:
        fh.write(render(runs, idx))

    print("\nbatch wall: %.1f s" % (time.perf_counter() - t_batch))
    bad_sel = sum(1 for r in runs if r["select_errors"])
    bad_out = sum(1 for r in runs if r["final_output_worse_than_repair"])
    not_adopted = sum(1 for r in runs if r["start"] == "warm" and not r["start_adopted"])
    print("selection errors: %d | outputs worse than repair: %d | warm runs without "
          "confirmed adoption: %d" % (bad_sel, bad_out, not_adopted))
    return 0 if (bad_sel == 0 and bad_out == 0 and not_adopted == 0) else 1


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:
        return "unknown"


def render(runs, idx) -> str:
    L: List[str] = []
    L.append("# stage05_details.md\n")
    L.append("MIP-start comparison. highspy %s, budget %.0f s, kappa=%g, threads=1, "
             "run order shuffled with seed %d.\n"
             % (_hs_version(), BUDGET, KAPPA, SHUFFLE_SEED))
    L.append("\n## Absolute costs, both seeds\n")
    L.append("| state | method | seed | J_repair | J_cold | J_warm | raw cold | raw warm | "
             "source cold | source warm | delta_start | delta_restrict |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|")
    for state in sorted({r["state"] for r in runs}):
        for method in METHODS:
            for seed in SEEDS:
                c = idx.get((state, method, "cold", seed))
                w = idx.get((state, method, "warm", seed))
                if c is None or w is None:
                    continue
                L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    state, method, seed, _g(c["repair_cost"]), _g(c["selected_cost"]),
                    _g(w["selected_cost"]), _g(c["raw_objective"]), _g(w["raw_objective"]),
                    c["selected_source"], w["selected_source"],
                    _g(c["delta_start"]), _g(w["delta_restrict"])))
    L.append("\n## MIP start adoption\n")
    L.append("| state | method | seed | setSolution | adopted | reported start cost | log message |")
    L.append("|---|---|---:|---|---:|---:|---|")
    for r in runs:
        if r["start"] != "warm":
            continue
        L.append("| %s | %s | %d | %s | %s | %s | %s |" % (
            r["state"], r["method"], r["seed"], r["set_solution_status"],
            r["start_adopted"], _g(r["start_reported_objective"]),
            (r["start_message"] or "").replace("|", "/")))
    L.append("\n## Raw incumbent validity and timing\n")
    L.append("| state | method | start | seed | raw valid | flips | mip_gap | dual bound | total s | overrun s |")
    L.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for r in runs:
        L.append("| %s | %s | %s | %d | %s | %s | %s | %s | %.2f | %.2f |" % (
            r["state"], r["method"], r["start"], r["seed"], r["raw_valid"],
            r["raw_flips_vs_repair"], _g(r["mip_gap"]), _g(r["dual_bound"]),
            r["timing"]["total_s"], r["timing"]["overrun_s"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
