#!/usr/bin/env python
"""
Stage 09B: which variable group is worth releasing?

    .venv-hs/Scripts/python.exe stage09b_run.py --resume
    .venv-hs/Scripts/python.exe stage09b_savecheck.py
    .venv-hs/Scripts/python.exe stage09b_analyse.py

The single question
-------------------
> Starting from the continuous optimum, which binary variable group can be released so that
> extra gain is obtained WITHIN 20 s, and which beats full EMPTY?

Variable groups
---------------
    A = { Y_ijt : t > tau }          far-horizon setups
    B = { Z_ijt : 1 <= t <= tau }    short-term carry-overs
    C = { Z_ijt : tau < t < T }      far-horizon carry-overs

Structurally pinned Z (Z_ij0 and Z_ijT under audited_v1) stay fixed in EVERY configuration, and
the short-term Y (t <= tau) stays fixed in every configuration.  The seven released subsets are
A, B, C, AB, AC, BC, ABC; ABC is EMPTY's full freedom.  LP (nothing released) is the stage 09A
baseline and needs no MIP run.

This is a complete combinatorial experiment over the three groups.  It checks at once whether a
single class of decisions can deliver gain, whether KEEPING some class fixed helps a
time-limited search, and whether some combination behaves better.  The last point can only be
called a TIME-LIMITED COMBINATION EFFECT: with unconverged objective values nothing here
establishes mathematical complementarity or necessity.

Frozen protocol
---------------
* the original 16 LARGE fault states; original model, kappa, tau, solver version, 1 thread
* every unreleased binary is fixed to S_r's NORMALISED value (integer tolerance checked first)
* EVERY MIP starts from the SAME stage 09A LP solution S_LP
* fixing conditions and the stability reference remain S_r -- never redefined
* solver seeds 0 and 1; 20 s total budget INCLUDING the LP initialisation and method setup
* no time slicing, no restart, no solver-parameter changes
* to avoid the stage 07 shared-prefix accounting failure, each logical run RE-RUNS the cheap LP
  itself (~0.1 s); its objective is checked against stage 09A
* the delivered output is the best VALID candidate among at least {solver, S_LP, S_r} -- it is
  never allowed to fall back to the worse S_r alone

16 states x 7 configurations x 2 seeds = 224 MIPs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import build_highs, repair_vector
from lsp_model import MODE_AUDITED, Solution, build_model, enumeration_space, unpack
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from lsp_select import evaluate_candidate, is_usable, select_output
from stage06_run import KAPPA, fixing_from_release, solution_from_record, stability_row

BUDGET = 20.0
LP_BUDGET = 5.0
SEEDS = (0, 1)
THREADS = 1
SHUFFLE_SEED = 20260930
CONFIGS = ("A", "B", "C", "AB", "AC", "BC", "ABC")
RUNS_FILE = "stage09b_runs.json"
CKPT_FILE = "stage09b_runs.partial.json"
TOL = 1e-6


def group_of(kind: str, t: int, tau: int, T: int) -> Optional[str]:
    """A = far Y, B = short Z, C = far Z (excluding the structurally pinned t=T)."""
    if kind == "Y":
        return "A" if t > tau else None          # short-term Y is never released
    if t <= tau:
        return "B"
    if t < T:
        return "C"
    return None                                   # Z_ijT is structurally pinned


def slot_sets(pert, tau: int) -> Dict[str, Set[Tuple[str, int, int, int]]]:
    out: Dict[str, Set[Tuple[str, int, int, int]]] = {"A": set(), "B": set(), "C": set()}
    for slot in enumeration_space(pert, MODE_AUDITED):
        kind, i, j, t = slot
        g = group_of(kind, t, tau, pert.T)
        if g:
            out[g].add(slot)
    return out


def vector_of(sol: Solution, bm) -> np.ndarray:
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


def solve_lp_fixed(bm, lb_pin, ub_pin, budget: float, log_path: str) -> Dict[str, Any]:
    """Stage 09A LP: every binary pinned by bounds, integrality dropped."""
    import dataclasses

    import highspy

    bm_lp = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality),
                               var_lb=lb_pin, var_ub=ub_pin)
    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm_lp, {}, ({}, 0.0))
    prep = time.perf_counter() - t0
    for stale in (log_path,):
        if os.path.exists(stale):
            os.remove(stale)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    h.passModel(lp)
    h.setOptionValue("time_limit", float(budget))
    h.setOptionValue("threads", int(THREADS))
    t1 = time.perf_counter()
    h.run()
    solved = time.perf_counter() - t1
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    obj, out = None, None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        x = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(x)):
            X, Y, Z, I, L = unpack(x, bm.idx)
            try:
                obj = float(info.objective_function_value)
            except Exception:                        # noqa: BLE001
                obj = None
            out = Solution(status=status, objective=obj if obj is not None else float("nan"),
                           X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
    return {"status": status, "objective": obj, "solution": out,
            "prep_s": prep, "solve_s": solved}


def solve_mip(bm, fixed: Dict[int, float], stab, budget: float, seed: int,
              start_values: np.ndarray, log_path: str) -> Dict[str, Any]:
    import highspy

    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm, fixed, stab)
    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    h.passModel(lp)
    h.setOptionValue("time_limit", float(max(0.1, budget)))
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("threads", int(THREADS))
    set_status = h.setSolution(n_cols, np.arange(n_cols, dtype=np.int32),
                               np.asarray(start_values, dtype=np.float64))
    prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    h.run()
    solved = time.perf_counter() - t1
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    obj, out = None, None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        x = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(x)):
            X, Y, Z, I, L = unpack(x, bm.idx)
            try:
                obj = float(info.objective_function_value)
            except Exception:                        # noqa: BLE001
                obj = None
            out = Solution(status=status, objective=obj if obj is not None else float("nan"),
                           X=X, Y=Y, Z=Z, I=I, L=L, mode=MODE_AUDITED)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log_txt = fh.read()
    except OSError:
        log_txt = ""
    return {"status": status, "objective": obj, "solution": out,
            "dual_bound": (float(info.mip_dual_bound)
                           if np.isfinite(info.mip_dual_bound) else None),
            "gap": (float(info.mip_gap) if np.isfinite(info.mip_gap) else None),
            "prep_s": prep, "solve_s": solved, "set_status": str(set_status),
            "adopted": "MIP start solution is feasible" in log_txt,
            "log": log_txt}


def flips_by_group(pert, sol: Solution, repair: Solution, tau: int) -> Dict[str, int]:
    out = {"A": 0, "B": 0, "C": 0, "short_Y": 0}
    for i in range(pert.N):
        for j in range(pert.M):
            for t in range(1, pert.T + 1):
                if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                    if t <= tau:
                        out["short_Y"] += 1
                    else:
                        out["A"] += 1
                if abs(float(sol.Z[i, j, t - 1]) - float(repair.Z[i, j, t - 1])) > 0.5:
                    g = group_of("Z", t, tau, pert.T)
                    if g:
                        out[g] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(s for s in ST if ST[s]["meta"]["scale"] == "large")
    if len(states) != 16:
        raise SystemExit("expected 16 large states, found %d" % len(states))
    with open("stage09a_runs.json", encoding="utf-8") as fh:
        ref_lp = {r["state"]: r for r in json.load(fh)["runs"]}

    plan = [(st, cfg, sd) for st in states for cfg in CONFIGS for sd in SEEDS]
    random.Random(SHUFFLE_SEED).shuffle(plan)
    print("states %d | configs %s | seeds %s | MIPs planned: %d"
          % (len(states), list(CONFIGS), list(SEEDS), len(plan)))
    print("each run re-runs its own LP (%.0f s cap) then gives the remainder to the MIP"
          % LP_BUDGET)

    runs: List[Dict[str, Any]] = []
    done: Set[str] = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {"%s|%s|%d" % (r["state"], r["config"], r["seed"]) for r in runs}
            print("resuming: %d already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA, "configs": list(CONFIGS),
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
        sets = build_release_sets(pert, repair, tau, dis)
        groups = slot_sets(pert, tau)
        # normalised S_r binary values + integer-tolerance check
        assign: Dict[Tuple[str, int, int, int], int] = {}
        norm_err = 0.0
        for slot in enumeration_space(pert, MODE_AUDITED):
            kind, i, j, t = slot
            raw = float(repair.Y[i, j, t - 1]) if kind == "Y" else float(repair.Z[i, j, t - 1])
            norm_err = max(norm_err, abs(raw - round(raw)))
            assign[slot] = int(round(raw))
        lb_pin, ub_pin = bm.var_lb.copy(), bm.var_ub.copy()
        for slot, val in assign.items():
            kind, i, j, t = slot
            col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
            lb_pin[col] = float(val)
            ub_pin[col] = float(val)
        cache[st] = {"meta": rec["meta"], "inst": pert, "repair": repair, "bm": bm, "tau": tau,
                     "stab": stability_row(bm, repair, tau), "x0": repair_vector(bm, repair),
                     "groups": groups, "assign": assign, "norm_err": norm_err,
                     "jr": float(repair.objective), "legal": sets["legal"]}

    os.makedirs("stage09b_logs", exist_ok=True)
    t_batch = time.perf_counter()
    for (state, config, seed) in plan:
        if "%s|%s|%d" % (state, config, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = C["jr"]
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), config, seed)

        # ---- phase 0: the cheap LP, re-run for THIS run (no shared prefix)
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage09b_logs", tag + "__lp.log"))
        lp_obj = lp["objective"]
        ref = ref_lp[state]["lp_objective"]
        lp_matches = (lp_obj is not None
                      and abs(lp_obj - ref) <= 1e-6 * (1 + abs(ref)))

        # ---- released set for this configuration
        rel: List[Tuple[str, int, int, int]] = []
        for g in config:
            rel.extend(sorted(C["groups"][g]))
        rel_set = set(rel)
        fixed_cols: Dict[int, float] = {}
        for slot, val in C["assign"].items():
            if slot in rel_set:
                continue
            kind, i, j, t = slot
            col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
            fixed_cols[col] = float(val)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}

        # ---- phase 1: MIP warm-started from S_LP, on the REMAINING budget
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_lp = vector_of(lp["solution"], bm) if lp["solution"] is not None else C["x0"]
        mip = solve_mip(bm, fixed_cols, C["stab"], remaining, seed, x_lp,
                        os.path.join("stage09b_logs", tag + ".mip.log"))

        # ---- output selection: best VALID among {solver, S_LP, S_r}
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
            best, source, reason = None, "none", "no_valid_candidate"
        else:
            best = min(usable, key=lambda c: c.cost)
            source = best.label
            reason = ("solver_better" if best.label == "solver"
                      else "lp_better_than_solver" if best.label == "lp" else "repair_fallback")
        total = time.perf_counter() - t_run0

        final_sol = None if best is None else best.solution
        fg = ({"A": 0, "B": 0, "C": 0, "short_Y": 0} if final_sol is None
              else flips_by_group(pert, final_sol, repair, tau))

        runs.append({
            "state": state, "config": config, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ST[state]["disruption"]["name"],
            "budget_s": BUDGET, "repair_cost": jr,
            "n_released": len(rel), "released_groups": list(config),
            "raw_binary_norm_max_error": C["norm_err"],
            "lp_status": lp["status"], "lp_objective": lp_obj,
            "lp_matches_stage09a": bool(lp_matches),
            "lp_ref_objective": ref,
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_start_adopted": bool(mip["adopted"]),
            "mip_set_status": mip["set_status"],
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": source, "final_reason": reason,
            "final_max_violation": (None if best is None else best.max_violation),
            "final_stability_flips": (None if best is None else best.stability_flips),
            "final_flips_by_group": fg,
            "candidate_costs": {c.label: c.cost for c in cands if c.cost is not None},
            "select_errors": errors,
            "final_solution": (None if final_sol is None
                               else {k: np.asarray(getattr(final_sol, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "mip_solution": (None if mip["solution"] is None
                             else {k: np.asarray(getattr(mip["solution"], k)).tolist()
                                   for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "mip_prep_s": mip["prep_s"], "mip_solve_s": mip["solve_s"],
                       "budget_given_s": remaining, "total_s": total,
                       "overrun_s": max(0.0, total - BUDGET)},
            "log_tail": mip["log"][-2500:],
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-28s %-4s s%d rel=%-5d final=%-11s (%s) extra=%+.4f t=%.1f"
              % (len(runs), len(plan), state, config, seed, len(rel),
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 source, (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)), total), flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "lp_budget_s": LP_BUDGET, "kappa": KAPPA,
                            "configs": list(CONFIGS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "threads": THREADS,
                            "scope": "16 frozen LARGE fault states from stage 06",
                            "solver": "highspy", "solver_version": _hs_version(),
                            "groups": {"A": "Y_ijt, t > tau", "B": "Z_ijt, 1 <= t <= tau",
                                       "C": "Z_ijt, tau < t < T (Z_ijT is structurally pinned)"},
                            "protocol": "every MIP warm-started from its own re-run stage 09A "
                                        "LP solution; unreleased binaries fixed to S_r; kappa "
                                        "and fixings relative to S_r; 20 s total including "
                                        "the LP initialisation",
                            "output_rule": "best valid among {solver incumbent, S_LP, S_r}"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "config", "seed",
              "repair_cost", "lp_objective", "lp_matches_stage09a", "mip_objective",
              "mip_status", "mip_dual_bound", "mip_gap", "mip_start_adopted",
              "n_released", "final_cost", "final_source", "final_stability_flips",
              "total_s", "overrun_s"]
    with open("stage09b_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            flat["total_s"] = r["timing"]["total_s"]
            flat["overrun_s"] = r["timing"]["overrun_s"]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    not_adopted = sum(1 for r in runs if not r["mip_start_adopted"])
    lp_mismatch = sum(1 for r in runs if not r["lp_matches_stage09a"])
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    print("\nMIPs: %d | wall %.1f s" % (len(runs), time.perf_counter() - t_batch))
    print("selection errors: %d | MIP start not adopted: %d | LP objective mismatch vs 09A: %d "
          "| output worse than repair: %d" % (bad, not_adopted, lp_mismatch, worse))
    print("final source counts: %s"
          % {s: sum(1 for r in runs if r["final_source"] == s)
             for s in ("solver", "lp", "repair", "none")})
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if (bad == 0 and not_adopted == 0 and worse == 0) else 1


def pinned_bounds(C):
    """Pinned bounds for the LP: every binary at S_r's normalised value.

    Memoised ON the per-state dict rather than in a module-level map keyed by id(): CPython
    reuses object ids after garbage collection, which already caused one bug in this project.
    """
    if "_pinned" not in C:
        bm = C["bm"]
        lb, ub = bm.var_lb.copy(), bm.var_ub.copy()
        for slot, val in C["assign"].items():
            kind, i, j, t = slot
            col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
            lb[col] = float(val)
            ub[col] = float(val)
        C["_pinned"] = (lb, ub)
    return C["_pinned"]


def _hs_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("highspy")
    except Exception:                                # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
