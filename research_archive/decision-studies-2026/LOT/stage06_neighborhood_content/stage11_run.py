#!/usr/bin/env python
"""
Stage 11: matched random Z-fixing control.

    .venv-hs/Scripts/python.exe stage11_run.py --resume
    .venv-hs/Scripts/python.exe stage11_analyse.py

Question
--------
> With the same NUMBER of fixed Z, the same machine distribution and the same S_r 0/1
> composition, is fixing the FAR-HORIZON Z still more effective than spreading the fixed Z
> randomly across the short-term and far-horizon positions?

Configurations (stage 09B development set: 16 large fault states, 8 nominal instances, seeds 2-5)
------------------------------------------------------------------------------------------------
    AB         short-term Y fixed, A released, B released, C fixed
    ABC        short-term Y fixed, A, B, C all released
    MR-Z-1/2/3 short-term Y fixed, A released, ALL Z released EXCEPT a matched random fixing set

The matched random sets are stratified by (machine, S_r binary value) with the SAME per-stratum
quota as C, drawn from the FULL stratum pool B u C (C is NOT removed from the pool first), and
were frozen by `stage11_matched_random.py` before any MIP ran.  They may fix some B slots and
leave some C slots free, which is exactly what makes them different from AB.  They are never
redrawn.

Protocol is identical to stages 09B/10: one shared continuous LP initialisation, the MIP started
from S_LP, every unreleased binary pinned to S_r, kappa and fixing conditions relative to S_r,
20 s total including initialisation, preparation and verification, seeds 0 and 1, and the
delivered output is the best valid candidate among {solver, S_LP, S_r}.

16 states x 5 configurations x 2 seeds = 160 MIPs.
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
from lsp_release import (CONFIGS, LP_BUDGET, build_state_context, flips_by_group,
                         pinned_bounds, solve_lp_fixed, solve_mip, stage11_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record, stability_row

BUDGET = 20.0
SEEDS = (0, 1)
SHUFFLE_SEED = 20261003
METHODS = ("AB", "ABC", "MR-Z-1", "MR-Z-2", "MR-Z-3")
MR_NAMES = ("MR-Z-1", "MR-Z-2", "MR-Z-3")
STATES_FILE = "stage06_states.json"
MATCHED_FILE = "stage11_matched_random.json"
RUNS_FILE = "stage11_runs.json"
CKPT_FILE = "stage11_runs.partial.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ST = json.load(fh)
    states = sorted(s for s in ST if ST[s]["meta"]["scale"] == "large")
    if len(states) != 16:
        raise SystemExit("expected the 16 large development states, found %d" % len(states))
    with open(MATCHED_FILE, encoding="utf-8") as fh:
        MATCH = json.load(fh)
    missing = [s for s in states if s not in MATCH]
    if missing:
        raise SystemExit("matched sets missing for %s" % missing)
    print("states: %d | methods %s | seeds %s" % (len(states), list(METHODS), list(SEEDS)))
    print("matched-random sets loaded from %s (frozen, never redrawn)" % MATCHED_FILE)

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
        mr_sets = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                   for n in MR_NAMES}
        C.update({"meta": rec["meta"], "inst": pert, "repair": repair, "tau": tau, "bm": bm,
                  "stab": stability_row(bm, repair, tau), "x0": repair_vector(bm, repair),
                  "jr": float(repair.objective), "mr_sets": mr_sets})
        cache[st] = C

    def release_and_fix(C, method):
        return stage11_columns(C, method)

    os.makedirs("stage11_logs", exist_ok=True)
    t_batch = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = C["jr"]
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), method, seed)

        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage11_logs", tag + "__lp.log"))

        rel, fixed_cols, fixed_Z = release_and_fix(C, method)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_lp = vector_of(lp["solution"], bm) if lp["solution"] is not None else C["x0"]
        mip = solve_mip(bm, fixed_cols, C["stab"], remaining, seed, x_lp,
                        os.path.join("stage11_logs", tag + ".mip.log"))

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
        best = min(usable, key=lambda c: c.cost) if usable else None
        total = time.perf_counter() - t_run0
        final_sol = None if best is None else best.solution
        fg = ({"A": 0, "B": 0, "C": 0, "short_Y": 0} if final_sol is None
              else flips_by_group(pert, final_sol, repair, tau))

        # Z bookkeeping straight from the pinned column set: a pinned Z column is a fixed Z.
        Z_cols = set()
        for (_k, i, j, t) in (C["groups"]["B"] | C["groups"]["C"]):
            Z_cols.add(bm.idx.Z(i, j, t))
        n_fixed_Z_actual = sum(1 for c in Z_cols if c in fixed_cols)
        n_free_Z = len(Z_cols) - n_fixed_Z_actual
        runs.append({
            "state": state, "method": method, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ST[state]["disruption"]["name"],
            "budget_s": BUDGET, "repair_cost": jr,
            "n_released": len(rel),
            "n_free_Z": n_free_Z,
            "n_fixed_Z": n_fixed_Z_actual,
            "n_fixed_columns": len(fixed_cols),
            "releases_short_term_Y": False,
            "raw_binary_norm_max_error": C["norm_err"],
            "lp_status": lp["status"], "lp_objective": lp["objective"],
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "mip_log": mip["log"],
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": ("none" if best is None else best.label),
            "strictly_below_lp": bool(best is not None and lp["objective"] is not None
                                      and float(best.cost) < lp["objective"]
                                      - 1e-9 * (1 + abs(lp["objective"]))),
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
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-30s %-7s s%d rel=%-5d freeZ=%-5d final=%-11s (%s) G=%+.4f t=%.1f"
              % (len(runs), len(plan), state, method, seed, len(rel), n_free_Z,
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)),
                 total), flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "lp_budget_s": LP_BUDGET, "kappa": KAPPA,
                            "methods": list(METHODS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "scope": "16 large development states (stage 09B), 8 nominal "
                                     "instances, seeds 2-5",
                            "solver": "highspy", "solver_version": _hs_version(),
                            "matched_random_file": MATCHED_FILE,
                            "protocol": "same LP for every method, then MIP from S_LP; "
                                        "unreleased binaries fixed to S_r; kappa and fixings "
                                        "relative to S_r; 20 s total including the LP",
                            "output_rule": "best valid among {solver incumbent, S_LP, S_r}",
                            "note": "this is a MECHANISM development experiment on the stage 09B "
                                    "development set; stage 10 remains the independent "
                                    "confirmation and this round is NOT a second confirmation"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "scale", "rho", "inst_seed", "disruption", "method", "seed",
              "repair_cost", "lp_objective", "mip_objective", "mip_status",
              "mip_proven_optimal", "mip_dual_bound", "mip_gap", "mip_start_adopted",
              "n_released", "n_free_Z", "n_fixed_Z", "final_cost", "final_source",
              "strictly_below_lp", "final_stability_flips", "total_s", "overrun_s"]
    with open("stage11_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
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
