"""Stage 15 runner: disruption breadth x duration crossed grid.

Scope: 8 frozen nominal instances x 4 disruption cells = 32 states, each solved with the four
configurations A / AB / RECOVERY / END at solver seed 0 and a 20 s budget -> 128 MIPs.

Protocol is the Stage 14 protocol, unchanged apart from the state set and the single seed:
* every run computes its OWN continuous LP (S_LP) and starts the MIP from it;
* no Stage 12/13 plan pool, no Q, no change-position guidance;
* the repair plan is the frozen Stage 15 repair (`stage15_states.json`), which for the two
  existing cells reproduces the frozen Stage 06 objective exactly.

The two EXISTING cells are re-run inside this batch so that no cell is confounded with a
different run batch.  This is a mechanism exploration over the state grid, NOT a new
independent confirmation.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED, build_model
from lsp_release import (LP_BUDGET, build_state_context, pinned_bounds, solve_lp_fixed,
                         solve_mip, stage14_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record, stability_row

BUDGET = 20.0
SEEDS = (0,)
SHUFFLE_SEED = 20261015
METHODS = ("A", "AB", "RECOVERY", "END")
STATES_FILE = "stage15_states.json"
RUNS_FILE = "stage15_runs.json"
CKPT_FILE = "stage15_runs.partial.json"


def disruption_of(rec) -> Disruption:
    """Rebuild the disruption from the frozen state record (pure data, no lookup table)."""
    d = rec["disruption"]
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)
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
        dis = disruption_of(rec)
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        tau = rec["meta"]["tau"]
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert,
                  "dis": dis, "mr_sets": {}})
        cache[st] = C

    os.makedirs("stage15_logs", exist_ok=True)
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
                            os.path.join("stage15_logs", tag + "__lp.log"))
        rel, fixed_cols, meta = stage14_columns(C, method)
        fixed_Y = {(i, j, t): float(repair.Y[i, j, t - 1])
                   for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)}
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else C["x0"])
        mip = solve_mip(bm, fixed_cols, stability_row(bm, repair, tau), remaining, seed, x_lp,
                        os.path.join("stage15_logs", tag + ".mip.log"))
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
            "cell": ALL[state]["cell"],
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
        print("  [%3d/%3d] %-26s %-6s %-3s %-9s relZ=%-5d final=%-11s (%s) G=%+.4f t=%.1f"
              % (len(runs), len(plan), state, ALL[state]["cell"]["breadth"],
                 ALL[state]["cell"]["duration"], method, meta["n_rel_Z"],
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)), wall),
              flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHODS), "seeds": list(SEEDS),
                            "n_runs": len(runs), "shuffle_seed": SHUFFLE_SEED,
                            "scope": "8 nominal instances x 4 disruption cells = 32 states",
                            "design": "breadth {single machine 0, all machines} x duration "
                                      "{1 period, 2 periods}, all outages starting at period 1",
                            "note": "mechanism exploration over the state grid, NOT a new "
                                    "independent confirmation; the two existing cells are "
                                    "re-run inside this batch"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "rho", "inst_seed", "disruption", "method", "seed",
              "repair_cost", "lp_objective", "mip_status", "mip_objective", "mip_dual_bound",
              "n_rel_Y", "n_rel_Z", "rel_size", "final_cost", "final_source",
              "strictly_below_lp", "total_s", "overrun_s"]
    with open("stage15_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
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
