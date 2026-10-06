#!/usr/bin/env python
"""
Stage 06 phase 1: generate and freeze the 16 new nominal instances, then the 32 states.

    .venv-hs/Scripts/python.exe stage06_nominal.py [--resume]

Frozen configuration
--------------------
scales        medium, large
rho           0.75, 1.10
new seeds     2, 3, 4, 5     -> 2 x 2 x 4 = 16 nominal instances
disruptions   the original D1 (machine 0 down periods 1-2) and D2 (all machines down
              period 1) -> 32 states
nominal solve 120 s per instance, ONE fixed protocol
repair        repair_minbatch_v3, frozen after generation

The 120 s nominal budget is a fixed starting-point generation protocol, NOT a guarantee of a
high-quality nominal plan. Conclusions in this round remain conditional on these starting
points; nominal-solution quality is deliberately NOT studied here.

The nominal solve is seeded with a submitted all-lost feasible plan (X = Y = Z = 0,
L = demand, I = 0). That plan is verified by the independent checker before use. Submitting it
is only a starting point, not a quality claim.

The old seeds 0 and 1 are used ONLY to check that the new code runs; they are excluded from
the new sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import SCALES, apply_disruption, build_instance, disruptions_for
from lsp_highs import repair_vector, solve_highs
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_repair3 import REPAIR_NAME, repair_minbatch_v3
from lsp_select import select_output

NEW_SEEDS = (2, 3, 4, 5)
NOMINAL_BUDGET = 120.0
KAPPA = 4.0
NOMINAL_FILE = "stage06_nominal.json"
STATES_FILE = "stage06_states.json"


def all_lost_plan(inst) -> Solution:
    """X = Y = Z = 0, I = 0, L = demand. Checked to be feasible before it is ever submitted."""
    N, M, T = inst.N, inst.M, inst.T
    L = np.zeros((N, T))
    for i in range(N):
        cur = float(inst.I0[i])
        for t in range(1, T + 1):
            served = min(cur, float(inst.d[i][t - 1]))
            L[i, t - 1] = float(inst.d[i][t - 1]) - served
            cur -= served
    sol = Solution(status="all_lost_witness", objective=float("nan"),
                   X=np.zeros((N, M, T)), Y=np.zeros((N, M, T)), Z=np.zeros((N, M, T)),
                   I=np.zeros((N, T)), L=L, mode=MODE_AUDITED)
    return sol


def load(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    nom = load(NOMINAL_FILE) if args.resume else {}
    states = load(STATES_FILE) if args.resume else {}

    # ---- phase 1a: nominal solves
    todo = [(s, rho, sd) for s in ("medium", "large") for rho in (0.75, 1.10)
            for sd in NEW_SEEDS]
    print("nominal instances planned: %d (seeds %s)" % (len(todo), list(NEW_SEEDS)))
    for scale, rho, sd in todo:
        if ("%s_rho%.2f_s%d" % (scale, rho, sd)) in nom:
            continue
        inst, meta = build_instance(scale, rho, sd)
        bm = build_model(inst, MODE_AUDITED)
        start = all_lost_plan(inst)
        chk_start = check_solution(inst, start, MODE_AUDITED)
        if not chk_start.ok:
            raise SystemExit("all-lost witness is NOT feasible for %s: %s"
                             % (meta["name"], chk_start.failed_checks()))
        x0 = repair_vector(bm, start)
        t0 = time.perf_counter()
        run = solve_highs(bm, {}, ({}, 0.0), NOMINAL_BUDGET, seed=0,
                          start_values=x0, threads=1)
        wall = time.perf_counter() - t0
        chk = check_solution(inst, run.solution, MODE_AUDITED) if run.solution else None
        rec = {
            "meta": {k: meta[k] for k in ("name", "scale", "rho", "seed", "N", "M", "T",
                                          "tau")},
            "nominal_budget_s": NOMINAL_BUDGET,
            "all_lost_start_feasible": bool(chk_start.ok),
            "start_adopted": run.start_adopted,
            "start_reported_objective": run.start_reported_objective,
            "start_message": run.start_message,
            "status": run.status,
            "objective": run.objective,
            "mip_gap": run.mip_gap,
            "dual_bound": run.dual_bound,
            "wall_s": wall,
            "feasible": bool(chk.ok) if chk else False,
            "max_violation": chk.max_violation if chk else None,
            "cost_recomputed": chk.cost_recomputed if chk else None,
            "solution": (None if run.solution is None else
                         {k: np.asarray(getattr(run.solution, k)).tolist()
                          for k in ("X", "Y", "Z", "I", "L")}),
        }
        nom[meta["name"]] = rec
        save(NOMINAL_FILE, nom)
        print("  %-22s obj=%-13s status=%-10s gap=%-10s feas=%-5s adopted=%-5s t=%.1fs"
              % (meta["name"], rec["objective"], rec["status"],
                 ("%.3g" % rec["mip_gap"]) if rec["mip_gap"] is not None else "-",
                 rec["feasible"], rec["start_adopted"], wall), flush=True)

    # ---- phase 1b: states (disruptions + v3 repair)
    print("\nstates planned: %d" % (len(todo) * 2))
    for scale, rho, sd in todo:
        inst, meta = build_instance(scale, rho, sd)
        nrec = nom[meta["name"]]
        if nrec["solution"] is None:
            raise SystemExit("nominal solve produced no plan for %s" % meta["name"])
        nom_sol = Solution(status=nrec["status"], objective=float(nrec["objective"]),
                           X=np.array(nrec["solution"]["X"], float).reshape(inst.N, inst.M, inst.T),
                           Y=np.array(nrec["solution"]["Y"], float).reshape(inst.N, inst.M, inst.T),
                           Z=np.array(nrec["solution"]["Z"], float).reshape(inst.N, inst.M, inst.T),
                           I=np.array(nrec["solution"]["I"], float).reshape(inst.N, inst.T),
                           L=np.array(nrec["solution"]["L"], float).reshape(inst.N, inst.T),
                           mode=MODE_AUDITED)
        for dis in disruptions_for(inst):
            state = "%s|%s" % (meta["name"], dis.name)
            if state in states:
                continue
            pert = apply_disruption(inst, dis)
            t0 = time.perf_counter()
            rep = repair_minbatch_v3(pert, nom_sol, dis)
            wall = time.perf_counter() - t0
            chk = check_solution(pert, rep.solution, MODE_AUDITED)
            states[state] = {
                "meta": {k: meta[k] for k in ("name", "scale", "rho", "seed", "N", "M", "T",
                                              "tau")},
                "disruption": dis.as_dict(),
                "nominal_objective": nrec["objective"],
                "nominal_status": nrec["status"],
                "nominal_gap": nrec["mip_gap"],
                "repair": {
                    "name": REPAIR_NAME,
                    "objective": float(rep.solution.objective),
                    "feasible": bool(chk.ok),
                    "max_violation": chk.max_violation,
                    "wall_s": wall,
                    "log_text": rep.log_text()[:2000],
                    "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                             for k in ("X", "Y", "Z", "I", "L")},
                },
            }
            save(STATES_FILE, states)
            print("  %-32s nominal=%-12s repair=%-12s feas=%-5s"
                  % (state, "%.6g" % nrec["objective"],
                     "%.6g" % rep.solution.objective, chk.ok), flush=True)

    print("\nnominal instances: %d | states: %d" % (len(nom), len(states)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
