#!/usr/bin/env python
"""
Stage 10 phase 0: generate and FREEZE the new large instances, nominal plans and repairs.

    .venv-hs/Scripts/python.exe stage10_generate.py [--resume]

Frozen protocol (identical to stage 06, scale restricted to large)
------------------------------------------------------------------
scale          large only                    (stage 06 used medium + large)
rho            0.75, 1.10
NEW seeds      6, 7, 8, 9, 10, 11, 12, 13   -> 2 x 8 = 16 nominal instances
disruptions    the original D1 and D2        -> 32 states
nominal solve  120 s per instance, the SAME protocol and the same all-lost witness start
repair         repair_minbatch_v3, unchanged
model          audited_v1, kappa = 4, tau = 6, unchanged

The 120 s nominal budget is a fixed GENERATION protocol, not a quality guarantee.  Instances are
NOT redrawn because a nominal gap is large, a repair cost is high, or an algorithm performs
badly: the whole point of this round is an independent confirmation, so selection on outcome
would destroy it.

Instances, nominal plans and repairs are written and fsynced BEFORE any method comparison runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_highs import repair_vector, solve_highs
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_repair3 import REPAIR_NAME, repair_minbatch_v3
from stage06_nominal import all_lost_plan

NEW_SEEDS = (6, 7, 8, 9, 10, 11, 12, 13)
SCALE = "large"
RHOS = (0.75, 1.10)
NOMINAL_BUDGET = 120.0
NOMINAL_FILE = "stage10_nominal.json"
STATES_FILE = "stage10_states.json"


def load(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    nom = load(NOMINAL_FILE) if args.resume else {}
    states = load(STATES_FILE) if args.resume else {}

    todo = [(SCALE, rho, sd) for rho in RHOS for sd in NEW_SEEDS]
    print("nominal instances planned: %d (scale=%s, seeds %s)"
          % (len(todo), SCALE, list(NEW_SEEDS)))
    assert len(todo) == 16, "expected 16 nominal instances"

    # ---------------- phase 1: nominal plans (120 s, fixed protocol)
    for scale, rho, sd in todo:
        name = "%s_rho%.2f_s%d" % (scale, rho, sd)
        if name in nom:
            continue
        inst, meta = build_instance(scale, rho, sd)
        bm = build_model(inst, MODE_AUDITED)
        start = all_lost_plan(inst)
        chk_start = check_solution(inst, start, MODE_AUDITED)
        if not chk_start.ok:
            raise SystemExit("all-lost witness is NOT feasible for %s: %s"
                             % (name, chk_start.failed_checks()))
        x0 = repair_vector(bm, start)
        t0 = time.perf_counter()
        run = solve_highs(bm, {}, ({}, 0.0), NOMINAL_BUDGET, seed=0,
                          start_values=x0, threads=1)
        wall = time.perf_counter() - t0
        chk = check_solution(inst, run.solution, MODE_AUDITED) if run.solution else None
        if chk is None or not chk.ok:
            raise SystemExit("nominal solve for %s produced no/invalid plan" % name)
        nom[name] = {
            "meta": {k: meta[k] for k in ("name", "scale", "rho", "seed", "N", "M", "T",
                                          "tau")},
            "nominal_budget_s": NOMINAL_BUDGET,
            "all_lost_start_feasible": bool(chk_start.ok),
            "start_adopted": run.start_adopted,
            "start_reported_objective": run.start_reported_objective,
            "status": run.status, "objective": run.objective,
            "mip_gap": run.mip_gap, "dual_bound": run.dual_bound, "wall_s": wall,
            "feasible": True, "max_violation": chk.max_violation,
            "cost_recomputed": chk.cost_recomputed,
            "solution": {k: np.asarray(getattr(run.solution, k)).tolist()
                         for k in ("X", "Y", "Z", "I", "L")},
        }
        save(NOMINAL_FILE, nom)
        print("  %-22s obj=%-13s status=%-20s gap=%-8s t=%.1fs"
              % (name, "%.6g" % run.objective, run.status,
                 ("%.3g" % run.mip_gap) if run.mip_gap is not None else "-", wall),
              flush=True)

    # ---------------- phase 2: disruptions + frozen repair
    print("\nstates planned: %d" % (len(todo) * 2))
    for scale, rho, sd in todo:
        inst, meta = build_instance(scale, rho, sd)
        nrec = nom[meta["name"]]
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
            if not chk.ok:
                raise SystemExit("repair for %s is INFEASIBLE: %s" % (state, chk.failed_checks()))
            states[state] = {
                "meta": {k: meta[k] for k in ("name", "scale", "rho", "seed", "N", "M", "T",
                                              "tau")},
                "disruption": dis.as_dict(),
                "nominal_objective": nrec["objective"],
                "nominal_status": nrec["status"],
                "nominal_gap": nrec["mip_gap"],
                "repair": {"name": REPAIR_NAME, "objective": float(rep.solution.objective),
                           "feasible": True, "max_violation": chk.max_violation,
                           "wall_s": wall, "log_text": rep.log_text()[:2000],
                           "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                                    for k in ("X", "Y", "Z", "I", "L")}},
            }
            save(STATES_FILE, states)
            print("  %-32s nominal=%-12s repair=%-12s feas=%s"
                  % (state, "%.6g" % nrec["objective"],
                     "%.6g" % rep.solution.objective, chk.ok), flush=True)

    print("\nnominal instances: %d | states: %d" % (len(nom), len(states)))
    gaps = [v["mip_gap"] for v in nom.values() if v["mip_gap"] is not None]
    if gaps:
        print("nominal gaps: min=%.3f max=%.3f mean=%.3f (recorded, NOT used to redraw)"
              % (min(gaps), max(gaps), sum(gaps) / len(gaps)))
    print("instances and repairs are FROZEN; no instance was redrawn on outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())
