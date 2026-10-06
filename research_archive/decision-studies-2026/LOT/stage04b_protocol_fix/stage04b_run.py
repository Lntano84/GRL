#!/usr/bin/env python
"""
Stage 04b: protocol fix and an 8-state retest.

    python stage04b_run.py            # run the 8 states x 4 methods at 20 s
    python stage04b_offline.py        # provisional re-aggregation of the stage-04 scalars

What is fixed relative to stage 04
---------------------------------
1. OUTPUT SELECTION. Every run now stores
       raw_solver_solution   the plan the solver actually returned (may be absent)
       selected_solution     the plan the method delivers
       selected_source       "solver" | "repair"
   with the rule   J_out = min(J_repair, J_valid_incumbent),
   both candidates independently validated (model, stability budget, fixings, recomputed
   cost). Stage 04 only used the repaired plan when there was NO incumbent.

2. TIMING. The clock starts BEFORE feature computation. Features, candidate ranking and
   the fixing-row construction come out of the budget, the remainder is handed to the solver.
   Solution verification and output selection are timed separately and counted into the
   method total; writing results to disk is recorded but excluded. Actual overrun is recorded.

3. FREE-VARIABLE COUNTING. Slot totals, variables left un-pinned by bounds, and variables
   pinned by the model's own structure are reported separately. The stage-04 statement
   "1296 free Z" counted slots and ignored that Z_ijT is structurally fixed.

Frozen inputs (NOT rebuilt)
---------------------------
nominal plans, disruptions, repaired plans, shortage scoring, release-set rules and the
stability budget all come from the stage-04 artefacts. The repairer is versioned separately
(`lsp_repair3`) but this retest deliberately keeps using the stage-04 repaired plans, so the
comparison baseline does not change at the same time.

Outputs: stage04b_results.csv, stage04b_runs.json, stage04b_details.md
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.sparse import coo_matrix, vstack as sp_vstack

from lsp_checker import check_solution
from lsp_gen import all_instances, apply_disruption, disruptions_for
from lsp_model import MODE_AUDITED, Bounds, LinearConstraint, Solution, build_model, milp, unpack
from lsp_repair3 import REPAIR_NAME, binary_change_count
from lsp_select import evaluate_candidate, select_output

BUDGET = 20.0
KAPPA = 4.0
METHODS = ("FULL", "EMPTY", "SHORTAGE-12", "SHORTAGE-24")
SHORTAGE_SIZES = {"SHORTAGE-12": 12, "SHORTAGE-24": 24}
SHUFFLE_SEED = 20260926
SOURCE_DIR = "../stage04_runtime_calibration"
STAGE04_STATES = os.path.join(SOURCE_DIR, "stage04_states.json")
STAGE04_NOMINAL = os.path.join(SOURCE_DIR, "stage04_nominal.json")

# medium and large, both rho, seed 0, both disruptions -> 8 states
SELECTED = [(scale, rho) for scale in ("medium", "large") for rho in (0.75, 1.10)]
SEED = 0


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
    return Solution(status=rec.get("status", "stored"), objective=float(rec["objective"]),
                    X=np.array(p["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


# --------------------------------------------------------------------------------------
# Constraint system with an honest free-variable count
# --------------------------------------------------------------------------------------


class System:
    def __init__(self, bm, repair: Solution, tau: int):
        self.bm = bm
        self.idx = bm.idx
        self.repair = repair
        self.tau = tau
        self.A = bm.A.tocsr()
        self.row_lb = bm.row_lb.copy()
        self.row_ub = bm.row_ub.copy()
        self.lb = bm.var_lb.copy()
        self.ub = bm.var_ub.copy()
        self.stab = self._stability_row()
        self.all_short = [(i, j, t) for i in range(bm.inst.N) for j in range(bm.inst.M)
                          for t in range(1, tau + 1)]

    def _stability_row(self):
        coef: Dict[int, float] = {}
        n_one = 0
        for i in range(self.bm.inst.N):
            for j in range(self.bm.inst.M):
                for t in range(1, self.tau + 1):
                    yref = float(self.repair.Y[i, j, t - 1])
                    if yref > 0.5:
                        n_one += 1
                        coef[self.idx.Y(i, j, t)] = -1.0
                    else:
                        coef[self.idx.Y(i, j, t)] = 1.0
        return coef, float(KAPPA) - float(n_one)

    # ---- counting -------------------------------------------------------------------
    def count_free(self, fixed: Dict[int, float]) -> Dict[str, Any]:
        """Separate three different things that stage 04 conflated:

        slots            every (i,j,t) binary position in the model
        struct_pinned    pinned by the MODEL's own definition: Z_ij0 from (1.13) and Z_ijT
                         under D2 (no carry-over out of the last period)
        method_pinned    pinned by this method's fixing conditions (t <= tau only)
        free_slots       slots - struct_pinned - method_pinned  == genuinely free
        bound_free       positions whose variable bounds alone leave them free (this is the
                         raw `lb < ub` count, kept for reference)
        """
        inst = self.bm.inst
        nY = nZ = 0
        free_Y = free_Z = 0
        struct_Z = 0
        method_Y = 0
        for i in range(inst.N):
            for j in range(inst.M):
                nY += inst.T
                nZ += inst.T
                for t in range(1, inst.T + 1):
                    cY = self.idx.Y(i, j, t)
                    cZ = self.idx.Z(i, j, t)
                    if self.lb[cY] < self.ub[cY] - 1e-12:
                        if cY not in fixed:
                            free_Y += 1
                        else:
                            method_Y += 1
                    if self.lb[cZ] < self.ub[cZ] - 1e-12:
                        free_Z += 1
                    else:
                        struct_Z += 1
        return {
            "Y_slots": nY, "Z_slots": nZ,
            "Z_structurally_pinned": struct_Z,
            "Y_method_pinned": method_Y,
            "free_Y_slots": free_Y,
            "free_Z_slots": free_Z,
            "Z_bound_free": sum(1 for i in range(inst.N) for j in range(inst.M)
                                for t in range(1, inst.T + 1)
                                if self.lb[self.idx.Z(i, j, t)]
                                < self.ub[self.idx.Z(i, j, t)] - 1e-12),
        }

    # ---- solve ----------------------------------------------------------------------
    def build_and_solve(self, fixed: Dict[int, float], deadline: float):
        """Return (solver_result, timings). Feature/rank work is done by the CALLER before
        this is entered; everything here is charged to the method as well."""
        t0 = time.perf_counter()
        n = self.idx.n
        lb = self.lb.copy()
        ub = self.ub.copy()
        blocks = [self.A]
        lob = [self.row_lb]
        upb = [self.row_ub]
        coef, rhs = self.stab
        cols = list(coef.keys())
        blocks.append(coo_matrix(([coef[c] for c in cols],
                                  ([0] * len(cols), cols)), shape=(1, n)).tocsr())
        lob.append(np.array([-np.inf]))
        upb.append(np.array([rhs]))
        if fixed:
            k = list(fixed.keys())
            blocks.append(coo_matrix((np.ones(len(k)), (np.arange(len(k)), np.array(k))),
                                     shape=(len(k), n)).tocsr())
            v = np.array([fixed[c] for c in k], dtype=float)
            lob.append(v)
            upb.append(v)
            for c in k:
                lb[c] = fixed[c]
                ub[c] = fixed[c]
        A = sp_vstack(blocks, format="csr") if len(blocks) > 1 else self.A
        rl = np.concatenate(lob)
        ru = np.concatenate(upb)
        t_build = time.perf_counter() - t0

        remaining = max(0.1, deadline - time.perf_counter())
        t1 = time.perf_counter()
        res = milp(c=self.bm.c, constraints=LinearConstraint(A, lb=rl, ub=ru),
                   integrality=self.bm.integrality, bounds=Bounds(lb=lb, ub=ub),
                   options={"time_limit": float(remaining)})
        t_solve = time.perf_counter() - t1
        return res, {"constraint_build_s": t_build, "solver_wall_s": t_solve,
                     "solver_budget_s": remaining}


# --------------------------------------------------------------------------------------
# Methods (identical scoring and release rules to stage 03/04)
# --------------------------------------------------------------------------------------


def possible(inst, u) -> bool:
    i, j, t = u
    if inst.w[i][j] == 0:
        return False
    return inst.s[i] <= inst.c[j][t - 1] + 1e-9


def shortage_scores(inst, repair: Solution, tau: int) -> Dict[Tuple[int, int, int], float]:
    out = {}
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                shortage = sum(float(repair.L[i, r - 1]) for r in range(t, inst.T + 1))
                out[(i, j, t)] = (inst.l[i] * shortage) / (inst.s[i] + inst.b[i] * inst.m[i])
    return out


def method_fixing(sysx: System, method: str):
    """Returns (fixed, info, fixed_Y_map). `fixed_Y_map` is the (i,j,t) -> value view used
    to check the method's own fixing conditions."""
    inst = sysx.bm.inst
    tau = sysx.tau
    feas = [u for u in sysx.all_short if possible(inst, u)]
    info: Dict[str, Any] = {"n_candidates_feasible": len(feas), "note": ""}
    if method == "FULL":
        return {}, {"release_set": "all short-term Y", "n_released": len(feas),
                    "n_candidates_feasible": len(feas), "note": ""}, {}
    if method == "EMPTY":
        release: List[Tuple[int, int, int]] = []
        note = ""
    else:
        k = SHORTAGE_SIZES[method]
        scores = shortage_scores(inst, sysx.repair, tau)
        order = sorted(feas, key=lambda u: (-scores[u], u))
        release = order[:k]
        note = ("" if len(order) >= k else
                "only %d legal candidates (< %d); all released" % (len(order), k))
    rel = set(release)
    fixed: Dict[int, float] = {}
    fixed_Y: Dict[Tuple[int, int, int], float] = {}
    for (i, j, t) in sysx.all_short:
        if (i, j, t) not in rel:
            val = float(sysx.repair.Y[i, j, t - 1])
            fixed[sysx.idx.Y(i, j, t)] = val
            fixed_Y[(i, j, t)] = val
    info = {"release_set": "%s (%d)" % (method, len(rel)), "n_released": len(rel),
            "n_candidates_feasible": len(feas), "note": note}
    return fixed, info, fixed_Y


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


CSV_FIELDS = ["state", "scale", "rho", "disruption", "method",
              "repair_cost", "raw_incumbent", "selected_cost", "selected_source",
              "select_reason", "raw_status", "raw_proven_optimal", "raw_gap",
              "total_time_s", "solver_wall_s", "constraint_build_s", "feature_rank_s",
              "verify_select_s", "overrun_s", "free_Y_slots", "free_Z_slots",
              "Z_structurally_pinned", "raw_valid", "selected_valid", "errors"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    with open(STAGE04_NOMINAL, encoding="utf-8") as fh:
        NOM = json.load(fh)
    with open(STAGE04_STATES, encoding="utf-8") as fh:
        ST = json.load(fh)

    insts = {m["name"]: (i, m) for i, m in all_instances()}
    todo: List[Tuple[str, Dict[str, Any], Any, Any]] = []
    for name, (inst0, meta) in insts.items():
        if meta["scale"] not in ("medium", "large") or meta["seed"] != SEED \
                or meta["rho"] not in (0.75, 1.10):
            continue
        for dis in disruptions_for(inst0):
            state = "%s|%s" % (name, dis.name)
            if state in ST:
                todo.append((state, meta, inst0, dis))
    todo.sort()

    runs: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for state, meta, inst0, dis in todo:
        rec = ST[state]
        pert = apply_disruption(inst0, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        sysx = System(bm, repair, meta["tau"])

        order = list(METHODS)
        random.Random(SHUFFLE_SEED).shuffle(order)
        for method in order:
            t_method0 = time.perf_counter()
            deadline = t_method0 + BUDGET

            # ---- feature and ranking work, INSIDE the budget
            t_feat0 = time.perf_counter()
            fixed, info, fixed_Y = method_fixing(sysx, method)
            t_feat = time.perf_counter() - t_feat0

            # ---- constraint construction + solve, remainder of the budget
            res, tinfo = sysx.build_and_solve(fixed, deadline)

            # ---- verification and output selection, also counted
            t_ver0 = time.perf_counter()
            raw_sol = None
            if res.x is not None:
                X, Y, Z, I, L = unpack(res.x, bm.idx)
                raw_sol = Solution(
                    status=("optimal" if res.status == 0 else "suboptimal"),
                    objective=float(res.fun), X=X, Y=Y, Z=Z, I=I, L=L,
                    mip_gap=float(res.mip_gap) if res.mip_gap is not None else None,
                    mode=MODE_AUDITED)
            raw_status = ("no_incumbent" if res.x is None
                          else ("optimal" if res.status == 0 else "suboptimal"))
            if res.status == 2:
                raw_status = "infeasible"
            sel = select_output(pert, raw_sol, repair, KAPPA, meta["tau"],
                                fixed_Y=fixed_Y, solver_status=raw_status)
            t_ver = time.perf_counter() - t_ver0

            total = time.perf_counter() - t_method0
            overrun = max(0.0, total - BUDGET)
            free = sysx.count_free(fixed)
            raw_cand = [c for c in sel.candidates if c.label == "solver"][0]

            runs.append({
                "state": state, "scale": meta["scale"], "rho": meta["rho"],
                "seed": meta["seed"], "method": method, "budget_s": BUDGET,
                "repair_cost": float(repair.objective),
                "raw_incumbent_cost": raw_cand.cost,
                "raw_incumbent_reported": raw_cand.reported_cost,
                "raw_incumbent_valid": bool(raw_cand.feasible),
                "raw_incumbent_problems": raw_cand.problems,
                "raw_status": raw_status,
                "raw_proven_optimal": bool(res.status == 0),
                "raw_mip_gap": (float(res.mip_gap) if res.mip_gap is not None else None),
                "raw_mip_dual_bound": _f(getattr(res, "mip_dual_bound", None)),
                "selected_cost": None if sel.chosen is None else float(sel.chosen.cost),
                "selected_source": sel.source,
                "select_reason": sel.reason,
                "select_errors": sel.errors,
                "selected_solution": (None if sel.chosen is None or sel.chosen.solution is None
                                      else {k: np.asarray(getattr(sel.chosen.solution, k)).tolist()
                                            for k in ("X", "Y", "Z", "I", "L")}),
                "raw_solver_solution": (None if raw_sol is None else
                                        {k: np.asarray(getattr(raw_sol, k)).tolist()
                                         for k in ("X", "Y", "Z", "I", "L")}),
                "selected_max_violation": (None if sel.chosen is None
                                           else sel.chosen.max_violation),
                "selected_stability_flips": (None if sel.chosen is None
                                             else sel.chosen.stability_flips),
                "timing": {"feature_rank_s": t_feat, "verify_select_s": t_ver,
                           "total_method_s": total, "overrun_s": overrun, **tinfo},
                "free": free,
                "fixing_info": info,
            })
            print("  %-28s %-12s raw=%-11s sel=%-11s %-7s %-22s t=%5.2f ovr=%.2f"
                  % (state, method, _g(raw_cand.cost), _g(runs[-1]["selected_cost"]),
                     sel.source, sel.reason, total, overrun), flush=True)

    # ---- write outputs
    with open(f"{args.outdir}/stage04b_runs.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "kappa": KAPPA,
                            "shuffle_seed": SHUFFLE_SEED, "methods": list(METHODS),
                            "repair_used_for_baseline": "stage04 repaired plans (frozen)",
                            "repair_module": REPAIR_NAME,
                            "scipy": scipy.__version__,
                            "python": platform.python_version(),
                            "states": [t[0] for t in todo]},
                   "runs": runs}, fh, indent=2, default=str)

    rows = []
    for r in runs:
        rows.append({
            "state": r["state"], "scale": r["scale"], "rho": r["rho"],
            "disruption": r["state"].split("|")[1], "method": r["method"],
            "repair_cost": r["repair_cost"],
            "raw_incumbent": r["raw_incumbent_cost"],
            "selected_cost": r["selected_cost"],
            "selected_source": r["selected_source"],
            "select_reason": r["select_reason"],
            "raw_status": r["raw_status"],
            "raw_proven_optimal": r["raw_proven_optimal"],
            "raw_gap": r["raw_mip_gap"],
            "total_time_s": r["timing"]["total_method_s"],
            "solver_wall_s": r["timing"]["solver_wall_s"],
            "constraint_build_s": r["timing"]["constraint_build_s"],
            "feature_rank_s": r["timing"]["feature_rank_s"],
            "verify_select_s": r["timing"]["verify_select_s"],
            "overrun_s": r["timing"]["overrun_s"],
            "free_Y_slots": r["free"]["free_Y_slots"],
            "free_Z_slots": r["free"]["free_Z_slots"],
            "Z_structurally_pinned": r["free"]["Z_structurally_pinned"],
            "raw_valid": r["raw_incumbent_valid"],
            "selected_valid": (r["selected_max_violation"] is not None
                               and r["selected_max_violation"] <= 1e-6),
            "errors": len(r["select_errors"]),
        })
    with open(f"{args.outdir}/stage04b_results.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    with open(f"{args.outdir}/stage04b_details.md", "w", encoding="utf-8") as fh:
        fh.write(render(runs, todo))

    print("runs: %d  wall %.1fs" % (len(runs), time.perf_counter() - t_start))
    n_worse = sum(1 for r in runs if r["selected_cost"] is not None
                  and r["selected_cost"] > r["repair_cost"] + 1e-6)
    n_err = sum(1 for r in runs if r["select_errors"])
    print("runs whose FINAL OUTPUT is worse than the repair plan: %d" % n_worse)
    print("runs with selection errors: %d" % n_err)
    return 0 if (n_worse == 0 and n_err == 0) else 1


def render(runs, todo) -> str:
    L: List[str] = []
    L.append("# stage04b_details.md\n")
    L.append("Protocol-fix retest. Budget %.0f s per method, kappa=%g, method order shuffled "
             "with seed %d.\n" % (BUDGET, KAPPA, SHUFFLE_SEED))
    L.append("Repaired reference plans are the FROZEN stage-04 ones, so the comparison "
             "baseline is unchanged.\n")
    L.append("\n## Per run\n")
    L.append("| state | method | repair | raw incumbent | raw valid | selected | source | reason | total s | overrun s |")
    L.append("|---|---|---:|---:|---|---:|---|---|---:|---:|")
    for r in runs:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %.2f | %.2f |" % (
            r["state"], r["method"], _g(r["repair_cost"]), _g(r["raw_incumbent_cost"]),
            r["raw_incumbent_valid"], _g(r["selected_cost"]), r["selected_source"],
            r["select_reason"], r["timing"]["total_method_s"], r["timing"]["overrun_s"]))
    L.append("\n## Free-variable accounting (three separate numbers)\n")
    L.append("| state | method | Y slots | Z slots | Z struct-pinned | free Y | free Z |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in runs:
        f = r["free"]
        L.append("| %s | %s | %d | %d | %d | %d | %d |" % (
            r["state"], r["method"], f["Y_slots"], f["Z_slots"],
            f["Z_structurally_pinned"], f["free_Y_slots"], f["free_Z_slots"]))
    L.append("\n`Z struct-pinned` counts Z_ij0 from (1.13) and Z_ijT under D2. Stage 04 "
             "reported `Z slots` as if all of them were free.\n")
    L.append("\n## Selection errors\n")
    any_err = False
    for r in runs:
        if r["select_errors"]:
            any_err = True
            L.append("- **%s / %s**: %s" % (r["state"], r["method"], r["select_errors"]))
    if not any_err:
        L.append("None.\n")
    L.append("\n## Raw-incumbent problems (invalid incumbents are errors, not timeouts)\n")
    any_p = False
    for r in runs:
        if r["raw_incumbent_problems"]:
            any_p = True
            L.append("- **%s / %s**: %s" % (r["state"], r["method"],
                                            r["raw_incumbent_problems"]))
    if not any_p:
        L.append("None.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
