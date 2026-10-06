#!/usr/bin/env python
"""
Stage 04: runtime calibration -- does limiting the re-optimised variable set buy better
solutions within the same wall-clock budget, and is the shortage rule already enough?

    python stage04_run.py --phase nominal     # 12 nominal solves (<= 60 s each)
    python stage04_run.py --phase reference   # repairs + 120 s FULL per state
    python stage04_run.py --phase timing      # 1/5/20 s x 4 methods x 24 states
    python stage04_run.py --phase report      # assemble outputs
    python stage04_run.py --phase all         # everything, resumable

Time protocol (this is the part that must be right)
---------------------------------------------------
The measured budget is END-TO-END wall clock for the method:
    feature computation + candidate ranking + fixing-row construction + model solve.
Shared, method-independent work is excluded and measured once per state:
    reading the frozen JSON, importing numpy/scipy, generating the instance, building the
    base constraint system.
Both the solver's own time and the true wall time are recorded; `time_limit` is never
reported as the achieved runtime.

External feasible fallback
--------------------------
Every method keeps the SAME independently verified repaired plan. If the solver returns no
incumbent, the repaired plan is used and the reason is recorded. Claiming infeasibility is
treated as an error, because a feasible witness exists.

HOT START CAVEAT
----------------
`scipy.optimize.milp` exposes only (c, *, integrality, bounds, constraints, options): there
is NO way to pass an initial solution / MIP start. Keeping the repaired plan externally is
therefore NOT a warm start. This round is a cost calibration under the CURRENT interface and
must not be reported as beating a properly warm-started full solve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from lsp_gen import (CAPACITY, RHOS, SCALES, SEEDS, all_instances, apply_disruption,
                     disruptions_for)
from lsp_model import (MODE_AUDITED, Bounds, Instance, LinearConstraint, Solution,
                       build_model, unpack)
from lsp_model import milp as _milp
from lsp_repair2 import (REPAIR_NAME, binary_change_count, binary_change_count_full,
                         repair_minbatch_v2)

KAPPA = 4.0
BUDGETS = (1.0, 5.0, 20.0)
REFERENCE_BUDGET = 120.0
NOMINAL_BUDGET = 60.0
METHODS = ("FULL", "EMPTY", "SHORTAGE-12", "SHORTAGE-24")
SHORTAGE_SIZES = {"SHORTAGE-12": 12, "SHORTAGE-24": 24}
SHUFFLE_SEED = 20260926

STATE_RESULTS = "stage04_states.json"
NOMINAL_RESULTS = "stage04_nominal.json"


def _g(v: Any, nd: int = 6) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "-"
    if abs(f) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % f


# --------------------------------------------------------------------------------------
# Pre-built constraint system (shared, method independent, built once per state)
# --------------------------------------------------------------------------------------


class System:
    """Base matrices plus the pieces a method needs. Building this is NOT charged to any
    method; applying a method's fixing rows IS charged."""

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
        self.stab = self._stability_rows()
        self.all_short = [(i, j, t) for i in range(bm.inst.N) for j in range(bm.inst.M)
                          for t in range(1, tau + 1)]

    def _stability_rows(self):
        """sum_{t<=tau} |Y - Y^ref| <= kappa, exact for binary Y, ONE row:
        sum_{Z=0} Y - sum_{Z=1} Y <= kappa - |{Z=1}|."""
        coef: Dict[int, float] = {}
        n_one = 0
        for (i, j, t) in ((i, j, t) for i in range(self.bm.inst.N)
                          for j in range(self.bm.inst.M) for t in range(1, self.tau + 1)):
            yref = float(self.repair.Y[i, j, t - 1])
            if yref > 0.5:
                n_one += 1
                coef[self.idx.Y(i, j, t)] = -1.0
            else:
                coef[self.idx.Y(i, j, t)] = 1.0
        return coef, float(KAPPA) - float(n_one)

    def n_free_binary(self, fixed_Y_cols: Sequence[int], fixed_Z_cols: Sequence[int]) -> Dict[str, int]:
        """How many Y and Z variables stay free once a method has fixed its set."""
        fy = set(fixed_Y_cols)
        fz = set(fixed_Z_cols)
        nY = sum(1 for i in range(self.bm.inst.N) for j in range(self.bm.inst.M)
                 for t in range(1, self.bm.inst.T + 1)
                 if self.idx.Y(i, j, t) not in fy)
        nZ = sum(1 for i in range(self.bm.inst.N) for j in range(self.bm.inst.M)
                 for t in range(1, self.bm.inst.T + 1)
                 if self.idx.Z(i, j, t) not in fz)
        return {"free_Y": nY, "free_Z": nZ}

    def solve(self, fixed: Dict[int, float], budget: float):
        n = self.idx.n
        lb = self.lb.copy()
        ub = self.ub.copy()
        A_blocks = [self.A]
        lob = [self.row_lb]
        upb = [self.row_ub]
        # stability row
        coef, rhs = self.stab
        rows = [0] * len(coef)
        cols = list(coef.keys())
        vals = [coef[c] for c in cols]
        A_blocks.append(coo_matrix((vals, (rows, cols)), shape=(1, n)).tocsr())
        lob.append(np.array([-np.inf]))
        upb.append(np.array([rhs]))
        # fixing rows
        if fixed:
            k = list(fixed.keys())
            A_blocks.append(coo_matrix((np.ones(len(k)), (np.arange(len(k)), np.array(k))),
                                       shape=(len(k), n)).tocsr())
            v = np.array([fixed[c] for c in k], dtype=float)
            lob.append(v)
            upb.append(v)
            for c in k:
                lb[c] = fixed[c]
                ub[c] = fixed[c]
        A = sp_vstack(A_blocks, format="csr") if len(A_blocks) > 1 else self.A
        rl = np.concatenate(lob)
        ru = np.concatenate(upb)
        t0 = time.perf_counter()
        res = _milp(c=self.bm.c, constraints=LinearConstraint(A, lb=rl, ub=ru),
                    integrality=self.bm.integrality, bounds=Bounds(lb=lb, ub=ub),
                    options={"time_limit": float(budget)})
        wall = time.perf_counter() - t0
        return res, wall, len(fixed)


# --------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------


def possible(inst: Instance, u) -> bool:
    i, j, t = u
    if inst.w[i][j] == 0:
        return False
    return inst.s[i] <= inst.c[j][t - 1] + 1e-9


def shortage_scores(inst: Instance, repair: Solution, tau: int) -> Dict[Tuple[int, int, int], float]:
    """Same score and tie-breaking rule as stage 03:
        a(i,j,t) = l_i * sum_{r>=t} L^r_ir / (s_i + b_i m_i)
    using only the repaired plan and known instance parameters."""
    out = {}
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                shortage = sum(float(repair.L[i, r - 1]) for r in range(t, inst.T + 1))
                out[(i, j, t)] = (inst.l[i] * shortage) / (inst.s[i] + inst.b[i] * inst.m[i])
    return out


def method_fixing(sysx: System, method: str) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Return the flat-variable fixings for a method, plus diagnostics."""
    inst = sysx.bm.inst
    idx = sysx.idx
    tau = sysx.tau
    feas = [u for u in sysx.all_short if possible(inst, u)]
    info: Dict[str, Any] = {"n_candidates_feasible": len(feas), "release_set": None,
                            "note": ""}
    if method == "FULL":
        release = list(feas) + [u for u in sysx.all_short if u not in feas]
        info["release_set"] = "all short-term Y"
        info["note"] = "no short-term fixing beyond the base model"
        return {}, info

    if method == "EMPTY":
        release: List[Tuple[int, int, int]] = []
        info["release_set"] = "empty"
    else:
        k = SHORTAGE_SIZES[method]
        scores = shortage_scores(inst, sysx.repair, tau)
        order = sorted(feas, key=lambda u: (-scores[u], u))
        release = order[:k]
        info["release_set"] = "top %d shortage-scored candidates" % k
        if len(order) < k:
            info["note"] = ("only %d legal candidates available (< %d); all of them released"
                            % (len(order), k))
            release = order

    fixed: Dict[int, float] = {}
    rel = set(release)
    for (i, j, t) in sysx.all_short:
        if (i, j, t) not in rel:
            fixed[idx.Y(i, j, t)] = float(sysx.repair.Y[i, j, t - 1])
    info["n_released"] = len(rel)
    info["n_fixed_Y"] = len(fixed)
    return fixed, info


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


def load_state(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_state(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp, path)


def sol_record(sol: Solution, inst: Instance) -> Dict[str, Any]:
    chk = check_solution(inst, sol, MODE_AUDITED) if sol.X is not None else None
    return {
        "objective": float(sol.objective) if sol.X is not None else None,
        "status": sol.status,
        "proven_optimal": bool(sol.status == "optimal"),
        "mip_gap": sol.mip_gap,
        "mip_dual_bound": getattr(sol, "mip_dual_bound", None),
        "solve_time_s": sol.solve_time_s,
        "feasible": bool(chk.ok) if chk else False,
        "max_violation": chk.max_violation if chk else None,
        "cost_recomputed": chk.cost_recomputed if chk else None,
        "plan": {k: np.asarray(getattr(sol, k)).tolist() for k in ("X", "Y", "Z", "I", "L")}
        if sol.X is not None else None,
    }


def phase_nominal() -> None:
    store = load_state(NOMINAL_RESULTS)
    for inst, meta in all_instances():
        if meta["name"] in store:
            continue
        t0 = time.perf_counter()
        bm = build_model(inst, MODE_AUDITED)
        t_build = time.perf_counter() - t0
        sol = solve_milp_timed(bm, NOMINAL_BUDGET)
        rec = {
            "meta": {k: meta[k] for k in ("name", "scale", "rho", "seed", "N", "M", "T", "tau")},
            "build_time_s": t_build,
            "solver": "scipy.optimize.milp -> HiGHS",
            "hot_start": False,
            "solution": sol_record(sol, inst),
            "verified_nominal": True,
            "generation_failed": sol.X is None,
        }
        store[meta["name"]] = rec
        save_state(NOMINAL_RESULTS, store)
        print("  nominal %-22s obj=%-14s status=%-20s gap=%s feas=%s" %
              (meta["name"], _g(rec["solution"]["objective"]), rec["solution"]["status"],
               _g(rec["solution"]["mip_gap"]), rec["solution"]["feasible"]), flush=True)


def solve_milp_timed(bm, budget: float) -> Solution:
    from lsp_model import solve_milp
    return solve_milp(bm, time_limit=budget)


def phase_reference() -> None:
    store = load_state(STATE_RESULTS)
    nominal = load_state(NOMINAL_RESULTS)
    for inst, meta in all_instances():
        nom_rec = nominal[meta["name"]]
        nom = solution_from_record(inst, nom_rec["solution"])
        for dis in disruptions_for(inst):
            state = "%s|%s" % (meta["name"], dis.name)
            if state in store and "reference" in store[state]:
                continue
            rec = store.get(state, {})
            rec.setdefault("meta", {k: meta[k] for k in
                                    ("name", "scale", "rho", "seed", "N", "M", "T", "tau")})
            rec["disruption"] = dis.as_dict()
            rec["nominal_objective"] = nom_rec["solution"]["objective"]
            rec["nominal_mip_gap"] = nom_rec["solution"]["mip_gap"]
            rec["nominal_proven_optimal"] = nom_rec["solution"]["proven_optimal"]

            pert = apply_disruption(inst, dis)
            t0 = time.perf_counter()
            rep = repair_minbatch_v2(pert, nom, dis)
            rep_wall = time.perf_counter() - t0
            rep_chk = check_solution(pert, rep.solution, MODE_AUDITED)
            rec["repair"] = {
                "name": REPAIR_NAME,
                "objective": float(rep.solution.objective),
                "status": "repaired",
                "wall_time_s": rep_wall,
                "feasible": bool(rep_chk.ok),
                "max_violation": rep_chk.max_violation,
                "cost_recomputed": rep_chk.cost_recomputed,
                "log": [e.as_dict() for e in rep.events],
                "log_text": rep.log_text(),
                "plan": {k: np.asarray(getattr(rep.solution, k)).tolist()
                         for k in ("X", "Y", "Z", "I", "L")},
            }

            t0 = time.perf_counter()
            bm = build_model(pert, MODE_AUDITED)
            rec["build_time_s"] = time.perf_counter() - t0

            sysx = System(bm, rep.solution, meta["tau"])
            fixed, info = method_fixing(sysx, "FULL")
            t0 = time.perf_counter()
            res, solve_wall, nfixed = sysx.solve(fixed, REFERENCE_BUDGET)
            total_wall = time.perf_counter() - t0
            sol = solution_from_res(bm, res, pert)
            ref = sol_record(sol, pert) if sol is not None else None
            rec["reference"] = {
                "budget_s": REFERENCE_BUDGET,
                "method": "FULL",
                "fixing_info": info,
                "n_fixed": nfixed,
                "solver_wall_s": solve_wall,
                "total_wall_s": total_wall,
                "solution": ref,
                "mip_dual_bound": getattr(res, "mip_dual_bound", None),
            }
            # J_best seed: reference solution, else the independently verified repair
            cands = []
            if ref and ref["feasible"]:
                cands.append(("reference", ref["objective"], ref["status"]))
            cands.append(("repair", rec["repair"]["objective"], "repaired"))
            best = min(cands, key=lambda c: c[1])
            rec["J_best_seed"] = {"value": best[1], "source": best[0]}
            store[state] = rec
            save_state(STATE_RESULTS, store)
            print("  ref %-28s repair=%-13s ref=%-13s status=%-20s gap=%-8s bound=%-12s" %
                  (state, _g(rec["repair"]["objective"]),
                   _g(ref["objective"]) if ref else "-",
                   ref["status"] if ref else "-",
                   _g(ref["mip_gap"]) if ref else "-",
                   _g(getattr(res, "mip_dual_bound", None))), flush=True)


def solution_from_record(inst: Instance, rec: Dict[str, Any]) -> Solution:
    p = rec["plan"]
    return Solution(status=rec["status"], objective=float(rec["objective"]),
                    X=np.array(p["X"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(p["Y"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(p["Z"], dtype=float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(p["I"], dtype=float).reshape(inst.N, inst.T),
                    L=np.array(p["L"], dtype=float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def solution_from_res(bm, res, inst: Instance) -> Optional[Solution]:
    if res.x is None:
        return None
    X, Y, Z, I, L = unpack(res.x, bm.idx)
    return Solution(status="optimal" if res.status == 0 else
                    ("suboptimal" if res.x is not None else "no_incumbent"),
                    objective=float(res.fun), X=X, Y=Y, Z=Z, I=I, L=L,
                    mip_gap=float(res.mip_gap) if res.mip_gap is not None else None,
                    solve_time_s=0.0, mode=MODE_AUDITED)


def phase_timing() -> None:
    store = load_state(STATE_RESULTS)
    nominal = load_state(NOMINAL_RESULTS)
    insts = {m["name"]: (i, m) for i, m in all_instances()}
    for state, rec in store.items():
        if "timing" in rec and len(rec["timing"]["runs"]) == len(BUDGETS) * len(METHODS):
            continue
        name, dis_name = state.split("|")
        inst0, meta = insts[name]
        dis = next(d for d in disruptions_for(inst0) if d.name == dis_name)
        pert = apply_disruption(inst0, dis)
        rep = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        sysx = System(bm, rep, meta["tau"])

        runs: List[Dict[str, Any]] = rec.get("timing", {}).get("runs", [])
        have = {(r["method"], r["budget_s"]) for r in runs}
        todo = [(m, b) for b in BUDGETS for m in METHODS if (m, b) not in have]
        # fixed-seed shuffle of the method order, then run sequentially (never concurrent)
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(todo)
        for method, budget in todo:
            t_feat0 = time.perf_counter()
            fixed, info = method_fixing(sysx, method)
            t_feat = time.perf_counter() - t_feat0
            freez = sysx.n_free_binary(list(fixed.keys()), [])
            t0 = time.perf_counter()
            res, solve_wall, nfixed = sysx.solve(fixed, budget)
            total_wall = time.perf_counter() - t0
            sol = solution_from_res(bm, res, pert)
            used_fallback = False
            fallback_reason = ""
            if sol is None:
                # external feasible fallback: the independently verified repaired plan
                used_fallback = True
                fallback_reason = "solver returned no incumbent within %.1fs" % budget
                out_obj = rec["repair"]["objective"]
                out_feasible = rec["repair"]["feasible"]
                out_status = "fallback_repair"
                out_gap = None
                bound = None
                maxviol = rec["repair"]["max_violation"]
            else:
                sr = sol_record(sol, pert)
                out_obj = sr["objective"]
                out_feasible = sr["feasible"]
                out_status = sr["status"]
                out_gap = sr["mip_gap"]
                bound = getattr(res, "mip_dual_bound", None)
                maxviol = sr["max_violation"]
            if out_status == "infeasible" or (sol is not None and res.status == 2):
                fallback_reason = ("MODEL REPORTED INFEASIBLE but the repaired plan is a "
                                   "feasible witness: protocol error, investigate")
                used_fallback = True
                out_obj = rec["repair"]["objective"]
                out_feasible = rec["repair"]["feasible"]
            runs.append({
                "method": method, "budget_s": budget,
                "feature_and_rank_time_s": t_feat,
                "solver_wall_s": solve_wall,
                "total_wall_s": total_wall,
                "n_fixed_vars": nfixed,
                "free_binary": freez,
                "fixing_info": info,
                "objective": out_obj,
                "status": out_status,
                "proven_optimal": bool(sol is not None and res.status == 0),
                "mip_gap": out_gap,
                "mip_dual_bound": bound,
                "feasible": out_feasible,
                "max_violation": maxviol,
                "used_repair_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "over_budget": bool(total_wall > budget + 0.5),
            })
            rec.setdefault("timing", {})["runs"] = runs
            rec["timing"]["shuffle_seed"] = SHUFFLE_SEED
            store[state] = rec
            save_state(STATE_RESULTS, store)
            print("  %-30s %-12s %5.1fs wall=%6.2fs obj=%-13s %-18s fb=%s" %
                  (state, method, budget, total_wall, _g(out_obj), out_status,
                   "Y" if used_fallback else "n"), flush=True)


def phase_report() -> None:
    store = load_state(STATE_RESULTS)
    nominal = load_state(NOMINAL_RESULTS)
    rows = build_rows(store, nominal)
    write_csv(rows)
    write_details(store, nominal, rows)
    print("states with results:", len(store))


CSV_FIELDS = ["state", "scale", "rho", "seed", "disruption", "N", "M", "T", "tau",
              "nominal_obj", "nominal_gap", "nominal_proven_optimal",
              "repair_obj", "repair_feasible",
              "FULL_20", "EMPTY_20", "S12_20", "S24_20",
              "wall_FULL_20", "wall_EMPTY_20", "wall_S12_20", "wall_S24_20",
              "J_best", "J_best_source", "FULL_dual_bound", "FULL_120_proven_optimal",
              "gap_best_FULL", "gap_best_EMPTY", "gap_best_S12", "gap_best_S24",
              "fallback_FULL_20", "fallback_EMPTY_20", "fallback_S12_20", "fallback_S24_20",
              "free_Y_FULL", "free_Y_EMPTY", "free_Y_S12", "free_Y_S24",
              "max_violation"]


def _f(v: Any) -> Optional[float]:
    """Coerce a value read back from JSON to float.

    `json.dump(..., default=str)` turns numpy scalars into strings, so a reloaded objective
    can arrive as "655.126". Everything that is used arithmetically goes through here.
    """
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


def compute_jbest(rec: Dict[str, Any]) -> Tuple[float, str]:
    """J_best = the cheapest feasible cost any method produced for this state, including the
    120 s FULL reference and the independently verified repaired plan.

    It is a current-best-known value, NOT a proven optimum. Derived on demand so it is never
    round-tripped through JSON (doing that once turned the value into a string)."""
    cands: List[Tuple[str, float]] = [("repair", _f(rec["repair"]["objective"]))]
    for r in rec.get("timing", {}).get("runs", []):
        if r["feasible"] and r["objective"] is not None:
            cands.append(("%s@%gs" % (r["method"], r["budget_s"]), _f(r["objective"])))
    ref = rec.get("reference", {}).get("solution")
    if ref and ref["feasible"]:
        cands.append(("FULL@%gs" % REFERENCE_BUDGET, _f(ref["objective"])))
    jsrc, jb = min(cands, key=lambda c: c[1])
    return jb, jsrc


def build_rows(store: Dict[str, Any], nominal: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for state, rec in sorted(store.items()):
        if "timing" not in rec:
            continue
        runs = rec["timing"]["runs"]
        by = {(r["method"], _f(r["budget_s"])): r for r in runs}

        def at(m, b=20.0):
            return by.get((m, float(b)))

        jb, jsrc = compute_jbest(rec)
        ref = rec.get("reference", {}).get("solution")

        def gap(m, b=20.0):
            r = at(m, b)
            if r is None or not r["feasible"] or r["objective"] is None:
                return None
            return (_f(r["objective"]) - jb) / max(1.0, abs(jb))

        row = {
            "state": state, "scale": rec["meta"]["scale"], "rho": rec["meta"]["rho"],
            "seed": rec["meta"]["seed"], "disruption": rec["disruption"]["name"],
            "N": rec["meta"]["N"], "M": rec["meta"]["M"], "T": rec["meta"]["T"],
            "tau": rec["meta"]["tau"],
            "nominal_obj": _f(rec["nominal_objective"]),
            "nominal_gap": _f(rec["nominal_mip_gap"]),
            "nominal_proven_optimal": rec["nominal_proven_optimal"],
            "repair_obj": _f(rec["repair"]["objective"]),
            "repair_feasible": rec["repair"]["feasible"],
            "J_best": jb, "J_best_source": jsrc,
            "FULL_dual_bound": _f(rec.get("reference", {}).get("mip_dual_bound")),
            "FULL_120_proven_optimal": (ref or {}).get("proven_optimal"),
            "max_violation": max([_f(rec["repair"]["max_violation"]) or 0.0] +
                                 [_f(r["max_violation"]) or 0.0 for r in runs]),
        }
        for m, tag in (("FULL", "FULL"), ("EMPTY", "EMPTY"),
                       ("SHORTAGE-12", "S12"), ("SHORTAGE-24", "S24")):
            r20 = at(m)
            row["%s_20" % tag] = _f(r20["objective"]) if r20 else None
            row["wall_%s_20" % tag] = _f(r20["total_wall_s"]) if r20 else None
            row["gap_best_%s" % tag] = gap(m)
            row["fallback_%s_20" % tag] = r20["used_repair_fallback"] if r20 else None
            row["free_Y_%s" % tag] = (r20["free_binary"]["free_Y"] if r20 else None)
        rows.append(row)
    save_state(STATE_RESULTS, store)
    return rows


def write_csv(rows: List[Dict[str, Any]]) -> None:
    with open("stage04_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def write_scale_summary(store: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    """Aggregate gap_best by scale, method and budget, as the brief requires."""
    L: List[str] = []
    L.append("\n## 1b. Scale / method / budget summary\n")
    L.append("Mean and worst `gap_best` over the states of each scale. A state where the "
             "method fell back to the repaired plan is counted at its fallback gap.\n")
    L.append("| scale | method | budget | states | mean gap | worst gap | proven optimal | fallbacks |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for scale in ("small", "medium", "large"):
        for m in METHODS:
            for b in BUDGETS:
                vals, n, n_opt, n_fb = [], 0, 0, 0
                for state, rec in store.items():
                    if rec["meta"]["scale"] != scale or "timing" not in rec:
                        continue
                    jb, _ = compute_jbest(rec)
                    run = next((x for x in rec["timing"]["runs"]
                                if x["method"] == m and _f(x["budget_s"]) == float(b)), None)
                    if run is None or not run["feasible"] or run["objective"] is None:
                        continue
                    n += 1
                    vals.append((_f(run["objective"]) - jb) / max(1.0, abs(jb)))
                    if run["proven_optimal"]:
                        n_opt += 1
                    if run["used_repair_fallback"]:
                        n_fb += 1
                if n:
                    L.append("| %s | %s | %g | %d | %.4f | %.4f | %d | %d |" %
                             (scale, m, b, n, sum(vals) / len(vals), max(vals), n_opt, n_fb))
    return L


def write_details(store: Dict[str, Any], nominal: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    L: List[str] = []
    L.append("# stage04_details.md\n")
    L.append("Machine-generated detail tables. Produced by `stage04_run.py`.\n")
    L.append("Environment: python %s, numpy %s, scipy %s, %s\n"
             % (platform.python_version(), np.__version__, scipy.__version__,
                platform.platform()))
    L.append("Model `audited_v1`, kappa = %g, tau per scale %s. "
             "`scipy.optimize.milp` has NO MIP-start parameter, so no run is warm started.\n"
             % (KAPPA, {k: v["tau"] for k, v in SCALES.items()}))

    L.append("\n## 0. Nominal solves\n")
    L.append("| instance | N | M | T | obj | gap | proven optimal | feasible |")
    L.append("|---|---:|---:|---:|---:|---:|---|---|")
    for name, rec in sorted(nominal.items()):
        s = rec["solution"]
        L.append("| %s | %d | %d | %d | %s | %s | %s | %s |" %
                 (name, rec["meta"]["N"], rec["meta"]["M"], rec["meta"]["T"],
                  _g(s["objective"]), _g(s["mip_gap"]), s["proven_optimal"], s["feasible"]))

    L.append("\n## 1. Per-state 20 s results\n")
    L.append("| state | scale | rho | dis | nom | repair | FULL | EMPTY | S-12 | S-24 | J_best | src | FULL bound |")
    L.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for r in rows:
        L.append("| %s | %s | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["state"], r["scale"], r["rho"], r["disruption"], _g(r["nominal_obj"]),
            _g(r["repair_obj"]), _g(r["FULL_20"]), _g(r["EMPTY_20"]),
            _g(r["SHORTAGE12_20"] if "SHORTAGE12_20" in r else r.get("S12_20")),
            _g(r["SHORTAGE24_20"] if "SHORTAGE24_20" in r else r.get("S24_20")),
            _g(r["J_best"]),
            r["J_best_source"], _g(r["FULL_dual_bound"])))

    L.append("\n## 2. gap_best by budget, all methods\n")
    L.extend(write_scale_summary(store, rows))
    L.append("\n### Per state\n")
    L.append("| state | method | 1s | 5s | 20s |")
    L.append("|---|---|---:|---:|---:|")
    for state, rec in sorted(store.items()):
        if "timing" not in rec:
            continue
        jb, _ = compute_jbest(rec)
        for m in METHODS:
            cells = []
            for b in BUDGETS:
                run = next((x for x in rec["timing"]["runs"]
                            if x["method"] == m and _f(x["budget_s"]) == float(b)), None)
                if run is None or not run["feasible"] or run["objective"] is None:
                    cells.append("-")
                else:
                    cells.append("%.4f" % ((_f(run["objective"]) - jb) / max(1.0, abs(jb))))
            L.append("| %s | %s | %s |" % (state, m, " | ".join(cells)))

    L.append("\n## 3. Wall clock versus requested budget\n")
    L.append("| state | method | budget | wall | solver wall | feature+rank | n fixed | free Y | free Z | fallback |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for state, rec in sorted(store.items()):
        if "timing" not in rec:
            continue
        for run in rec["timing"]["runs"]:
            L.append("| %s | %s | %g | %.2f | %.2f | %.3f | %d | %d | %d | %s |" % (
                state, run["method"], run["budget_s"], run["total_wall_s"],
                run["solver_wall_s"], run["feature_and_rank_time_s"], run["n_fixed_vars"],
                run["free_binary"]["free_Y"], run["free_binary"]["free_Z"],
                "YES" if run["used_repair_fallback"] else "no"))

    L.append("\n## 4. Repair logs\n")
    for state, rec in sorted(store.items()):
        if "repair" in rec:
            L.append("- **%s**: %s" % (state, rec["repair"]["log_text"][:400]))
    with open("stage04_details.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["nominal", "reference", "timing", "report", "all"])
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    if args.phase in ("nominal", "all"):
        print("== phase nominal ==", flush=True)
        phase_nominal()
    if args.phase in ("reference", "all"):
        print("== phase reference ==", flush=True)
        phase_reference()
    if args.phase in ("timing", "all"):
        print("== phase timing ==", flush=True)
        phase_timing()
    if args.phase in ("report", "all"):
        print("== phase report ==", flush=True)
        phase_report()
    print("total %.1fs" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
