#!/usr/bin/env python
"""
Shared machinery for the variable-group release experiments (stage 09B / stage 10).

Group definitions
-----------------
    A = { Y_ijt : t > tau }          far-horizon setups
    B = { Z_ijt : 1 <= t <= tau }    short-term carry-overs
    C = { Z_ijt : tau < t < T }      far-horizon carry-overs

Structurally pinned Z (Z_ij0 and Z_ijT under audited_v1) are fixed in EVERY configuration, and
the short-term Y (t <= tau) is fixed in every configuration except FULL.

`FULL` is deliberately a SEPARATE switch rather than another group letter: it releases the
short-term Y as well, which is the one thing EMPTY/ABC keeps fixed.  Keeping it separate stops
"ABC" from silently becoming "FULL" and keeps the comparison against the original task honest.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

import highspy
from lsp_highs import build_highs
from lsp_model import MODE_AUDITED, Solution, enumeration_space, unpack

LP_BUDGET = 5.0
THREADS = 1
GROUPS = ("A", "B", "C")
# config -> (released groups, short-term Y released?)
CONFIGS: Dict[str, Tuple[str, bool]] = {
    "A": ("A", False),
    "B": ("B", False),
    "C": ("C", False),
    "AB": ("AB", False),
    "AC": ("AC", False),
    "BC": ("BC", False),
    "ABC": ("ABC", False),
    "FULL": ("ABC", True),
}


def group_of(kind: str, t: int, tau: int, T: int) -> Optional[str]:
    if kind == "Y":
        return "A" if t > tau else None
    if t <= tau:
        return "B"
    if t < T:
        return "C"
    return None


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


def pinned_bounds(C: Dict[str, Any]):
    """LP bounds: every binary pinned at S_r's normalised value.

    Memoised on the per-state dict, NOT in a module map keyed by id(): CPython reuses ids after
    garbage collection, which already caused one bug in this project.
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


def solve_lp_fixed(bm, lb_pin, ub_pin, budget: float, log_path: str) -> Dict[str, Any]:
    """The stage 09A LP: every binary pinned by bounds, integrality dropped."""
    bm_lp = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality),
                               var_lb=lb_pin, var_ub=ub_pin)
    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm_lp, {}, ({}, 0.0))
    prep = time.perf_counter() - t0
    if os.path.exists(log_path):
        os.remove(log_path)
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


def build_state_context(inst, pert, repair, bm, tau) -> Dict[str, Any]:
    """Everything a release experiment needs for one state."""
    assign: Dict[Tuple[str, int, int, int], int] = {}
    norm_err = 0.0
    for slot in enumeration_space(pert, MODE_AUDITED):
        kind, i, j, t = slot
        raw = float(repair.Y[i, j, t - 1]) if kind == "Y" else float(repair.Z[i, j, t - 1])
        norm_err = max(norm_err, abs(raw - round(raw)))
        assign[slot] = int(round(raw))
    return {"bm": bm, "assign": assign, "norm_err": norm_err,
            "groups": slot_sets(pert, tau), "all_slots": set(assign)}


def stage14_columns(C: Dict[str, Any], method: str):
    """Stage 14 group-release configurations.

        A         far-horizon Y released, no short-term Z released
        AB        far-horizon Y released, every non-pinned short-term Z released
        RECOVERY  far-horizon Y released, only Z_{i,j,r_j} released, where r_j is the first
                  period in which machine j has positive capacity again after its outage
        END       far-horizon Y released, only Z_{i,j,tau} released

    RECOVERY and END release the same NUMBER of machine-period groups; only the position differs.
    Both are computed from fault-time information only (the outage length per machine), so the
    rule is deployable and transferable.
    """
    from lsp_neighborhood import recovery_period as _rec

    bm, pert, tau = C["bm"], C["inst"], C["tau"]
    A_slots = sorted(C["groups"]["A"])
    B_coords = {tuple(u[1:]) for u in C["groups"]["B"]}
    released_Z: Set[Tuple[int, int, int]] = set()
    # `affected` = the machines that actually had an outage.  delta_for returns 0 for a machine
    # that was never down, which would otherwise place it at period 1 and make the two rules
    # release different NUMBERS of groups for D1.
    affected = sorted(j for j in range(pert.M) if C["dis"].delta_for(j) > 0)
    if method == "AB":
        released_Z = set(B_coords)
    elif method in ("RECOVERY", "END"):
        r = {j: _rec(C["dis"], j, pert.T) for j in affected}
        for (i, j, t) in B_coords:
            if j not in r:
                continue
            if method == "RECOVERY":
                if r[j] <= tau and t == r[j]:
                    released_Z.add((i, j, t))
            else:
                if t == tau:
                    released_Z.add((i, j, t))
    elif method != "A":
        raise SystemExit("unknown stage 14 method %r" % method)

    rel = list(A_slots) + [("Z",) + tuple(u) for u in sorted(released_Z)]
    rel_set = set(rel)
    fixed: Dict[int, float] = {}
    for slot, val in C["assign"].items():
        if slot in rel_set:
            continue
        kind, i, j, t = slot
        col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
        fixed[col] = float(val)
    meta = {"n_rel_Y": len(A_slots), "n_rel_Z": len(released_Z), "rel_size": len(rel),
            "affected_machines": affected,
            "recovery_periods": {str(j): _rec(C["dis"], j, pert.T) for j in affected}}
    return rel, fixed, meta


def fixed_columns(C: Dict[str, Any], config: str) -> Tuple[List[Tuple[str, int, int, int]],
                                                            Dict[int, float]]:
    """Released slot list and the pinned columns for a configuration."""
    rel_groups, release_short_y = CONFIGS[config]
    rel: List[Tuple[str, int, int, int]] = []
    for g in rel_groups:
        rel.extend(sorted(C["groups"][g]))
    if release_short_y:
        rel.extend([("Y", i, j, t)
                    for i in range(C["bm"].inst.N) for j in range(C["bm"].inst.M)
                    for t in range(1, C["tau"] + 1)])
    rel_set = set(rel)
    fixed: Dict[int, float] = {}
    for slot, val in C["assign"].items():
        if slot in rel_set:
            continue
        kind, i, j, t = slot
        col = C["bm"].idx.Y(i, j, t) if kind == "Y" else C["bm"].idx.Z(i, j, t)
        fixed[col] = float(val)
    return rel, fixed


# --------------------------------------------------------------------------------------
# Stage 11: matched random Z-fixing control
# --------------------------------------------------------------------------------------
#
# Group slots already carry their tag: ('Y', i, j, t) for group A and ('Z', i, j, t) for B and C,
# with t 1-based.  The frozen matched-random sets store bare (i, j, t) triples, so comparisons
# are made on `u[1:]`.
#
#   AB    release A and B, fix the short-term Y and all of C
#   ABC   release A, B and C, fix the short-term Y
#   MR-r  release A and (B u C) minus the frozen matched fixing set, fix the short-term Y
#
# The short-term Y stays fixed in EVERY one of these, AB, ABC and MR alike.  What the matched
# control is allowed to shuffle is the fixed/released identity of the Z slots, including the
# short-term Z group B.

def stage11_columns(C: Dict[str, Any], method: str):
    """Released slot list, pinned columns and the matched fixing set for a stage 11 method."""
    if method in CONFIGS:
        rel, fixed = fixed_columns(C, method)
        return rel, fixed, None
    A_slots = sorted(C["groups"]["A"])
    B_coords = {tuple(u[1:]) for u in C["groups"]["B"]}
    C_coords = {tuple(u[1:]) for u in C["groups"]["C"]}
    Z_all = [("Z",) + tuple(u) for u in sorted(B_coords | C_coords)]
    fixed_Z = C["mr_sets"][method]
    if not fixed_Z <= (B_coords | C_coords):
        raise SystemExit("%s: matched fixing set is not a subset of B u C (%d stray)"
                         % (C["meta"]["name"], len(fixed_Z - (B_coords | C_coords))))
    rel = list(A_slots) + [u for u in Z_all if u[1:] not in fixed_Z]
    rel_set = set(rel)
    fixed: Dict[int, float] = {}
    for slot, val in C["assign"].items():
        if slot in rel_set:
            continue
        kind, i, j, t = slot
        col = C["bm"].idx.Y(i, j, t) if kind == "Y" else C["bm"].idx.Z(i, j, t)
        fixed[col] = float(val)
    return rel, fixed, fixed_Z


# --------------------------------------------------------------------------------------
# Stage 16: single-shot LP-guided neighbourhood (RINS-LP-ONE)
# --------------------------------------------------------------------------------------
#
# The rule ("fix the integer variables on which the relaxation and the incumbent agree, then
# re-optimise the rest") is the single-shot version of the RINS idea.  This implementation has
# exactly ONE relaxation solve and ONE neighbourhood solve -- there is no repeated invocation
# inside a search tree, so it is NOT a reproduction of the full RINS algorithm and is named
# `RINS-LP-ONE` accordingly.
#
# Reference for the idea: Rothberg, "Heuristics for MIP", lecture slides, pp. 9-11.
# The neighbourhood is built from the FULL relaxation (integrality dropped, every structural
# fixing kept, stability row kept).  The all-binary-pinned stage 09A LP is NOT a valid source
# for this neighbourhood.

RINS_TOL = 1e-6


def stability_row_local(bm, repair, tau, kappa: float = 4.0):
    """The same stability row the runners use: sum of short-term Y changes <= kappa.

    Duplicated here (rather than imported from stage06_run) so that lsp_release stays free of
    runner-level imports.
    """
    coef: Dict[int, float] = {}
    n_one = 0
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, tau + 1):
                if float(repair.Y[i, j, t - 1]) > 0.5:
                    n_one += 1
                    coef[bm.idx.Y(i, j, t)] = -1.0
                else:
                    coef[bm.idx.Y(i, j, t)] = 1.0
    return coef, float(kappa) - float(n_one)


def binary_slot_map(C: Dict[str, Any]) -> Dict[int, Tuple[str, int, int, int]]:
    """column -> slot for every BINARY column of the audited model (covers structural pins)."""
    bm = C["bm"]
    out: Dict[int, Tuple[str, int, int, int]] = {}
    for slot in C["assign"]:
        kind, i, j, t = slot
        col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
        out[int(col)] = slot
    return out


def structural_pin_columns(C: Dict[str, Any]) -> Set[int]:
    """Binary columns pinned by the MODEL itself (lb == ub), e.g. Z_ij0 = Z_ijT = 0.

    These are structurally zero-width and are NOT part of `C["assign"]` (the enumeration space
    excludes them), so this normally returns the empty set; it is retained as a guard so that a
    structural pin can never be mistaken for a RINS decision.
    """
    bm = C["bm"]
    pins: Set[int] = set()
    for col in binary_slot_map(C):
        if float(bm.var_lb[col]) == float(bm.var_ub[col]):
            pins.add(col)
    return pins


def group_of_column(C: Dict[str, Any], col: int) -> str:
    """A / B / C / short_Y / structural for a binary column, using the frozen tau."""
    kind, i, j, t = binary_slot_map(C)[col]
    if kind == "Y":
        return "short_Y" if t <= C["tau"] else "A"
    if t <= C["tau"]:
        return "B"
    if t < C["bm"].inst.T:
        return "C"
    return "structural"


def solve_lp_relax(bm, stab, budget: float, log_path: str) -> Dict[str, Any]:
    """FULL LP relaxation: integrality dropped, structural fixings + stability row kept.

    No column is pinned to the repaired plan, so this is the relaxation of the FULL task and
    its objective is a valid LOWER BOUND on `J_LP`.
    """
    bm_lp = dataclasses.replace(bm, integrality=np.zeros_like(bm.integrality))
    t0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm_lp, {}, stab)
    prep = time.perf_counter() - t0
    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)
    h.passModel(lp)
    h.setOptionValue("time_limit", float(max(0.1, budget)))
    h.setOptionValue("threads", int(THREADS))
    t1 = time.perf_counter()
    h.run()
    solved = time.perf_counter() - t1
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    obj, x = None, None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        cand = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(cand)):
            x = cand
            try:
                obj = float(info.objective_function_value)
            except Exception:                        # noqa: BLE001
                obj = None
    # "optimal" is the only status that guarantees the relaxation was solved to optimality
    # inside the budget; anything else means "no valid optimal relaxation", and the caller must
    # fall back to FULL.
    is_optimal = status.lower().startswith("optimal")
    return {"status": status, "objective": obj, "x": x, "is_optimal": bool(is_optimal),
            "prep_s": prep, "solve_s": solved, "n_cols": n_cols, "n_rows": n_rows}


def stage16_rins_columns(C: Dict[str, Any], x_lp: Optional[np.ndarray],
                         fallback: bool = False):
    """Fixings of RINS-LP-ONE.

        |x^LP_v - S_{r,v}| <= RINS_TOL   ->   v = S_{r,v}

    for every NON-structural binary v.  Continuous variables stay free; A, C and the
    short-term Y are not forced either way; the short-term Y stays under the original kappa
    row.  The threshold is never tuned to match AB's released-variable count.
    """
    bm = C["bm"]
    pins = structural_pin_columns(C)
    slot_of = binary_slot_map(C)
    s_r = C["assign"]
    n_by_group = {"A": 0, "B": 0, "C": 0, "short_Y": 0, "structural": 0}
    fixed: Dict[int, float] = {}
    n_nonstruct = 0
    n_clamped = 0

    if x_lp is not None and not fallback:
        for col in sorted(slot_of):
            if col in pins:
                continue                              # structural pin, not a RINS decision
            n_nonstruct += 1
            slot = slot_of[col]
            ref = float(s_r[slot])
            if abs(float(x_lp[col]) - ref) <= RINS_TOL:
                val = ref
                # safety only: this branch already guarantees agreement within the tolerance,
                # so no clamp should ever be needed.  It is applied and COUNTED, never silent.
                if abs(float(x_lp[col]) - val) > 0.5:
                    n_clamped += 1
                fixed[col] = val
                n_by_group[group_of_column(C, col)] += 1

    n_by_group["structural"] = len(pins)
    meta = {
        "n_binary_total": len(slot_of),
        "n_structural_pins": len(pins),
        "n_free_binary_raw": n_nonstruct,
        "n_fixed_total": len(fixed),
        "n_free_binary_after": n_nonstruct - len(fixed),
        "n_fixed_by_group": n_by_group,
        "n_clamped": n_clamped,
        "fallback": bool(fallback or x_lp is None),
        "rel_size": int(bm.idx.n - len(fixed)),
        "rins_tol": RINS_TOL,
    }
    return meta["rel_size"], fixed, meta
