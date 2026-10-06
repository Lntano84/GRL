#!/usr/bin/env python
"""
Stage 01 runner: build, solve, independently verify.

One command reproduces every number in stage01_report.md:

    python stage01_run.py

Outputs (written next to this script):
    stage01_results.csv   one row per check, machine readable
    stage01_details.md    the same content as readable markdown tables, plus
                          per-row constraint residuals for the diverging cases

No software is installed, no network access is used, and no learning method is
implemented: this round only builds and audits the experimental base.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

from lsp_checker import check_solution
from lsp_instances import hand_instances, micro_instances
from lsp_model import (MODE_AUDITED, MODE_PAPER, MODES, Instance, Solution,
                       assignment_of_solution, build_model, cost_breakdown,
                       enumerate_optimal, enumeration_space, solve_milp,
                       solve_lp_with_binaries_fixed, unpack)

HERE = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else "."

# Hand-computed expectations supplied for this round. These are recorded, NOT enforced:
# the runner reports agreement or disagreement and never adjusts a result to match them.
HAND_EXPECTATIONS: Dict[str, Dict[str, Optional[float]]] = {
    "A_ample": {MODE_AUDITED: 5.0, MODE_PAPER: None},
    "B_short": {MODE_AUDITED: 14.0, MODE_PAPER: None},
    "C_incompat": {MODE_AUDITED: 30.0, MODE_PAPER: None},
    "D_down": {MODE_AUDITED: 30.0, MODE_PAPER: None},
    "E_early": {MODE_AUDITED: 8.0, MODE_PAPER: None},
    "F_carryover": {MODE_AUDITED: 1.0, MODE_PAPER: 11.0},
    "G_terminal_Z": {MODE_AUDITED: 10.0, MODE_PAPER: 1.0},
}

TOL_OBJ = 1e-6  # objective agreement tolerance, scaled by (1 + |objective|)

# Hand-computed optimal costs for the micro instances. These are an EXTRA check on top of
# the MILP-versus-enumeration agreement; a mismatch is reported, never silently absorbed.
MICRO_EXPECTATIONS: Dict[str, float] = {
    "MI0_basic": 21.0,
    "MI1_incompat": 39.0,
    "MI2_zero_demand": 0.0,
    "MI3_initial_inv": 3.0,
    "MI4_tight": 57.0,
    "MI5_zero_cap": 42.0,
    "MI6_cheap_lost": 12.0,
    "MI7_setup_time": 112.0,
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _g(v: float, nd: int = 6) -> str:
    """Format a number, collapsing -0 and near-zero noise."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if abs(v) < 10 ** (-nd):
        return "0"
    return ("%." + str(nd) + "g") % v


def fmt_vec(arr: np.ndarray) -> str:
    return "[" + ",".join(_g(v) for v in np.asarray(arr).ravel()) + "]"


def plan_string(inst: Instance, sol: Solution) -> str:
    """Compact, human-readable rendering of a plan."""
    parts = []
    for i in range(inst.N):
        for j in range(inst.M):
            xs = ",".join(_g(sol.X[i, j, t]) for t in range(inst.T))
            ys = ",".join(_g(sol.Y[i, j, t]) for t in range(inst.T))
            zs = ",".join(_g(sol.Z[i, j, t]) for t in range(inst.T))
            parts.append(f"(i{i},j{j}) X=[{xs}] Y=[{ys}] Z=[{zs}]")
        ls = ",".join(_g(sol.L[i, t]) for t in range(inst.T))
        iis = ",".join(_g(sol.I[i, t]) for t in range(inst.T))
        parts.append(f"i{i} I=[{iis}] L=[{ls}]")
    return " ".join(parts)


class Rows:
    """Accumulates CSV rows."""

    def __init__(self, fieldnames: List[str]):
        self.fieldnames = fieldnames
        self.rows: List[Dict[str, Any]] = []

    def add(self, **kw: Any) -> None:
        missing = [f for f in self.fieldnames if f not in kw]
        if missing:
            raise KeyError(f"row is missing fields {missing}")
        self.rows.append({k: kw[k] for k in self.fieldnames})

    def write(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writeheader()
            for row in self.rows:
                w.writerow(row)


CSV_FIELDS = [
    "block", "instance", "mode", "check", "passed", "metric", "value",
    "milp_objective", "enum_objective", "objective_delta", "max_violation",
    "n_rule_failures", "status", "notes", "plan",
]


# --------------------------------------------------------------------------------------
# block 1: hand-checkable instances
# --------------------------------------------------------------------------------------


def run_hand(rows: Rows) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, inst in hand_instances().items():
        per_mode: Dict[str, Any] = {}
        for mode in MODES:
            bm = build_model(inst, mode)
            sol = solve_milp(bm)
            rep = check_solution(inst, sol, mode)
            expect = HAND_EXPECTATIONS.get(name, {}).get(mode)
            if expect is None:
                agreement = "not-specified"
            elif np.isnan(sol.objective):
                agreement = "mismatch"
            elif abs(sol.objective - expect) <= TOL_OBJ * (1.0 + abs(expect)):
                agreement = "match"
            else:
                agreement = "mismatch"
            plan = plan_string(inst, sol)
            per_mode[mode] = {
                "objective": sol.objective, "expectation": expect,
                "agreement": agreement, "check_ok": rep.ok,
                "max_violation": rep.max_violation,
                "failed": rep.failed_checks(), "plan": plan,
                "status": sol.status,
                "cost": rep.cost_recomputed,
                "notes": bm.notes,
            }
            rows.add(block="hand", instance=name, mode=mode, check="milp_solve",
                     passed=rep.ok, metric="objective", value=sol.objective,
                     milp_objective=sol.objective, enum_objective="",
                     objective_delta="", max_violation=rep.max_violation,
                     n_rule_failures=len(rep.failed_checks()),
                     status=sol.status,
                     notes=("hand expectation %s -> %s" % (expect, agreement)),
                     plan=plan)
        # does the two-mode distinction change anything at all?
        a, p = per_mode[MODE_AUDITED]["objective"], per_mode[MODE_PAPER]["objective"]
        differs = (not np.isnan(a)) and (not np.isnan(p)) and abs(a - p) > TOL_OBJ
        results[name] = {"modes": per_mode,
                         "mode_differs": bool(differs),
                         "delta_audited_minus_paper": (a - p) if not (np.isnan(a) or np.isnan(p)) else None}
    return results


# --------------------------------------------------------------------------------------
# block 2: independent enumeration
# --------------------------------------------------------------------------------------


def run_enumeration(rows: Rows, max_configs: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    micro = micro_instances()
    # Structural sanity check attached to the enumeration. With non-negative parameters the
    # instance is ALWAYS feasible: setting every Y and Z to 0 admits X = 0, and the flow
    # balance (1.2) is then satisfied by L_it = max(0, d_it - I_i(t-1) + I_it) with I >= 0,
    # which (1.3) allows because L_it <= d_it. Lost sales carry no capacity cost, so no
    # amount of missing capacity can make the model infeasible. The all-zero assignment is
    # therefore a witness that must always be found; if it is not, the audit itself is
    # broken. (A deliberately infeasible instance was attempted and had to be dropped: every
    # candidate turned out feasible for exactly this reason.)

    for name, inst in micro.items():
        mode = MODE_AUDITED
        bm = build_model(inst, mode)
        sol = solve_milp(bm)
        rep = check_solution(inst, sol, mode)
        plan = plan_string(inst, sol)

        rows.add(block="enum", instance=name, mode=mode, check="milp_solve",
                 passed=rep.ok, metric="objective", value=sol.objective,
                 milp_objective=sol.objective, enum_objective="",
                 objective_delta="", max_violation=rep.max_violation,
                 n_rule_failures=len(rep.failed_checks()), status=sol.status,
                 notes="MILP path", plan=plan)

        slots = enumeration_space(inst, mode)
        n_cfg = 2 ** len(slots)
        if n_cfg > max_configs:
            results[name] = {"skipped": f"{n_cfg} configurations exceed cap {max_configs}"}
            rows.add(block="enum", instance=name, mode=mode, check="enumeration",
                     passed=False, metric="skipped", value=n_cfg,
                     milp_objective=sol.objective, enum_objective="", objective_delta="",
                     max_violation="", n_rule_failures="", status="skipped",
                     notes=f"configuration cap {max_configs}", plan="")
            continue

        t0 = time.perf_counter()
        enum = enumerate_optimal(bm, max_configs=max_configs)
        enum_time = time.perf_counter() - t0
        eo = enum["best_objective"]
        delta = abs(eo - sol.objective) if np.isfinite(eo) else float("nan")
        agree = np.isfinite(eo) and np.isfinite(sol.objective) and \
            delta <= TOL_OBJ * (1.0 + abs(sol.objective))

        # the all-zero assignment must always be feasible (see the note above)
        st_zero, obj_zero, _ = solve_lp_with_binaries_fixed(
            bm, {slot: 0 for slot in enumeration_space(inst, mode)})
        witness_ok = st_zero == "optimal"
        rows.add(block="enum", instance=name, mode=mode, check="all_zero_witness_feasible",
                 passed=bool(witness_ok), metric="all_binary_zero_objective", value=obj_zero,
                 milp_objective="", enum_objective="", objective_delta="",
                 max_violation="", n_rule_failures="", status=st_zero,
                 notes="X=Y=Z=0 with L absorbing demand must be feasible; otherwise the "
                       "independent LP path or the assembled model is wrong",
                 plan="")

        expect = MICRO_EXPECTATIONS.get(name)
        if expect is None:
            expect_agree = "not-specified"
        elif abs(sol.objective - expect) <= TOL_OBJ * (1.0 + abs(expect)):
            expect_agree = "match"
        else:
            expect_agree = "mismatch"

        # cross-check: the MILP's own assignment must be feasible in the enumeration path
        assign = assignment_of_solution(sol, inst, mode)
        st_milp_plan, obj_milp_plan, _ = solve_lp_with_binaries_fixed(bm, assign)

        rows.add(block="enum", instance=name, mode=mode, check="enumeration_vs_milp",
                 passed=bool(agree), metric="objective_delta", value=delta,
                 milp_objective=sol.objective, enum_objective=eo,
                 objective_delta=delta, max_violation=rep.max_violation,
                 n_rule_failures=len(rep.failed_checks()), status=sol.status,
                 notes=(f"{enum['n_configs']} configs, {enum['n_feasible']} feasible, "
                        f"{enum_time:.3f}s; expected cost {expect} -> {expect_agree}; "
                        f"MILP assignment re-solved as LP -> "
                        f"{st_milp_plan} obj={obj_milp_plan:.10g}"),
                 plan=plan)

        results[name] = {
            "milp_objective": sol.objective, "enum_objective": eo,
            "delta": delta, "agree": bool(agree),
            "expectation": expect, "expectation_agreement": expect_agree,
            "all_zero_witness_ok": bool(witness_ok),
            "n_slots": enum["n_binary_slots"], "n_configs": enum["n_configs"],
            "n_feasible": enum["n_feasible"], "enum_time_s": enum_time,
            "check_ok": rep.ok, "max_violation": rep.max_violation,
            "failed": rep.failed_checks(), "plan": plan,
            "milp_assignment_status": st_milp_plan,
            "milp_assignment_objective": obj_milp_plan,
        }
    return results


# --------------------------------------------------------------------------------------
# block 3: fix-and-release interface
# --------------------------------------------------------------------------------------


def run_fix_release(rows: Rows) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, inst in micro_instances().items():
        mode = MODE_AUDITED
        bm = build_model(inst, mode)
        ref = solve_milp(bm)                      # verified feasible reference plan
        ref_check = check_solution(inst, ref, mode)
        if not ref.ok or not ref_check.ok:
            rows.add(block="fixrelease", instance=name, mode=mode, check="reference_plan",
                     passed=False, metric="status", value=ref.status,
                     milp_objective=ref.objective, enum_objective="", objective_delta="",
                     max_violation=ref_check.max_violation,
                     n_rule_failures=len(ref_check.failed_checks()), status=ref.status,
                     notes="no verified feasible reference plan, so the interface cannot be "
                           "exercised on this instance", plan="")
            results[name] = {"reference_ok": False, "reference_status": ref.status,
                             "reference_check_ok": ref_check.ok,
                             "reference_objective": ref.objective,
                             "n_Y": 0, "fix_all_ok": False, "fix_Y_only_ok": False,
                             "monotone": False, "sweep": [],
                             "reference_feasible_under_Y_fix": False}
            continue
        idx = bm.idx
        plan = plan_string(inst, ref)

        # (a) fix every variable to the reference plan -> cost must be identical
        fix_all: Dict[int, float] = {}
        for i in range(inst.N):
            for j in range(inst.M):
                for t in range(1, inst.T + 1):
                    fix_all[idx.X(i, j, t)] = float(ref.X[i, j, t - 1])
                    fix_all[idx.Y(i, j, t)] = float(ref.Y[i, j, t - 1])
                    fix_all[idx.Z(i, j, t)] = float(ref.Z[i, j, t - 1])
            for t in range(1, inst.T + 1):
                fix_all[idx.I(i, t)] = float(ref.I[i, t - 1])
                fix_all[idx.L(i, t)] = float(ref.L[i, t - 1])
        sol_a = solve_milp(bm, fix=fix_all)
        delta_a = abs(sol_a.objective - ref.objective)
        ok_a = delta_a <= TOL_OBJ * (1.0 + abs(ref.objective))
        rows.add(block="fixrelease", instance=name, mode=mode, check="fix_all_variables",
                 passed=bool(ok_a), metric="objective_delta", value=delta_a,
                 milp_objective=ref.objective, enum_objective=sol_a.objective,
                 objective_delta=delta_a, max_violation=ref_check.max_violation,
                 n_rule_failures=len(ref_check.failed_checks()), status=sol_a.status,
                 notes="fixing every variable must reproduce the reference cost",
                 plan=plan)

        # (b) fix only the Y variables -> reference plan must remain feasible, cost may drop
        sol_b = solve_milp(bm, free_Y_subset=[], Y_reference=ref.Y)
        ref_feasible = _is_assignment_feasible(bm, ref)
        ok_b = ref_feasible and sol_b.ok and sol_b.objective <= ref.objective + TOL_OBJ
        rows.add(block="fixrelease", instance=name, mode=mode, check="fix_Y_only",
                 passed=bool(ok_b), metric="objective", value=sol_b.objective,
                 milp_objective=ref.objective, enum_objective=sol_b.objective,
                 objective_delta=sol_b.objective - ref.objective,
                 max_violation=sol_b.meta.get("check_violation", ""),
                 n_rule_failures="", status=sol_b.status,
                 notes=("reference plan feasible with Y fixed: %s ; cost may decrease but "
                        "must not increase" % ref_feasible),
                 plan=plan_string(inst, sol_b))

        # (c) grow the set of free Y variables -> optimal cost must not increase.
        # The paper's fix-and-optimize fixes the setup variables of the short-term horizon
        # t <= tau and leaves later periods free. We mirror that: all_Y holds the Y
        # variables of the short-term horizon tau = T - 1 that may be released, while the Y
        # variables of period T are always free. k = 0 is the fully restricted model in
        # which every short-term setup is pinned to the reference plan, so the sweep starts
        # from a genuinely restricted problem instead of from the full model.
        tau = max(1, inst.T - 1)
        all_Y = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
                 for t in range(1, tau + 1)]
        sweep = []
        prev = None
        monotone = True
        for k in range(len(all_Y) + 1):
            best = float("inf")
            for combo in itertools.combinations(all_Y, k):
                s = solve_milp(bm, free_Y_subset=list(combo), Y_reference=ref.Y)
                if s.ok and s.objective < best:
                    best = s.objective
            sweep.append({"k": k, "objective": best,
                          "n_subsets": math.comb(len(all_Y), k)})
            if prev is not None and best > prev + TOL_OBJ * (1.0 + abs(prev)):
                monotone = False
            if np.isfinite(best):
                prev = best
        rows.add(block="fixrelease", instance=name, mode=mode, check="grow_free_Y_monotone",
                 passed=bool(monotone), metric="sweep",
                 value=";".join("k%d=%.6g" % (e["k"], e["objective"]) for e in sweep),
                 milp_objective=ref.objective,
                 enum_objective=sweep[-1]["objective"], objective_delta="",
                 max_violation="", n_rule_failures="", status="optimal",
                 notes=("tau=%d, %d releaseable short-term Y; k=0 pins all of them to the "
                        "reference plan and the optimum must be non-increasing in k"
                        % (tau, len(all_Y))),
                 plan=plan)

        results[name] = {"reference_objective": ref.objective, "reference_check_ok": ref_check.ok,
                         "fix_all_delta": delta_a, "fix_all_ok": bool(ok_a),
                         "fix_Y_only_objective": sol_b.objective, "fix_Y_only_ok": bool(ok_b),
                         "reference_feasible_under_Y_fix": bool(ref_feasible),
                         "sweep": sweep, "monotone": bool(monotone),
                         "tau": tau, "n_Y": len(all_Y)}
    return results


def _is_assignment_feasible(bm, sol: Solution) -> bool:
    """Independently re-solve the continuous problem with the plan's binaries pinned."""
    inst = bm.inst
    assign = assignment_of_solution(sol, inst, bm.mode)
    status, _, _ = solve_lp_with_binaries_fixed(bm, assign)
    return status == "optimal"


# --------------------------------------------------------------------------------------
# diagnostics: where exactly do the two modes diverge?
# --------------------------------------------------------------------------------------


def mode_divergence_diagnostics(max_slots: int = 12) -> List[Dict[str, Any]]:
    """For each hand instance, compare the two modes and, where they differ, enumerate the
    full binary space so the divergence is reproducible evidence rather than an assertion."""
    out = []
    for name, inst in hand_instances().items():
        models = {m: build_model(inst, m) for m in MODES}
        sols = {m: solve_milp(models[m]) for m in MODES}
        entries = []
        for m in MODES:
            bm, sol = models[m], sols[m]
            x = _flatten(bm, sol)
            entries.append({
                "mode": m,
                "objective": sol.objective,
                "plan": plan_string(inst, sol),
                "upper_bounds": {f"i{i},j{j},t{t}": bm.upper_bounds[(i, j, t)]
                                 for i in range(inst.N) for j in range(inst.M)
                                 for t in range(1, inst.T + 1)},
                "zT_fixed_to_zero": m == MODE_AUDITED,
                "row_activity": _row_activity(bm, x),
            })
        same = (not np.isnan(entries[0]["objective"])) and \
            abs(entries[0]["objective"] - entries[1]["objective"]) <= TOL_OBJ

        # full binary enumeration per mode, when small enough
        enum_by_mode: Dict[str, List[Dict[str, Any]]] = {}
        for m in MODES:
            slots = enumeration_space(inst, m)
            if len(slots) > max_slots:
                continue
            rows_here = []
            for bits in itertools.product((0, 1), repeat=len(slots)):
                assign = {slots[k]: int(bits[k]) for k in range(len(slots))}
                st, val, _ = solve_lp_with_binaries_fixed(models[m], assign)
                rows_here.append({
                    "assignment": {("%s%d%d%d" % (s[0], s[1], s[2], s[3])): a
                                   for s, a in assign.items()},
                    "status": st, "objective": val,
                })
            rows_here.sort(key=lambda r: (not np.isfinite(r["objective"]), r["objective"]))
            enum_by_mode[m] = rows_here

        out.append({"instance": name, "identical_objective": bool(same), "modes": entries,
                    "enumeration": enum_by_mode,
                    "notes": {m: models[m].notes for m in MODES}})
    return out


def _flatten(bm, sol: Solution) -> np.ndarray:
    inst = bm.inst
    x = np.zeros(bm.idx.n)
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                x[bm.idx.X(i, j, t)] = sol.X[i, j, t - 1]
                x[bm.idx.Y(i, j, t)] = sol.Y[i, j, t - 1]
                x[bm.idx.Z(i, j, t)] = sol.Z[i, j, t - 1]
        for t in range(1, inst.T + 1):
            x[bm.idx.I(i, t)] = sol.I[i, t - 1]
            x[bm.idx.L(i, t)] = sol.L[i, t - 1]
    return x


def _row_activity(bm, x: np.ndarray) -> Dict[str, Dict[str, float]]:
    ax = bm.A @ x
    out: Dict[str, Dict[str, float]] = {}
    for k, nm in enumerate(bm.row_names):
        lo, hi = bm.row_lb[k], bm.row_ub[k]
        slack_ub = (hi - ax[k]) if np.isfinite(hi) else float("inf")
        slack_lb = (ax[k] - lo) if np.isfinite(lo) else float("inf")
        out[nm] = {"activity": float(ax[k]), "lb": float(lo), "ub": float(hi),
                   "slack_ub": float(slack_ub), "slack_lb": float(slack_lb)}
    return out


# --------------------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------------------


def render_details(hand, enum, fixrel, diag) -> str:
    L: List[str] = []
    L.append("# stage01_details.md\n")
    L.append("Machine-generated detail tables. Produced by `python stage01_run.py`.\n")
    L.append("Environment: python %s, numpy %s, scipy %s, %s\n"
             % (platform.python_version(), np.__version__, scipy.__version__,
                platform.platform()))

    # ---- headline summary, the table requested for the report
    hand_rows = [(n, m) for n, info in hand.items() for m in MODES]
    hand_max = max((info["modes"][m]["max_violation"] for n, info in hand.items()
                    for m in MODES), default=0.0)
    hand_mismatch = [n for n, info in hand.items() for m in MODES
                     if info["modes"][m]["agreement"] == "mismatch"]
    enum_rows = [r for r in enum.values() if "skipped" not in r]
    enum_max = max((r["max_violation"] for r in enum_rows), default=0.0)
    enum_delta = max((r["delta"] for r in enum_rows if np.isfinite(r["delta"])), default=0.0)
    fx_rows = [r for r in fixrel.values() if r.get("reference_ok", True)]
    fx_max = max((r["fix_all_delta"] for r in fx_rows), default=0.0)
    fx_ok = all(r["fix_all_ok"] and r["fix_Y_only_ok"] and r["monotone"] for r in fx_rows)

    L.append("\n## 0. Summary\n")
    L.append("| 检查 | 是否通过 | 最大误差/违反量 | 未解决问题 |")
    L.append("|---|---|---:|---|")
    L.append("| 手工实例 (A-G, 两种模式, %d 次求解) | %s | %.1e | %s |" % (
        len(hand_rows), "PASS" if not hand_mismatch else "FAIL", hand_max,
        "无" if not hand_mismatch else "与手算预期不符: " + ", ".join(hand_mismatch)))
    L.append("| 独立枚举 (%d 实例, MILP vs 全二元枚举+LP) | %s | 目标 %.1e / 约束 %.1e | %s |" % (
        len(enum_rows),
        "PASS" if all(r["agree"] and r["check_ok"] and r["all_zero_witness_ok"]
                      for r in enum_rows) else "FAIL",
        enum_delta, enum_max,
        "无" if all(r["expectation_agreement"] != "mismatch" for r in enum_rows)
        else "回归基线与求解值不符"))
    L.append("| 固定与释放接口 (%d 实例) | %s | %.1e | %s |" % (
        len(fx_rows), "PASS" if fx_ok else "FAIL", fx_max,
        "无。全部 Y 变量的单调性只对最优解成立，"
        "未来限时求解结果不保证单调" ))

    L.append("\n## 1. Hand-checkable instances A-G\n")
    L.append("| instance | mode | objective | hand expectation | agreement | checker | max violation | status |")
    L.append("|---|---|---:|---:|---|---|---:|---|")
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            exp = "-" if m["expectation"] is None else "%g" % m["expectation"]
            L.append("| %s | %s | %.6g | %s | %s | %s | %.1e | %s |" % (
                name, mode, m["objective"], exp, m["agreement"],
                "PASS" if m["check_ok"] else "FAIL", m["max_violation"], m["status"]))
    L.append("\n### Optimal plans\n")
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            L.append("- **%s / %s** (obj %.6g): `%s`" % (name, mode, m["objective"], m["plan"]))
        if info["mode_differs"]:
            L.append("- %s: **the two modes differ** by %g" %
                     (name, info["delta_audited_minus_paper"]))
        else:
            L.append("- %s: same objective in both modes" % name)

    L.append("\n## 2. Independent enumeration (audited_v1)\n")
    L.append("| instance | slots | configs | feasible | MILP obj | enum obj | delta | agree | expected | expectation | checker | max viol | enum time (s) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|---:|")
    for name, r in enum.items():
        if "skipped" in r:
            L.append("| %s | - | - | - | - | - | - | SKIPPED | - | - | - | - | - |" % name)
            continue
        L.append("| %s | %d | %d | %d | %.6g | %.6g | %.2e | %s | %s | %s | %s | %.1e | %.3f |" % (
            name, r["n_slots"], r["n_configs"], r["n_feasible"],
            r["milp_objective"], r["enum_objective"], r["delta"],
            "yes" if r["agree"] else "NO",
            "-" if r["expectation"] is None else "%.6g" % r["expectation"],
            r["expectation_agreement"],
            "PASS" if r["check_ok"] else "FAIL",
            r["max_violation"], r["enum_time_s"]))
    L.append("\n### Optimal plans\n")
    for name, r in enum.items():
        if "plan" in r:
            L.append("- **%s** (obj %.6g): `%s`" % (name, r["milp_objective"], r["plan"]))

    L.append("\n## 3. Fix-and-release interface\n")
    L.append("| instance | free Y | ref obj | fix-all delta | fix-all ok | fix Y only obj | ok | ref feasible | monotone | sweep |")
    L.append("|---|---:|---:|---:|---|---:|---|---|---|---|")
    for name, r in fixrel.items():
        if not r.get("reference_ok", True):
            L.append("| %s | - | - | - | SKIPPED | - | - | - | - | no feasible reference plan |"
                     % name)
            continue
        L.append("| %s | %d | %.6g | %.2e | %s | %.6g | %s | %s | %s | %s |" % (
            name, r["n_Y"], r["reference_objective"], r["fix_all_delta"],
            "PASS" if r["fix_all_ok"] else "FAIL", r["fix_Y_only_objective"],
            "PASS" if r["fix_Y_only_ok"] else "FAIL",
            r["reference_feasible_under_Y_fix"], "yes" if r["monotone"] else "NO",
            "; ".join("k%d=%.6g" % (e["k"], e["objective"]) for e in r["sweep"])))

    L.append("\n## 4. Where the two modes diverge\n")
    for d in diag:
        if d["identical_objective"]:
            continue
        L.append("\n### %s\n" % d["instance"])
        for entry in d["modes"]:
            L.append("- **%s** obj %s: `%s`" % (entry["mode"], _g(entry["objective"]),
                                                entry["plan"]))
            L.append("  - activation upper bounds U_ijt: `%s`" % entry["upper_bounds"])
            L.append("  - Z_ijT pinned to zero: %s" % entry["zT_fixed_to_zero"])
            tight = [nm for nm, v in entry["row_activity"].items()
                     if np.isfinite(v["slack_ub"]) and v["slack_ub"] < 1e-9]
            L.append("  - binding upper-bound rows: `%s`" % tight)
        for mode, table in d["enumeration"].items():
            L.append("\n#### %s / %s : all %d binary configurations, best first\n"
                     % (d["instance"], mode, len(table)))
            L.append("| rank | assignment | optimal objective | status |")
            L.append("|---:|---|---:|---|")
            for rank, r in enumerate(table, 1):
                assign = ", ".join("%s=%d" % (k, v) for k, v in sorted(r["assignment"].items()))
                L.append("| %d | `%s` | %s | %s |" % (rank, assign, _g(r["objective"]),
                                                     r["status"]))
    L.append("\n### Mode-specific notes\n")
    for d in diag:
        for mode in MODES:
            for note in d["notes"][mode]:
                L.append("- %s / %s: %s" % (d["instance"], mode, note))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 01 model audit")
    ap.add_argument("--max-configs", type=int, default=4096,
                    help="cap on enumerated binary configurations per instance")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    rows = Rows(CSV_FIELDS)

    hand = run_hand(rows)
    enum = run_enumeration(rows, args.max_configs)
    fixrel = run_fix_release(rows)
    diag = mode_divergence_diagnostics()

    csv_path = args.outdir + "/stage01_results.csv"
    rows.write(csv_path)
    details_path = args.outdir + "/stage01_details.md"
    with open(details_path, "w", encoding="utf-8") as fh:
        fh.write(render_details(hand, enum, fixrel, diag))

    meta = {
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "solver": "scipy.optimize.milp / linprog -> HiGHS",
        "wall_time_s": time.perf_counter() - t_start,
        "hand": hand, "enum": enum, "fixrelease": fixrel,
    }
    with open(args.outdir + "/stage01_raw.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ---- console summary
    print("=" * 78)
    print("hand instances: expectation agreement")
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            print("  %-16s %-14s obj=%-10.6g expect=%-6s %-11s checker=%s" % (
                name, mode, m["objective"],
                "-" if m["expectation"] is None else "%g" % m["expectation"],
                m["agreement"], "PASS" if m["check_ok"] else "FAIL"))
    print("\nenumeration vs MILP (audited_v1)")
    for name, r in enum.items():
        if "skipped" in r:
            print("  %-30s SKIPPED (%s)" % (name, r["skipped"]))
        else:
            print("  %-30s milp=%-10.6g enum=%-10.6g delta=%.2e agree=%s checker=%s expect=%s/%s witness=%s" % (
                name, r["milp_objective"], r["enum_objective"], r["delta"],
                r["agree"], r["check_ok"],
                "-" if r["expectation"] is None else "%.6g" % r["expectation"],
                r["expectation_agreement"], r["all_zero_witness_ok"]))
    print("\nfix-and-release")
    for name, r in fixrel.items():
        if not r.get("reference_ok", True):
            print("  %-30s SKIPPED (no feasible reference plan)" % name)
            continue
        print("  %-30s fix-all=%s fix-Y=%s monotone=%s" % (
            name, r["fix_all_ok"], r["fix_Y_only_ok"], r["monotone"]))
    print("\nwrote: %s\n       %s\n       %s" % (csv_path, details_path,
                                                args.outdir + "/stage01_raw.json"))
    print("total wall time: %.2fs" % (time.perf_counter() - t_start))
    print("=" * 78)

    failures = 0
    for name, info in hand.items():
        for mode in MODES:
            m = info["modes"][mode]
            if not m["check_ok"] or m["agreement"] == "mismatch":
                failures += 1
    for r in enum.values():
        if "skipped" in r:
            continue
        if not r["agree"] or not r["all_zero_witness_ok"]:
            failures += 1
    for r in fixrel.values():
        if not r.get("reference_ok", True):
            continue
        if not (r["fix_all_ok"] and r["fix_Y_only_ok"] and r["monotone"]):
            failures += 1
    print("hard failures: %d" % failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
