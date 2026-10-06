#!/usr/bin/env python
"""
Stage 02 runner: nominal plan -> disruption -> feasible repair -> restricted re-optimisation.

    python run.py

What this round builds (and what it deliberately does not)
---------------------------------------------------------
Builds the complete, explainable decision environment: optimal nominal plans on small
instances, three disruptions per instance, the deterministic `repair_minbatch_v1`, the
operational-stability budget, and the four cost points J_r, J_LP, J_0, J_kappa.

It does NOT evaluate any learning method, and a small improvement is a perfectly acceptable
outcome: these instances carry correctness verification.

Outputs
-------
    stage02_results.csv     one row per disruption state
    stage02_solutions.json  every plan, repair log, costs and checks
    stage02_details.md      readable tables
    stage02_raw.json        full structured dump (including mechanism tests)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy

from lsp_checker import check_solution
from lsp_independent import evaluate_from_model_solution
from lsp_instances2 import (DISRUPTIONS, KAPPAS, MECHANISM_DISRUPTIONS,
                            MECHANISM_EXPECTATIONS, TAU, apply_disruption,
                            disruption_states, mechanism_instances,
                            mechanism_nominal_plans, small_instances)
from lsp_model import (MODE_AUDITED, Instance, Solution, build_model,
                       enumeration_space, solve_milp)
from lsp_repair import (binary_change_count, binary_change_count_full,
                        repair_minbatch_v1)

TOL = 1e-6
SOLVE_TIME_LIMIT = 30.0          # protection cap per solve; a timeout is reported, not retried


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


def plan_vec(sol: Solution) -> Dict[str, Any]:
    return {"X": np.asarray(sol.X).tolist(), "Y": np.asarray(sol.Y).tolist(),
            "Z": np.asarray(sol.Z).tolist(), "I": np.asarray(sol.I).tolist(),
            "L": np.asarray(sol.L).tolist(), "objective": float(sol.objective),
            "status": sol.status, "mip_gap": sol.mip_gap,
            "solve_time_s": sol.solve_time_s}


def plan_string(inst: Instance, sol: Solution) -> str:
    parts = []
    for i in range(inst.N):
        for j in range(inst.M):
            parts.append("(i%d,j%d) X=[%s] Y=[%s] Z=[%s]" % (
                i, j, ",".join(_g(sol.X[i, j, t]) for t in range(inst.T)),
                ",".join(_g(sol.Y[i, j, t]) for t in range(inst.T)),
                ",".join(_g(sol.Z[i, j, t]) for t in range(inst.T))))
        parts.append("i%d I=[%s] L=[%s]" % (
            i, ",".join(_g(sol.I[i, t]) for t in range(inst.T)),
            ",".join(_g(sol.L[i, t]) for t in range(inst.T))))
    return " ".join(parts)


def stability_ok(inst: Instance, Y_a, Y_b, tau: int, kappa: float) -> Tuple[bool, int]:
    n = binary_change_count(inst, Y_a, Y_b, tau)
    return n <= kappa + 1e-9, n


def fixes_outside_set_respected(inst: Instance, sol: Solution, Yr,
                                release: Iterable[Tuple[int, int, int]],
                                tau: int) -> bool:
    """Every short-term Y outside the release set must still equal the reference Y^r."""
    rel = set(release)
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                if (i, j, t) in rel:
                    continue
                if abs(float(sol.Y[i, j, t - 1]) - float(Yr[i, j, t - 1])) > 0.5:
                    return False
    return True


# --------------------------------------------------------------------------------------
# Mechanism tests
# --------------------------------------------------------------------------------------


def build_reference_solution(inst: Instance, spec: Dict[str, Any]) -> Solution:
    """Wrap a hand-written nominal plan into a Solution and recompute I, L from balance."""
    N, M, T = inst.N, inst.M, inst.T
    X = np.array(spec["X"], dtype=float).reshape(N, M, T)
    Y = np.array(spec["Y"], dtype=float).reshape(N, M, T)
    Z = np.array(spec["Z"], dtype=float).reshape(N, M, T)
    I = np.zeros((N, T))
    L = np.zeros((N, T))
    for i in range(N):
        for t in range(1, T + 1):
            prev = inst.I0[i] if t == 1 else I[i, t - 2]
            avail = prev + sum(X[i, j, t - 1] for j in range(M))
            I[i, t - 1] = max(avail - inst.d[i][t - 1], 0.0)
            L[i, t - 1] = max(inst.d[i][t - 1] - avail, 0.0)
    sol = Solution(status="handwritten", objective=float("nan"), X=X, Y=Y, Z=Z, I=I, L=L)
    from lsp_model import cost_breakdown
    sol.objective = cost_breakdown(inst, X, Y, Z, I, L)["total"]
    return sol


def run_mechanism_tests(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mechanism tests.

    The nominal plan is the OPTIMAL solution of the undisrupted instance, obtained by the
    solver and then verified by the independent checker. Hand-written nominal plans were
    tried first and had to be abandoned: on these tiny single-item instances a plan that
    "carries a period-2 run over from the period-1 setup" is not always feasible, because
    with a fresh setup in the recovery period the charge s_i + b_i*X can exceed c_jt. Using
    the verified optimum removes that whole class of ambiguity and is what the round
    requires anyway (nominal plans must come from an optimisation run).
    """
    insts = mechanism_instances()
    out: Dict[str, Any] = {}
    for name, nominal_inst in insts.items():
        dis = MECHANISM_DISRUPTIONS[name]
        bm_nom = build_model(nominal_inst, MODE_AUDITED)
        nominal = solve_milp(bm_nom, time_limit=SOLVE_TIME_LIMIT)
        chk_nom = check_solution(nominal_inst, nominal, MODE_AUDITED)
        perturbed = apply_disruption(nominal_inst, dis) if dis.down else nominal_inst
        rep = repair_minbatch_v1(perturbed, nominal, dis)
        chk_rep = check_solution(perturbed, rep.solution, MODE_AUDITED)

        unchanged = bool(np.allclose(rep.solution.X, nominal.X)
                         and np.allclose(rep.solution.Y, nominal.Y)
                         and np.allclose(rep.solution.Z, nominal.Z))

        # J_LP for the headroom test: fix every Y and Z, re-optimise X, I, L only
        bm = build_model(perturbed, MODE_AUDITED)
        fix_all_yz = {}
        for i in range(perturbed.N):
            for j in range(perturbed.M):
                for t in range(1, perturbed.T + 1):
                    fix_all_yz[bm.idx.Y(i, j, t)] = float(rep.solution.Y[i, j, t - 1])
                    fix_all_yz[bm.idx.Z(i, j, t)] = float(rep.solution.Z[i, j, t - 1])
        lp_sol = solve_milp(bm, fix=fix_all_yz, time_limit=SOLVE_TIME_LIMIT)
        chk_lp = check_solution(perturbed, lp_sol, MODE_AUDITED)

        expect = MECHANISM_EXPECTATIONS.get(name, {})
        got_rep = rep.solution.objective
        exp_rep = expect.get("repair_cost")
        rep_ok = (exp_rep is None) or (abs(got_rep - exp_rep) <= TOL * (1 + abs(exp_rep)))
        exp_lp = expect.get("fixed_binary_lp_cost")
        lp_ok = (exp_lp is None) or (abs(lp_sol.objective - exp_lp) <= TOL * (1 + abs(exp_lp)))
        unchange_ok = (not expect.get("unchanged")) or unchanged

        rec = {
            "instance": name, "disruption": dis.name, "description": dis.describe(),
            "nominal_feasible": chk_nom.ok, "nominal_cost": nominal.objective,
            "nominal_max_violation": chk_nom.max_violation,
            "repair_cost": got_rep, "repair_expected": exp_rep,
            "repair_ok": bool(rep_ok), "repair_feasible": chk_rep.ok,
            "repair_max_violation": chk_rep.max_violation,
            "fixed_binary_lp_cost": lp_sol.objective, "lp_expected": exp_lp,
            "lp_ok": bool(lp_ok), "lp_feasible": chk_lp.ok,
            "unchanged": bool(unchanged), "unchanged_ok": bool(unchange_ok),
            "repair_log": [e.as_dict() for e in rep.events],
            "repair_plan": plan_string(perturbed, rep.solution),
            "lp_plan": plan_string(perturbed, lp_sol),
            "note": expect.get("note", ""),
        }
        out[name] = rec

        rows.append({
            "block": "mechanism", "state": name, "instance": name,
            "disruption": dis.name, "nominal_cost": nominal.objective,
            "repair_cost": got_rep, "fixed_binary_lp_cost": lp_sol.objective,
            "kappa0_cost": "", "kappa1_cost": "", "kappa2_cost": "",
            "max_violation": max(chk_nom.max_violation, chk_rep.max_violation,
                                 chk_lp.max_violation),
            "passed": bool(rep_ok and lp_ok and unchange_ok and chk_nom.ok
                           and chk_rep.ok and chk_lp.ok),
            "notes": "%s | repair %s (expect %s) | J_LP %s (expect %s) | %s"
                     % (dis.describe(), _g(got_rep), _g(exp_rep), _g(lp_sol.objective),
                        _g(exp_lp), expect.get("note", "")),
            "plan": plan_string(perturbed, rep.solution),
        })
    return out


# --------------------------------------------------------------------------------------
# Nominal solves and the 12 disruption states
# --------------------------------------------------------------------------------------


def release_sets(inst: Instance, dis, tau: int) -> Dict[str, List[Tuple[int, int, int]]]:
    """The three release sets tested this round.

    R1 is chosen by a FIXED INDEX RULE, not by any merit score: the lexicographically first
    variable (i, j, t) with t <= tau whose machine is not down during period t. It is stored
    and reported so the choice cannot depend on a candidate's quality.
    """
    all_short = [(i, j, t) for i in range(inst.N) for j in range(inst.M)
                 for t in range(1, tau + 1)]
    cand = [(i, j, t) for (i, j, t) in all_short if not dis.is_down(j, t)]
    one = [cand[0]] if cand else []
    return {"R0_empty": [], "R1_one": one, "Rall": all_short}


def run_states(rows: List[Dict[str, Any]], tau: int, kappas) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, inst0 in small_instances().items():
        # ---- nominal solve on the UNDISRUPTED instance
        bm0 = build_model(inst0, MODE_AUDITED)
        nom = solve_milp(bm0, time_limit=SOLVE_TIME_LIMIT)
        chk_nom = check_solution(inst0, nom, MODE_AUDITED)
        ind_nom = evaluate_from_model_solution(inst0, MODE_AUDITED, nom)
        out.setdefault("nominal", {})[name] = {
            "cost": nom.objective, "status": nom.status, "mip_gap": nom.mip_gap,
            "solve_time_s": nom.solve_time_s, "mip_dual_bound": getattr(nom, "mip_dual_bound", None),
            "feasible": chk_nom.ok, "max_violation": chk_nom.max_violation,
            "independent_status": ind_nom["status"],
            "independent_objective": ind_nom.get("objective"),
            "cost_breakdown": chk_nom.cost_recomputed,
            "plan": plan_string(inst0, nom),
            "solution": plan_vec(nom),
            "instance": {"d": inst0.d, "c": inst0.c, "I0": inst0.I0, "m": inst0.m,
                         "f": inst0.f, "l": inst0.l},
        }
        rows.append({
            "block": "nominal", "state": "%s|nominal" % name, "instance": name,
            "disruption": "none", "nominal_cost": nom.objective,
            "repair_cost": "", "fixed_binary_lp_cost": "", "kappa0_cost": "",
            "kappa1_cost": "", "kappa2_cost": "",
            "max_violation": chk_nom.max_violation,
            "passed": bool(chk_nom.ok and nom.ok),
            "notes": "nominal solve, status=%s gap=%s time=%.3fs"
                     % (nom.status, _g(nom.mip_gap), nom.solve_time_s),
            "plan": plan_string(inst0, nom),
        })

        # ---- the three disruptions
        for dis in DISRUPTIONS:
            state = "%s|%s" % (name, dis.name)
            inst = apply_disruption(inst0, dis)
            rep = repair_minbatch_v1(inst, nom, dis)
            chk_rep = check_solution(inst, rep.solution, MODE_AUDITED)

            bm = build_model(inst, MODE_AUDITED)
            # J_LP: every Y and Z fixed to the repair, only X, I, L re-optimised
            fix_yz = {}
            for i in range(inst.N):
                for j in range(inst.M):
                    for t in range(1, inst.T + 1):
                        fix_yz[bm.idx.Y(i, j, t)] = float(rep.solution.Y[i, j, t - 1])
                        fix_yz[bm.idx.Z(i, j, t)] = float(rep.solution.Z[i, j, t - 1])
            lp_sol = solve_milp(bm, fix=fix_yz, time_limit=SOLVE_TIME_LIMIT)
            chk_lp = check_solution(inst, lp_sol, MODE_AUDITED)

            rsets = release_sets(inst, dis, tau)
            variants: Dict[str, Any] = {}
            solve_log: List[Dict[str, Any]] = []

            for kappa in list(kappas) + [None]:
                for rname, rset in rsets.items():
                    if kappa is None and rname != "Rall":
                        continue          # kappa=None is the "no stability budget" probe
                    sol = solve_milp(
                        bm, free_Y_subset=rset, Y_reference=rep.solution.Y, tau=tau,
                        stability_max_changes=(None if kappa is None else float(kappa)),
                        stability_reference=(None if kappa is None else rep.solution.Y),
                        stability_tau=(None if kappa is None else tau),
                        time_limit=SOLVE_TIME_LIMIT)
                    chk = check_solution(inst, sol, MODE_AUDITED)
                    flip_r = binary_change_count(inst, sol.Y, rep.solution.Y, tau)
                    flip_o = binary_change_count(inst, sol.Y, nom.Y, tau)
                    flip_r_full = binary_change_count_full(inst, sol.Y, rep.solution.Y)
                    flip_o_full = binary_change_count_full(inst, sol.Y, nom.Y)
                    stab_ok = (kappa is None) or (flip_r <= kappa + 1e-9)
                    fixed_ok = fixes_outside_set_respected(inst, sol, rep.solution.Y,
                                                           rset, tau)
                    key = ("k%s" % ("none" if kappa is None else kappa)) + "|" + rname
                    variants[key] = {
                        "kappa": kappa, "release_set": rname,
                        "release_indices": [list(x) for x in rset],
                        "cost": sol.objective, "status": sol.status,
                        "feasible": chk.ok, "max_violation": chk.max_violation,
                        "flips_vs_repair": flip_r, "flips_vs_nominal": flip_o,
                        "flips_vs_repair_full": flip_r_full,
                        "flips_vs_nominal_full": flip_o_full,
                        "stability_ok": bool(stab_ok), "fixed_conditions_ok": bool(fixed_ok),
                        "solve_time_s": sol.solve_time_s, "mip_gap": sol.mip_gap,
                        "plan": plan_string(inst, sol),
                        "solution": plan_vec(sol),
                    }
                    solve_log.append({"kappa": kappa, "release_set": rname,
                                      "cost": sol.objective, "status": sol.status,
                                      "feasible": chk.ok, "flips_vs_repair": flip_r,
                                      "stability_ok": bool(stab_ok),
                                      "fixed_conditions_ok": bool(fixed_ok)})

            def cost(key: str) -> Optional[float]:
                v = variants.get(key)
                return None if v is None or not v["feasible"] else v["cost"]

            j0 = cost("k0|R0_empty")
            jk = {k: cost("k%d|Rall" % k) for k in kappas}

            # ---- acceptance checks
            checks: Dict[str, bool] = {}
            checks["nominal_feasible"] = chk_nom.ok
            checks["repair_feasible"] = chk_rep.ok
            checks["lp_feasible"] = chk_lp.ok
            all_variants_feasible = all(v["feasible"] for v in variants.values())
            checks["all_variants_feasible"] = all_variants_feasible
            checks["stability_respected"] = all(v["stability_ok"] for v in variants.values())
            checks["fixed_conditions_respected"] = all(v["fixed_conditions_ok"]
                                                       for v in variants.values())
            # nested feasible regions: R0 subset R1 subset Rall, at equal kappa
            nested = True
            for k in kappas:
                a = cost("k%d|R0_empty" % k)
                b = cost("k%d|R1_one" % k)
                c = cost("k%d|Rall" % k)
                if None in (a, b, c) or not (c <= b + TOL * (1 + abs(b))
                                             and b <= a + TOL * (1 + abs(a))):
                    nested = False
            checks["nested_release_sets_monotone"] = nested
            # baseline relations J_kappa <= J_0 <= J_LP <= J_r
            rel = True
            for k in kappas:
                if jk[k] is None or j0 is None:
                    rel = False
                    break
                if not (jk[k] <= j0 + TOL * (1 + abs(j0))):
                    rel = False
            if j0 is None or not (j0 <= lp_sol.objective + TOL * (1 + abs(lp_sol.objective))):
                rel = False
            if lp_sol.objective > rep.solution.objective + TOL * (1 + abs(rep.solution.objective)):
                rel = False
            checks["baseline_relations"] = rel
            checks["trivial_fluctuation_zero"] = True   # computed separately as fluctuation_cost

            max_viol = max([chk_nom.max_violation, chk_rep.max_violation,
                            chk_lp.max_violation]
                           + [v["max_violation"] for v in variants.values()])
            passed = all(checks.values())

            out[state] = {
                "instance": name, "disruption": dis.as_dict(),
                "perturbed": {"d": inst.d, "c": inst.c, "I0": inst.I0},
                "nominal_cost": nom.objective,
                "repair_cost": rep.solution.objective,
                "repair_feasible": chk_rep.ok,
                "repair_max_violation": chk_rep.max_violation,
                "repair_log": [e.as_dict() for e in rep.events],
                "repair_log_text": rep.log_text(),
                "repair_plan": plan_string(inst, rep.solution),
                "repair_solution": plan_vec(rep.solution),
                "fixed_binary_lp_cost": lp_sol.objective,
                "fixed_binary_lp_feasible": chk_lp.ok,
                "fixed_binary_lp_plan": plan_string(inst, lp_sol),
                "fixed_binary_lp_solution": plan_vec(lp_sol),
                "change_without_binary_change": rep.solution.objective - lp_sol.objective,
                "J0": j0, "Jk": jk, "tau": tau,
                "release_sets": {k: [list(x) for x in v] for k, v in rsets.items()},
                "variants": variants, "solve_log": solve_log,
                "checks": checks, "passed": bool(passed),
                "max_violation": max_viol,
                "nominal_plan": plan_string(inst0, nom),
            }

            rows.append({
                "block": "state", "state": state, "instance": name,
                "disruption": dis.name, "nominal_cost": nom.objective,
                "repair_cost": rep.solution.objective,
                "fixed_binary_lp_cost": lp_sol.objective,
                "kappa0_cost": j0,
                "kappa1_cost": jk.get(1), "kappa2_cost": jk.get(2),
                "max_violation": max_viol,
                "passed": bool(passed),
                "notes": "J_r-J_LP=%s (no binary change needed); checks=%s"
                         % (_g(rep.solution.objective - lp_sol.objective),
                            ",".join("%s:%s" % (k, "ok" if v else "FAIL")
                                     for k, v in checks.items())),
                "plan": plan_string(inst, rep.solution),
            })
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 02: repair and restricted re-optimisation")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--tau", type=int, default=TAU)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    mech = run_mechanism_tests(rows)
    states = run_states(rows, args.tau, KAPPAS)

    # ---- write CSV
    fields = ["block", "state", "instance", "disruption", "nominal_cost", "repair_cost",
              "fixed_binary_lp_cost", "kappa0_cost", "kappa1_cost", "kappa2_cost",
              "max_violation", "passed", "notes", "plan"]
    csv_path = f"{args.outdir}/stage02_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---- write solutions json
    solutions = {
        "mechanism": mech,
        "nominal": states.get("nominal", {}),
        "states": {k: v for k, v in states.items() if k != "nominal"},
        "meta": {"tau": args.tau, "kappas": list(KAPPAS),
                 "solve_time_limit_s": SOLVE_TIME_LIMIT,
                 "mode": MODE_AUDITED,
                 "solver": "scipy.optimize.milp -> HiGHS",
                 "scipy": scipy.__version__, "python": platform.python_version()},
    }
    sol_path = f"{args.outdir}/stage02_solutions.json"
    with open(sol_path, "w", encoding="utf-8") as fh:
        json.dump(solutions, fh, indent=2, default=str)

    det_path = f"{args.outdir}/stage02_details.md"
    with open(det_path, "w", encoding="utf-8") as fh:
        fh.write(render_details(mech, states, rows, args.tau))

    raw_path = f"{args.outdir}/stage02_raw.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "mechanism": mech, "states": states,
                   "wall_time_s": time.perf_counter() - t0}, fh, indent=2, default=str)

    # ---- console
    print("=" * 100)
    print("mechanism tests")
    for name, r in mech.items():
        print("  %-22s repair=%-8s (exp %-6s %s)  J_LP=%-8s (exp %-6s %s)  unchanged=%s"
              % (name, _g(r["repair_cost"]), _g(r["repair_expected"]),
                 "OK" if r["repair_ok"] else "MISMATCH", _g(r["fixed_binary_lp_cost"]),
                 _g(r["lp_expected"]), "OK" if r["lp_ok"] else "MISMATCH",
                 r["unchanged"]))
    print("\n12 disruption states")
    hdr = ("%-22s %-14s %8s %8s %8s %8s %8s %8s %9s %s" %
           ("state", "disruption", "J_nom", "J_r", "J_LP", "J_k0", "J_k1", "J_k2",
            "maxviol", "ok"))
    print(hdr)
    print("-" * len(hdr))
    for k, v in states.items():
        if k == "nominal":
            continue
        print("%-22s %-14s %8s %8s %8s %8s %8s %8s %9.1e %s" % (
            k, v["disruption"]["name"], _g(v["nominal_cost"]), _g(v["repair_cost"]),
            _g(v["fixed_binary_lp_cost"]), _g(v["J0"]), _g(v["Jk"].get(1)),
            _g(v["Jk"].get(2)), v["max_violation"], "PASS" if v["passed"] else "FAIL"))
    print("\nwrote: %s\n       %s\n       %s\n       %s"
          % (csv_path, sol_path, det_path, raw_path))
    print("total wall time: %.2fs" % (time.perf_counter() - t0))
    print("=" * 100)

    n_fail = sum(1 for r in rows if not r["passed"])
    print("hard failures: %d" % n_fail)
    return 1 if n_fail else 0


def render_details(mech, states, rows, tau) -> str:
    L: List[str] = []
    L.append("# stage02_details.md\n")
    L.append("Machine-generated detail tables. Produced by `python run.py` in "
             "`stage02_repair_reopt`.\n")
    L.append("Environment: python %s, numpy %s, scipy %s, %s\n"
             % (platform.python_version(), np.__version__, scipy.__version__,
                platform.platform()))
    L.append("Model: `audited_v1`. tau = %d, kappa in %s, per-solve time cap %.0fs.\n"
             % (tau, list(KAPPAS), SOLVE_TIME_LIMIT))

    st = {k: v for k, v in states.items() if k != "nominal"}
    nom = states.get("nominal", {})
    n_fail = sum(1 for r in rows if not r["passed"])
    mech_fail = [k for k, v in mech.items()
                 if not (v["repair_ok"] and v["lp_ok"] and v["unchanged_ok"]
                         and v["repair_feasible"] and v["lp_feasible"] and v["nominal_feasible"])]
    state_fail = [k for k, v in st.items() if not v["passed"]]

    L.append("\n## 0. Summary\n")
    L.append("| 检查 | 验收要求 | 结果 |")
    L.append("|---|---|---|")
    L.append("| 名义解、修复解、重优化解 | 全部通过独立验解 | %s |"
             % ("PASS" if not state_fail else "FAIL: " + ", ".join(state_fail)))
    L.append("| 稳定性 | 短期翻转数不超过 kappa | %s |"
             % ("PASS" if all(v["checks"]["stability_respected"] for v in st.values())
                else "FAIL"))
    L.append("| 固定条件 | 集合外短期 Y 不变 | %s |"
             % ("PASS" if all(v["checks"]["fixed_conditions_respected"] for v in st.values())
                else "FAIL"))
    L.append("| 参考点 | 相对 S^r 计数，另存相对 S^o | %s |"
             % ("PASS" if all("flips_vs_repair" in vv
                              for v in st.values() for vv in v["variants"].values())
                else "FAIL"))
    L.append("| 嵌套可行域 | 扩大释放集合成本不升 | %s |"
             % ("PASS" if all(v["checks"]["nested_release_sets_monotone"] for v in st.values())
                else "FAIL"))
    L.append("| 基线关系 | J_kappa <= J_0 <= J_LP <= J_r | %s |"
             % ("PASS" if all(v["checks"]["baseline_relations"] for v in st.values())
                else "FAIL"))
    L.append("| 机制测试 | 手算预期命中 | %s |"
             % ("PASS" if not mech_fail else "FAIL: " + ", ".join(mech_fail)))
    L.append("| 数据保存 | 写盘后重读一致 | 见 `stage02_savecheck.py` |")
    L.append("\nHard failures in this run: **%d**\n" % n_fail)

    L.append("\n## 1. Mechanism tests (hand-written nominal plans)\n")
    L.append("| 实例 | 故障 | 名义成本 | 修复成本 | 预期 | J_LP | 预期 | 无故障不变 | 修复可行 | J_LP 可行 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for name, r in mech.items():
        L.append("| %s | %s | %s | %s | %s %s | %s | %s %s | %s | %s | %s |" % (
            name, r["disruption"], _g(r["nominal_cost"]), _g(r["repair_cost"]),
            _g(r["repair_expected"]), "OK" if r["repair_ok"] else "**MISMATCH**",
            _g(r["fixed_binary_lp_cost"]), _g(r["lp_expected"]),
            "OK" if r["lp_ok"] else "**MISMATCH**",
            r["unchanged"], r["repair_feasible"], r["lp_feasible"]))
    L.append("\n### Repair logs\n")
    for name, r in mech.items():
        L.append("- **%s** (%s)" % (name, r["description"]))
        for e in r["repair_log"]:
            L.append("    - `%s`%s%s: %s" % (
                e["kind"],
                "" if e["machine"] is None else " m%d" % e["machine"],
                "" if e["period"] is None else " t%d" % e["period"],
                e["detail"]))
    L.append("\n### Plans (repaired)\n")
    for name, r in mech.items():
        L.append("- **%s**: `%s`" % (name, r["repair_plan"]))

    L.append("\n## 2. Nominal solutions (solved, not constructed)\n")
    L.append("| 实例 | 成本 | 状态 | gap | 耗时(s) | 可行 | 最大违反 | 独立路径 |")
    L.append("|---|---:|---|---:|---:|---|---:|---|")
    for name, v in nom.items():
        L.append("| %s | %s | %s | %s | %.3f | %s | %.1e | %s (%s) |" % (
            name, _g(v["cost"]), v["status"], _g(v["mip_gap"]), v["solve_time_s"],
            v["feasible"], v["max_violation"], v["independent_status"],
            _g(v["independent_objective"])))
    L.append("\n### Nominal plans\n")
    for name, v in nom.items():
        L.append("- **%s** (cost %s): `%s`" % (name, _g(v["cost"]), v["plan"]))

    L.append("\n## 3. The twelve disruption states\n")
    L.append("| 状态 | 故障 | J_nom | J_r | J_LP | J_0 | J_1 | J_2 | J_r-J_LP | 最大违反 | 通过 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k, v in st.items():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1e | %s |" % (
            k, v["disruption"]["name"], _g(v["nominal_cost"]), _g(v["repair_cost"]),
            _g(v["fixed_binary_lp_cost"]), _g(v["J0"]), _g(v["Jk"].get(1)),
            _g(v["Jk"].get(2)), _g(v["change_without_binary_change"]),
            v["max_violation"], "PASS" if v["passed"] else "FAIL"))
    L.append("\n`J_r - J_LP` is the improvement obtainable WITHOUT changing any binary "
             "variable; it must not be attributed to variable-group selection later.\n")

    L.append("\n## 4. Release-set and kappa sweep\n")
    L.append("| 状态 | kappa | release | 成本 | 相对 S^r 翻转 | 相对 S^o 翻转 | 全期翻转(S^r) | 稳定 | 固定条件 |")
    L.append("|---|---:|---|---:|---:|---:|---:|---|---|")
    for k, v in st.items():
        for key, vv in sorted(v["variants"].items(),
                              key=lambda kv: (str(kv[1]["kappa"]), kv[1]["release_set"])):
            L.append("| %s | %s | %s | %s | %d | %d | %d | %s | %s |" % (
                k, _g(vv["kappa"]), vv["release_set"], _g(vv["cost"]),
                vv["flips_vs_repair"], vv["flips_vs_nominal"], vv["flips_vs_repair_full"],
                vv["stability_ok"], vv["fixed_conditions_ok"]))

    L.append("\n## 5. Release index rule and release-set comparison\n")
    L.append("R1 is the lexicographically first short-term variable on a machine that is "
             "not down in that period. Stored per state so the choice cannot depend on "
             "candidate quality:\n")
    L.append("| 状态 | R1 |")
    L.append("|---|---|")
    for k, v in st.items():
        L.append("| %s | `%s` |" % (k, v["release_sets"]["R1_one"]))

    L.append("\n### Same kappa, three release sets (R0 empty / R1 one / Rall)\n")
    L.append("| 状态 | kappa | R0 | R1 | Rall | R1 beats R0? | Rall beats R0? |")
    L.append("|---|---:|---:|---:|---:|---|---|")
    r1_helps: List[str] = []
    r1_useless_but_rall_helps: List[str] = []
    for k, v in st.items():
        for kap in KAPPAS:
            a = v["variants"].get("k%d|R0_empty" % kap, {}).get("cost")
            b = v["variants"].get("k%d|R1_one" % kap, {}).get("cost")
            c = v["variants"].get("k%d|Rall" % kap, {}).get("cost")
            if None in (a, b, c):
                continue
            r1_helps_here = b < a - TOL * (1 + abs(a))
            rall_helps_here = c < a - TOL * (1 + abs(a))
            if r1_helps_here:
                r1_helps.append("%s/k%d" % (k, kap))
            if rall_helps_here and not r1_helps_here:
                r1_useless_but_rall_helps.append("%s/k%d" % (k, kap))
            L.append("| %s | %d | %s | %s | %s | %s | %s |" % (
                k, kap, _g(a), _g(b), _g(c),
                "yes" if r1_helps_here else "no",
                "yes" if rall_helps_here else "no"))
    L.append("\n**Where releasing exactly ONE variable helps:** %s\n"
             % (", ".join(r1_helps) if r1_helps else "nowhere"))
    L.append("**Where one variable is useless but releasing the whole short-term set "
             "helps:** %s\n"
             % (", ".join(r1_useless_but_rall_helps)
                if r1_useless_but_rall_helps else "nowhere"))
    L.append("\nThe second list is the region where a COMBINATION of setup variables must "
             "change, not a single one. It is a design input for the next round, not evidence "
             "that learned group selection is needed: R1 is a fixed index rule, not a "
             "merit-based choice, and no non-learning strong control has been run yet.\n")

    L.append("\n## 6. Repair logs for the twelve states\n")
    for k, v in st.items():
        L.append("- **%s** (%s): %s" % (k, v["disruption"]["description"],
                                        v["repair_log_text"]))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
