"""Stage 16 runner: AB vs FULL vs RINS-LP-ONE on the 32 frozen Stage 15 states.

Scope: 32 states x 3 methods x 1 solver seed = 96 main runs, 20 s total budget each.
This is a BASELINE SCREEN on already-developed states, NOT a new independent confirmation.

Protocol (identical to Stage 14/15 apart from the methods):
* every run computes its OWN continuous polishing solution S_LP;
* AB and FULL are RE-RUN inside this batch -- no old-batch objective values are reused in the
  main comparison;
* the stability row and the fixing reference are always relative to the original S_r;
* end-to-end timing: LP, variable selection, assembly, submission and verification all count.

RINS-LP-ONE (single LP-guided neighbourhood, NOT a reproduction of the full RINS algorithm):
  1. solve and verify S_LP;
  2. solve the FULL LP relaxation (integrality dropped, structural fixings + stability row
     kept) within LP_BUDGET, counted against the same 20 s;
  3. fix every non-structural binary v with |x^LP_v - S_{r,v}| <= RINS_TOL to S_{r,v};
     everything else stays free, continuous variables stay free;
  4. submit S_LP as the MIP start and solve the neighbourhood with the remaining time.
If no valid OPTIMAL relaxation is obtained, fall back to FULL with the remaining budget and
mark the run `fallback` explicitly.  Interface/model errors abort the batch instead.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED, build_model
from lsp_release import (LP_BUDGET, RINS_TOL, build_state_context, fixed_columns,
                         pinned_bounds, solve_lp_fixed, solve_lp_relax, solve_mip,
                         stability_row_local, stage16_rins_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record

BUDGET = 20.0
SEED = 0
# Interleaving is FIXED here, before any result exists.  Rotating the starting method per state
# spreads each method across the batch without ever reacting to outcomes.
METHOD_CYCLE = ("AB", "RINS-LP-ONE", "FULL")
STATES_FILE = "stage15_states.json"
RUNS_FILE = "stage16_runs.json"
CKPT_FILE = "stage16_runs.partial.json"


def disruption_of(rec) -> Disruption:
    d = rec["disruption"]
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


def method_order_for(state_index: int) -> List[str]:
    k = len(METHOD_CYCLE)
    off = state_index % k
    return [METHOD_CYCLE[(off + i) % k] for i in range(k)]


def y_fixings_of(C: Dict[str, Any], fixed: Dict[int, float]) -> Dict[tuple, float]:
    """The short-term-Y fixings actually imposed by THIS run, as (i,j,t) -> value.

    `evaluate_candidate` validates a candidate against this dict, so it must describe the run's
    own fixing set -- otherwise a legal solution is reported invalid and the delivered plan is
    silently downgraded to a worse candidate.
    """
    bm = C["bm"]
    out: Dict[tuple, float] = {}
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, C["tau"] + 1):
                col = bm.idx.Y(i, j, t)
                if col in fixed:
                    out[(i, j, t)] = float(fixed[col])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N runs (debug only)")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)

    plan: List[tuple] = []
    for idx, st in enumerate(states):
        for m in method_order_for(idx):
            plan.append((st, m, SEED))
    print("states: %d | methods %s | seed %d | MIPs %d | interleave: %s"
          % (len(states), list(METHOD_CYCLE), SEED, len(plan),
             "rotate start by state index over %s" % (METHOD_CYCLE,)))

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
            json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHOD_CYCLE),
                                "seed": SEED, "n_runs": len(runs)},
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

    os.makedirs("stage16_logs", exist_ok=True)
    t_all = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = float(repair.objective)
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), method, seed)

        # ---- phase 0: this run's own S_LP -------------------------------------------------
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage16_logs", tag + "__lp.log"))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else None)
        j_lp = lp["objective"]

        stab = stability_row_local(bm, repair, tau, KAPPA)
        relaxed: Optional[Dict[str, Any]] = None
        sel: Dict[str, Any] = {}
        fixed: Dict[int, float] = {}

        if method == "AB":
            _, fixed = fixed_columns(C, "AB")
            sel = {"rule": "AB", "n_fixed_total": len(fixed),
                   "n_free_binary_after": len(C["assign"]) - len(fixed)}
        elif method == "FULL":
            _, fixed = fixed_columns(C, "FULL")
            sel = {"rule": "FULL", "n_fixed_total": len(fixed),
                   "n_free_binary_after": len(C["assign"]) - len(fixed)}
        elif method == "RINS-LP-ONE":
            relaxed = solve_lp_relax(bm, stab, LP_BUDGET,
                                     os.path.join("stage16_logs", tag + "__relax.log"))
            usable_relax = bool(relaxed["is_optimal"] and relaxed["x"] is not None
                                and relaxed["objective"] is not None)
            if usable_relax:
                _rs, fixed, meta = stage16_rins_columns(C, relaxed["x"], fallback=False)
                sel = dict(meta)
                sel["rule"] = "RINS-LP-ONE"
            else:
                # frozen rule: fall back to FULL with the remaining budget
                _, fixed = fixed_columns(C, "FULL")
                sel = {"rule": "RINS-LP-ONE", "fallback": True,
                       "fallback_reason": "no valid optimal relaxation (status=%s)"
                                          % relaxed["status"],
                       "n_fixed_total": len(fixed),
                       "n_free_binary_after": len(C["assign"]) - len(fixed)}
        else:
            raise SystemExit("unknown stage 16 method %r" % method)

        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_start = x_lp if x_lp is not None else C["x0"]
        mip = solve_mip(bm, fixed, stab, remaining, seed, x_start,
                        os.path.join("stage16_logs", tag + ".mip.log"))
        wall = time.perf_counter() - t_run0

        # Validate every candidate against THIS run's own fixing set.  Passing another
        # configuration's fixings here would reject legal solutions and silently downgrade the
        # delivered plan, so the set is derived from `fixed` itself and never from a constant.
        fixed_Y = y_fixings_of(C, fixed)
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
        start_lines = [ln.strip() for ln in str(mip.get("log", "")).splitlines()
                       if "MIP start solution is feasible" in ln]

        runs.append({
            "state": state, "method": method, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ALL[state]["disruption"]["name"],
            "cell": ALL[state]["cell"],
            "repair_cost": jr, "budget_s": BUDGET,
            # release / fixing structure
            "sel": sel,
            "n_fixed_total": int(sel.get("n_fixed_total", 0)),
            "n_free_binary_raw": sel.get("n_free_binary_raw",
                                         len(C["assign"]) - int(sel.get("n_fixed_total", 0))),
            "n_free_binary_after": int(sel.get("n_free_binary_after", 0)),
            "n_fixed_by_group": sel.get("n_fixed_by_group"),
            "fallback": bool(sel.get("fallback", False)),
            # LP stage
            "lp_objective": j_lp, "lp_status": lp["status"],
            # NOTE: S_LP pins every binary at S_r and solves the resulting LP to optimality, so
            # it is NOT a lower bound for the full problem -- the integer optimum can be well
            # below it.  These three fields are DIAGNOSTIC ONLY and never affect selection.
            "lp_is_optimal": bool(str(lp["status"]).lower().startswith("optimal")),
            "mip_improves_over_s_lp": (None if mip["objective"] is None or j_lp is None
                                       else bool(float(mip["objective"]) < float(j_lp)
                                                 - 1e-6 * (1 + abs(j_lp)))),
            "relax_status": (relaxed["status"] if relaxed else None),
            "relax_objective": (relaxed["objective"] if relaxed else None),
            "relax_is_optimal": (relaxed["is_optimal"] if relaxed else None),
            "relax_prep_s": (relaxed["prep_s"] if relaxed else None),
            "relax_solve_s": (relaxed["solve_s"] if relaxed else None),
            "relax_le_j_lp": (None if not relaxed or relaxed["objective"] is None
                              or j_lp is None
                              else bool(relaxed["objective"] <= j_lp
                                        + 1e-6 * (1.0 + abs(j_lp)))),
            # MIP stage
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "mip_start_lines": start_lines[:3],
            "presolve": pre,
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": ("none" if best is None else best.label),
            "strictly_below_lp": bool(best is not None and j_lp is not None
                                      and float(best.cost) < j_lp - 1e-9 * (1 + abs(j_lp))),
            "select_errors": errors,
            "final_solution": (None if best is None or best.solution is None
                               else {k: np.asarray(getattr(best.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "relax_s": ((relaxed["prep_s"] + relaxed["solve_s"]) if relaxed else 0.0),
                       "mip_s": mip["prep_s"] + mip["solve_s"],
                       "total_s": wall, "budget_given_s": remaining,
                       "overrun_s": max(0.0, wall - BUDGET)},
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-26s %-11s %-6s %-3s fix=%-5d free=%-5d final=%-11s (%s) "
              "G=%+.4f t=%.1f%s"
              % (len(runs), len(plan), state, method, ALL[state]["cell"]["breadth"],
                 ALL[state]["cell"]["duration"], r["n_fixed_total"], r["n_free_binary_after"],
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)), wall,
                 " FALLBACK" if r["fallback"] else ""), flush=True)
        if args.limit and len(runs) >= args.limit:
            print("stopping early at --limit %d" % args.limit)
            return 0

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHOD_CYCLE), "seed": SEED,
                            "n_runs": len(runs),
                            "scope": "32 frozen stage 15 states (8 nominal instances x 4 "
                                     "disruption cells)",
                            "design": "baseline screen: AB vs FULL vs RINS-LP-ONE",
                            "rins_tol": RINS_TOL, "lp_budget_s": LP_BUDGET,
                            "method_cycle": list(METHOD_CYCLE),
                            "note": "AB and FULL are re-run inside this batch; no old-batch "
                                    "objective values are used. RINS-LP-ONE is a single-shot "
                                    "LP-guided neighbourhood, NOT the full RINS algorithm."},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "rho", "inst_seed", "disruption", "method", "seed", "repair_cost",
              "lp_objective", "relax_status", "relax_objective", "relax_is_optimal",
              "relax_le_j_lp", "n_fixed_total", "n_free_binary_after", "fallback",
              "mip_status", "mip_objective", "mip_start_adopted", "final_cost",
              "final_source", "strictly_below_lp", "total_s", "overrun_s"]
    with open("stage16_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
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
    fb = sum(1 for r in runs if r["fallback"])
    print("\nruns %d | wall %.1f s | select errors %d | worse than repair %d | fallbacks %d"
          % (len(runs), time.perf_counter() - t_all, bad, worse, fb))
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
