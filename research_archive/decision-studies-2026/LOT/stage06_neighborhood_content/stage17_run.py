"""Stage 17 runner: seed 1 + the targeted RINS-intersect-AB control.

Sole MAIN question: with a different solver seed, does RINS-LP-ONE's advantage over AB hold?
Sole secondary diagnostic: if ALL of AB's fixing conditions are kept, does the LP-agreement
rule still improve on AB within the remaining variables?

Three arms, 32 frozen Stage 15 states, ONE solver seed (1), 20 s total each -> 96 runs:

    AB           short-term Y and C fixed to S_r (the original AB rule)
    RINS-LP-ONE  F_R = {v : |x^LP_v - S_{r,v}| <= RINS_TOL}, built from the FULL LP relaxation
    RINS-AB      F_R u F_AB  (feasible regions INTERSECTED), every fixed value still from S_r

`F_R` is built exactly as in Stage 16: solve the FULL LP relaxation (integrality dropped,
structural fixings + stability row kept).  It is NOT the AB relaxation -- changing the
relaxation would change both the information source and the rule, answering a different
question.

Timing: the end-to-end clock runs until the DELIVERED candidate has been chosen, so candidate
verification and selection are inside the budget; the overrun is reported separately.

Nothing is inherited from earlier stages: no plan pool, no shared incumbent or solve time.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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
SEED = 1
METHOD_CYCLE = ("AB", "RINS-LP-ONE", "RINS-AB")
STATES_FILE = "stage15_states.json"
RUNS_FILE = "stage17_runs.json"
CKPT_FILE = "stage17_runs.partial.json"


def disruption_of(rec) -> Disruption:
    d = rec["disruption"]
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


def method_order_for(state_index: int) -> List[str]:
    k = len(METHOD_CYCLE)
    off = state_index % k
    return [METHOD_CYCLE[(off + i) % k] for i in range(k)]


def fixings_of(C: Dict[str, Any], fixed: Dict[int, float]):
    """Split a column->value fixing dict into (i,j,t) -> value maps for Y and Z.

    The FULL fixing set must be validated, not just the short-term Y part.
    """
    bm = C["bm"]
    tau, T = C["tau"], bm.inst.T
    fy: Dict[Tuple[int, int, int], float] = {}
    fz: Dict[Tuple[int, int, int], float] = {}
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, tau + 1):
                col = bm.idx.Y(i, j, t)
                if col in fixed:
                    fy[(i, j, t)] = float(fixed[col])
            for t in range(1, T + 1):
                col = bm.idx.Z(i, j, t)
                if col in fixed:
                    fz[(i, j, t)] = float(fixed[col])
    return fy, fz


def union_fixings(C: Dict[str, Any], f_r: Dict[int, float], f_ab: Dict[int, float]):
    """F_R u F_AB.  Values come from `C["assign"]` (S_r) for BOTH sets, so any overlap must
    agree exactly; a disagreement is a real error and is reported, never silently resolved."""
    conflicts = []
    merged: Dict[int, float] = dict(f_r)
    for col, val in f_ab.items():
        if col in merged:
            if abs(merged[col] - val) > 1e-9:
                conflicts.append((col, merged[col], val))
        else:
            merged[col] = val
    meta = {"n_F_R": len(f_r), "n_F_AB": len(f_ab),
            "n_overlap": len(set(f_r) & set(f_ab)),
            "n_union": len(merged), "conflicts": conflicts}
    return merged, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)

    plan: List[tuple] = []
    for idx, st in enumerate(states):
        for m in method_order_for(idx):
            plan.append((st, m, SEED))
    print("states: %d | methods %s | seed %d | runs %d"
          % (len(states), list(METHOD_CYCLE), SEED, len(plan)))

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

    os.makedirs("stage17_logs", exist_ok=True)
    t_all = time.perf_counter()
    for (state, method, seed) in plan:
        if "%s|%s|%d" % (state, method, seed) in done:
            continue
        C = cache[state]
        bm, pert, repair, tau = C["bm"], C["inst"], C["repair"], C["tau"]
        jr = float(repair.objective)
        t_run0 = time.perf_counter()
        tag = "%s__%s__s%d" % (state.replace("|", "_"), method, seed)

        # ---- this run's own S_LP ----------------------------------------------------------
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage17_logs", tag + "__lp.log"))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else None)
        j_lp = lp["objective"]
        stab = stability_row_local(bm, repair, tau, KAPPA)

        relaxed: Optional[Dict[str, Any]] = None
        sel: Dict[str, Any] = {}
        fixed: Dict[int, float] = {}

        if method == "AB":
            _, fixed = fixed_columns(C, "AB")
            sel = {"rule": "AB", "n_fixed_total": len(fixed)}
        elif method in ("RINS-LP-ONE", "RINS-AB"):
            relaxed = solve_lp_relax(bm, stab, LP_BUDGET,
                                     os.path.join("stage17_logs", tag + "__relax.log"))
            ok_relax = bool(relaxed["is_optimal"] and relaxed["x"] is not None
                            and relaxed["objective"] is not None)
            if ok_relax:
                _rs, f_r, meta_r = stage16_rins_columns(C, relaxed["x"], fallback=False)
                if method == "RINS-LP-ONE":
                    fixed = f_r
                    sel = dict(meta_r)
                    sel["rule"] = "RINS-LP-ONE"
                else:
                    _, f_ab = fixed_columns(C, "AB")
                    fixed, umeta = union_fixings(C, f_r, f_ab)
                    if umeta["conflicts"]:
                        raise SystemExit("RINS-AB fixing conflict on %s: %s"
                                         % (state, umeta["conflicts"][:3]))
                    sel = dict(meta_r)
                    sel.update(umeta)
                    sel["rule"] = "RINS-AB"
                    sel["n_fixed_total"] = umeta["n_union"]
            else:
                # frozen fallback: RINS-LP-ONE -> FULL, RINS-AB -> AB, remaining budget only
                fb = "FULL" if method == "RINS-LP-ONE" else "AB"
                _, fixed = fixed_columns(C, fb)
                sel = {"rule": method, "fallback": True, "fallback_to": fb,
                       "fallback_reason": "no valid optimal relaxation (status=%s)"
                                          % relaxed["status"],
                       "n_fixed_total": len(fixed)}
        else:
            raise SystemExit("unknown stage 17 method %r" % method)

        sel.setdefault("n_free_binary_after", len(C["assign"]) - int(sel["n_fixed_total"]))

        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_start = x_lp if x_lp is not None else C["x0"]
        mip = solve_mip(bm, fixed, stab, remaining, seed, x_start,
                        os.path.join("stage17_logs", tag + ".mip.log"))
        t_solve_done = time.perf_counter() - t_run0

        # ---- delivery: verification and selection are INSIDE the end-to-end clock ---------
        fy, fz = fixings_of(C, fixed)
        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", pert, mip["solution"], repair, KAPPA, tau,
                                            fy, fz))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", pert, lp["solution"], repair, KAPPA, tau,
                                            fy, fz))
        cands.append(evaluate_candidate("repair", pert, repair, repair, KAPPA, tau, None, None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        best = min(usable, key=lambda c: c.cost) if usable else None
        errors = [("INVALID %s: %s" % (c.label, c.problems[:2]))
                  for c in cands if c.present and not is_usable(c)]
        wall = time.perf_counter() - t_run0           # clock stops AFTER the choice is made
        pre = _presolve(mip["log"])
        start_lines = [ln.strip() for ln in str(mip.get("log", "")).splitlines()
                       if "MIP start solution is feasible" in ln]

        runs.append({
            "state": state, "method": method, "seed": seed,
            "scale": C["meta"]["scale"], "rho": C["meta"]["rho"],
            "inst_seed": C["meta"]["seed"], "disruption": ALL[state]["disruption"]["name"],
            "cell": ALL[state]["cell"],
            "repair_cost": jr, "budget_s": BUDGET,
            "sel": sel,
            "n_fixed_total": int(sel["n_fixed_total"]),
            "n_free_binary_after": int(sel["n_free_binary_after"]),
            "n_overlap": sel.get("n_overlap"),
            "fixing_conflicts": len(sel.get("conflicts", [])),
            "fallback": bool(sel.get("fallback", False)),
            "lp_objective": j_lp, "lp_status": lp["status"],
            "lp_is_optimal": bool(str(lp["status"]).lower().startswith("optimal")),
            "mip_improves_over_s_lp": (None if mip["objective"] is None or j_lp is None
                                       else bool(float(mip["objective"]) < float(j_lp)
                                                 - 1e-6 * (1 + abs(j_lp)))),
            "relax_status": (relaxed["status"] if relaxed else None),
            "relax_objective": (relaxed["objective"] if relaxed else None),
            "relax_is_optimal": (relaxed["is_optimal"] if relaxed else None),
            "relax_s": ((relaxed["prep_s"] + relaxed["solve_s"]) if relaxed else 0.0),
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
            "fixed_Y": {("%d,%d,%d" % k): v for k, v in fy.items()},
            "fixed_Z": {("%d,%d,%d" % k): v for k, v in fz.items()},
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "relax_s": ((relaxed["prep_s"] + relaxed["solve_s"])
                                   if relaxed else 0.0),
                       "mip_s": mip["prep_s"] + mip["solve_s"],
                       "solve_done_s": t_solve_done,
                       "total_s": wall,
                       # end-to-end: everything up to and including delivery selection
                       "overrun_s": max(0.0, wall - BUDGET),
                       "budget_given_s": remaining},
        })
        write_ckpt()
        r = runs[-1]
        print("  [%3d/%3d] %-26s %-11s s%d fix=%-5d free=%-5d final=%-11s (%s) G=%+.4f "
              "t=%.1f ovr=%.2f%s"
              % (len(runs), len(plan), state, method, seed, r["n_fixed_total"],
                 r["n_free_binary_after"],
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], (jr - (r["final_cost"] or jr)) / max(1.0, abs(jr)),
                 wall, r["timing"]["overrun_s"],
                 " FALLBACK" if r["fallback"] else ""), flush=True)
        if args.limit and len(runs) >= args.limit:
            print("stopping early at --limit %d" % args.limit)
            return 0

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "methods": list(METHOD_CYCLE), "seed": SEED,
                            "n_runs": len(runs), "rins_tol": RINS_TOL,
                            "lp_budget_s": LP_BUDGET, "method_cycle": list(METHOD_CYCLE),
                            "scope": "32 frozen stage 15 states",
                            "design": "seed-1 confirmation of RINS-LP-ONE vs AB, plus the "
                                      "targeted RINS-intersect-AB control",
                            "timing": "end-to-end clock covers LP, selection, assembly, "
                                      "submission, verification and DELIVERY SELECTION; "
                                      "overrun is reported separately",
                            "note": "no FULL arm; nothing inherited from earlier stages"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "rho", "inst_seed", "disruption", "method", "seed", "repair_cost",
              "lp_objective", "relax_status", "relax_objective", "n_fixed_total",
              "n_free_binary_after", "n_overlap", "fallback", "mip_status", "mip_objective",
              "mip_start_adopted", "final_cost", "final_source", "strictly_below_lp",
              "total_s", "solve_done_s", "overrun_s"]
    with open("stage17_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            for k in ("total_s", "solve_done_s", "overrun_s"):
                flat[k] = r["timing"][k]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    worse = sum(1 for r in runs if r["final_cost"] is not None
                and r["final_cost"] > r["repair_cost"] + 1e-6 * (1 + abs(r["repair_cost"])))
    fb = sum(1 for r in runs if r["fallback"])
    ovr = max((r["timing"]["overrun_s"] for r in runs), default=0.0)
    print("\nruns %d | wall %.1f s | select errors %d | worse than repair %d | fallbacks %d | "
          "max overrun %.3f s"
          % (len(runs), time.perf_counter() - t_all, bad, worse, fb, ovr))
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
