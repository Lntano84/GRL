"""Stage 19 runner: start-quality sensitivity check on the frozen `S_N+` plans.

Two questions only:
  Q1  starting from S_N+, is post-disruption repair + continuous polishing still clearly behind
      RINS?
  Q2  can the same S_N+ still be clearly improved by RINS when there is no disruption?

40 runs: 32 fault states (8 instances x 4 disruptions) + 8 no-fault states, 20 s each.

Stability anchors (explicit, never mixed):
  fault states     kappa = 4 relative to the NEW repair solution
  no-fault states  kappa = 4 relative to S_N+ itself
Cumulative changes relative to the OLD S_N are NOT constrained.  Old repair solutions, old
fixing sets and old starting points are never used.

Each run computes its own continuous polishing solution and delivers the best of
{reference, polishing LP, valid MIP candidate}.  Only RINS-LP-ONE is run: no FULL, no AB, no
other algorithm.

Metric scale is FIXED per instance at the OLD nominal cost D_n = max(1, |J_{N,n}|), so that a
change in repair cost cannot move the denominator.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import Disruption, apply_disruption, build_instance
from lsp_model import MODE_AUDITED, build_model
from lsp_release import (LP_BUDGET, RINS_TOL, build_state_context, pinned_bounds,
                         solve_lp_fixed, solve_lp_relax, solve_mip, stability_row_local,
                         stage16_rins_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record
from stage17_run import fixings_of

BUDGET = 20.0
SEED = 1
TAU = 6
STATES_FILE = "stage19_states.json"
RUNS_FILE = "stage19_runs.json"
CKPT_FILE = "stage19_runs.partial.json"


def disruption_of(rec) -> Optional[Disruption]:
    d = rec["disruption"]
    if not d.get("down"):
        return None
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)
    n_fault = sum(1 for s in states if disruption_of(ALL[s]) is not None)
    print("Stage 19 | %d states (%d fault + %d no-fault) | seed %d | budget %.1f s"
          % (len(states), n_fault, len(states) - n_fault, SEED, BUDGET))

    runs: List[Dict[str, Any]] = []
    done = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {r["state"] for r in runs}
            print("resuming: %d already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            runs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "n_runs": len(runs)},
                       "runs": runs}, fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    os.makedirs("stage19_logs", exist_ok=True)
    t_all = time.perf_counter()
    for state in states:
        if state in done:
            continue
        rec = ALL[state]
        meta = rec["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        dis = disruption_of(rec)
        pert = apply_disruption(inst, dis) if dis is not None else inst
        reference = solution_from_record(pert, rec["repair"])   # new repair, or S_N+ itself
        bm = build_model(pert, MODE_AUDITED)
        D = float(rec["D_scale"])
        t_run0 = time.perf_counter()
        tag = "%s__s%d" % (state.replace("|", "_"), SEED)

        C = build_state_context(inst, pert, reference, bm, TAU)
        C.update({"bm": bm, "tau": TAU, "meta": meta, "repair": reference, "inst": pert,
                  "dis": dis, "mr_sets": {}})

        # 1. own continuous polishing solution (all binaries pinned to the reference)
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage19_logs", tag + "__lp.log"))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else None)
        j_lp = lp["objective"]

        # 2. LP relaxation, stability row relative to THIS state's reference
        stab = stability_row_local(bm, reference, TAU, KAPPA)
        rlx = solve_lp_relax(bm, stab, LP_BUDGET, os.path.join("stage19_logs", tag + "__relax.log"))
        fallback = not bool(rlx["is_optimal"] and rlx["x"] is not None
                            and rlx["objective"] is not None)

        # 3. agreement rule
        if fallback:
            _rs, fixed, meta_s = stage16_rins_columns(C, None, fallback=True)
            sel = {"rule": "RINS-LP-ONE", "fallback": True,
                   "fallback_reason": "no valid optimal relaxation (status=%s)" % rlx["status"]}
        else:
            _rs, fixed, meta_s = stage16_rins_columns(C, rlx["x"], fallback=False)
            sel = dict(meta_s)
        sel["rule"] = "RINS-LP-ONE"
        sel.setdefault("n_fixed_total", len(fixed))
        sel.setdefault("n_free_binary_after", len(C["assign"]) - len(fixed))

        # 4. submit and solve the neighbourhood
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_start = x_lp if x_lp is not None else vector_of(reference, bm)
        mip = solve_mip(bm, fixed, stab, remaining, SEED, x_start,
                        os.path.join("stage19_logs", tag + ".mip.log"))
        t_solve_done = time.perf_counter() - t_run0

        # 5. deliver the cheapest valid plan
        fy, fz = fixings_of(C, fixed)
        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", pert, mip["solution"], reference, KAPPA,
                                            TAU, fy, fz))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", pert, lp["solution"], reference, KAPPA, TAU,
                                            fy, fz))
        cands.append(evaluate_candidate("reference", pert, reference, reference, KAPPA, TAU,
                                        None, None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        best = min(usable, key=lambda c: c.cost) if usable else None
        errors = [("INVALID %s: %s" % (c.label, c.problems[:2]))
                  for c in cands if c.present and not is_usable(c)]
        wall = time.perf_counter() - t_run0
        start_lines = [ln.strip() for ln in str(mip.get("log", "")).splitlines()
                       if "MIP start solution is feasible" in ln]

        j_ref = float(reference.objective)
        j_out = None if best is None else float(best.cost)
        g_cont = (j_ref - j_lp) / D if j_lp is not None else None
        g_bin = ((j_lp - j_out) / D) if (j_lp is not None and j_out is not None) else None

        runs.append({
            "state": state, "instance": state.split("|")[0],
            "disruption": rec["disruption"]["name"],
            "arm": "fault" if dis is not None else "no_fault",
            "cell": rec["cell"], "rho": meta["rho"], "inst_seed": meta["seed"],
            "seed": SEED, "tau": TAU, "kappa": KAPPA, "budget_s": BUDGET, "D_n": D,
            "old_nominal_objective": rec["old_nominal_objective"],
            "nominal_plus_objective": rec["nominal_plus_objective"],
            "J_ref": j_ref, "J_LP": j_lp,
            "lp_status": lp["status"],
            "relax_status": rlx["status"], "relax_objective": rlx["objective"],
            "relax_is_optimal": rlx["is_optimal"], "relax_s": rlx["prep_s"] + rlx["solve_s"],
            "fallback": bool(fallback),
            "sel": sel, "n_fixed_total": int(sel["n_fixed_total"]),
            "n_free_binary_after": int(sel["n_free_binary_after"]),
            "n_fixed_by_group": sel.get("n_fixed_by_group"),
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "mip_start_lines": start_lines[:3],
            "final_cost": j_out, "final_source": ("none" if best is None else best.label),
            "G_cont": g_cont, "G_bin": g_bin,
            "G_total": ((j_ref - j_out) / D if j_out is not None else None),
            "short_y_changes": (None if best is None or best.solution is None else
                                _short_y_changes(pert, best.solution, reference, TAU)),
            "select_errors": errors,
            "final_solution": (None if best is None or best.solution is None
                               else {k: np.asarray(getattr(best.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "fixed_Y": {("%d,%d,%d" % k): v for k, v in fy.items()},
            "fixed_Z": {("%d,%d,%d" % k): v for k, v in fz.items()},
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "relax_s": rlx["prep_s"] + rlx["solve_s"],
                       "mip_s": mip["prep_s"] + mip["solve_s"],
                       "solve_done_s": t_solve_done, "total_s": wall,
                       "overrun_s": max(0.0, wall - BUDGET), "budget_given_s": remaining},
        })
        write_ckpt()
        r = runs[-1]
        print("  [%2d/%2d] %-26s %-6s J_ref=%-10.2f J_out=%-10s (%s) Gcont=%s Gbin=%s t=%.1f%s"
              % (len(runs), len(states), state, r["arm"], j_ref,
                 "-" if j_out is None else "%.6g" % j_out, r["final_source"],
                 "%+.4f" % g_cont if g_cont is not None else "-",
                 "%+.4f" % g_bin if g_bin is not None else "-", wall,
                 " FALLBACK" if r["fallback"] else ""), flush=True)
        if args.limit and len(runs) >= args.limit:
            print("stopping early at --limit %d" % args.limit)
            return 0

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "n_runs": len(runs),
                            "tau": TAU, "kappa": KAPPA, "rins_tol": RINS_TOL,
                            "lp_budget_s": LP_BUDGET,
                            "scope": "32 fault states (from S_N+) + 8 no-fault states",
                            "reference": "S_N+ frozen in stage19_nominal_plus.json; fault "
                                         "states use repair_minbatch_v3 applied to S_N+",
                            "metric_scale": "fixed per instance at the OLD nominal cost "
                                            "D_n = max(1,|J_{N,n}|)",
                            "primary": "fault-arm G_bin, averaged equally over the 4 "
                                       "disruptions per instance then over the 8 instances",
                            "warning": "the fault and no-fault arms must NOT be subtracted: "
                                       "their feasible regions and reference solutions differ"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "instance", "disruption", "arm", "rho", "inst_seed", "D_n", "J_ref",
              "J_LP", "lp_status", "relax_status", "relax_objective", "fallback",
              "n_fixed_total", "n_free_binary_after", "mip_status", "mip_start_adopted",
              "final_cost", "final_source", "G_cont", "G_bin", "G_total", "short_y_changes",
              "total_s", "solve_done_s", "overrun_s"]
    with open("stage19_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            for k in ("total_s", "solve_done_s", "overrun_s"):
                flat[k] = r["timing"][k]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    ovr = max((r["timing"]["overrun_s"] for r in runs), default=0.0)
    print("\nruns %d | wall %.1f s | select errors %d | fallbacks %d | max overrun %.3f s"
          % (len(runs), time.perf_counter() - t_all, bad,
             sum(1 for r in runs if r["fallback"]), ovr))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if bad == 0 else 1


def _short_y_changes(inst, sol, reference, tau) -> int:
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                if abs(float(sol.Y[i, j, t - 1]) - float(reference.Y[i, j, t - 1])) > 0.5:
                    n += 1
    return n


if __name__ == "__main__":
    sys.exit(main())
