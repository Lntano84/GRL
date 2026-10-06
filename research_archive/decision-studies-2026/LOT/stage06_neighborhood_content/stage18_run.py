"""Stage 18: does RINS-LP-ONE also improve the NO-DISRUPTION nominal plan?

Sole question: with no disruption at all and the short-term plan still restricted, can
RINS-LP-ONE clearly improve the frozen nominal plan within 20 s?

Scope: the 8 large nominal instances (seeds 2-5, two rho), solver seed 1, 20 s end-to-end each
-> 8 main runs.

* capacity and demand come from the ORIGINAL, undisrupted instance;
* the reference plan is the frozen nominal plan S_N from `stage06_nominal.json`;
* the repairer is NOT called -- the reference solution IS S_N;
* stability is tau = 6, kappa = 4, measured against S_N;
* ONLY RINS-LP-ONE is run: no AB, no FULL, no multi-round method;
* no post-disruption plan is used as a start or a candidate.

Flow (the existing method, unchanged):
  1. pin all binaries of S_N and solve the continuous polishing LP -> S_{N,LP};
  2. solve the LP relaxation of the NO-DISRUPTION FULL problem, keeping the stability row
     relative to S_N;
  3. apply the original 1e-6 agreement rule to fix variables;
  4. submit S_{N,LP} as the MIP start and solve the neighbourhood with the remaining time;
  5. deliver the cheapest valid plan among {S_N, S_{N,LP}, S_solver}.

Timing is end-to-end and covers delivery selection.  Nothing from earlier stages is inherited
beyond the frozen nominal plans themselves, which are never rewritten.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import build_instance
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_release import (LP_BUDGET, RINS_TOL, build_state_context, pinned_bounds,
                         solve_lp_fixed, solve_lp_relax, solve_mip, stability_row_local,
                         stage16_rins_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record
from stage17_run import fixings_of

BUDGET = 20.0
SEED = 1
TAU = 6
NOMINAL_FILE = "stage06_nominal.json"
STATES_FILE = "stage06_states.json"
RUNS_FILE = "stage18_runs.json"
CKPT_FILE = "stage18_runs.partial.json"


def nominal_instances() -> List[str]:
    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    return sorted({v["meta"]["name"] for v in ALL.values() if v["meta"]["scale"] == "large"})


def load_nominal(inst, rec) -> Solution:
    return Solution(status=rec["status"], objective=float(rec["objective"]),
                    X=np.array(rec["solution"]["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(rec["solution"]["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(rec["solution"]["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(rec["solution"]["I"], float).reshape(inst.N, inst.T),
                    L=np.array(rec["solution"]["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    with open(NOMINAL_FILE, encoding="utf-8") as fh:
        NOM = json.load(fh)
    insts = nominal_instances()
    print("Stage 18 | no-disruption nominal check | instances %d | seed %d | budget %.1f s"
          % (len(insts), SEED, BUDGET))

    runs: List[Dict[str, Any]] = []
    done = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                runs = list(json.load(fh).get("runs", []))
            done = {r["instance"] for r in runs}
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

    os.makedirs("stage18_logs", exist_ok=True)
    t_all = time.perf_counter()
    for name in insts:
        if name in done:
            continue
        rec = NOM[name]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        assert inst.name == name, (inst.name, name)
        nominal = load_nominal(inst, rec)                 # S_N, the reference plan
        bm = build_model(inst, MODE_AUDITED)              # NO disruption
        t_run0 = time.perf_counter()
        tag = "%s__NOMINAL__s%d" % (name, SEED)

        C = build_state_context(inst, inst, nominal, bm, TAU)
        C.update({"bm": bm, "tau": TAU, "meta": rec["meta"], "repair": nominal, "inst": inst,
                  "dis": None, "mr_sets": {}})

        # ---- 1. continuous polishing solution with every binary pinned to S_N --------------
        lb_pin, ub_pin = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                            os.path.join("stage18_logs", tag + "__lp.log"))
        x_lp = (vector_of(lp["solution"], bm) if lp["solution"] is not None else None)
        j_nlp = lp["objective"]

        # ---- 2. LP relaxation of the no-disruption FULL problem, stability vs S_N ---------
        stab = stability_row_local(bm, nominal, TAU, KAPPA)
        rlx = solve_lp_relax(bm, stab, LP_BUDGET, os.path.join("stage18_logs", tag + "__relax.log"))
        fallback = not bool(rlx["is_optimal"] and rlx["x"] is not None
                            and rlx["objective"] is not None)

        # ---- 3. agreement rule ------------------------------------------------------------
        if fallback:
            _rs, fixed, meta = stage16_rins_columns(C, None, fallback=True)
            sel = {"rule": "RINS-LP-ONE", "fallback": True,
                   "fallback_reason": "no valid optimal relaxation (status=%s)"
                                      % rlx["status"]}
        else:
            _rs, fixed, meta = stage16_rins_columns(C, rlx["x"], fallback=False)
            sel = dict(meta)
        sel["rule"] = "RINS-LP-ONE"
        sel.setdefault("n_fixed_total", len(fixed))
        sel.setdefault("n_free_binary_after", len(C["assign"]) - len(fixed))

        # ---- 4. submit S_{N,LP} and solve the neighbourhood --------------------------------
        remaining = max(0.1, BUDGET - (time.perf_counter() - t_run0))
        x_start = x_lp if x_lp is not None else vector_of(nominal, bm)
        mip = solve_mip(bm, fixed, stab, remaining, SEED, x_start,
                        os.path.join("stage18_logs", tag + ".mip.log"))
        t_solve_done = time.perf_counter() - t_run0

        # ---- 5. deliver the cheapest valid plan among the three ----------------------------
        fy, fz = fixings_of(C, fixed)
        cands = []
        if mip["solution"] is not None:
            cands.append(evaluate_candidate("solver", inst, mip["solution"], nominal, KAPPA,
                                            TAU, fy, fz))
        if lp["solution"] is not None:
            cands.append(evaluate_candidate("lp", inst, lp["solution"], nominal, KAPPA, TAU,
                                            fy, fz))
        cands.append(evaluate_candidate("nominal", inst, nominal, nominal, KAPPA, TAU, None,
                                        None))
        usable = [c for c in cands if is_usable(c) and c.cost is not None]
        best = min(usable, key=lambda c: c.cost) if usable else None
        errors = [("INVALID %s: %s" % (c.label, c.problems[:2]))
                  for c in cands if c.present and not is_usable(c)]
        wall = time.perf_counter() - t_run0
        start_lines = [ln.strip() for ln in str(mip.get("log", "")).splitlines()
                       if "MIP start solution is feasible" in ln]

        j_n = float(nominal.objective)
        d_n = max(1.0, abs(j_n))
        j_out = None if best is None else float(best.cost)
        g_total = (j_n - j_out) / d_n if j_out is not None else None
        g_cont = (j_n - j_nlp) / d_n if j_nlp is not None else None
        g_bin = ((j_nlp - j_out) / d_n) if (j_nlp is not None and j_out is not None) else None

        runs.append({
            "instance": name, "seed": SEED, "rho": rec["meta"]["rho"],
            "inst_seed": rec["meta"]["seed"], "tau": TAU, "kappa": KAPPA, "budget_s": BUDGET,
            "nominal_objective": j_n, "nominal_gap": rec.get("mip_gap"),
            "nominal_status": rec["status"], "D_N": d_n,
            "lp_objective": j_nlp, "lp_status": lp["status"],
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
            "G_total": g_total, "G_continuous": g_cont, "G_binary": g_bin,
            # short-term Y changes actually used, relative to S_N
            "short_y_changes": (None if best is None or best.solution is None else
                                _short_y_changes(inst, best.solution, nominal, TAU)),
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
        print("  [%d/%d] %-19s fix=%-5d free=%-5d J_N=%-10.2f J_out=%-10s (%s) "
              "Gtot=%s Gcont=%s Gbin=%s t=%.1f ovr=%.2f%s"
              % (len(runs), len(insts), name, r["n_fixed_total"], r["n_free_binary_after"],
                 j_n, "-" if j_out is None else "%.6g" % j_out, r["final_source"],
                 "%+.4f" % g_total if g_total is not None else "-",
                 "%+.4f" % g_cont if g_cont is not None else "-",
                 "%+.4f" % g_bin if g_bin is not None else "-",
                 wall, r["timing"]["overrun_s"], " FALLBACK" if r["fallback"] else ""),
              flush=True)
        if args.limit and len(runs) >= args.limit:
            print("stopping early at --limit %d" % args.limit)
            return 0

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "n_runs": len(runs),
                            "tau": TAU, "kappa": KAPPA, "rins_tol": RINS_TOL,
                            "lp_budget_s": LP_BUDGET,
                            "scope": "8 large nominal instances (seeds 2-5, two rho), "
                                     "NO disruption",
                            "reference": "frozen nominal plan S_N from %s; repairer NOT called"
                                         % NOMINAL_FILE,
                            "design": "does RINS-LP-ONE also improve the no-disruption "
                                      "nominal plan?",
                            "metrics": "G_total = G_continuous + G_binary, all normalised by "
                                       "D_N = max(1,|J_N|)",
                            "warning": "do NOT subtract this from the post-disruption "
                                       "percentages: different reference plans, denominators "
                                       "and feasible regions"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["instance", "rho", "inst_seed", "nominal_objective", "nominal_gap", "D_N",
              "lp_objective", "lp_status", "relax_status", "relax_objective", "fallback",
              "n_fixed_total", "n_free_binary_after", "mip_status", "mip_start_adopted",
              "final_cost", "final_source", "G_total", "G_continuous", "G_binary",
              "short_y_changes", "total_s", "solve_done_s", "overrun_s"]
    with open("stage18_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
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


def _short_y_changes(inst, sol, nominal, tau) -> int:
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                if abs(float(sol.Y[i, j, t - 1]) - float(nominal.Y[i, j, t - 1])) > 0.5:
                    n += 1
    return n


if __name__ == "__main__":
    sys.exit(main())
