"""Stage 20 part 1: the online 20 s RINS-LP-ONE baseline on the 16 frozen states.

For every state this saves everything the offline diagnostics need to reuse the SAME start:
the delivered plan, the complete fixing set F_R (columns + (i,j,t) maps + values), the repair
reference S_r, and the logs.

The online budget stays 20 s.  The 60 s diagnostics are a SEPARATE offline budget and are never
reported as a 20 s deployment result.
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
STATES_FILE = "stage20_states.json"
RUNS_FILE = "stage20_base.json"
CKPT_FILE = "stage20_base.partial.json"


def disruption_of(rec) -> Disruption:
    d = rec["disruption"]
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)
    print("Stage 20 baseline | %d states | RINS-LP-ONE | seed %d | online budget %.1f s"
          % (len(states), SEED, BUDGET))

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

    os.makedirs("stage20_logs", exist_ok=True)
    t_all = time.perf_counter()
    for state in states:
        if state in done:
            continue
        rec = ALL[state]
        meta = rec["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        dis = disruption_of(rec)
        pert = apply_disruption(inst, dis)
        reference = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        t0 = time.perf_counter()
        tag = "base__%s__s%d" % (state.replace("|", "_"), SEED)

        C = build_state_context(inst, pert, reference, bm, TAU)
        C.update({"bm": bm, "tau": TAU, "meta": meta, "repair": reference, "inst": pert,
                  "dis": dis, "mr_sets": {}})

        lb, ub = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb, ub, LP_BUDGET, os.path.join("stage20_logs", tag + "__lp.log"))
        x_lp = vector_of(lp["solution"], bm) if lp["solution"] is not None else None
        stab = stability_row_local(bm, reference, TAU, KAPPA)
        rlx = solve_lp_relax(bm, stab, LP_BUDGET,
                             os.path.join("stage20_logs", tag + "__relax.log"))
        fallback = not bool(rlx["is_optimal"] and rlx["x"] is not None
                            and rlx["objective"] is not None)
        if fallback:
            _rs, fixed, ms = stage16_rins_columns(C, None, fallback=True)
            sel = {"rule": "RINS-LP-ONE", "fallback": True}
        else:
            _rs, fixed, ms = stage16_rins_columns(C, rlx["x"], fallback=False)
            sel = dict(ms)
        sel["rule"] = "RINS-LP-ONE"
        sel.setdefault("n_fixed_total", len(fixed))
        sel.setdefault("n_free_binary_after", len(C["assign"]) - len(fixed))

        remaining = max(0.1, BUDGET - (time.perf_counter() - t0))
        x_start = x_lp if x_lp is not None else vector_of(reference, bm)
        mip = solve_mip(bm, fixed, stab, remaining, SEED, x_start,
                        os.path.join("stage20_logs", tag + ".mip.log"))
        t_solve_done = time.perf_counter() - t0

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
        wall = time.perf_counter() - t0

        runs.append({
            "state": state, "instance": state.split("|")[0],
            "disruption": rec["disruption"]["name"], "cell": rec["cell"],
            "rho": meta["rho"], "inst_seed": meta["seed"], "seed": SEED,
            "tau": TAU, "kappa": KAPPA, "budget_s": BUDGET, "D_n": rec["D_n"],
            "J_ref": float(reference.objective), "J_LP": lp["objective"],
            "lp_status": lp["status"], "relax_status": rlx["status"],
            "fallback": bool(fallback), "sel": sel,
            "n_fixed_total": int(sel["n_fixed_total"]),
            "n_free_binary_after": int(sel["n_free_binary_after"]),
            "mip_status": mip["status"], "mip_objective": mip["objective"],
            "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
            "mip_proven_optimal": bool(mip["status"].lower().startswith("optimal")),
            "mip_start_adopted": bool(mip["adopted"]),
            "final_cost": (None if best is None else float(best.cost)),
            "final_source": ("none" if best is None else best.label),
            "select_errors": errors,
            # everything the offline diagnostics need to reuse the SAME start
            "final_solution": (None if best is None or best.solution is None
                               else {k: np.asarray(getattr(best.solution, k)).tolist()
                                     for k in ("X", "Y", "Z", "I", "L")}),
            "fixed_cols": {str(c): float(v) for c, v in fixed.items()},
            "fixed_Y": {("%d,%d,%d" % k): v for k, v in fy.items()},
            "fixed_Z": {("%d,%d,%d" % k): v for k, v in fz.items()},
            "timing": {"lp_s": lp["prep_s"] + lp["solve_s"],
                       "relax_s": rlx["prep_s"] + rlx["solve_s"],
                       "mip_s": mip["prep_s"] + mip["solve_s"],
                       "solve_done_s": t_solve_done, "total_s": wall,
                       "overrun_s": max(0.0, wall - BUDGET)},
        })
        write_ckpt()
        r = runs[-1]
        print("  [%2d/%2d] %-26s fix=%-5d J_ref=%-10.2f J_out=%-10s (%s) t=%.1f"
              % (len(runs), len(states), state, r["n_fixed_total"], r["J_ref"],
                 "-" if r["final_cost"] is None else "%.6g" % r["final_cost"],
                 r["final_source"], wall), flush=True)

    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "n_runs": len(runs),
                            "tau": TAU, "kappa": KAPPA, "rins_tol": RINS_TOL,
                            "scope": "16 frozen states on Sbar_N (NOM-160)",
                            "role": "online 20 s baseline; the 60 s diagnostics reuse these "
                                    "saved fixed sets and starting plans"},
                   "runs": runs}, fh, indent=2, default=str)

    fields = ["state", "instance", "disruption", "rho", "seed", "J_ref", "J_LP", "final_cost",
              "final_source", "n_fixed_total", "n_free_binary_after", "mip_status",
              "mip_dual_bound", "mip_gap", "fallback", "total_s", "overrun_s"]
    with open("stage20_base_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            flat = dict(r)
            flat["total_s"] = r["timing"]["total_s"]
            flat["overrun_s"] = r["timing"]["overrun_s"]
            w.writerow(flat)

    bad = sum(1 for r in runs if r["select_errors"])
    print("\nruns %d | wall %.1f s | select errors %d | fallbacks %d"
          % (len(runs), time.perf_counter() - t_all, bad,
             sum(1 for r in runs if r["fallback"])))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
