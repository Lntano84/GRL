"""Stage 20 part 2: the two 60 s OFFLINE diagnostics per state.

    RINS-BOUND   F_R kept EXACTLY as generated online;  stability anchored on the original S_r
    FULL-BOUND   all RINS fixings withdrawn, every original task constraint and structural
                 fixing kept (the repair-based reference and the kappa row stay); anchor S_r

Both start from the SAME saved online-delivered plan, so the comparison is within one state.
Neither diagnostic may regenerate fixings from a new incumbent, and the kappa reference is never
moved.

Bounds: U = best VALID solution cost found (validated with check_solution, and against F_R for
the RINS side); L = the solver's dual bound.  A candidate is only accepted as a witness if it is
feasible for the original model and within kappa; for the RINS side it must additionally satisfy
F_R.

60 s is a DIAGNOSTIC budget.  It does not change the online 20 s budget and these runs are never
reported as 20 s deployment results.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import apply_disruption, build_instance
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_release import (build_state_context, solve_mip, stability_row_local, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record
from stage17_run import fixings_of
from stage20_run_base import disruption_of

DIAG_BUDGET = 60.0
SEED = 1
TAU = 6
STATES_FILE = "stage20_states.json"
BASE_FILE = "stage20_base.json"
OUT_FILE = "stage20_diag.json"
CKPT_FILE = "stage20_diag.partial.json"
DIAGS = ("RINS-BOUND", "FULL-BOUND")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(BASE_FILE, encoding="utf-8") as fh:
        BASE = {r["state"]: r for r in json.load(fh)["runs"]}
    states = sorted(ALL)
    print("Stage 20 diagnostics | %d states x %d diagnostics | %.0f s each | seed %d"
          % (len(states), len(DIAGS), DIAG_BUDGET, SEED))

    recs: List[Dict[str, Any]] = []
    done = set()
    if args.resume and os.path.exists(CKPT_FILE):
        try:
            with open(CKPT_FILE, encoding="utf-8") as fh:
                recs = list(json.load(fh).get("runs", []))
            done = {"%s|%s" % (r["state"], r["diagnostic"]) for r in recs}
            print("resuming: %d already complete" % len(done))
        except (OSError, ValueError) as exc:
            print("checkpoint unreadable (%s); starting fresh" % exc)
            recs, done = [], set()

    def write_ckpt():
        tmp = CKPT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"budget_s": DIAG_BUDGET, "n_runs": len(recs)}, "runs": recs},
                      fh, indent=2, default=str)
        os.replace(tmp, CKPT_FILE)

    os.makedirs("stage20_logs", exist_ok=True)
    t_all = time.perf_counter()
    print("\n%-26s %-10s | %11s %11s %11s | %10s %10s | %s"
          % ("state", "diag", "J_ref", "U", "L", "valid sol", "proven", "witness"))
    for state in states:
        rec = ALL[state]
        b = BASE[state]
        meta = rec["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        dis = disruption_of(rec)
        pert = apply_disruption(inst, dis)
        reference = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        D = float(rec["D_n"])

        C = build_state_context(inst, pert, reference, bm, TAU)
        C.update({"bm": bm, "tau": TAU, "meta": meta, "repair": reference, "inst": pert,
                  "dis": dis, "mr_sets": {}})

        # the online fixing set, frozen: rebuilt from the SAVED column->value map
        fixed_cols = {int(k): float(v) for k, v in b["fixed_cols"].items()}
        fy, fz = fixings_of(C, fixed_cols)
        stab = stability_row_local(bm, reference, TAU, KAPPA)

        # the SAME starting plan as the online run delivered
        x_start = _vec(pert, b["final_solution"])
        J_online = float(b["final_cost"])

        for diag in DIAGS:
            key = "%s|%s" % (state, diag)
            if key in done:
                continue
            fixed = dict(fixed_cols) if diag == "RINS-BOUND" else {}
            # C46: the fixing conditions must be validated ONLY for the arm that imposes them.
            # FULL-BOUND deliberately withdraws F_R, so checking a FULL plan against F_R would
            # reject exactly the plans that ignore the fixings -- i.e. the evidence being sought.
            val_fy, val_fz = (fy, fz) if diag == "RINS-BOUND" else (None, None)
            t0 = time.perf_counter()
            mip = solve_mip(bm, fixed, stab, DIAG_BUDGET, SEED, x_start,
                            os.path.join("stage20_logs",
                                         "diag__%s__%s.log" % (state.replace("|", "_"), diag)))
            wall = time.perf_counter() - t0

            # ---- validate the solver's answer; only a VALID plan becomes U -----------------
            U, valid, note = None, False, ""
            if mip["solution"] is not None:
                cand = evaluate_candidate("solver", pert, mip["solution"], reference, KAPPA,
                                          TAU, val_fy, val_fz)
                if is_usable(cand) and cand.cost is not None:
                    U, valid = float(cand.cost), True
                else:
                    note = "solver plan rejected: %s" % (cand.problems[:2],)
            if not valid:
                # fall back to the online-delivered plan, which is valid for BOTH diagnostics
                U, note = J_online, (note + " ; using the online plan as U").strip(" ;")
            L = mip["dual_bound"]
            proven = bool(mip["status"].lower().startswith("optimal"))

            recs.append({
                "state": state, "instance": state.split("|")[0],
                "disruption": rec["disruption"]["name"], "diagnostic": diag,
                "seed": SEED, "budget_s": DIAG_BUDGET, "D_n": D,
                "J_ref": float(reference.objective), "J_online": J_online,
                "n_fixed": len(fixed),
                "U": U, "L": L, "solver_valid": valid, "note": note,
                "mip_status": mip["status"], "mip_objective": mip["objective"],
                "mip_dual_bound": mip["dual_bound"], "mip_gap": mip["gap"],
                "proven_optimal": proven, "wall_s": wall,
                "fixed_cols": {str(c): float(v) for c, v in fixed.items()},
                # ALWAYS keep the plan, so a witness is never lost again
                "witness": ({k: np.asarray(getattr(mip["solution"], k)).tolist()
                             for k in ("X", "Y", "Z", "I", "L")}
                            if mip["solution"] is not None else None),
                "witness_valid": valid,
            })
            write_ckpt()
            print("%-26s %-10s | %11.2f %11.2f %11s | %10s %10s | %s"
                  % (state, diag, reference.objective, U,
                     "%.2f" % L if L is not None else "-",
                     "yes" if valid else "no", "yes" if proven else "no",
                     recs[-1]["mip_status"]), flush=True)

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": DIAG_BUDGET, "seed": SEED, "tau": TAU, "kappa": KAPPA,
                            "n_runs": len(recs),
                            "role": "offline bound diagnostics; NOT a 20 s deployment result",
                            "anchor": "kappa row always relative to the original repair S_r; "
                                      "the reference solution never moves",
                            "fixings": "RINS-BOUND keeps the online F_R unchanged; FULL-BOUND "
                                       "withdraws only the RINS fixings"},
                   "runs": recs}, fh, indent=2, default=str)
    print("\ndiagnostics %d | wall %.1f s" % (len(recs), time.perf_counter() - t_all))
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)
    return 0


def _vec(inst, plan) -> np.ndarray:
    """Dense variable vector from a stored (X,Y,Z,I,L) plan dict."""
    from lsp_model import Indexer
    idx = Indexer(inst)
    x = np.zeros(idx.n)
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                x[idx.X(i, j, t)] = float(plan["X"][i][j][t - 1])
                x[idx.Y(i, j, t)] = float(plan["Y"][i][j][t - 1])
                x[idx.Z(i, j, t)] = float(plan["Z"][i][j][t - 1])
            x[idx.Z(i, j, 0)] = 0.0
        for t in range(1, inst.T + 1):
            x[idx.I(i, t)] = float(plan["I"][i][t - 1])
            x[idx.L(i, t)] = float(plan["L"][i][t - 1])
    return x


if __name__ == "__main__":
    sys.exit(main())
