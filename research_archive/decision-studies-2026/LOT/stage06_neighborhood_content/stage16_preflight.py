"""Stage 16 preflight: verify the RINS-LP-ONE interface on ONE frozen state.

The three checks the round requires, before any main run:

  1. the FULL relaxation solves validly AND its objective is not above `J_LP` (within a uniform
     numerical tolerance);
  2. `S_LP` satisfies every fixing condition of RINS-LP-ONE and is actually submitted as the
     MIP start;
  3. the final plan, re-read FROM DISK, satisfies the original constraints, the kappa limit and
     this run's fixing set.

Interface or model errors stop the round here; they are NOT treated as a normal fallback.
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_release import (CONFIGS, LP_BUDGET, build_state_context, fixed_columns, pinned_bounds,
                         solve_lp_fixed, solve_lp_relax, solve_mip, stability_row_local,
                         stage16_rins_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record

STATES_FILE = "stage15_states.json"
STATE = "large_rho0.75_s2|D2_all_1p"
BUDGET = 20.0
REL_TOL = 1e-6


def main() -> int:
    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    rec = ALL[STATE]
    inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
    dis = _dis(rec)
    pert = apply_disruption(inst, dis)
    repair = solution_from_record(pert, rec["repair"])
    bm = build_model(pert, MODE_AUDITED)
    tau = rec["meta"]["tau"]
    C = build_state_context(inst, pert, repair, bm, tau)
    C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert,
              "dis": dis, "mr_sets": {}})
    os.makedirs("stage16_logs", exist_ok=True)
    print("preflight state: %s | tau=%d | J_r=%.4f" % (STATE, tau, repair.objective))

    # ---- S_LP: the same continuous polishing solution every method starts from -------------
    lb_pin, ub_pin = pinned_bounds(C)
    lp = solve_lp_fixed(bm, lb_pin, ub_pin, LP_BUDGET,
                        os.path.join("stage16_logs", "preflight__S_LP.log"))
    assert lp["solution"] is not None, "S_LP produced no solution"
    x_lp_polish = vector_of(lp["solution"], bm)
    j_lp = float(lp["objective"])
    print("  S_LP status=%s objective=%.6f" % (lp["status"], j_lp))

    # ---- check 1: FULL relaxation ---------------------------------------------------------
    stab = stability_row_local(bm, repair, tau, KAPPA)
    rlx = solve_lp_relax(bm, stab, LP_BUDGET, os.path.join("stage16_logs", "preflight__relax.log"))
    print("  FULL relaxation status=%s is_optimal=%s objective=%s"
          % (rlx["status"], rlx["is_optimal"], rlx["objective"]))
    ok1 = False
    if rlx["is_optimal"] and rlx["objective"] is not None and rlx["x"] is not None:
        tol = REL_TOL * (1.0 + abs(j_lp))
        ok1 = rlx["objective"] <= j_lp + tol
        print("  check 1: relax_obj - J_LP = %+.6e (tol %.2e) -> %s"
              % (rlx["objective"] - j_lp, tol, "PASS" if ok1 else "FAIL"))
    else:
        print("  check 1: FAIL (no valid optimal relaxation)")
    if not ok1:
        raise SystemExit("PREFLIGHT FAILED at check 1 (FULL relaxation)")

    # ---- check 2: S_LP satisfies the fixings and is submitted -----------------------------
    rel_size, fixed, meta = stage16_rins_columns(C, rlx["x"])
    print("  RINS-LP-ONE fixings: total=%d free_after=%d by_group=%s clamped=%d fallback=%s"
          % (meta["n_fixed_total"], meta["n_free_binary_after"], meta["n_fixed_by_group"],
             meta["n_clamped"], meta["fallback"]))
    viol = [c for c, v in fixed.items() if abs(float(x_lp_polish[c]) - v) > 0.5]
    print("  check 2: fixings violated by S_LP = %d -> %s"
          % (len(viol), "PASS" if not viol else "FAIL %s" % viol[:5]))
    if viol:
        raise SystemExit("PREFLIGHT FAILED at check 2 (S_LP violates its own fixing set)")

    mip = solve_mip(bm, fixed, stab, BUDGET - 1.0, 0, x_lp_polish,
                    os.path.join("stage16_logs", "preflight__rins.mip.log"))
    submitted = [ln.strip() for ln in str(mip.get("log", "")).splitlines()
                 if "MIP start solution is feasible" in ln]
    print("  MIP start log lines (%d): %s" % (len(submitted), submitted[:2]))
    print("  check 2b: start submitted and acknowledged -> %s"
          % ("PASS" if submitted else "FAIL"))
    if not submitted:
        raise SystemExit("PREFLIGHT FAILED at check 2b (MIP start not acknowledged)")

    # ---- check 3: final plan re-read and re-checked ---------------------------------------
    cands = []
    if mip["solution"] is not None:
        cands.append(evaluate_candidate("solver", pert, mip["solution"], repair, KAPPA, tau, None))
    cands.append(evaluate_candidate("repair", pert, repair, repair, KAPPA, tau, None))
    usable = [c for c in cands if is_usable(c) and c.cost is not None]
    best = min(usable, key=lambda c: c.cost) if usable else None
    assert best is not None and best.solution is not None, "no usable candidate"
    dump = {k: np.asarray(getattr(best.solution, k)).tolist()
            for k in ("X", "Y", "Z", "I", "L")}
    path = "stage16_logs/preflight__final_plan.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh)
    with open(path, encoding="utf-8") as fh:                 # RE-READ FROM DISK
        back = json.load(fh)
    sol_back = Solution(status="disk", objective=float(best.cost),
                        X=np.array(back["X"], float).reshape(pert.N, pert.M, pert.T),
                        Y=np.array(back["Y"], float).reshape(pert.N, pert.M, pert.T),
                        Z=np.array(back["Z"], float).reshape(pert.N, pert.M, pert.T),
                        I=np.array(back["I"], float).reshape(pert.N, pert.T),
                        L=np.array(back["L"], float).reshape(pert.N, pert.T),
                        mode=MODE_AUDITED)
    chk = check_solution(pert, sol_back, MODE_AUDITED)
    # fixing-set conformance on the re-read plan
    fix_bad = []
    xb = vector_of(sol_back, bm)
    for c, v in fixed.items():
        if abs(float(xb[c]) - v) > 1e-6:
            fix_bad.append(c)
    kappa = _kappa(pert, sol_back, repair, tau)
    print("  check 3: check_solution ok=%s max_violation=%.3e | fixing-set violations=%d | "
          "kappa=%d (limit %d)"
          % (chk.ok, chk.max_violation, len(fix_bad), kappa, KAPPA))
    ok3 = chk.ok and not fix_bad and kappa <= KAPPA
    print("  check 3 -> %s" % ("PASS" if ok3 else "FAIL"))
    if not ok3:
        raise SystemExit("PREFLIGHT FAILED at check 3")

    print("\nPREFLIGHT PASSED on %s" % STATE)
    return 0


def _kappa(pert, sol, repair, tau) -> int:
    n = 0
    for i in range(pert.N):
        for j in range(pert.M):
            for t in range(1, tau + 1):
                if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                    n += 1
    return n


def _dis(rec):
    from lsp_gen import Disruption
    d = rec["disruption"]
    return Disruption(d["name"], {int(j): list(ts) for j, ts in d["down"].items()},
                      d.get("note", ""))


if __name__ == "__main__":
    raise SystemExit(main())
