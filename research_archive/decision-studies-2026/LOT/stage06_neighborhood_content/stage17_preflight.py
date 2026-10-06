"""Stage 17 preflight: the three implementation checks required before the main run.

  1. the RINS-AB fixing set is EXACTLY the union of F_R and F_AB, with no value conflict on any
     shared variable;
  2. `S_LP` satisfies that union (so it is a legitimate MIP start);
  3. the final plan is validated against the COMPLETE Y and Z fixing set, not only `fixed_Y`.

Run over all 32 states (cheap: LP relaxations only, no neighbourhood MIP).
"""

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance
from lsp_model import MODE_AUDITED, build_model
from lsp_release import (LP_BUDGET, build_state_context, fixed_columns, pinned_bounds,
                         solve_lp_fixed, solve_lp_relax, stability_row_local,
                         stage16_rins_columns, vector_of)
from lsp_select import evaluate_candidate, is_usable
from stage06_run import KAPPA, solution_from_record
from stage17_run import fixings_of, union_fixings, disruption_of

STATES_FILE = "stage15_states.json"


def main() -> int:
    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    states = sorted(ALL)
    os.makedirs("stage17_logs", exist_ok=True)
    print("preflight over %d states" % len(states))
    print("%-26s %6s %6s %7s %7s %6s | %s"
          % ("state", "|F_R|", "|F_AB|", "overlap", "|union|", "conf", "checks 1/2/3"))

    n_ok1 = n_ok2 = n_ok3 = 0
    n_allfix = 0
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

        lb, ub = pinned_bounds(C)
        lp = solve_lp_fixed(bm, lb, ub, LP_BUDGET,
                            os.path.join("stage17_logs", "_pf_lp.log"))
        x_lp = vector_of(lp["solution"], bm)
        stab = stability_row_local(bm, repair, tau, KAPPA)
        rlx = solve_lp_relax(bm, stab, LP_BUDGET, os.path.join("stage17_logs", "_pf_rlx.log"))
        _rs, f_r, _mr = stage16_rins_columns(C, rlx["x"], fallback=False)
        _, f_ab = fixed_columns(C, "AB")
        merged, umeta = union_fixings(C, f_r, f_ab)

        # ---- check 1: exact union, no conflict ------------------------------------------
        c1 = (merged.keys() == (set(f_r) | set(f_ab))) and not umeta["conflicts"]
        # every union member keeps the S_r value
        slot_of = {}
        for slot in C["assign"]:
            k, i, j, t = slot
            slot_of[int(bm.idx.Y(i, j, t) if k == "Y" else bm.idx.Z(i, j, t))] = slot
        c1 = c1 and all(abs(merged[c] - float(C["assign"][slot_of[c]])) < 1e-9
                        for c in merged if c in slot_of)

        # ---- check 2: S_LP satisfies the union ------------------------------------------
        fy, fz = fixings_of(C, merged)
        cand = evaluate_candidate("s_lp", pert, lp["solution"], repair, KAPPA, tau, fy, fz)
        c2 = is_usable(cand)

        # ---- check 3: validation covers the complete Y and Z fixing set ------------------
        # build a deliberately damaged copy of S_LP: flip one fixed Z and one fixed Y
        c3 = True
        bad = cand.solution
        if bad is None:
            c3 = False
        else:
            from lsp_model import Solution
            fz_list = sorted(fz)
            if fz_list:
                i, j, t = fz_list[0]
                Zb = np.array(bad.Z, dtype=float)
                Zb[i, j, t - 1] = 1.0 - Zb[i, j, t - 1]
                damaged = Solution(status="damaged", objective=bad.objective, X=bad.X, Y=bad.Y,
                                   Z=Zb, I=bad.I, L=bad.L, mode=MODE_AUDITED)
                dmg = evaluate_candidate("dmg", pert, damaged, repair, KAPPA, tau, fy, fz)
                c3 = not dmg.fixing_ok            # the Z violation MUST be caught
            fy_list = sorted(fy)
            if fy_list:
                i, j, t = fy_list[0]
                Yb = np.array(bad.Y, dtype=float)
                Yb[i, j, t - 1] = 1.0 - Yb[i, j, t - 1]
                damaged2 = Solution(status="damaged", objective=bad.objective, X=bad.X, Y=Yb,
                                    Z=bad.Z, I=bad.I, L=bad.L, mode=MODE_AUDITED)
                dmg2 = evaluate_candidate("dmg", pert, damaged2, repair, KAPPA, tau, fy, fz)
                c3 = c3 and not dmg2.fixing_ok

        n_ok1 += c1
        n_ok2 += c2
        n_ok3 += c3
        if umeta["n_union"] == len(C["assign"]):
            n_allfix += 1
        print("%-26s %6d %6d %7d %7d %6d | %s %s %s"
              % (st, umeta["n_F_R"], umeta["n_F_AB"], umeta["n_overlap"], umeta["n_union"],
                 len(umeta["conflicts"]),
                 "OK" if c1 else "FAIL", "OK" if c2 else "FAIL", "OK" if c3 else "FAIL"))

    print("\ncheck 1 (exact union, no conflict, S_r values) : %d/%d" % (n_ok1, len(states)))
    print("check 2 (S_LP satisfies the union)              : %d/%d" % (n_ok2, len(states)))
    print("check 3 (Y and Z fixings both enforced)         : %d/%d" % (n_ok3, len(states)))
    print("states where the union fixes EVERY binary slot  : %d/%d (would be a single point)"
          % (n_allfix, len(states)))
    if n_ok1 != len(states) or n_ok2 != len(states) or n_ok3 != len(states):
        print("\nPREFLIGHT FAILED -- do not start the main run")
        return 1
    print("\nPREFLIGHT PASSED on all %d states" % len(states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
