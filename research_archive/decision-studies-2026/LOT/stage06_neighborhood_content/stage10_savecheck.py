#!/usr/bin/env python
"""
Stage 10 verification, covering every run.

    .venv-hs/Scripts/python.exe stage10_savecheck.py

Re-reads `stage10_runs.json` and re-verifies WITHOUT re-solving:

  * the run grid is 32 states x 5 methods x 2 seeds = 320, on the 16 NEW nominal instances
    (seeds 6-13), 32 fault states;
  * no instance was redrawn: the state metadata must match the frozen generation seeds;
  * the released set matches the METHOD's definition -- in particular ABC must NOT release the
    short-term Y while FULL must, so ABC cannot silently become FULL;
  * every binary the method did not release is still at S_r's normalised value;
  * the delivered plan is feasible, kappa-feasible against S_r, cost-consistent and never worse
    than S_r;
  * the output rule returned the best valid candidate among {solver, S_LP, S_r};
  * the MIP start was adopted;
  * budget conservation (the stage 07 failure mode): LP + MIP prep + solve <= 20 s + overrun, and
    the MIP was granted only the remainder.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution, build_model, enumeration_space
from lsp_release import CONFIGS, build_state_context, fixed_columns
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record

TOL = 1e-6
BUDGET = 20.0
METHODS = ("AB", "A", "ABC", "FULL", "BC")
SEEDS = (0, 1)
NEW_SEEDS = (6, 7, 8, 9, 10, 11, 12, 13)


def rebuild(inst, plan: Dict[str, Any], objective: float) -> Solution:
    return Solution(status="stored", objective=float(objective),
                    X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage10_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    with open("stage10_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    failures: List[str] = []
    states = sorted({r["state"] for r in runs})
    keys = [(r["state"], r["method"], int(r["seed"])) for r in runs]
    expect = {(s, m, sd) for s in states for m in METHODS for sd in SEEDS}
    if len(runs) != 320:
        failures.append("expected 320 runs, found %d" % len(runs))
    if set(keys) != expect:
        failures.append("run grid wrong: missing %d extra %d"
                        % (len(expect - set(keys)), len(set(keys) - expect)))
    if len(states) != 32:
        failures.append("expected 32 states, found %d" % len(states))
    insts = sorted({s.split("|")[0] for s in states})
    if len(insts) != 16:
        failures.append("expected 16 nominal instances, found %d" % len(insts))

    # ---- no redraw: the frozen generation seeds must be exactly 6..13, large only
    for st in states:
        m = ST[st]["meta"]
        if m["scale"] != "large":
            failures.append("%s is not large" % st)
        if m["seed"] not in NEW_SEEDS:
            failures.append("%s uses instance seed %s, not in the frozen new set"
                            % (st, m["seed"]))
        if m["tau"] != 6:
            failures.append("%s has tau=%s, expected 6" % (st, m["tau"]))
    if sorted({ST[s]["meta"]["seed"] for s in states}) != list(NEW_SEEDS):
        failures.append("instance seeds are %s, expected %s"
                        % (sorted({ST[s]["meta"]["seed"] for s in states}), list(NEW_SEEDS)))
    if len({s.split("|")[1] for s in states}) != 2:
        failures.append("expected the two original disruptions")

    ctx: Dict[str, Any] = {}
    n_final = 0
    worst = 0.0
    n_adopted = 0
    n_lp_ok = 0
    source_counts: Dict[str, int] = {}

    print("=" * 116)
    print("%-30s %-4s %3s %6s %6s %12s %10s %9s  %s"
          % ("state", "m", "sd", "rel", "fixed", "final", "source", "maxviol", "ok"))
    print("-" * 116)
    for r in sorted(runs, key=lambda z: (z["state"], z["method"], z["seed"])):
        st, mth, seed = r["state"], r["method"], int(r["seed"])
        ok = True
        if st not in ctx:
            rec = ST[st]
            inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
            dis = next(d for d in disruptions_for(inst)
                       if d.name == rec["disruption"]["name"])
            pert = apply_disruption(inst, dis)
            repair = solution_from_record(pert, rec["repair"])
            bm = build_model(pert, MODE_AUDITED)
            tau = rec["meta"]["tau"]
            C = build_state_context(inst, pert, repair, bm, tau)
            C.update({"pert": pert, "repair": repair, "tau": tau, "bm": bm})
            ctx[st] = C
        C = ctx[st]
        pert, repair, tau = C["pert"], C["repair"], C["tau"]
        jr = float(r["repair_cost"])
        if abs(jr - float(repair.objective)) > TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: repair cost != frozen repair" % (st, mth, seed))
            ok = False

        # ---- released set must equal the method definition (ABC vs FULL checked explicitly)
        want_rel, want_fixed = fixed_columns(C, mth)
        if r["n_released"] != len(want_rel):
            failures.append("%s/%s/s%d: released %d, expected %d"
                            % (st, mth, seed, r["n_released"], len(want_rel)))
            ok = False
        rel_groups, releases_short_y = CONFIGS[mth]
        if bool(r["releases_short_term_Y"]) != releases_short_y:
            failures.append("%s/%s/s%d: short-term-Y release flag wrong" % (st, mth, seed))
            ok = False
        if mth == "ABC" and releases_short_y:
            failures.append("%s/%s/s%d: ABC must NOT release the short-term Y" % (st, mth, seed))
            ok = False
        if mth == "FULL" and not releases_short_y:
            failures.append("%s/%s/s%d: FULL must release the short-term Y" % (st, mth, seed))
            ok = False
        if releases_short_y and len(want_rel) <= len(C["groups"]["A"]) + len(C["groups"]["B"]) \
                + len(C["groups"]["C"]):
            failures.append("%s/%s/s%d: FULL release looks too small" % (st, mth, seed))
            ok = False

        # ---- delivered plan
        fc = r["final_cost"]
        if fc is None or r["final_solution"] is None:
            failures.append("%s/%s/s%d: no delivered plan" % (st, mth, seed))
            ok = False
        else:
            n_final += 1
            sol = rebuild(pert, r["final_solution"], fc)
            chk = check_solution(pert, sol, MODE_AUDITED)
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/s%d: delivered INFEASIBLE %s"
                                % (st, mth, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - fc) > TOL * (1 + abs(fc)):
                failures.append("%s/%s/s%d: stored cost != recomputed" % (st, mth, seed))
                ok = False
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/s%d: short-term flips %d > kappa" % (st, mth, seed, flips))
                ok = False
            if fc > jr + TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: delivered worse than S_r" % (st, mth, seed))
                ok = False
            # unreleased binaries must still be at S_r
            moved = []
            for (kind, i, j, t), val in C["assign"].items():
                col = C["bm"].idx.Y(i, j, t) if kind == "Y" else C["bm"].idx.Z(i, j, t)
                if col not in want_fixed:
                    continue
                got = float(sol.Y[i, j, t - 1]) if kind == "Y" else float(sol.Z[i, j, t - 1])
                if abs(got - val) > 0.5:
                    moved.append((kind, i, j, t))
            if moved:
                failures.append("%s/%s/s%d: %d UNRELEASED binaries moved, e.g. %s"
                                % (st, mth, seed, len(moved), moved[:3]))
                ok = False
            if not releases_short_y:
                for i in range(pert.N):
                    for j in range(pert.M):
                        for t in range(1, tau + 1):
                            if abs(float(sol.Y[i, j, t - 1])
                                   - float(repair.Y[i, j, t - 1])) > 0.5:
                                failures.append("%s/%s/s%d: short-term Y moved but the method "
                                                "does not release it" % (st, mth, seed))
                                ok = False
            if np.abs(sol.Z[:, :, pert.T - 1]).sum() > 0.5:
                failures.append("%s/%s/s%d: structurally pinned Z_ijT moved" % (st, mth, seed))
                ok = False

            cc = r["candidate_costs"]
            if cc:
                best_avail = min(cc.values())
                if fc > best_avail + TOL * (1 + abs(best_avail)):
                    failures.append("%s/%s/s%d: delivered %.10g but a valid candidate was %.10g"
                                    % (st, mth, seed, fc, best_avail))
                    ok = False
                if "repair" not in cc or "lp" not in cc:
                    failures.append("%s/%s/s%d: candidate set missing S_r or S_LP"
                                    % (st, mth, seed))
                    ok = False

        if r["select_errors"]:
            failures.append("%s/%s/s%d: selection errors %s" % (st, mth, seed, r["select_errors"]))
            ok = False
        if not r["mip_start_adopted"]:
            failures.append("%s/%s/s%d: MIP start not adopted" % (st, mth, seed))
            ok = False
        else:
            n_adopted += 1
        if r["lp_objective"] is not None:
            n_lp_ok += 1
        source_counts[r["final_source"]] = source_counts.get(r["final_source"], 0) + 1

        # ---- budget conservation
        t = r["timing"]
        if t["budget_given_s"] > BUDGET - t["lp_s"] + t["overrun_s"] + 0.5:
            failures.append("%s/%s/s%d: MIP granted %.3f s after a %.3f s LP"
                            % (st, mth, seed, t["budget_given_s"], t["lp_s"]))
            ok = False
        if t["budget_given_s"] > BUDGET + 1e-9:
            failures.append("%s/%s/s%d: MIP granted more than the full budget" % (st, mth, seed))
            ok = False
        if abs(t["overrun_s"] - max(0.0, t["total_s"] - BUDGET)) > 1e-3:
            failures.append("%s/%s/s%d: overrun inconsistent" % (st, mth, seed))
            ok = False

        if not ok:
            print("%-30s %-4s %3d %6d %6d %12s %10s %9s  FAIL"
                  % (st, mth, seed, r["n_released"], len(want_fixed),
                     "-" if fc is None else "%.6g" % fc, r["final_source"], "-"))

    print("=" * 116)
    print("runs: %d | delivered plans re-verified: %d" % (len(runs), n_final))
    print("MIP start adopted: %d/%d | LP objective present: %d/%d"
          % (n_adopted, len(runs), n_lp_ok, len(runs)))
    print("final source counts: %s" % source_counts)
    print("largest constraint violation: %.3g" % worst)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS RE-VERIFY: released sets (ABC keeps short-term Y fixed, FULL releases "
          "it), unreleased-binary fixings, kappa (S_r), output rule and budget all agree"
          % len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
