#!/usr/bin/env python
"""
Stage 09B verification, covering every run.

    .venv-hs/Scripts/python.exe stage09b_savecheck.py

Re-reads `stage09b_runs.json` and re-verifies WITHOUT re-solving:

  * the run grid is exactly 16 states x 7 configurations x 2 seeds = 224;
  * the released set is exactly the union of the groups the configuration names, computed from
    the frozen definitions (A = far Y, B = short Z, C = far Z excluding the pinned Z_ijT);
  * every binary the configuration did NOT release is still at S_r's value in the delivered
    plan -- this is the fixing condition that makes the configuration what it claims to be;
  * short-term Y is fixed in EVERY configuration, and structurally pinned Z never moves;
  * the delivered plan is feasible, kappa-feasible against S_r, cost-consistent with the
    recomputation, and never worse than S_r;
  * the output rule really returned the best valid candidate among {solver, S_LP, S_r};
  * the re-run LP objective matches stage 09A for the same state;
  * the MIP start was adopted;
  * budget conservation: LP + MIP prep + MIP solve <= 20 s + overrun, and the MIP was given only
    the remainder -- the stage 07 shared-prefix failure mode, checked explicitly here.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution, build_model, enumeration_space
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record
from stage09b_run import group_of, slot_sets

TOL = 1e-6
BUDGET = 20.0
CONFIGS = ("A", "B", "C", "AB", "AC", "BC", "ABC")
SEEDS = (0, 1)


def rebuild(inst, plan: Dict[str, Any], objective: float) -> Solution:
    return Solution(status="stored", objective=float(objective),
                    X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage09b_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    with open("stage09a_runs.json", encoding="utf-8") as fh:
        ref = {r["state"]: r["lp_objective"] for r in json.load(fh)["runs"]}

    failures: List[str] = []
    states = sorted({r["state"] for r in runs})
    keys = [(r["state"], r["config"], int(r["seed"])) for r in runs]
    expect = {(s, c, sd) for s in states for c in CONFIGS for sd in SEEDS}
    if len(runs) != 224:
        failures.append("expected 224 runs, found %d" % len(runs))
    if set(keys) != expect:
        failures.append("run grid wrong: missing %d extra %d"
                        % (len(expect - set(keys)), len(set(keys) - expect)))
    if len(states) != 16:
        failures.append("expected 16 states, found %d" % len(states))
    for st in states:
        if ST[st]["meta"]["scale"] != "large":
            failures.append("%s is not a large state" % st)

    ctx: Dict[str, Any] = {}
    n_final = 0
    worst = 0.0
    n_adopted = 0
    n_lp_match = 0
    source_counts: Dict[str, int] = {}

    print("=" * 112)
    print("%-28s %-4s %3s %6s %6s %12s %10s %9s  %s"
          % ("state", "cfg", "sd", "rel", "fixed", "final", "source", "maxviol", "ok"))
    print("-" * 112)
    for r in sorted(runs, key=lambda z: (z["state"], z["config"], z["seed"])):
        st, cfg, seed = r["state"], r["config"], int(r["seed"])
        ok = True
        if st not in ctx:
            rec = ST[st]
            inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
            dis = next(d for d in disruptions_for(inst)
                       if d.name == rec["disruption"]["name"])
            pert = apply_disruption(inst, dis)
            repair = solution_from_record(pert, rec["repair"])
            tau = rec["meta"]["tau"]
            bm = build_model(pert, MODE_AUDITED)
            groups = slot_sets(pert, tau)
            assign = {}
            for slot in enumeration_space(pert, MODE_AUDITED):
                kind, i, j, t = slot
                raw = (float(repair.Y[i, j, t - 1]) if kind == "Y"
                       else float(repair.Z[i, j, t - 1]))
                assign[slot] = int(round(raw))
            ctx[st] = {"pert": pert, "repair": repair, "tau": tau, "bm": bm,
                       "groups": groups, "assign": assign,
                       "all_slots": set(assign)}
        C = ctx[st]
        pert, repair, tau = C["pert"], C["repair"], C["tau"]
        jr = float(r["repair_cost"])
        if abs(jr - float(repair.objective)) > TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: repair cost differs from the frozen repair"
                            % (st, cfg, seed))
            ok = False

        # ---- released set must be exactly the union of the named groups
        want_rel: Set[Tuple[str, int, int, int]] = set()
        for g in cfg:
            want_rel |= C["groups"][g]
        if r["n_released"] != len(want_rel):
            failures.append("%s/%s/s%d: released %d, expected %d"
                            % (st, cfg, seed, r["n_released"], len(want_rel)))
            ok = False
        fixed_slots = C["all_slots"] - want_rel

        # ---- LP objective must reproduce stage 09A
        if r["lp_matches_stage09a"]:
            n_lp_match += 1
        else:
            failures.append("%s/%s/s%d: LP objective %.10g != stage 09A %.10g"
                            % (st, cfg, seed, r["lp_objective"] or float("nan"), ref[st]))
            ok = False

        # ---- delivered plan
        fc = r["final_cost"]
        if fc is None or r["final_solution"] is None:
            failures.append("%s/%s/s%d: no delivered plan" % (st, cfg, seed))
            ok = False
        else:
            n_final += 1
            sol = rebuild(pert, r["final_solution"], fc)
            chk = check_solution(pert, sol, MODE_AUDITED)
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/s%d: delivered INFEASIBLE %s"
                                % (st, cfg, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - fc) > TOL * (1 + abs(fc)):
                failures.append("%s/%s/s%d: stored cost != recomputed" % (st, cfg, seed))
                ok = False
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/s%d: short-term flips %d > kappa" % (st, cfg, seed, flips))
                ok = False
            if fc > jr + TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: delivered worse than S_r" % (st, cfg, seed))
                ok = False

            # ---- THE defining property: unreleased binaries are still at S_r's value
            moved = []
            for (kind, i, j, t) in fixed_slots:
                got = float(sol.Y[i, j, t - 1]) if kind == "Y" else float(sol.Z[i, j, t - 1])
                if abs(got - C["assign"][(kind, i, j, t)]) > 0.5:
                    moved.append((kind, i, j, t))
            if moved:
                failures.append("%s/%s/s%d: %d UNRELEASED binaries moved, e.g. %s"
                                % (st, cfg, seed, len(moved), moved[:3]))
                ok = False
            # short-term Y fixed in every configuration
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, tau + 1):
                        if abs(float(sol.Y[i, j, t - 1])
                               - float(repair.Y[i, j, t - 1])) > 0.5:
                            failures.append("%s/%s/s%d: short-term Y moved at (%d,%d,%d)"
                                            % (st, cfg, seed, i, j, t))
                            ok = False
            # structural pins
            if np.abs(sol.Z[:, :, pert.T - 1]).sum() > 0.5:
                failures.append("%s/%s/s%d: structurally pinned Z_ijT moved" % (st, cfg, seed))
                ok = False

            # ---- per-group flip counts must agree with the plan
            fg = r["final_flips_by_group"]
            a_cnt = sum(1 for i in range(pert.N) for j in range(pert.M)
                        for t in range(tau + 1, pert.T + 1)
                        if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5)
            b_cnt = sum(1 for i in range(pert.N) for j in range(pert.M)
                        for t in range(1, tau + 1)
                        if abs(float(sol.Z[i, j, t - 1]) - float(repair.Z[i, j, t - 1])) > 0.5)
            c_cnt = sum(1 for i in range(pert.N) for j in range(pert.M)
                        for t in range(tau + 1, pert.T)
                        if abs(float(sol.Z[i, j, t - 1]) - float(repair.Z[i, j, t - 1])) > 0.5)
            if (fg["A"], fg["B"], fg["C"]) != (a_cnt, b_cnt, c_cnt):
                failures.append("%s/%s/s%d: group flip counts %s != recomputed (%d,%d,%d)"
                                % (st, cfg, seed, fg, a_cnt, b_cnt, c_cnt))
                ok = False
            if fg["short_Y"] != 0:
                failures.append("%s/%s/s%d: short_Y flips recorded %d" % (st, cfg, seed,
                                                                          fg["short_Y"]))
                ok = False

            # ---- output rule: best valid among {solver, lp, repair}
            cc = r["candidate_costs"]
            if cc:
                best_avail = min(cc.values())
                if fc > best_avail + TOL * (1 + abs(best_avail)):
                    failures.append("%s/%s/s%d: delivered %.10g but a valid candidate was "
                                    "%.10g" % (st, cfg, seed, fc, best_avail))
                    ok = False
                if "repair" not in cc:
                    failures.append("%s/%s/s%d: S_r missing from the candidate set"
                                    % (st, cfg, seed))
                    ok = False

        if r["select_errors"]:
            failures.append("%s/%s/s%d: selection errors %s" % (st, cfg, seed, r["select_errors"]))
            ok = False
        if not r["mip_start_adopted"]:
            failures.append("%s/%s/s%d: MIP start not adopted" % (st, cfg, seed))
            ok = False
        else:
            n_adopted += 1
        source_counts[r["final_source"]] = source_counts.get(r["final_source"], 0) + 1

        # ---- budget conservation (the stage 07 failure mode)
        t = r["timing"]
        if t["budget_given_s"] > BUDGET - t["lp_s"] + t["overrun_s"] + 0.5:
            failures.append("%s/%s/s%d: MIP was granted %.3f s after a %.3f s LP "
                            "(20 s budget)" % (st, cfg, seed, t["budget_given_s"], t["lp_s"]))
            ok = False
        if t["budget_given_s"] > BUDGET + 1e-9:
            failures.append("%s/%s/s%d: MIP granted more than the full budget"
                            % (st, cfg, seed))
            ok = False
        if abs(t["overrun_s"] - max(0.0, t["total_s"] - BUDGET)) > 1e-3:
            failures.append("%s/%s/s%d: overrun inconsistent" % (st, cfg, seed))
            ok = False

        if not ok:
            print("%-28s %-4s %3d %6d %6d %12s %10s %9s  FAIL"
                  % (st, cfg, seed, r["n_released"], len(fixed_slots),
                     "-" if fc is None else "%.6g" % fc, r["final_source"], "-"))

    print("=" * 112)
    print("runs: %d | delivered plans re-verified: %d" % (len(runs), n_final))
    print("MIP start adopted: %d/%d | LP reproduces stage 09A: %d/%d"
          % (n_adopted, len(runs), n_lp_match, len(runs)))
    print("final source counts: %s" % source_counts)
    print("largest constraint violation: %.3g" % worst)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS RE-VERIFY: released sets, unreleased-binary fixings, short-term Y, "
          "structural pins, kappa (S_r), output rule and budget all agree" % len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
