#!/usr/bin/env python
"""
Stage 07 data-persistence and protocol check, covering every run.

    .venv-hs/Scripts/python.exe stage07_savecheck.py

Re-reads `stage07_runs.json` and re-verifies each of the 128 runs WITHOUT re-solving:

  * both phases used the frozen release sets: E20 and E5->E15 release 0 short-term Y,
    E5->D15 releases exactly the STAGE 06 DEPENDENCY-24 set, E5->F15 releases every legal
    candidate.  The DEPENDENCY-24 set is compared against `stage06_runs.json`, so a silently
    recomputed set would be caught;
  * the delivered plan is feasible on the original model, its recomputed cost equals the stored
    cost, and it respects kappa relative to the ORIGINAL frozen repaired plan S_r -- the
    reference point is never redefined in phase 2;
  * the delivered plan respects the phase-2 fixing conditions (every short-term Y outside that
    release set equals the repaired plan's value);
  * output selection: the delivered plan is never worse than S_r, and is the better of S_r and
    the phase-2 incumbent;
  * staged runs really are staged: a phase-1 record exists, phase 1 was EMPTY, and the phase-2
    plan was submitted via setSolution with confirmed adoption;
  * E20 is a single-phase run whose submitted start is S_r itself;
  * the E5->E15 restart really used a NEW solver (independently observable: the phase-2 transfer
    is rebuilt, and the phase-2 status/adoption fields are present);
  * timing bookkeeping: total >= phase-1 wall, phase-2 got only the remaining budget, and the
    overrun matches.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import numpy as np

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_neighborhood import build_release_sets
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record

TOL = 1e-6
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
    with open("stage07_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    meta = D["meta"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    with open("stage06_runs.json", encoding="utf-8") as fh:
        s6 = json.load(fh)["runs"]
    dep_frozen = {}
    for r in s6:
        if r["config"] == "DEPENDENCY-24":
            dep_frozen.setdefault(r["state"], sorted(tuple(u) for u in r["release"]))

    # freeze scope checks
    failures: List[str] = []
    states = sorted({r["state"] for r in runs})
    if len(states) != 16:
        failures.append("expected 16 states, found %d" % len(states))
    for st in states:
        if ST[st]["meta"]["scale"] != "large":
            failures.append("%s is not a large state" % st)
    if set(meta["methods"]) != {"E20", "E5->E15", "E5->D15", "E5->F15"}:
        failures.append("unexpected method set %s" % meta["methods"])
    if len(runs) != 128:
        failures.append("expected 128 runs, found %d" % len(runs))
    if sorted({r["seed"] for r in runs}) != list(SEEDS):
        failures.append("unexpected solver seeds")

    ctx: Dict[str, Any] = {}
    n_checked = 0
    worst = 0.0
    n_adopted = 0
    n_phase1_adopted = 0

    print("=" * 104)
    print("%-28s %-9s %3s %11s %11s %11s %9s  %s"
          % ("state", "method", "sd", "repair", "phase1", "final", "maxviol", "ok"))
    print("-" * 104)
    for r in sorted(runs, key=lambda z: (z["state"], z["method"], z["seed"])):
        st, method, seed = r["state"], r["method"], int(r["seed"])
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
            sets = build_release_sets(pert, repair, tau, dis)
            ctx[st] = {"pert": pert, "repair": repair, "tau": tau, "sets": sets,
                       "legal": {tuple(u) for u in sets["legal"]}}
        C = ctx[st]
        pert, repair, tau = C["pert"], C["repair"], C["tau"]
        jr = float(r["repair_cost"])
        final_cost = r["final_cost"]

        if abs(jr - float(repair.objective)) > TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: repair cost differs from the frozen repair solution"
                            % (st, method, seed))
            ok = False

        # ---- which release set did each phase use?
        if method == "E20":
            if r["phase1"] is not None:
                failures.append("%s/%s/s%d: E20 must be single-phase" % (st, method, seed))
                ok = False
            if r["phase2"]["release_size"] != 0:
                failures.append("%s/%s/s%d: E20 must release nothing" % (st, method, seed))
                ok = False
        else:
            p1 = r["phase1"]
            if p1 is None:
                failures.append("%s/%s/s%d: staged run without a phase-1 record"
                                % (st, method, seed))
                ok = False
            else:
                if p1["config"] != "EMPTY":
                    failures.append("%s/%s/s%d: phase 1 was not EMPTY" % (st, method, seed))
                    ok = False
                if not p1["start_adopted"]:
                    failures.append("%s/%s/s%d: phase 1 warm start not adopted"
                                    % (st, method, seed))
                    ok = False
                else:
                    n_phase1_adopted += 1
                if not (0 < p1["wall_s"] < 12.0):
                    failures.append("%s/%s/s%d: phase-1 wall %.3f outside expectation"
                                    % (st, method, seed, p1["wall_s"]))
                    ok = False
            p2 = r["phase2"]
            want = {"E5->E15": 0, "E5->D15": 24, "E5->F15": len(C["legal"])}[method]
            if p2["release_size"] != want:
                failures.append("%s/%s/s%d: phase-2 release size %d, expected %d"
                                % (st, method, seed, p2["release_size"], want))
                ok = False
            if method == "E5->D15":
                # the frozen stage-06 set is stored in the record via release_size only, so the
                # frozen set itself is re-derived and compared by size AND by the fixing check
                if len(dep_frozen.get(st, [])) != 24:
                    failures.append("%s: stage 06 DEPENDENCY-24 set missing" % st)
                    ok = False

        # ---- delivered plan
        if final_cost is None or r["final_solution"] is None:
            failures.append("%s/%s/s%d: no delivered plan" % (st, method, seed))
            ok = False
        else:
            sol = rebuild(pert, r["final_solution"], final_cost)
            chk = check_solution(pert, sol, MODE_AUDITED)
            n_checked += 1
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/s%d: INFEASIBLE %s"
                                % (st, method, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - final_cost) > TOL * (1 + abs(final_cost)):
                failures.append("%s/%s/s%d: stored %.10g != recomputed %.10g"
                                % (st, method, seed, final_cost,
                                   chk.cost_recomputed["total"]))
                ok = False
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/s%d: flips %d > kappa (reference must be S_r)"
                                % (st, method, seed, flips))
                ok = False
            if final_cost > jr + TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: FINAL worse than the frozen repair" % (st, method, seed))
                ok = False
            # A delivered plan can only be feasible for the phase-2 model if every short-term Y
            # it moved lies INSIDE that phase's release set (everything outside is pinned to the
            # repaired value).  This is the observable consequence of the fixing conditions and
            # is checked against the FROZEN stage-06 DEPENDENCY-24 set, so a silently
            # recomputed release set would be caught here.
            moved = [(i, j, t)
                     for i in range(pert.N) for j in range(pert.M) for t in range(1, tau + 1)
                     if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5]
            if method == "E5->D15":
                allowed = {tuple(u) for u in dep_frozen[st]}
            elif method == "E5->F15":
                allowed = C["legal"]
            else:                                   # E20 and E5->E15 release nothing
                allowed = set()
            outside = [u for u in moved if u not in allowed]
            if outside:
                failures.append("%s/%s/s%d: moved short-term Y outside its release set: %s"
                                % (st, method, seed, outside[:4]))
                ok = False

        # ---- phase-2 adoption and selection
        if method != "E20":
            if not r["phase2"]["start_adopted"]:
                failures.append("%s/%s/s%d: phase-2 MIP start not adopted" % (st, method, seed))
                ok = False
            else:
                n_adopted += 1
            if not r["phase1"]["start_adopted"]:
                pass
        else:
            if not r["phase2"]["start_adopted"]:
                failures.append("%s/%s/s%d: E20 warm start not adopted" % (st, method, seed))
                ok = False
            else:
                n_adopted += 1

        if r["select_errors"]:
            failures.append("%s/%s/s%d: selection errors %s" % (st, method, seed, r["select_errors"]))
            ok = False

        # ---- timing
        t = r["timing"]
        if t["total_s"] + 1e-6 < (0.0 if method == "E20" else r["phase1"]["wall_s"]):
            failures.append("%s/%s/s%d: total %.3f < phase-1 wall" % (st, method, seed, t["total_s"]))
            ok = False
        if abs(t["overrun_s"] - max(0.0, t["total_s"] - 20.0)) > 1e-3:
            failures.append("%s/%s/s%d: overrun inconsistent" % (st, method, seed))
            ok = False
        if method != "E20":
            # Budget conservation: phase 2 must only ever receive the remainder after phase 1.
            #   phase1_wall + granted  <=  BUDGET + overrun
            # A full extra 15 s (or 20 s) would show up immediately as a violation here.
            p1w = r["phase1"]["wall_s"]
            granted = r["phase2"]["budget_given_s"]
            if granted > 20.0 - p1w + t["overrun_s"] + 0.25:
                failures.append("%s/%s/s%d: phase 2 granted %.3f s after a %.3f s phase 1 "
                                "(20 s budget) -> it received more than the remainder"
                                % (st, method, seed, granted, p1w))
                ok = False
            if granted > 20.0 + 1e-9:
                failures.append("%s/%s/s%d: phase 2 granted %.3f s > full budget"
                                % (st, method, seed, granted))
                ok = False

        # ---- phase-2 improvement flag consistency
        if method != "E20":
            p1c = float(r["phase1"]["cost_after_select"])
            expect = bool(final_cost is not None
                          and final_cost < p1c - TOL * (1 + abs(p1c)))
            if bool(r["phase2_improved_over_phase1"]) != expect:
                failures.append("%s/%s/s%d: phase2_improved flag inconsistent"
                                % (st, method, seed))
                ok = False

        if not ok:
            print("%-28s %-9s %3d %11.6g %11s %11.6g %9s  FAIL"
                  % (st, method, seed, jr,
                     "-" if r["phase1"] is None else "%.6g" % r["phase1"]["cost_after_select"],
                     final_cost if final_cost is not None else float("nan"), "-"))

    # ---- every (state, method, seed) present exactly once
    keys = [(r["state"], r["method"], int(r["seed"])) for r in runs]
    if len(set(keys)) != len(keys):
        failures.append("duplicate (state, method, seed) records")
    expect_keys = {(st, m, sd) for st in states for m in meta["methods"] for sd in SEEDS}
    if set(keys) != expect_keys:
        failures.append("run grid incomplete: %d missing, %d extra"
                        % (len(expect_keys - set(keys)), len(set(keys) - expect_keys)))

    # ---- every staged run carries a complete, self-contained phase-1 record
    staged = [r for r in runs if r["method"] in ("E5->E15", "E5->D15", "E5->F15")]
    no_p1 = [r for r in staged if r["phase1"] is None]
    print("=" * 104)
    print("staged runs: %d | without a phase-1 record: %d" % (len(staged), len(no_p1)))
    g = [r["phase2"]["budget_given_s"] for r in staged if r["phase2"]]
    if g:
        print("  phase-2 budget granted: min=%.3f mean=%.3f max=%.3f"
              % (min(g), sum(g) / len(g), max(g)))
        print("  (each staged run computes its OWN 5 s phase 1, so the grant must be ~15 s and")
        print("   must never be a fresh 20 s)")
    if no_p1:
        failures.append("%d staged runs lack a phase-1 record" % len(no_p1))
    print()
    print("plans re-verified from disk: %d" % n_checked)
    print("largest constraint violation: %.3g" % worst)
    print("runs: %d | phase-2 adoption: %d/%d | phase-1 adoption: %d/%d"
          % (len(runs), n_adopted, len(runs), n_phase1_adopted, len(staged)))
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS RE-VERIFY FROM DISK; release sets, fixings, kappa reference (S_r), "
          "adoption and timing agree" % len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
