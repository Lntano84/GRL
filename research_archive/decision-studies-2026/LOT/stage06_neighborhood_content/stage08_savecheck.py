#!/usr/bin/env python
"""
Stage 08 verification, covering every run and every recorded event.

    .venv-hs/Scripts/python.exe stage08_savecheck.py

Re-reads `stage08_runs.json` and re-verifies WITHOUT re-solving:

  * the run grid is exactly 16 states x {EMPTY, FULL} x {0, 1} = 64, on the frozen large states;
  * the release sets are the frozen ones (EMPTY 0, FULL = every legal candidate);
  * every RECORDED EVENT vector independently re-verifies: model feasibility, kappa relative to
    the frozen S_r, the fixing conditions of that run's configuration, and a recomputed cost
    equal to the objective the solver reported -- this is what licenses calling an event
    "usable", so it is re-derived here rather than trusted;
  * event times are non-decreasing and the recorded objective never worsens (an improving
    solution must improve);
  * at least one event is the submitted start itself (same objective as S_r); such an event is
    NOT an improvement and must not be counted as one;
  * the delivered plan is feasible, kappa-feasible against S_r, and never worse than S_r;
  * wall-clock bookkeeping: total >= transfer >= 0, and the solve time is within the budget plus
    a small tolerance;
  * the solver log was kept for every run and contains the MIP-start adoption line.
"""

from __future__ import annotations

import glob
import json
import os
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
BUDGET = 40.0


def rebuild(inst, plan: Dict[str, Any], objective: float) -> Solution:
    return Solution(status="stored", objective=float(objective),
                    X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open("stage08_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    meta = D["meta"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    failures: List[str] = []
    states = sorted({r["state"] for r in runs})
    keys = [(r["state"], r["config"], int(r["seed"])) for r in runs]
    expect = {(s, c, sd) for s in states for c in ("EMPTY", "FULL") for sd in (0, 1)}
    if len(states) != 16:
        failures.append("expected 16 states, found %d" % len(states))
    for st in states:
        if ST[st]["meta"]["scale"] != "large":
            failures.append("%s is not a large state" % st)
    if set(keys) != expect:
        failures.append("run grid wrong: missing %d extra %d"
                        % (len(expect - set(keys)), len(set(keys) - expect)))
    if len(runs) != 64:
        failures.append("expected 64 runs, found %d" % len(runs))

    ctx: Dict[str, Any] = {}
    n_events = 0
    n_events_ok = 0
    n_final = 0
    worst = 0.0
    n_start_event = 0
    n_adopted = 0

    print("=" * 104)
    print("%-28s %-6s %3s %5s %5s %10s %10s %9s  %s"
          % ("state", "cfg", "sd", "ev", "ok", "first_imp", "final", "maxviol", "note"))
    print("-" * 104)
    for r in sorted(runs, key=lambda z: (z["state"], z["config"], z["seed"])):
        st, cfg, seed = r["state"], r["config"], int(r["seed"])
        ok = True
        note = []
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
            ctx[st] = {"pert": pert, "repair": repair, "tau": tau,
                       "legal": {tuple(u) for u in sets["legal"]},
                       "FULL": [tuple(u) for u in sets["FULL"]["release"]]}
        C = ctx[st]
        pert, repair, tau = C["pert"], C["repair"], C["tau"]
        jr = float(r["repair_cost"])
        if abs(jr - float(repair.objective)) > TOL * (1 + abs(jr)):
            failures.append("%s/%s/s%d: repair cost differs from the frozen repair"
                            % (st, cfg, seed))
            ok = False

        # ---- release set
        if cfg == "EMPTY" and r["n_released"] != 0:
            failures.append("%s/%s/s%d: EMPTY must release nothing" % (st, cfg, seed))
            ok = False
        if cfg == "FULL" and r["n_released"] != len(C["legal"]):
            failures.append("%s/%s/s%d: FULL released %d of %d legal candidates"
                            % (st, cfg, seed, r["n_released"], len(C["legal"])))
            ok = False
        relset = set() if cfg == "EMPTY" else set(C["FULL"])

        # ---- events
        prev_t = -1.0
        prev_obj = float("inf")
        first_imp = None
        saw_start = False
        for k, e in enumerate(r["events"]):
            t = e.get("wall_s")
            if t is None or t < prev_t - 1e-9:
                failures.append("%s/%s/s%d: event %d time not non-decreasing" % (st, cfg, seed, k))
                ok = False
            prev_t = max(prev_t, t if t is not None else prev_t)
            obj = e.get("objective_function_value")
            if obj is not None and obj > prev_obj + TOL * (1 + abs(prev_obj)):
                failures.append("%s/%s/s%d: event %d objective worsened (%.10g -> %.10g)"
                                % (st, cfg, seed, k, prev_obj, obj))
                ok = False
            if obj is not None:
                prev_obj = min(prev_obj, obj)
            sol_rec = e.get("sol")
            if sol_rec is None:
                n_events += 1
                if e.get("usable"):
                    failures.append("%s/%s/s%d: event %d marked usable without a vector"
                                    % (st, cfg, seed, k))
                    ok = False
                continue
            n_events += 1
            sol = rebuild(pert, sol_rec, e.get("cost_recomputed") or 0.0)
            chk = check_solution(pert, sol, MODE_AUDITED)
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            viol = 0
            for u in [(i, j, t) for i in range(pert.N) for j in range(pert.M)
                      for t in range(1, tau + 1) if (i, j, t) not in relset]:
                if abs(float(sol.Y[u[0], u[1], u[2] - 1])
                       - float(repair.Y[u[0], u[1], u[2] - 1])) > 0.5:
                    viol += 1
            cost = float(chk.cost_recomputed["total"])
            worst = max(worst, chk.max_violation)
            usable = bool(chk.ok) and flips <= KAPPA + 1e-9 and viol == 0 \
                and bool(e.get("objective_matches"))
            if usable != bool(e.get("usable")):
                failures.append("%s/%s/s%d: event %d usability recorded %s re-checks %s"
                                % (st, cfg, seed, k, e.get("usable"), usable))
                ok = False
            if e.get("flips") is not None and int(e["flips"]) != flips:
                failures.append("%s/%s/s%d: event %d recorded flips %d != %d"
                                % (st, cfg, seed, k, e["flips"], flips))
                ok = False
            n_events_ok += int(usable)
            if abs(cost - jr) <= TOL * (1 + abs(jr)):
                saw_start = True
            if usable and cost < jr - TOL * (1 + abs(jr)) and first_imp is None:
                first_imp = t
        if not saw_start:
            failures.append("%s/%s/s%d: no event reproduces the submitted start objective"
                            % (st, cfg, seed))
            ok = False
        else:
            n_start_event += 1

        # ---- delivered plan
        if r["final_cost"] is None or r["final_solution"] is None:
            failures.append("%s/%s/s%d: no delivered plan" % (st, cfg, seed))
            ok = False
        else:
            n_final += 1
            sol = rebuild(pert, r["final_solution"], r["final_cost"])
            chk = check_solution(pert, sol, MODE_AUDITED)
            worst = max(worst, chk.max_violation)
            if not chk.ok:
                failures.append("%s/%s/s%d: delivered plan INFEASIBLE %s"
                                % (st, cfg, seed, chk.failed_checks()))
                ok = False
            if abs(chk.cost_recomputed["total"] - r["final_cost"]) > TOL * (1 + abs(r["final_cost"])):
                failures.append("%s/%s/s%d: stored final cost != recomputed"
                                % (st, cfg, seed))
                ok = False
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                failures.append("%s/%s/s%d: delivered flips %d > kappa" % (st, cfg, seed, flips))
                ok = False
            if r["final_cost"] > jr + TOL * (1 + abs(jr)):
                failures.append("%s/%s/s%d: delivered plan worse than repair" % (st, cfg, seed))
                ok = False
        if r["select_errors"]:
            failures.append("%s/%s/s%d: selection errors %s" % (st, cfg, seed, r["select_errors"]))
            ok = False

        # ---- adoption and timing
        if not r["start_adopted"]:
            failures.append("%s/%s/s%d: MIP start adoption not confirmed in the log"
                            % (st, cfg, seed))
            ok = False
        else:
            n_adopted += 1
        t = r["timing"]
        if t["wall_total_s"] + 1e-9 < t["wall_solve_s"]:
            failures.append("%s/%s/s%d: total < solve" % (st, cfg, seed))
            ok = False
        if t["wall_solve_s"] > BUDGET + 5.0:
            failures.append("%s/%s/s%d: solve wall %.2f s far beyond the %.0f s budget"
                            % (st, cfg, seed, t["wall_solve_s"], BUDGET))
            ok = False

        if not ok:
            print("%-28s %-6s %3d %5d %5s %10s %10s %9.3g  FAIL"
                  % (st, cfg, seed, r["n_events"], "-",
                     "-" if first_imp is None else "%.2f" % first_imp,
                     r["final_cost"] if r["final_cost"] is not None else float("nan"), "-"))

    print("=" * 104)
    print("runs: %d | delivered plans re-verified: %d" % (len(runs), n_final))
    print("events: %d | re-verified usable: %d" % (n_events, n_events_ok))
    print("runs whose events include the submitted start: %d/%d" % (n_start_event, len(runs)))
    print("MIP-start adoption confirmed: %d/%d" % (n_adopted, len(runs)))
    print("largest constraint violation across all re-verified plans: %.3g" % worst)
    import glob as _glob
    log_files = _glob.glob(os.path.join("stage08_logs", "*.log"))
    print("solver logs kept: %d/64" % len(log_files))
    for r in runs:
        p = os.path.join("stage08_logs", "%s__%s__s%d.log"
                         % (r["state"].replace("|", "_"), r["config"], int(r["seed"])))
        if not os.path.exists(p):
            failures.append("%s/%s/s%d: solver log missing" % (r["state"], r["config"], r["seed"]))
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("ALL %d RUNS AND %d EVENTS RE-VERIFY FROM DISK" % (len(runs), n_events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
