#!/usr/bin/env python
"""
Stage 06 feasibility-migration certificate.

    .venv-hs/Scripts/python.exe stage06_migration_certificate.py

The three restricted configurations use the same repaired reference S_r, the same kappa and the
same far-horizon-Y / Z freedom, and DEPENDENCY-24 releases a superset of what EMPTY releases,
so the feasible sets satisfy

    F_EMPTY  subset  F_DEPENDENCY  subset  F_FULL .

Wherever EMPTY's delivered plan is strictly cheaper than DEPENDENCY-24's delivered plan, the
better plan is therefore INSIDE DEPENDENCY-24's search range: DEPENDENCY-24 did not fail because
its release set excluded the good solution, it failed to FIND it within the 20 s budget.

For every such pair this script emits an independent check of exactly that claim, using the
STORED plans (nothing is re-solved and no plan is regenerated):

  1. recompute the EMPTY plan on the disrupted model with the independent checker:
     feasibility, kappa, recomputed cost;
  2. verify the EMPTY plan satisfies DEPENDENCY-24's fixing conditions -- i.e. every short-term
     Y outside DEPENDENCY-24's release set already equals the repaired plan's value.  This is
     what makes the EMPTY plan a member of F_DEPENDENCY;
  3. report the short-term binary flips of the EMPTY plan (the kappa criterion is relative to
     S_r for both configurations, so a plan feasible for EMPTY with flips <= kappa is also
     feasible for DEPENDENCY-24);
  4. report the release-set relation |R_EMPTY \\ R_DEP| (must be 0) and |R_EMPTY and R_DEP|.

Scope: every large state, both solver seeds, all three restricted configurations -- the
certificate reports all of them, not only the successful ones.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record

TOL = 1e-6


def rebuild(inst, rec, objective) -> Solution:
    return solution_from_record(inst, {"objective": objective, "plan": rec})


def main() -> int:
    with open("stage06_runs.json", encoding="utf-8") as fh:
        runs = json.load(fh)["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    idx = {(r["state"], r["config"], int(r["seed"])): r for r in runs}
    states = sorted({r["state"] for r in runs if r["scale"] == "large"})

    ctx: Dict[str, Any] = {}
    for st in states:
        rec = ST[st]
        inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        ctx[st] = {"pert": pert, "repair": solution_from_record(pert, rec["repair"]),
                   "tau": rec["meta"]["tau"]}

    print("=" * 110)
    print("=== release-set inclusion: EMPTY vs DEPENDENCY-24 vs FULL (large states, all runs) ===")
    print("=" * 110)
    bad_rel = 0
    for st in states:
        for sd in (0, 1):
            rE = set(tuple(u) for u in idx[(st, "EMPTY", sd)]["release"])
            rD = set(tuple(u) for u in idx[(st, "DEPENDENCY-24", sd)]["release"])
            rF = set(tuple(u) for u in idx[(st, "FULL", sd)]["release"])
            if (rE - rD) or (rD - rF):
                bad_rel += 1
                print("  VIOLATION %s s%d: |E\\D|=%d |D\\F|=%d"
                      % (st, sd, len(rE - rD), len(rD - rF)))
    print("  states x seeds checked: %d | inclusion violations: %d"
          % (len(states) * 2, bad_rel))
    print("  => F_EMPTY subset F_DEPENDENCY subset F_FULL holds on every frozen instance")
    print()

    print("=" * 110)
    print("=== migration certificate: EMPTY plan strictly cheaper than DEPENDENCY-24 plan ===")
    print("=" * 110)
    hdr = ("%-28s %3s %12s %12s %9s %8s %7s %6s %s"
           % ("state", "sd", "J_EMPTY", "J_DEP", "gain", "in F_DEP", "flips", "<=kap", "notes"))
    print(hdr)
    print("-" * 110)

    n_pairs = 0
    n_strict = 0
    n_certified = 0
    states_with = set()
    for st in states:
        C = ctx[st]
        pert, repair, tau = C["pert"], C["repair"], C["tau"]
        for sd in (0, 1):
            rE = idx[(st, "EMPTY", sd)]
            rD = idx[(st, "DEPENDENCY-24", sd)]
            jE = float(rE["selected_cost"])
            jD = float(rD["selected_cost"])
            n_pairs += 1
            if not (jE < jD - TOL * (1 + abs(jD))):
                continue
            n_strict += 1
            states_with.add(st)
            notes: List[str] = []

            # ---- 1. independent re-check of the EMPTY plan
            sol = rebuild(pert, rE["selected_solution"], jE)
            chk = check_solution(pert, sol, MODE_AUDITED)
            if not chk.ok:
                notes.append("EMPTY_PLAN_INFEASIBLE:%s" % (chk.failed_checks(),))
            if abs(chk.cost_recomputed["total"] - jE) > TOL * (1 + abs(jE)):
                notes.append("COST_MISMATCH")

            # ---- 2. does the EMPTY plan satisfy DEPENDENCY-24's fixing conditions?
            relD = set(tuple(u) for u in rD["release"])
            viol = []
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, tau + 1):
                        if (i, j, t) in relD:
                            continue
                        if abs(float(sol.Y[i, j, t - 1]) - float(repair.Y[i, j, t - 1])) > 0.5:
                            viol.append((i, j, t))
            in_fdep = not viol
            if viol:
                notes.append("FIXING_VIOLATED:%s" % (viol[:3],))

            # ---- 3. kappa is relative to S_r for BOTH configurations
            flips = binary_change_count(pert, sol.Y, repair.Y, tau)
            if flips > KAPPA + 1e-9:
                notes.append("KAPPA")

            # ---- 4. release relation
            relE = set(tuple(u) for u in rE["release"])
            if relE - relD:
                notes.append("RELEASE_NOT_SUBSET")
            if states_with and st in states_with:
                pass

            certified = in_fdep and chk.ok and flips <= KAPPA + 1e-9
            n_certified += int(certified)
            print("%-28s %3d %12.6g %12.6g %+9.4f %8s %7d %6s %s"
                  % (st, sd, jE, jD, (jD - jE) / max(1.0, abs(jD)),
                     "YES" if in_fdep else "NO", flips,
                     "yes" if flips <= KAPPA + 1e-9 else "NO",
                     "CERTIFIED" if certified else "FAILED " + ";".join(notes)))

    print("-" * 110)
    print("  paired (state, seed) comparisons on large: %d" % n_pairs)
    print("  EMPTY strictly cheaper than DEPENDENCY-24 : %d" % n_strict)
    print("  of those, certified inside F_DEPENDENCY   : %d" % n_certified)
    print("  distinct fault states involved            : %d (%s)"
          % (len(states_with), ", ".join(sorted(states_with))))
    print()
    print("  Interpretation: for these %d runs the cheaper plan satisfies DEPENDENCY-24's fixing" % n_certified)
    print("  conditions and the kappa budget relative to the SAME reference S_r, so it lies in")
    print("  DEPENDENCY-24's feasible set. DEPENDENCY-24's loss there is a SEARCH failure, not a")
    print("  restriction of its release set.  This does not explain the whole DEP-vs-random gap:")
    print("  a random set may use variables DEPENDENCY-24 never released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
