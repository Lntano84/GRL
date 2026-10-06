#!/usr/bin/env python
"""
Stage 11 preflight: check the release/fixing construction WITHOUT solving anything.

    .venv-hs/Scripts/python.exe stage11_preflight.py

For every state and every configuration it asserts:

  * the released slots are exactly the intended set and every one is a LEGAL candidate;
  * AB releases 864 far-horizon Y + 432 short-term Z = 1296, and fixes 792 far-horizon Z;
  * ABC releases 864 + 1224 = 2088 and fixes no Z;
  * each MR-r releases 864 + (1224 - 792) = 1296 and fixes exactly its frozen 792-slot set,
    which must be a subset of B u C;
  * the short-term Y (432 slots) is fixed in EVERY configuration, AB, ABC and MR alike;
  * the number of pinned columns equals 2520 - released, and the pinned value is always S_r's
    normalised 0/1 value.

Nothing here is a result: it exists so that the 160-run batch is not launched on a mis-specified
release set.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np

import stage11_run as S
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, build_model, enumeration_space
from lsp_release import LP_BUDGET, build_state_context, pinned_bounds, stage11_columns
from stage06_run import solution_from_record

EXPECT = {
    "AB": {"rel": 1296, "rel_Y": 864, "rel_Z": 432, "free_Z": 432},
    "ABC": {"rel": 2088, "rel_Y": 864, "rel_Z": 1224, "free_Z": 1224},
    "MR-Z-1": {"rel": 1296, "rel_Y": 864, "rel_Z": 432, "free_Z": 432},
    "MR-Z-2": {"rel": 1296, "rel_Y": 864, "rel_Z": 432, "free_Z": 432},
    "MR-Z-3": {"rel": 1296, "rel_Y": 864, "rel_Z": 432, "free_Z": 432},
}


def main() -> int:
    with open("stage06_states.json", encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open("stage11_matched_random.json", encoding="utf-8") as fh:
        MATCH = json.load(fh)
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    failures: List[str] = []
    n_checked = 0

    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"]})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in S.MR_NAMES}
        legal = {tuple(u) for u in enumeration_space(pert, MODE_AUDITED)}
        B_coords = {tuple(u[1:]) for u in C["groups"]["B"]}
        C_coords = {tuple(u[1:]) for u in C["groups"]["C"]}
        short_Y = {("Y", i, j, t) for i in range(pert.N) for j in range(pert.M)
                   for t in range(1, tau + 1)}

        for method in S.METHODS:
            n_checked += 1
            rel, fixed, fixed_Z = stage11_columns(C, method)
            relset = {tuple(u) for u in rel}
            exp = EXPECT[method]

            if len(rel) != exp["rel"]:
                failures.append("%s/%s: released %d, expected %d"
                                % (st, method, len(rel), exp["rel"]))
            if len([u for u in rel if u[0] == "Y"]) != exp["rel_Y"]:
                failures.append("%s/%s: released Y %d, expected %d"
                                % (st, method, len([u for u in rel if u[0] == "Y"]), exp["rel_Y"]))
            if len([u for u in rel if u[0] == "Z"]) != exp["rel_Z"]:
                failures.append("%s/%s: released Z %d, expected %d"
                                % (st, method, len([u for u in rel if u[0] == "Z"]), exp["rel_Z"]))
            if not relset <= legal:
                failures.append("%s/%s: %d released slots are NOT legal candidates"
                                % (st, method, len(relset - legal)))
            # short-term Y fixed everywhere
            if relset & short_Y:
                failures.append("%s/%s: short-term Y is in the release set" % (st, method))
            # Pinned MODEL COLUMNS: the release set is entirely binary, and the continuous
            # columns X, I, L are neither released nor pinned by a method.  So
            #   pinned = (#binary slots) - |released|
            # NOT (n_columns - |released|), which would wrongly include the 2304 continuous
            # columns (4824 total = 2520 binary + 2304 continuous).
            if len(fixed) != len(C["assign"]) - len(rel):
                failures.append("%s/%s: pinned %d columns, expected %d"
                                % (st, method, len(fixed), len(C["assign"]) - len(rel)))
            if not relset <= {tuple(u) for u in C["all_slots"]}:
                failures.append("%s/%s: release set contains non-binary columns" % (st, method))
            # pinned values are S_r's normalised values
            for slot, val in C["assign"].items():
                kind, i, j, t = slot
                col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
                if col in fixed and abs(fixed[col] - float(val)) > 1e-12:
                    failures.append("%s/%s: pinned column %d not at S_r value" % (st, method, col))
                    break
            # per-configuration structure
            if method == "AB":
                if relset & {("Z",) + tuple(u) for u in C_coords}:
                    failures.append("%s/AB: a far-horizon Z is released" % st)
            if method == "ABC":
                if not relset >= {("Z",) + tuple(u) for u in (B_coords | C_coords)}:
                    failures.append("%s/ABC: not every non-pinned Z is released" % st)
            if method.startswith("MR"):
                if len(fixed_Z) != 792:
                    failures.append("%s/%s: fixed Z %d, expected 792"
                                    % (st, method, len(fixed_Z)))
                if not fixed_Z <= (B_coords | C_coords):
                    failures.append("%s/%s: fixing set not a subset of B u C" % (st, method))
                # the released Z must be exactly the complement inside B u C
                if {tuple(u[1:]) for u in rel if u[0] == "Z"} != (B_coords | C_coords) - fixed_Z:
                    failures.append("%s/%s: released Z is not the complement of the fixing set"
                                    % (st, method))
                # a genuinely different control: at least one B slot fixed and one C slot free
                if not (fixed_Z & B_coords):
                    failures.append("%s/%s: fixes NO B slot -- control degenerates" % (st, method))
                if not (C_coords - fixed_Z):
                    failures.append("%s/%s: leaves NO C slot free -- control equals AB"
                                    % (st, method))
            # LP pins every BINARY slot at S_r's value and leaves every continuous column free.
            #
            # Do NOT assert a predicted count of `lb != ub` columns: some FREE columns already
            # carry ub = 0 (the structurally pinned Z_ij0 / Z_ijT), while some CONTINUOUS
            # columns are legitimately zero-width after pinning (an X, Y or Z column pinned to
            # 0 by S_r).  Both effects are state dependent, so the check is the invariant:
            #   * all 2520 binary slots are pinned to S_r's normalised value;
            #   * no continuous column (X, I, L) is pinned.
            lb, ub = pinned_bounds(C)
            n_binary = len(C["assign"])
            if n_binary != 2520:
                failures.append("%s: expected 2520 binary slots, found %d" % (st, n_binary))
            for slot, val in C["assign"].items():
                kind, i, j, t = slot
                col = bm.idx.Y(i, j, t) if kind == "Y" else bm.idx.Z(i, j, t)
                if lb[col] != float(val) or ub[col] != float(val):
                    failures.append("%s: LP did not pin binary column %d to %g"
                                    % (st, col, float(val)))
                    break
            idx = bm.idx
            cont_cols = []
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, pert.T + 1):
                        cont_cols.append(idx.X(i, j, t))
                for t in range(1, pert.T + 1):
                    cont_cols.append(idx.I(i, t))
                    cont_cols.append(idx.L(i, t))
            pinned_cont = [c for c in cont_cols
                           if lb[c] != bm.var_lb[c] or ub[c] != bm.var_ub[c]]
            if pinned_cont:
                failures.append("%s: LP pinned %d continuous columns" % (st, len(pinned_cont)))

    print("checked %d (state, configuration) combinations" % n_checked)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:40]:
            print("   -", f)
        return 1
    print("PREFLIGHT PASSED: release sets, short-term-Y fixing, pinned complements and the")
    print("matched-random controls (each fixing some B and leaving some C free) are all correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
