#!/usr/bin/env python
"""
Stage 11 phase 0: build and FREEZE the matched-random Z-fixing sets.

    .venv-hs/Scripts/python.exe stage11_matched_random.py

The question this round
-----------------------
> When the NUMBER of fixed Z, their MACHINE distribution and their S_r 0/1 composition are all
> matched, is fixing the FAR-HORIZON Z still more effective?

Hypothesis to falsify
---------------------
> AB's gain does not come merely from reducing the nominal count of free binaries: the POSITION
> of the fixed far-horizon carry-overs, together with the constraint propagation it causes, has
> extra value.

This is NOT a pure causal experiment on "time position", because different positions connect
differently to the already-fixed short-term Y.  What is measured is exactly whether that
structural difference is worth explaining.

Matched-random protocol (per state)
-----------------------------------
Write

    U   = B u C                    every NON-structurally-pinned Z slot
    U_h = { z in U : (machine_j, S_r-value of z) = h }
    q_h = | C and U_h |            how many C slots AB fixes inside stratum h

1. For every stratum h, draw q_h slots from the FULL U_h WITHOUT replacement and fix them to
   S_r's value; release every other Z.  The pool is the whole stratum -- C is NOT removed from
   the pool first.  Because C and U_h is a subset of U_h we always have q_h <= |U_h|, so the
   quota can never run short; the matched-random set may keep many C slots, and may even draw C
   back exactly.  This is deliberate: the control is "the same NUMBER of fixed Z, with the same
   machine distribution and the same fixed-value composition, but randomly spread across the
   short-term and far-horizon positions", not "as far from C as possible".
2. Consequently a matched-random set can fix some B slots and leave some C slots free; the
   counts of each are reported, and they sum to |C| = 792.
3. The short-term Y is fixed to S_r in EVERY configuration, AB and matched-random alike.  Note
   that "the short-term Y is untouched" is NOT the same as "B is untouched": B is the SHORT-TERM
   Z group and its fixed/released identity is exactly what the control is allowed to shuffle.
4. Three sets are generated and written to disk BEFORE any MIP runs; both solver seeds use the
   same three sets.
5. Overlap with AB's fixing set is reported and the sets are NEVER redrawn because of a result or
   because an overlap looks unfavourable.
6. A state is flagged DEGENERATE if some stratum has q_h = |U_h| and every other stratum has
   q_h = 0: then the only way to meet the quotas is to fix C itself, i.e. the control collapses
   onto AB.  The condition 0 < q_h < |U_h| for at least one stratum is what guarantees that a
   matched set DIFFERENT from AB exists.  Degenerate states are reported as such and no further
   relaxation is applied.

Binary values are taken through the audited normalisation (round to 0/1 after an integer-tolerance
check), never from raw floats.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, build_model
from lsp_release import build_state_context
from stage06_run import solution_from_record

RANDOM_SEEDS = {"MR-Z-1": 7101, "MR-Z-2": 7102, "MR-Z-3": 7103}
STATES_FILE = "stage09b_states.json"        # stage 09B used the stage 06 frozen states
FALLBACK_STATES = "stage06_states.json"
OUT_FILE = "stage11_matched_random.json"


def main() -> int:
    path = STATES_FILE if os.path.exists(STATES_FILE) else FALLBACK_STATES
    with open(path, encoding="utf-8") as fh:
        ALL = json.load(fh)
    # development set of stage 09B: the 16 LARGE fault states (8 nominal instances, seeds 2-5)
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")
    print("state source: %s" % path)
    print("large fault states: %d" % len(states))
    if len(states) != 16:
        raise SystemExit("expected 16 large states, found %d" % len(states))

    out: Dict[str, Any] = {}
    for st in states:
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)

        # ---- audited normalisation of S_r's Z
        zval: Dict[Tuple[int, int, int], int] = {}
        norm_err = 0.0
        for (kind, i, j, t) in C["all_slots"]:
            if kind != "Z":
                continue
            raw = float(repair.Z[i, j, t - 1])
            norm_err = max(norm_err, abs(raw - round(raw)))
            zval[(i, j, t)] = int(round(raw))

        B = {(i, j, t) for (_k, i, j, t) in C["groups"]["B"]}
        Cset = {(i, j, t) for (_k, i, j, t) in C["groups"]["C"]}
        # U = every non-structurally-pinned Z slot.  The groups partition it: group A holds only
        # far-horizon Y, so the non-pinned Z slots are exactly B u C.
        U = sorted(B | Cset)
        if B & Cset:
            raise SystemExit("%s: B and C overlap" % st)

        # ---- strata: (machine, S_r value) -> quota drawn from C, pool = the FULL stratum
        want: Dict[Tuple[int, int], int] = defaultdict(int)
        by_stratum: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)
        for u in U:
            by_stratum[(u[1], zval[u])].append(u)
        for u in Cset:
            want[(u[1], zval[u])] += 1
        for key, n_want in want.items():
            have = len(by_stratum.get(key, []))
            if have < n_want:
                raise SystemExit("%s: stratum %s has %d slots but the quota is %d"
                                 % (st, key, have, n_want))
        # degenerate when the quota forces fixing C exactly
        degenerate = all(want.get(k, 0) in (0, len(v)) for k, v in by_stratum.items())
        n_distinct_strata = sum(1 for k, v in by_stratum.items() if 0 < want.get(k, 0) < len(v))
        if Cset & B:
            raise SystemExit("%s: B and C overlap" % st)

        # ---- report record
        rec_out = {
            "meta": rec["meta"], "disruption": rec["disruption"].name
            if hasattr(rec["disruption"], "name") else rec["disruption"]["name"],
            "tau": tau, "T": pert.T,
            "norm_max_error": norm_err,
            "n_C_AB_fixes": len(Cset),
            "n_U": len(U),
            "n_nonpinned_Z": len(U),
            "degenerate": bool(degenerate),
            "n_strata_with_choice": n_distinct_strata,
            "strata": {"%d|%d" % k: v for k, v in sorted(want.items())},
            "strata_pool": {"%d|%d" % k: len(v) for k, v in sorted(by_stratum.items())},
            "C_slots": sorted(list(u) for u in Cset),
            "sets": {},
        }
        for name, sd in RANDOM_SEEDS.items():
            rng = random.Random(sd + abs(hash(st)) % 100000)
            fixed: List[Tuple[int, int, int]] = []
            for key in sorted(by_stratum):
                need = want.get(key, 0)
                if need:
                    fixed.extend(rng.sample(by_stratum[key], need))
            fixed = sorted(fixed)
            fset = set(fixed)
            ov = len(fset & Cset)
            rec_out["sets"][name] = {
                "seed": sd, "n_fixed": len(fixed),
                "n_fixed_in_C": ov,
                "n_fixed_in_B": len(fset & B),
                "n_C_left_free": len(Cset - fset),
                "overlap_with_C": ov,
                "overlap_fraction_of_C": ov / len(Cset),
                "per_stratum_fixed": {"%d|%d" % k: sum(1 for u in fixed
                                                       if (u[1], zval[u]) == k)
                                      for k in sorted(want)},
                "fixed_slots": [list(u) for u in fixed],
            }
            # quota bookkeeping: per stratum the fixed count must equal the quota
            for k in want:
                got = rec_out["sets"][name]["per_stratum_fixed"]["%d|%d" % k]
                if got != want[k]:
                    raise SystemExit("%s/%s: stratum %s fixed %d, quota %d"
                                     % (st, name, k, got, want[k]))
        out[st] = rec_out
        print("  %-30s |C|=%-4d |U|=%-5d strata=%d choice=%d degen=%-5s "
              "C∩MR=%s B∩MR=%s (norm %.1g)"
              % (st, len(Cset), len(U), len(want), n_distinct_strata, degenerate,
                 "/".join(str(rec_out["sets"][n]["n_fixed_in_C"]) for n in RANDOM_SEEDS),
                 "/".join(str(rec_out["sets"][n]["n_fixed_in_B"]) for n in RANDOM_SEEDS),
                 norm_err))

    # ---- global checks
    print()
    sizes = {n: sorted({out[s]["sets"][n]["n_fixed"] for s in states}) for n in RANDOM_SEEDS}
    print("fixed counts per random set (must be 792 everywhere, = |C|): %s" % sizes)
    print("degenerate states (control collapses onto AB): %d/%d"
          % (sum(1 for s in states if out[s]["degenerate"]), len(states)))
    print("strata with a genuine choice (0 < q_h < |U_h|), per state: min=%d max=%d"
          % (min(out[s]["n_strata_with_choice"] for s in states),
             max(out[s]["n_strata_with_choice"] for s in states)))
    print()
    print("  %-30s %8s %8s %8s %8s %10s" %
          ("state", "fix in C", "fix in B", "C free", "B free", "overlap w/AB"))
    for s in states:
        r = out[s]["sets"]["MR-Z-1"]
        print("  %-30s %8d %8d %8d %8d %9.3f"
              % (s, r["n_fixed_in_C"], r["n_fixed_in_B"], r["n_C_left_free"],
                 432 - r["n_fixed_in_B"], r["overlap_fraction_of_C"]))
    ovs = {n: [out[s]["sets"][n]["overlap_with_C"] for s in states] for n in RANDOM_SEEDS}
    for n in RANDOM_SEEDS:
        v = ovs[n]
        print("  %s overlap with C: min=%d max=%d mean=%.1f  (fraction of |C|: %.3f)"
              % (n, min(v), max(v), sum(v) / len(v), sum(v) / len(v) / 792))
    # per-state: the three sets must differ from each other (otherwise "3 sets" is a sham)
    ident = 0
    for s in states:
        sets = [frozenset(tuple(u) for u in out[s]["sets"][n]["fixed_slots"])
                for n in RANDOM_SEEDS]
        if len(set(sets)) < 3:
            ident += 1
    print("states where the three random sets are not all distinct: %d" % ident)
    print("max binary normalisation error across states: %.3g"
          % max(out[s]["norm_max_error"] for s in states))

    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, OUT_FILE)
    print("\nwrote %s (frozen; NOT to be redrawn on outcome or on overlap)" % OUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
