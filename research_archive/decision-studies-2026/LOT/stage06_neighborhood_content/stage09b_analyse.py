#!/usr/bin/env python
"""
Stage 09B analysis: which variable group is worth releasing?

    .venv-hs/Scripts/python.exe stage09b_analyse.py

Two main indicators, both keeping the ORIGINAL state-level normaliser so the comparison basis
does not move:

    G_extra(a)    = (J_LP  - J_a) / max(1, |J_r|)     extra gain after continuous re-optimisation
    G_restrict(a) = (J_ABC - J_a) / max(1, |J_r|)     whether restricting the variable classes
                                                      beats full EMPTY

Computed per (state, seed) FIRST, then averaged over the two disruptions and two seeds to give
the 8 nominal instances.  There is no "subtract 5.34% from the global mean", and no per-instance
best-configuration oracle presented as a strategy.

Reading the results (pre-agreed)
--------------------------------
* a fixed subset consistently beating ABC  -> a non-learning algorithmic baseline worth pursuing
* different instances needing different subsets, with real loss when wrong -> grounds for
  studying state-conditional selection, but only after a legal-feature test
* ABC best and restrictions useless        -> coarse fixed restriction does not improve search
* every configuration stuck at the LP cost -> these ranges still give no extra gain in 20 s
* a SMALL subset already obtaining the gain -> evidence that this subset SUFFICES; a smaller
  subset failing in the time limit does NOT prove it cannot suffice, unless an optimum or valid
  bound supports that.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

CONFIGS = ("A", "B", "C", "AB", "AC", "BC", "ABC")
SEEDS = (0, 1)
TOL = 1e-6


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def gain(jr: float, j: float) -> float:
    return (jr - j) / max(1.0, abs(jr))


def main() -> int:
    D = load("stage09b_runs.json")
    runs = D["runs"]
    A9 = load("stage09a_runs.json")["runs"]
    lp_by_state = {r["state"]: r["lp_objective"] for r in A9}

    print("Stage 09B | %d MIPs | groups A=far Y, B=short Z, C=far Z (Z_ijT pinned)"
          % len(runs))
    print("every MIP warm-started from its own re-run stage 09A LP; unreleased binaries fixed to S_r")
    print()

    # ------------------------------------------------------------------ per (state, seed)
    cell: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for r in runs:
        cell[(r["state"], int(r["seed"]))][r["config"]] = float(r["final_cost"])
    jr_of = {r["state"]: float(r["repair_cost"]) for r in runs}

    # ------------------------------------------------------------------ deliverable table
    print("=== per state: final cost by configuration (2 seeds shown as s0/s1) ===")
    hdr = ("%-28s %11s %11s" % ("state", "J_repair", "J_LP")
           + "".join("%21s" % c for c in CONFIGS))
    print(hdr)
    print("-" * len(hdr))
    for st in sorted({r["state"] for r in runs}):
        row = "%-28s %11.6g %11.6g" % (st, jr_of[st], lp_by_state[st])
        for c in CONFIGS:
            a = cell[(st, 0)].get(c)
            b = cell[(st, 1)].get(c)
            row += "%21s" % ("%.6g/%.6g" % (a, b))
        print(row)
    print()

    # ------------------------------------------------------------------ G_extra
    print("=== indicator 1: G_extra(a) = (J_LP - J_a)/max(1,|J_r|), per state x seed then averaged ===")
    def agg(fn) -> Dict[str, Dict[str, float]]:
        """config -> instance -> averaged value over (2 disruptions x 2 seeds)."""
        per_state = defaultdict(lambda: defaultdict(list))
        for (st, sd), d in cell.items():
            for c in CONFIGS:
                if c not in d:
                    continue
                v = fn(st, sd, c, d)
                if v is not None:
                    per_state[c][st].append(v)
        per_inst = defaultdict(dict)
        for c, by_state in per_state.items():
            acc = defaultdict(list)
            for st, vals in by_state.items():
                acc[st.split("|")[0]].append(sum(vals) / len(vals))
            for inst, vals in acc.items():
                per_inst[c][inst] = sum(vals) / len(vals)
        return per_inst

    ge = agg(lambda st, sd, c, d: (lp_by_state[st] - d[c]) / max(1.0, abs(jr_of[st])))
    print("  %-6s %10s %10s %10s %10s %8s %8s"
          % ("config", "mean", "median", "min", "max", "pos", "n"))
    for c in CONFIGS:
        v = sorted(ge[c].values())
        med = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
        print("  %-6s %+10.4f %+10.4f %+10.4f %+10.4f %8d %8d"
              % (c, sum(v) / len(v), med, v[0], v[-1], sum(1 for x in v if x > TOL), len(v)))
    print()

    # ------------------------------------------------------------------ G_restrict
    print("=== indicator 2: G_restrict(a) = (J_ABC - J_a)/max(1,|J_r|)  (positive = a beats ABC) ===")
    gr = agg(lambda st, sd, c, d: (d["ABC"] - d[c]) / max(1.0, abs(jr_of[st])))
    print("  %-6s %10s %10s %10s %10s %8s %8s   %s"
          % ("config", "mean", "median", "min", "max", "pos", "neg", "verdict"))
    for c in CONFIGS:
        v = sorted(gr[c].values())
        med = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
        npos = sum(1 for x in v if x > TOL)
        nneg = sum(1 for x in v if x < -TOL)
        if c == "ABC":
            verdict = "(reference)"
        elif npos == len(v) and sum(v) / len(v) > TOL:
            verdict = "BEATS ABC on every instance"
        elif npos >= 6:
            verdict = "beats ABC on %d/%d" % (npos, len(v))
        elif nneg >= 6:
            verdict = "loses to ABC on %d/%d" % (nneg, len(v))
        else:
            verdict = "mixed"
        print("  %-6s %+10.4f %+10.4f %+10.4f %+10.4f %8d %8d   %s"
              % (c, sum(v) / len(v), med, v[0], v[-1], npos, nneg, verdict))
    print()

    # ------------------------------------------------------------------ per instance detail
    print("=== per nominal instance (8): J_repair, J_LP, and each configuration's cost ===")
    insts = sorted({st.split("|")[0] for st in jr_of})
    inst_jr = {}
    for st, v in jr_of.items():
        inst_jr.setdefault(st.split("|")[0], []).append(v)
    inst_jr = {k: sum(v) / len(v) for k, v in inst_jr.items()}
    print("  %-22s %10s %10s" % ("instance", "J_repair*", "J_LP*")
          + "".join("%11s" % c for c in CONFIGS))
    for inst in insts:
        sts = [st for st in jr_of if st.split("|")[0] == inst]
        lpv = sum(lp_by_state[st] for st in sts) / len(sts)
        row = "  %-22s %10.6g %10.6g" % (inst, inst_jr[inst], lpv)
        for c in CONFIGS:
            vals = [cell[(st, sd)][c] for st in sts for sd in SEEDS if c in cell[(st, sd)]]
            row += "%11.6g" % (sum(vals) / len(vals))
        print(row)
    print("  * averages of the disruption-level values, for orientation only")
    print()

    # ------------------------------------------------------------------ where does the gain come from
    print("=== decomposition on the same normaliser (means over the 8 instances) ===")
    all_states = sorted(jr_of)
    def mean_over_states(fn):
        vals = []
        for st in all_states:
            for sd in SEEDS:
                vals.append(fn(st, sd))
        return sum(vals) / len(vals)
    cont = mean_over_states(lambda st, sd: (jr_of[st] - lp_by_state[st]) / max(1.0, abs(jr_of[st])))
    abc_extra = mean_over_states(lambda st, sd: (lp_by_state[st] - cell[(st, sd)]["ABC"])
                                 / max(1.0, abs(jr_of[st])))
    abc_total = mean_over_states(lambda st, sd: (jr_of[st] - cell[(st, sd)]["ABC"])
                                 / max(1.0, abs(jr_of[st])))
    print("  continuous LP improvement over S_r            : %+.4f" % cont)
    print("  ABC extra improvement beyond the LP           : %+.4f" % abc_extra)
    print("  ABC total improvement over S_r                : %+.4f" % abc_total)
    ab_extra = mean_over_states(lambda st, sd: (lp_by_state[st] - cell[(st, sd)]["AB"])
                                / max(1.0, abs(jr_of[st])))
    ab_total = mean_over_states(lambda st, sd: (jr_of[st] - cell[(st, sd)]["AB"])
                                / max(1.0, abs(jr_of[st])))
    print("  AB  extra improvement beyond the LP           : %+.4f" % ab_extra)
    print("  AB  total improvement over S_r                : %+.4f" % ab_total)
    print("  (same denominators = |J_r|; these are NOT optimality losses and NOT a fraction of")
    print("   'all potential gain')")
    print()
    e40_extra = None
    try:
        s8 = load("stage08_runs.json")["runs"]
        vals = []
        for st in all_states:
            for sd in SEEDS:
                rr = next(x for x in s8 if x["state"] == st and x["config"] == "EMPTY"
                          and int(x["seed"]) == sd)
                jr = rr["repair_cost"]
                ev = sorted((e for e in rr["events"]
                             if e.get("usable") and e.get("sol") is not None),
                            key=lambda e: e["wall_s"])
                best = None
                for e in ev:
                    if e["wall_s"] > 40.0 + 1e-9:
                        break
                    if best is None or e["cost_recomputed"] < best["cost_recomputed"] - TOL:
                        best = e
                j = best["cost_recomputed"] if best is not None else jr
                vals.append((lp_by_state[st] - j) / max(1.0, abs(jr)))
        e40_extra = sum(vals) / len(vals)
        print("  for reference, E@40 extra improvement beyond the LP: %+.4f" % e40_extra)
        print("  (E@40 is a 40 s trajectory prefix; ABC here gets only 20 s, so ABC is not")
        print("   expected to match it)")
    except (OSError, StopIteration):
        pass
    print()

    # ------------------------------------------------------------------ source / group usage
    print("=== what the selected plans actually changed (per configuration, averaged) ===")
    print("  %-6s %8s %10s %10s %10s %10s" % ("config", "released", "flips A", "flips B",
                                              "flips C", "solver wins"))
    for c in CONFIGS:
        sub = [r for r in runs if r["config"] == c]
        fa = sum(r["final_flips_by_group"]["A"] for r in sub) / len(sub)
        fb = sum(r["final_flips_by_group"]["B"] for r in sub) / len(sub)
        fc = sum(r["final_flips_by_group"]["C"] for r in sub) / len(sub)
        win = sum(1 for r in sub if r["final_source"] == "solver")
        print("  %-6s %8d %10.1f %10.1f %10.1f %10d/%d"
              % (c, sub[0]["n_released"], fa, fb, fc, win, len(sub)))
    print()

    # ------------------------------------------------------------------ pre-agreed reading
    print("=== pre-agreed reading ===")
    best_cfg = max((c for c in CONFIGS),
                   key=lambda c: sum(ge[c].values()) / len(ge[c]))
    beat_all = [c for c in CONFIGS
                if all(gr[c][i] > TOL for i in gr[c]) and sum(gr[c].values()) / len(gr[c]) > TOL]
    stuck = [c for c in CONFIGS
             if all(abs(x) <= TOL for x in ge[c].values())]
    print("  best configuration by mean G_extra : %s (%+.4f)"
          % (best_cfg, sum(ge[best_cfg].values()) / len(ge[best_cfg])))
    print("  configurations beating ABC on EVERY instance: %s" % (beat_all or "none"))
    print("  configurations stuck at the LP cost on every instance: %s" % (stuck or "none"))
    if beat_all:
        print("  -> a fixed subset beats full EMPTY everywhere: non-learning baseline worth")
        print("     pursuing; confirm across instances before adding a learner")
    if stuck:
        print("  -> these ranges still produce no extra gain within 20 s in any instance")
    print("  NOTE: with unconverged objectives this is a TIME-LIMITED COMBINATION EFFECT.")
    print("        Nothing here establishes mathematical complementarity or necessity.")

    with open("stage09b_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"g_extra": {c: ge[c] for c in CONFIGS},
                   "g_restrict": {c: gr[c] for c in CONFIGS},
                   "decomposition": {"continuous": cont, "abc_extra": abc_extra,
                                     "abc_total": abc_total, "ab_extra": ab_extra,
                                     "ab_total": ab_total, "e40_extra": e40_extra},
                   "best_config_by_g_extra": best_cfg,
                   "beats_abc_everywhere": beat_all,
                   "stuck_at_lp": stuck,
                   "meta": D["meta"]}, fh, indent=2, default=str)
    print("\nwrote stage09b_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
