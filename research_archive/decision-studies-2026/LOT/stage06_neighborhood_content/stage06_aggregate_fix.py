#!/usr/bin/env python
"""
Stage 06 aggregation re-check (corrected).

    .venv-hs/Scripts/python.exe stage06_aggregate_fix.py

The original per-instance aggregation was
    inst[name][cfg] = sum(vals)/len(vals)     # vals = the CURRENT state's values
    inst_jr[name]  = jr[state]
executed while looping over states, so for an instance with two disruptions the SECOND
disruption silently overwrote the first. The instance mean therefore did NOT average the two
disruptions. This script computes the paired form the report should have used:

    G_n(a,b) = (1/4) * sum_{d in {D1,D2}} sum_{s in {0,1}}
                   (J_{n,d,s,b} - J_{n,d,s,a}) / max(1, |J_r(n,d)|)

Two readings of the normaliser are computed, because they are different claims:

    jr_state    normalise by the repair cost OF THAT DISRUPTION  (J_r depends on d)
    jr_instance normalise by the instance-level repair cost (as the old inst_jr intended)

Also reports whether the repair cost actually differs between the two disruptions.
"""

from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict
from typing import Dict, List, Tuple

CONFIGS = ("FULL", "EMPTY", "SHORTAGE-24", "WINDOW-24", "DEPENDENCY-24",
           "MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")
RANDOMS = ("MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")
THRESHOLD = 0.02


def load():
    with open("stage06_runs.json", encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def buckets(runs):
    """(instance, disruption, seed) -> config -> (selected_cost, repair_cost)"""
    b: Dict[Tuple[str, str, int], Dict[str, Tuple[float, float]]] = defaultdict(dict)
    for r in runs:
        inst, dis = r["state"].split("|")
        b[(inst, dis, r["seed"])][r["config"]] = (float(r["selected_cost"]),
                                                  float(r["repair_cost"]))
    return b


def g_paired(b, a: str, bcfg: str, mode: str) -> Dict[str, float]:
    """Return instance -> G_n(a, bcfg).  mode in {jr_state, jr_instance}."""
    jr_inst = {}
    for (inst, dis, seed), d in b.items():
        jr_inst.setdefault(inst, {})[dis] = d["FULL"][1]

    acc = defaultdict(list)
    for (inst, dis, seed), d in b.items():
        if a not in d or bcfg not in d:
            continue
        ja, jra = d[a]
        jb, jrb = d[bcfg]
        jr = jra if mode == "jr_state" else jr_inst[inst][dis]
        acc[inst].append((jrb - ja) / max(1.0, abs(jr)))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def g_random_mean(b, a: str, mode: str) -> Dict[str, float]:
    """DEPENDENCY-24 vs the mean of the three matched-random configs, paired per
    (disruption, seed) FIRST, then averaged over the four pairs inside the instance."""
    jr_inst = {}
    for (inst, dis, seed), d in b.items():
        jr_inst.setdefault(inst, {})[dis] = d["FULL"][1]

    acc = defaultdict(list)
    for (inst, dis, seed), d in b.items():
        if a not in d:
            continue
        ja, jra = d[a]
        rnd = sum(d[c][0] for c in RANDOMS) / 3.0
        jr = jra if mode == "jr_state" else jr_inst[inst][dis]
        acc[inst].append((rnd - ja) / max(1.0, abs(jr)))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def old_style(runs, a: str, bcfg_or_none: str):
    """Reproduce the BUGGY aggregation, for the record."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        acc[r["state"]][r["config"]].append(float(r["selected_cost"]))
    inst, inst_jr = {}, {}
    for state, by_cfg in acc.items():
        name = state.split("|")[0]
        for cfg, vals in by_cfg.items():
            inst.setdefault(name, {})[cfg] = sum(vals) / len(vals)
        inst_jr[name] = float(next(r["repair_cost"] for r in runs if r["state"] == state))
    out = {}
    for n in inst:
        if bcfg_or_none is None:
            rnd = sum(inst[n][c] for c in RANDOMS) / 3.0
            bv = rnd
        else:
            bv = inst[n][bcfg_or_none]
        out[n] = (bv - inst[n][a]) / max(1.0, abs(inst_jr[n]))
    return inst, inst_jr, out


def report(label: str, vals: List[float]) -> None:
    sv = sorted(vals)
    med = sv[len(sv) // 2] if len(sv) % 2 else (sv[len(sv) // 2 - 1] + sv[len(sv) // 2]) / 2
    print("  %-34s n=%-3d mean=%+.4f med=%+.4f min=%+.4f max=%+.4f pos=%d neg=%d >=2%%=%d"
          % (label, len(vals), sum(vals) / len(vals), med, min(vals), max(vals),
             sum(1 for v in vals if v > 1e-9), sum(1 for v in vals if v < -1e-9),
             sum(1 for v in vals if v >= THRESHOLD)))


def main() -> int:
    runs = load()
    b = buckets(runs)

    # ---- does J_r differ between the two disruptions of an instance?
    print("=== does the repair cost depend on the disruption? ===")
    diff = 0
    for inst in sorted({k[0] for k in b}):
        js = {dis: b[(inst, dis, 0)]["FULL"][1] for dis in ("D1_m0_2p", "D2_all_1p")}
        d1, d2 = js["D1_m0_2p"], js["D2_all_1p"]
        rel = abs(d1 - d2) / max(1.0, abs(d1))
        if rel > 1e-9:
            diff += 1
        print("  %-22s D1=%-12.8g D2=%-12.8g  rel.diff=%.3g" % (inst, d1, d2, rel))
    print("  instances where J_r differs by disruption: %d/16" % diff)
    print()

    print("=== Q1  DEPENDENCY-24 vs mean of 3 matched-random ===")
    for mode in ("jr_state", "jr_instance"):
        v = g_random_mean(b, "DEPENDENCY-24", mode)
        report("paired, %s, ALL 16" % mode, list(v.values()))
        for scale in ("medium", "large"):
            vals = [g for n, g in v.items() if n.startswith(scale)]
            report("  paired, %s, %s" % (mode, scale), vals)
    print("  --- buggy original, for comparison ---")
    _, _, old = old_style(runs, "DEPENDENCY-24", None)
    report("original (overwriting) aggregation", list(old.values()))
    report("  original medium", [g for n, g in old.items() if n.startswith("medium")])
    report("  original large", [g for n, g in old.items() if n.startswith("large")])
    print()

    print("=== Q2  always DEPENDENCY-24 vs the frozen simple strategy ===")
    jr_inst = {}
    for (inst, dis, seed), d in b.items():
        jr_inst.setdefault(inst, {})[dis] = d["FULL"][1]
    for mode in ("jr_state", "jr_instance"):
        acc = defaultdict(list)
        for (inst, dis, seed), d in b.items():
            pick = "FULL" if inst.startswith("medium") else "EMPTY"
            ja, jra = d["DEPENDENCY-24"]
            jb = d[pick][0]
            jr = jra if mode == "jr_state" else jr_inst[inst][dis]
            acc[inst].append((jb - ja) / max(1.0, abs(jr)))
        v = {k: sum(x) / len(x) for k, x in acc.items()}
        report("paired, %s, ALL 16" % mode, list(v.values()))
    print()

    # ---- C3 done properly: do the two SOLVER seeds agree in direction?
    print("=== C3 done properly: per-solver-seed direction of DEP vs random-mean ===")
    jr_of = {}
    for (inst, dis, seed), d in b.items():
        jr_of[(inst, dis, seed)] = d["DEPENDENCY-24"][1]
    for mode in ("jr_state",):
        for sd in (0, 1):
            acc = defaultdict(list)
            for (inst, dis, seed), d in b.items():
                if seed != sd:
                    continue
                ja, jra = d["DEPENDENCY-24"]
                rnd = sum(d[c][0] for c in RANDOMS) / 3.0
                jr = jra if mode == "jr_state" else jr_inst[inst][dis]
                acc[inst].append((rnd - ja) / max(1.0, abs(jr)))
            v = {k: sum(x) / len(x) for k, x in acc.items()}
            report("solver seed %d, ALL 16" % sd, list(v.values()))
            report("  solver seed %d, medium" % sd,
                   [g for n, g in v.items() if n.startswith("medium")])
            report("  solver seed %d, large" % sd,
                   [g for n, g in v.items() if n.startswith("large")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
