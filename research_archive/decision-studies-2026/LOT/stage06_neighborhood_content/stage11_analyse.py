#!/usr/bin/env python
"""
Stage 11 analysis: matched-random Z-fixing control.

    .venv-hs/Scripts/python.exe stage11_analyse.py

Primary indicator, on the ORIGINAL denominator

    G_n = mean_{d,s} ( (1/3) * sum_r J_{MR-r,n,d,s} - J_{AB,n,d,s} ) / max(1, |J_r,n,d|)

so the three random sets are averaged INSIDE each (disruption, seed) pair and only then
aggregated -- the primary control is their MEAN performance, never the best-looking one.

Aggregation unit is the nominal instance (8 of them); the two disruptions and two solver seeds
are averaged inside an instance.  Reported with it: the two rho groups, the direction of each
solver seed, and the spread across the three random sets.

AB vs ABC is re-measured as a control.

Pre-agreed EXPLORATORY engineering criteria
-------------------------------------------
* AB beats the matched random mean by >= 2%, at least 6/8 instances positive, both rho groups
  positive  -> supports continuing to study the fixed-POSITION and constraint-propagation
  mechanism; NOT an independent confirmation
* the random fixings obtain comparable gains, AB has no stable extra benefit -> explain general
  fixing rather than making "far-horizon position" the core contribution
* the random fixings are stably BETTER -> keep the stronger control, fix AB's research framing,
  do not abandon the problem
* small effect, reversed groups, or large spread across the random sets -> record as
  inconclusive; do not start learning and do not pick a favourable random set after the fact
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")
SEEDS = (0, 1)
TOL = 1e-6


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    D = load("stage11_runs.json")
    runs = D["runs"]
    print("Stage 11 | %d MIPs | matched-random Z-fixing control" % len(runs))
    print("scope: %s" % D["meta"]["scope"])
    print()

    jr = {r["state"]: float(r["repair_cost"]) for r in runs}
    lp = {r["state"]: r["lp_objective"] for r in runs}
    cost: Dict[Tuple[str, str, int], float] = {}
    for r in runs:
        cost[(r["state"], r["method"], int(r["seed"]))] = float(r["final_cost"])
    states = sorted(jr)
    insts = sorted({s.split("|")[0] for s in states})

    def per_instance(a: str, b) -> Dict[str, float]:
        """G_n(a, b): b may be a method name or the literal 'MR_MEAN'."""
        acc: Dict[str, List[float]] = defaultdict(list)
        for st in states:
            for sd in SEEDS:
                if b == "MR_MEAN":
                    bv = sum(cost[(st, m, sd)] for m in MR) / len(MR)
                else:
                    bv = cost[(st, b, sd)]
                acc[st.split("|")[0]].append((bv - cost[(st, a, sd)]) / max(1.0, abs(jr[st])))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def per_seed(a: str, b, sd: int) -> Dict[str, float]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for st in states:
            if b == "MR_MEAN":
                bv = sum(cost[(st, m, sd)] for m in MR) / len(MR)
            else:
                bv = cost[(st, b, sd)]
            acc[st.split("|")[0]].append((bv - cost[(st, a, sd)]) / max(1.0, abs(jr[st])))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def stats(v: Dict[str, float]) -> Dict[str, Any]:
        vals = [v[i] for i in insts]
        sv = sorted(vals)
        n = len(sv)
        med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        out = {"mean": sum(vals) / n, "median": med, "min": sv[0], "max": sv[-1],
               "pos": sum(1 for x in vals if x > TOL),
               "neg": sum(1 for x in vals if x < -TOL),
               "tie": sum(1 for x in vals if abs(x) <= TOL)}
        for rho in ("0.75", "1.10"):
            g = [v[i] for i in insts if ("rho%s" % rho) in i]
            out["rho" + rho] = {"n": len(g), "mean": sum(g) / len(g), "pos": sum(1 for x in g if x > TOL)}
        return out

    def show(tag: str, v: Dict[str, float], s: Dict[str, Any]) -> None:
        print("=== %s ===" % tag)
        print("  n=%d  mean=%+.4f  median=%+.4f  min=%+.4f  max=%+.4f  pos/neg/tie=%d/%d/%d"
              % (len(insts), s["mean"], s["median"], s["min"], s["max"],
                 s["pos"], s["neg"], s["tie"]))
        for rho in ("0.75", "1.10"):
            d = s["rho" + rho]
            print("    rho%-5s n=%d mean=%+.4f positive=%d" % (rho, d["n"], d["mean"], d["pos"]))
        for i in insts:
            print("    %-22s %+.4f" % (i, v[i]))
        print()

    # ---------------- primary
    v_prim = per_instance("AB", "MR_MEAN")
    s_prim = stats(v_prim)
    print("=== PRIMARY: AB vs the MEAN of the three matched-random fixings ===")
    print("  (positive => AB is cheaper than the matched random mean)")
    print("  n=%d  mean=%+.4f  median=%+.4f  min=%+.4f  max=%+.4f  pos/neg/tie=%d/%d/%d"
          % (len(insts), s_prim["mean"], s_prim["median"], s_prim["min"], s_prim["max"],
             s_prim["pos"], s_prim["neg"], s_prim["tie"]))
    for rho in ("0.75", "1.10"):
        d = s_prim["rho" + rho]
        print("    rho%-5s n=%d mean=%+.4f positive=%d" % (rho, d["n"], d["mean"], d["pos"]))
    print("  per nominal instance:")
    for i in insts:
        print("    %-22s %+.4f" % (i, v_prim[i]))
    same = opp = 0
    for i in insts:
        v0 = per_seed("AB", "MR_MEAN", 0)[i]
        v1 = per_seed("AB", "MR_MEAN", 1)[i]
        if (v0 > 0) == (v1 > 0):
            same += 1
        else:
            opp += 1
        print("    seed direction %-22s seed0=%+.4f seed1=%+.4f %s"
              % (i, v0, v1, "same" if (v0 > 0) == (v1 > 0) else "OPPOSITE"))
    print("  seed direction: same=%d opposite=%d" % (same, opp))
    print()

    # ---------------- spread across the three random sets
    print("=== spread across the three matched-random sets (is one of them carrying it?) ===")
    per_r = {}
    for m in MR:
        v = per_instance("AB", m)
        per_r[m] = stats(v)
        print("  AB vs %-8s mean=%+.4f median=%+.4f min=%+.4f max=%+.4f pos=%d"
              % (m, per_r[m]["mean"], per_r[m]["median"], per_r[m]["min"], per_r[m]["max"],
                 per_r[m]["pos"]))
    means = [per_r[m]["mean"] for m in MR]
    print("  range across the three means: %.4f  (min=%+.4f max=%+.4f)"
          % (max(means) - min(means), min(means), max(means)))
    # per-state best/worst random set, to show how much the choice matters
    worst_case = []
    for st in states:
        for sd in SEEDS:
            ab = cost[(st, "AB", sd)]
            vals = [(cost[(st, m, sd)] - ab) / max(1.0, abs(jr[st])) for m in MR]
            worst_case.append(max(vals))
    print("  if the WORST-looking random set were chosen per run (an oracle we do NOT use):")
    print("    mean=%+.4f  (this is an upper bound on how much set choice could matter)"
          % (sum(worst_case) / len(worst_case)))
    print()

    # ---------------- control: AB vs ABC
    v_abc = per_instance("AB", "ABC")
    s_abc = stats(v_abc)
    print("=== control re-measurement: AB vs ABC (short-term Y fixed in both) ===")
    print("  n=%d  mean=%+.4f  median=%+.4f  min=%+.4f  max=%+.4f  pos/neg/tie=%d/%d/%d"
          % (len(insts), s_abc["mean"], s_abc["median"], s_abc["min"], s_abc["max"],
             s_abc["pos"], s_abc["neg"], s_abc["tie"]))
    for rho in ("0.75", "1.10"):
        d = s_abc["rho" + rho]
        print("    rho%-5s n=%d mean=%+.4f positive=%d" % (rho, d["n"], d["mean"], d["pos"]))
    print()

    # ---------------- orientation: everything against the repair cost
    print("=== G over the repair cost, per method (orientation only) ===")
    print("  %-8s %10s %10s %10s %10s" % ("method", "mean", "rho0.75", "rho1.10", "proven opt"))
    for m in ("AB", "ABC") + MR:
        vals, g = [], {"0.75": [], "1.10": []}
        for st in states:
            for sd in SEEDS:
                x = (jr[st] - cost[(st, m, sd)]) / max(1.0, abs(jr[st]))
                vals.append(x)
                g["1.10" if "rho1.10" in st else "0.75"].append(x)
        opt = sum(1 for r in runs if r["method"] == m and r["mip_proven_optimal"])
        tot = sum(1 for r in runs if r["method"] == m)
        print("  %-8s %+10.4f %+10.4f %+10.4f %10s"
              % (m, sum(vals) / len(vals), sum(g["0.75"]) / len(g["0.75"]),
                 sum(g["1.10"]) / len(g["1.10"]), "%d/%d" % (opt, tot)))
    print()

    # ---------------- source vs strict improvement
    print("=== source vs strict improvement (kept apart) ===")
    for m in ("AB", "ABC") + MR:
        sub = [r for r in runs if r["method"] == m]
        s = sum(1 for r in sub if r["strictly_below_lp"])
        ts = sorted(r["timing"]["total_s"] for r in sub)
        print("  %-8s strictly<LP %2d/%-2d | mean total %.2f s" % (m, s, len(sub),
                                                                   sum(ts) / len(ts)))
    print()

    # ---------------- pre-agreed criteria
    print("=== pre-agreed EXPLORATORY criteria ===")
    c1 = s_prim["mean"] >= 0.02 and s_prim["pos"] >= 6
    c2 = s_prim["rho0.75"]["mean"] > 0 and s_prim["rho1.10"]["mean"] > 0
    c3 = all(per_r[m]["mean"] > 0 for m in MR)
    print("  mean >= +2%%                    : %+.4f -> %s" % (s_prim["mean"], c1 and s_prim["mean"] >= 0.02))
    print("  at least 6/8 instances positive : %d/8" % s_prim["pos"])
    print("  both rho groups positive        : rho0.75 %+.4f, rho1.10 %+.4f -> %s"
          % (s_prim["rho0.75"]["mean"], s_prim["rho1.10"]["mean"], c2))
    print("  every one of the three sets gives AB the edge: %s" % c3)
    if c1 and c2 and c3:
        print("  => SUPPORTS continuing to study the fixed-POSITION / constraint-propagation")
        print("     mechanism.  This is NOT an independent confirmation.")
    elif s_prim["mean"] <= TOL:
        print("  => matched random fixings obtain comparable gains: explain GENERAL fixing")
        print("     optimisation rather than making far-horizon position the core contribution.")
    else:
        print("  => inconclusive by the pre-agreed criteria: record as uncertain, do not start")
        print("     learning, and do not pick a favourable random set after the fact.")
    print("  2%% remains an engineering reference against J_r, not a publication standard.")

    with open("stage11_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"primary_ab_vs_mr_mean": {"stats": s_prim, "per_instance": v_prim},
                   "per_random_set": {m: per_r[m] for m in MR},
                   "control_ab_vs_abc": {"stats": s_abc, "per_instance": v_abc},
                   "seed_direction_same": same, "seed_direction_opposite": opp,
                   "worst_random_choice_mean": sum(worst_case) / len(worst_case),
                   "meta": D["meta"]}, fh, indent=2, default=str)
    print("\nwrote stage11_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
