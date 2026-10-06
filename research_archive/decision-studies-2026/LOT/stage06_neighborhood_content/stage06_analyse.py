#!/usr/bin/env python
"""
Stage 06 analysis (CORRECTED aggregation).

    .venv-hs/Scripts/python.exe stage06_analyse.py

    Q1  Does the CONTENT of a fixed-size neighbourhood matter?
        DEPENDENCY-24 vs the mean of the three MATCHED-RANDOM sets
        DEPENDENCY-24 vs SHORTAGE-24 / WINDOW-24 / EMPTY / FULL
        WINDOW-24    vs SHORTAGE-24

    Q2  Is it worth deploying?  Each FIXED rule vs the frozen simple strategy
        "medium -> FULL, large -> EMPTY".

    G_n(a, b) = (1/4) * sum_{d} sum_{s} (J_{n,d,s,b} - J_{n,d,s,a}) / max(1, |J_r(n,d)|)

Pairs (disruption, solver seed) are formed INSIDE the instance and only then averaged, and the
normaliser is the repair cost OF THAT DISRUPTION.  See `stage06_aggregate.py` for the defect in
the first version of this script and why this definition is the correct one.

Aggregation unit is the NOMINAL INSTANCE (16 of them), never the 64 runs.
`G` is a relative cost difference against the frozen repair cost -- NOT an optimality gap.
The 2% threshold is an ENGINEERING SCREEN chosen for this round, not a literature standard.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict

from stage06_aggregate import (CONFIGS, RANDOMS, SIMPLE_PICK, THRESHOLD, describe,
                               instance_g, instance_g_random_mean, instance_g_simple,
                               load_runs, onemetric, pair_cells, scale_of)

SEEDS = (0, 1)


def label(tag: str, s: Dict[str, Any]) -> str:
    txt = ("  %-46s n=%-3d mean=%+.4f median=%+.4f min=%+.4f max=%+.4f  "
           "pos=%d neg=%d zero=%d  >=2%%: %d"
           % (tag, s["n"], s["mean"], s["median"], s["min"], s["max"],
              s["pos"], s["neg"], s["zero"], s["ge2pct"]))
    for key in ("by_scale", "by_scale_rho"):
        for k, v in s.get(key, {}).items():
            txt += "\n        %-14s mean=%+.4f (n=%d)" % (k, v["mean"], v["n"])
    return txt


def main() -> int:
    runs = load_runs()
    cells = pair_cells(runs)
    instances = sorted({k[0] for k in cells})
    print("runs: %d | nominal instances: %d | pair cells (inst,dis,seed): %d"
          % (len(runs), len(instances), len(cells)))
    print("aggregation: PAIRED per (disruption, solver seed), then averaged inside the "
          "instance; normaliser = repair cost of that disruption")
    print()

    # ---------------- absolute levels, per instance
    print("=== per-instance J_repair (the normaliser, per disruption) ===")
    dis_names = sorted({k[1] for k in cells})
    jr = {(k[0], k[1]): v for k, v in
          ((k, c["FULL"]["repair"]) for k, c in cells.items())}
    print("  %-24s %14s %14s" % ("instance", dis_names[0][:14], dis_names[1][:14]))
    for n in instances:
        print("  %-24s %14.6g %14.6g"
              % (n, jr[(n, dis_names[0])], jr[(n, dis_names[1])]))
    print()

    print("=== per-instance mean selected J (over 2 disruptions x 2 solver seeds) ===")
    print("  %-24s %10s" % ("instance", "J_repair*") + "".join("%11s" % c[:10] for c in CONFIGS))
    for n in instances:
        ks = [k for k in cells if k[0] == n]
        row = "  %-24s %10.6g" % (n, sum(cells[k]["FULL"]["repair"] for k in ks) / len(ks))
        for c in CONFIGS:
            vals = [cells[k][c]["cost"] for k in ks]
            row += "%11.6g" % (sum(vals) / len(vals))
        print(row)
    print("  * mean of the two disruptions' repair costs, for reference only; every G below")
    print("    uses the disruption-specific J_r.")
    print()

    # ---------------- Q1
    print("=== Q1: does the CONTENT of a fixed-size neighbourhood matter? ===")
    q1: Dict[str, Any] = {}
    base = instance_g_random_mean(cells, "DEPENDENCY-24")
    b = describe(list(base.values()))
    b["per_instance"] = sorted(base.items())
    for key, sel in (("by_scale", scale_of),
                     ("by_scale_rho", lambda n: "%s|%s" % tuple(n.split("_")[0:2]))):
        g: Dict[str, list] = defaultdict(list)
        for n, x in base.items():
            g[sel(n)].append(x)
        b[key] = {k: {"n": len(x), "mean": sum(x) / len(x)} for k, x in sorted(g.items())}
    q1["DEPENDENCY-24 vs mean of 3 matched-random"] = b
    print(label("DEPENDENCY-24 vs mean of 3 matched-random", b))

    for a, bb in (("DEPENDENCY-24", "SHORTAGE-24"), ("WINDOW-24", "SHORTAGE-24"),
                  ("DEPENDENCY-24", "WINDOW-24"), ("DEPENDENCY-24", "EMPTY"),
                  ("DEPENDENCY-24", "FULL"), ("WINDOW-24", "EMPTY"), ("WINDOW-24", "FULL")):
        s = onemetric(cells, a, bb)
        q1["%s vs %s" % (a, bb)] = s
        print(label("%s vs %s" % (a, bb), s))
    print()

    # per-random-control, so the control is not just an average
    print("  per matched-random control (DEPENDENCY-24 vs that single set):")
    for c in RANDOMS:
        s = onemetric(cells, "DEPENDENCY-24", c)
        q1["DEPENDENCY-24 vs %s" % c] = s
        print(label("DEPENDENCY-24 vs %s" % c, s))
    print()

    # ---------------- Q2
    print("=== Q2: each FIXED rule vs the frozen simple strategy "
          "(medium->FULL, large->EMPTY) ===")
    q2: Dict[str, Any] = {}
    for c in CONFIGS:
        v = instance_g_simple(cells, c)
        s = describe(list(v.values()))
        s["per_instance"] = sorted(v.items())
        g: Dict[str, list] = defaultdict(list)
        for n, x in v.items():
            g["%s|%s" % tuple(n.split("_")[0:2])].append(x)
        s["by_scale_rho"] = {k: {"n": len(x), "mean": sum(x) / len(x)} for k, x in sorted(g.items())}
        g2: Dict[str, list] = defaultdict(list)
        for n, x in v.items():
            g2[scale_of(n)].append(x)
        s["by_scale"] = {k: {"n": len(x), "mean": sum(x) / len(x)} for k, x in sorted(g2.items())}
        q2["always %s vs simple" % c] = s
        print(label("always %s vs simple" % c, s))
    print()
    print("  NOTE: 'best config per instance' is an ORACLE (hindsight) number and is not a")
    print("        deployable strategy; it is deliberately not computed here as a result.")
    print()

    # ---------------- criterion checks
    print("=== pre-agreed verdict criteria (DEPENDENCY-24 vs matched-random mean) ===")
    per_seed = {}
    for sd in SEEDS:
        sub = {k: v for k, v in cells.items() if k[2] == sd}
        v = instance_g_random_mean(sub, "DEPENDENCY-24")
        per_seed[sd] = describe(list(v.values()))
    c1 = b["ge2pct"] >= 4
    combo = [k for k, v in b["by_scale_rho"].items() if v["mean"] >= THRESHOLD]
    c2 = len(combo) >= 2
    same_sign = all(per_seed[s]["mean"] > 0 for s in SEEDS)
    per_scale_sign = {}
    for sd in SEEDS:
        sub = {k: v for k, v in cells.items() if k[2] == sd}
        vv = instance_g_random_mean(sub, "DEPENDENCY-24")
        for sc in ("medium", "large"):
            xs = [x for n, x in vv.items() if scale_of(n) == sc]
            per_scale_sign.setdefault(sc, []).append(sum(xs) / len(xs))
    c3 = same_sign and all(all(x > 0 for x in per_scale_sign[sc]) or
                           all(x < 0 for x in per_scale_sign[sc]) for sc in per_scale_sign)
    c4 = b["mean"] >= THRESHOLD
    print("  C1 >=4 nominal instances with G>=2%%           : %d      -> %s"
          % (b["ge2pct"], "PASS" if c1 else "FAIL"))
    print("  C2 >=2 scale-rho combos with mean G>=2%%       : %d (%s) -> %s"
          % (len(combo), ", ".join(combo), "PASS" if c2 else "FAIL"))
    print("  C3 both SOLVER SEEDS agree in direction       : per-seed mean %s -> %s"
          % (["%+.4f" % per_seed[s]["mean"] for s in SEEDS], "PASS" if c3 else "FAIL"))
    for sc, xs in per_scale_sign.items():
        print("        %-7s seed0=%+.4f seed1=%+.4f" % (sc, xs[0], xs[1]))
    print("  C4 instance-mean gain >=2%%                    : %+.4f -> %s"
          % (b["mean"], "PASS" if c4 else "FAIL"))
    print("  CONTENT verdict (C1..C4): %s"
          % ("PASSED" if (c1 and c2 and c3 and c4) else "NOT PASSED"))
    print()

    # ---------------- deployment verdict, stated separately
    print("=== deployment verdict (kept SEPARATE from the content criteria) ===")
    dep_simple = q2["always DEPENDENCY-24 vs simple"]
    best_rule = max(q2.items(), key=lambda kv: kv[1]["mean"])
    print("  always DEPENDENCY-24 vs simple : %+.4f" % dep_simple["mean"])
    print("  best fixed rule by mean        : %s (%+.4f)"
          % (best_rule[0], best_rule[1]["mean"]))
    any_pos = [k for k, v in q2.items() if v["mean"] > 0]
    if any_pos:
        print("  fixed rules with POSITIVE mean : %s" % ", ".join(any_pos))
    else:
        print("  fixed rules with POSITIVE mean : NONE (the best fixed rule is still negative)")
    print("  => deployment: %s"
          % ("SUPPORTED" if dep_simple["mean"] > 0 else
             "NOT SUPPORTED (no fixed information rule beats the simple strategy)"))
    print()

    # ---------------- flat / improvement bookkeeping
    print("=== improved-over-repair rate, and runs that produced no improvement ===")
    for c in CONFIGS:
        sub = [r for r in runs if r["config"] == c]
        imp = sum(1 for r in sub if r["improved_over_repair"])
        flat = sum(1 for r in sub
                   if abs(float(r["selected_cost"]) - float(r["repair_cost"])) <= 1e-9)
        fm = sum(1 for r in sub if abs(float(r["selected_cost"]) - float(r["repair_cost"])) <= 1e-9
                 and r["scale"] == "medium")
        fl = flat - fm
        print("  %-16s improved %3d/%3d | flat %3d (medium %d, large %d)"
              % (c, imp, len(sub), flat, fm, fl))

    with open("stage06_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"aggregation": "paired per (disruption, seed), averaged in instance; "
                                  "normaliser = disruption-specific repair cost",
                   "q1": q1, "q2": q2,
                   "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
                                "content_verdict": "PASSED" if (c1 and c2 and c3 and c4)
                                else "NOT PASSED",
                                "deployment_supported": bool(dep_simple["mean"] > 0),
                                "per_solver_seed_mean": {str(s): per_seed[s]["mean"]
                                                         for s in SEEDS}}},
                  fh, indent=2, default=str)
    print("\nwrote stage06_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
