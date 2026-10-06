#!/usr/bin/env python
"""
Stage 10 analysis: independent confirmation of AB.

    .venv-hs/Scripts/python.exe stage10_analyse.py

PRIMARY comparison, fixed before the new results were seen: AB vs FULL.

    G_n = (1/4) * sum_d sum_s (J_FULL,n,d,s - J_AB,n,d,s) / max(1, |J_r,n,d|)

The statistical unit is the 16 NOMINAL INSTANCES.  Faults and solver seeds are never treated as
independent samples: they are averaged inside an instance first.

Reported for the primary comparison: mean, median, positive/negative/tie instance counts, the two
rho groups, the direction of each solver seed per instance, and the worst loss.  The uncertainty
of the mean is a bootstrap interval resampling NOMINAL INSTANCES within each rho group.

Secondary, pre-specified: AB vs ABC, AB vs A, AB vs BC.  The primary is not replaced because a
secondary looks better, and no per-instance best configuration is presented as a strategy.

2% remains an engineering reference against the repair cost, not a general publication standard.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

PRIMARY = ("AB", "FULL")
SECONDARY = (("AB", "ABC"), ("AB", "A"), ("AB", "BC"))
SEEDS = (0, 1)
TOL = 1e-6
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20261002


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    D = load("stage10_runs.json")
    runs = D["runs"]
    print("Stage 10 | %d MIPs | independent confirmation on NEW instances" % len(runs))
    print("scope: %s" % D["meta"]["scope"])
    print("PRIMARY comparison (pre-registered): %s vs %s" % PRIMARY)
    print("secondary (pre-specified): %s" % ", ".join("%s vs %s" % p for p in SECONDARY))
    print()

    jr = {r["state"]: float(r["repair_cost"]) for r in runs}
    lp = {r["state"]: r["lp_objective"] for r in runs}
    cell: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for r in runs:
        cell[(r["state"], int(r["seed"]))][r["method"]] = float(r["final_cost"])
    states = sorted(jr)
    instances = sorted({s.split("|")[0] for s in states})

    def per_instance(a: str, b: str) -> Dict[str, float]:
        """G_n(a,b) > 0 means a is cheaper; 4 pairs averaged inside the instance."""
        acc: Dict[str, List[float]] = defaultdict(list)
        for st in states:
            inst = st.split("|")[0]
            for sd in SEEDS:
                d = cell[(st, sd)]
                if a not in d or b not in d:
                    continue
                acc[inst].append((d[b] - d[a]) / max(1.0, abs(jr[st])))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def per_seed(a: str, b: str, sd: int) -> Dict[str, float]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for st in states:
            d = cell[(st, sd)]
            if a not in d or b not in d:
                continue
            acc[st.split("|")[0]].append((d[b] - d[a]) / max(1.0, abs(jr[st])))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def bootstrap(vals: List[float]) -> Tuple[float, float]:
        """STRATIFIED by rho: each draw takes 8 instances from each rho group, then pools.

        The first version resampled all 16 instances together, which is NOT stratified and must
        not have been labelled as such.  The point estimate and the 16/16 positive count are
        unaffected by this choice; only the interval is.
        """
        rng = random.Random(BOOTSTRAP_SEED)
        groups = [g for g in (
            [v for i, v in zip(instances, vals) if "rho0.75" in i],
            [v for i, v in zip(instances, vals) if "rho1.10" in i],
        ) if g]
        n = sum(len(g) for g in groups)
        means = []
        for _ in range(BOOTSTRAP_DRAWS):
            tot = 0.0
            for g in groups:
                tot += sum(g[rng.randrange(len(g))] for _ in range(len(g)))
            means.append(tot / n)
        means.sort()
        return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]

    def bootstrap_unstratified(vals: List[float]) -> Tuple[float, float]:
        """Kept only to show the size of the difference; not used for any claim."""
        rng = random.Random(BOOTSTRAP_SEED)
        n = len(vals)
        means = []
        for _ in range(BOOTSTRAP_DRAWS):
            means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]

    def report(tag: str, a: str, b: str, is_primary: bool) -> Dict[str, Any]:
        v = per_instance(a, b)
        vals = [v[i] for i in instances]
        sv = sorted(vals)
        n = len(sv)
        med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        print("%s=== %s: %s vs %s (positive => %s cheaper) ===" % (
            "PRIMARY " if is_primary else "", tag, a, b, a))
        print("  n=%d  mean=%+.4f  median=%+.4f  min=%+.4f  max=%+.4f"
              % (n, sum(vals) / n, med, sv[0], sv[-1]))
        print("  positive=%d  negative=%d  tie=%d   (of %d nominal instances)"
              % (sum(1 for x in vals if x > TOL), sum(1 for x in vals if x < -TOL),
                 sum(1 for x in vals if abs(x) <= TOL), n))
        lo, hi = bootstrap(vals)
        print("  bootstrap 95%% interval of the mean: [%+.4f, %+.4f]  (STRATIFIED by rho; unit = "
              "nominal instance)" % (lo, hi))
        ulo, uhi = bootstrap_unstratified(vals)
        print("  same, unstratified (NOT used for any claim): [%+.4f, %+.4f]" % (ulo, uhi))
        for rho in ("0.75", "1.10"):
            g = [v[i] for i in instances if ("rho%s" % rho) in i]
            glo, ghi = bootstrap(g)
            print("    rho%-5s n=%d mean=%+.4f median=%+.4f  bootstrap [%+.4f, %+.4f]"
                  % (rho, len(g), sum(g) / len(g), sorted(g)[len(g) // 2], glo, ghi))
        # per-instance seed direction
        same = opp = 0
        rows = []
        for i in instances:
            v0 = per_seed(a, b, 0)[i]
            v1 = per_seed(a, b, 1)[i]
            ok = (v0 > 0) == (v1 > 0)
            same += ok
            opp += (not ok)
            rows.append((i, v0, v1, ok))
        print("  per-instance seed direction: same=%d  opposite=%d" % (same, opp))
        for i, v0, v1, ok in rows:
            print("    %-24s seed0=%+.4f seed1=%+.4f  %s"
                  % (i, v0, v1, "same" if ok else "OPPOSITE"))
        print("  worst loss for %s: %+.4f (%s)" % (
            a, sv[0], [i for i in instances if abs(v[i] - sv[0]) < 1e-12][0]))
        print()
        return {"comparison": "%s vs %s" % (a, b), "mean": sum(vals) / n, "median": med,
                "min": sv[0], "max": sv[-1],
                "positive": sum(1 for x in vals if x > TOL),
                "negative": sum(1 for x in vals if x < -TOL),
                "tie": sum(1 for x in vals if abs(x) <= TOL),
                "bootstrap_ci": [lo, hi], "seed_same": same, "seed_opposite": opp,
                "per_instance": rows}

    out = {"primary": report("primary", PRIMARY[0], PRIMARY[1], True)}
    out["secondary"] = [report("secondary", a, b, False) for a, b in SECONDARY]

    # ---------------- rho group means for every method vs the repair cost
    print("=== G over the repair cost, per method and rho (orientation only) ===")
    print("  %-6s %10s %10s %10s %12s" % ("method", "mean", "rho0.75", "rho1.10", "proven opt"))
    for m in D["meta"]["methods"]:
        vals, g = [], {"0.75": [], "1.10": []}
        for st in states:
            for sd in SEEDS:
                c = cell[(st, sd)].get(m)
                if c is None:
                    continue
                x = (jr[st] - c) / max(1.0, abs(jr[st]))
                vals.append(x)
                g["1.10" if "rho1.10" in st else "0.75"].append(x)
        opt = sum(1 for r in runs if r["method"] == m and r["mip_proven_optimal"])
        tot = sum(1 for r in runs if r["method"] == m)
        print("  %-6s %+10.4f %+10.4f %+10.4f %12s"
              % (m, sum(vals) / len(vals), sum(g["0.75"]) / len(g["0.75"]),
                 sum(g["1.10"]) / len(g["1.10"]), "%d/%d" % (opt, tot)))
    print()

    # ---------------- timing and source/improvement split
    print("=== running time, and source vs strict improvement (kept separate) ===")
    print("  %-6s %12s %12s %14s %10s" % ("method", "mean_total", "median_total",
                                           "strictly<LP", "=LP"))
    for m in D["meta"]["methods"]:
        sub = [r for r in runs if r["method"] == m]
        ts = sorted(r["timing"]["total_s"] for r in sub)
        med = ts[len(ts) // 2]
        s = sum(1 for r in sub if r["strictly_below_lp"])
        print("  %-6s %12.2f %12.2f %14s %10s"
              % (m, sum(ts) / len(ts), med, "%d/%d" % (s, len(sub)),
                 "%d/%d" % (len(sub) - s, len(sub))))
    src = defaultdict(int)
    for r in runs:
        src[r["final_source"]] += 1
    print("  final source: %s" % dict(src))
    print("  NOTE: source=solver does NOT mean strict improvement; the two are counted apart.")
    print()

    # ---------------- pre-agreed reading
    print("=== pre-agreed reading ===")
    p = out["primary"]
    insts_ok = p["positive"]
    beats_full = p["mean"] > TOL and insts_ok >= 12 and p["seed_opposite"] == 0
    sec = {("%s vs %s" % (s["comparison"].split(" vs ")[0], s["comparison"].split(" vs ")[1])):
           s for s in out["secondary"]}
    ab_abc = sec["AB vs ABC"]
    ab_a = sec["AB vs A"]
    ab_bc = sec["AB vs BC"]
    print("  AB vs FULL  : mean=%+.4f positive=%d/%d seed_opposite=%d"
          % (p["mean"], p["positive"], p["positive"] + p["negative"] + p["tie"],
             p["seed_opposite"]))
    print("  AB vs ABC   : mean=%+.4f positive=%d" % (ab_abc["mean"], ab_abc["positive"]))
    print("  AB vs A     : mean=%+.4f positive=%d" % (ab_a["mean"], ab_a["positive"]))
    print("  AB vs BC    : mean=%+.4f positive=%d" % (ab_bc["mean"], ab_bc["positive"]))
    if beats_full and ab_abc["mean"] > TOL:
        print("  -> AB keeps a clear, cross-instance advantage over BOTH FULL and ABC:")
        print("     grounds to move into algorithmic-contribution work")
    elif ab_abc["mean"] > TOL and not beats_full:
        print("  -> AB beats ABC but not FULL: a useful subproblem simplification, but no")
        print("     deployment advantage over the original task is established")
    if ab_a["mean"] <= TOL:
        print("  -> A is as good as or better than AB: prefer the SIMPLER A; do not keep the")
        print("     short-term-Z release just to preserve a mechanism story")
    if ab_bc["mean"] <= 0.02:
        print("  -> BC is close to AB at a fraction of the time: evaluate the quality-time")
        print("     trade-off; AB's extra computation must be shown to be worth it")
    if p["mean"] <= TOL or insts_ok < 10:
        print("  -> the advantage does NOT clearly survive on new instances: keep the original")
        print("     result as a development finding, stop extending AB, and do NOT keep tuning")
        print("     rules on this same new data until it wins")
    print("  2% remains an engineering reference against J_r, not a publication standard.")

    with open("stage10_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nwrote stage10_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
