#!/usr/bin/env python
"""
Stage 06 deployment-level checks.

    .venv-hs/Scripts/python.exe stage06_verdict.py

The aggregation lives in `stage06_aggregate.py` (paired per disruption and solver seed, with the
disruption-specific repair cost as the normaliser) and is imported here, so this script can no
longer drift from `stage06_analyse.py`.

What this adds beyond `stage06_analyse.py`:

  * the full "each fixed rule vs the frozen simple strategy" table with the scale breakdown;
  * DEPENDENCY-24 vs each matched-random control INDIVIDUALLY, per solver seed, so seed
    stability is visible control by control rather than only on the average;
  * the content-layer criteria evaluated literally, plus the deployment-layer verdict stated
    separately (they answer different questions and are not merged).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List

from stage06_aggregate import (CONFIGS, RANDOMS, SIMPLE_PICK, THRESHOLD, describe,
                               instance_g, instance_g_random_mean, instance_g_simple,
                               load_runs, pair_cells, scale_of, scale_rho)

SEEDS = (0, 1)


def show(label: str, s: Dict[str, Any]) -> None:
    print("  %-52s n=%-3d mean=%+.4f med=%+.4f min=%+.4f max=%+.4f "
          "pos=%d neg=%d zero=%d >=2%%=%d"
          % (label, s["n"], s["mean"], s["median"], s["min"], s["max"],
             s["pos"], s["neg"], s["zero"], s["ge2pct"]))
    for key in ("by_scale", "by_scale_rho"):
        for k, v in s.get(key, {}).items():
            print("        %-16s mean=%+.4f (n=%d)" % (k, v["mean"], v["n"]))


def breakdown(v: Dict[str, float], sel) -> Dict[str, Dict[str, Any]]:
    g: Dict[str, List[float]] = defaultdict(list)
    for n, x in v.items():
        g[sel(n)].append(x)
    return {k: {"n": len(x), "mean": sum(x) / len(x)} for k, x in sorted(g.items())}


def main() -> int:
    runs = load_runs()
    cells = pair_cells(runs)
    instances = sorted({k[0] for k in cells})

    # ---------------------------------------------------------------- 1. deployable
    print("=== 1. DEPLOYABLE: each fixed rule vs the frozen simple strategy ===")
    print("    simple = medium->FULL, large->EMPTY ; G>0 means the fixed rule is cheaper")
    q2: Dict[str, Any] = {}
    for c in CONFIGS:
        v = instance_g_simple(cells, c)
        s = describe(list(v.values()))
        s["by_scale"] = breakdown(v, scale_of)
        s["by_scale_rho"] = breakdown(v, lambda n: "%s|%s" % scale_rho(n))
        q2[c] = s
        show("always %-16s vs simple" % c, s)
    print()
    pos = [c for c in CONFIGS if q2[c]["mean"] > 0]
    print("  fixed rules with a POSITIVE mean: %s"
          % (", ".join(pos) if pos else "NONE"))
    print("  => deployment layer: %s"
          % ("SUPPORTED" if q2["DEPENDENCY-24"]["mean"] > 0 else
             "NOT SUPPORTED (no fixed information rule beats the simple strategy)"))
    print()

    # ---------------------------------------------------------------- 2. per control, per seed
    print("=== 2. CONTENT: DEPENDENCY-24 vs each matched-random control, per solver seed ===")
    per_control: Dict[str, Any] = {}
    for c in RANDOMS:
        v_all = instance_g(cells, "DEPENDENCY-24", c)
        per_control[c] = {"all": describe(list(v_all.values())),
                          "by_scale": breakdown(v_all, scale_of)}
        print("  DEPENDENCY-24 vs %-18s ALL: mean=%+.4f pos=%d neg=%d zero=%d >=2%%=%d"
              % (c, per_control[c]["all"]["mean"], per_control[c]["all"]["pos"],
                 per_control[c]["all"]["neg"], per_control[c]["all"]["zero"],
                 per_control[c]["all"]["ge2pct"]))
        for sc in ("medium", "large"):
            print("        %-8s mean=%+.4f (n=%d)"
                  % (sc, per_control[c]["by_scale"][sc]["mean"],
                     per_control[c]["by_scale"][sc]["n"]))
        for sd in SEEDS:
            sub = {k: x for k, x in cells.items() if k[2] == sd}
            v = instance_g(sub, "DEPENDENCY-24", c)
            d = describe(list(v.values()))
            print("        seed%d    mean=%+.4f pos=%d neg=%d zero=%d"
                  % (sd, d["mean"], d["pos"], d["neg"], d["zero"]))
        print()

    # ---------------------------------------------------------------- 3. criteria
    print("=== 3. pre-registered CONTENT criteria (DEPENDENCY-24 vs matched-random mean) ===")
    base_v = instance_g_random_mean(cells, "DEPENDENCY-24")
    base = describe(list(base_v.values()))
    base["by_scale"] = breakdown(base_v, scale_of)
    base["by_scale_rho"] = breakdown(base_v, lambda n: "%s|%s" % scale_rho(n))
    combo = [k for k, v in base["by_scale_rho"].items() if v["mean"] >= THRESHOLD]
    per_seed = {}
    for sd in SEEDS:
        sub = {k: x for k, x in cells.items() if k[2] == sd}
        per_seed[sd] = instance_g_random_mean(sub, "DEPENDENCY-24")
    c1 = base["ge2pct"] >= 4
    c2 = len(combo) >= 2
    c3_overall = all(sum(per_seed[s].values()) / len(per_seed[s]) > 0 for s in SEEDS)
    c3_scales = True
    for sc in ("medium", "large"):
        means = []
        for sd in SEEDS:
            xs = [x for n, x in per_seed[sd].items() if scale_of(n) == sc]
            means.append(sum(xs) / len(xs))
        if not (all(m > 0 for m in means) or all(m < 0 for m in means)):
            c3_scales = False
    c3 = c3_overall and c3_scales
    c4 = base["mean"] >= THRESHOLD
    print("  C1 >=4 nominal instances with G>=2%%      : %d      -> %s"
          % (base["ge2pct"], "PASS" if c1 else "FAIL"))
    print("  C2 >=2 scale-rho combos >=2%%             : %d (%s) -> %s"
          % (len(combo), ", ".join(combo), "PASS" if c2 else "FAIL"))
    print("  C3 both SOLVER SEEDS agree in direction   : overall=%s scales=%s -> %s"
          % (c3_overall, c3_scales, "PASS" if c3 else "FAIL"))
    print("  C4 instance-mean gain >=2%%               : %+.4f -> %s"
          % (base["mean"], "PASS" if c4 else "FAIL"))
    print("  CONTENT layer: %s" % ("PASSED" if (c1 and c2 and c3 and c4) else "NOT PASSED"))
    print()
    print("  NOTE: the deployment layer above is a SEPARATE question and is NOT covered by")
    print("        C1-C4.  'the content matters' and 'it is worth deploying instead of a")
    print("        one-line rule' can both be answered, and here they differ.")
    print()
    print("  per solver seed, DEPENDENCY-24 vs matched-random mean:")
    for sd in SEEDS:
        v = per_seed[sd]
        d = describe(list(v.values()))
        bs = breakdown(v, scale_of)
        print("        seed%d ALL mean=%+.4f | medium=%+.4f large=%+.4f"
              % (sd, d["mean"], bs["medium"]["mean"], bs["large"]["mean"]))
    print()
    print("  per-instance, DEPENDENCY-24 vs matched-random mean:")
    print("  %-24s %10s" % ("instance", "G_n"))
    for n in instances:
        print("  %-24s %+10.4f" % (n, base_v[n]))

    with open("stage06_verdict.json", "w", encoding="utf-8") as fh:
        json.dump({"aggregation": "paired per (disruption, solver seed); normaliser = "
                                  "disruption-specific repair cost",
                   "content_vs_random_mean": {k: v for k, v in base.items()},
                   "per_control": {k: v for k, v in per_control.items()},
                   "per_solver_seed": {str(s): describe(list(per_seed[s].values()))
                                       for s in SEEDS},
                   "deployable_vs_simple": {k: v for k, v in q2.items()},
                   "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4,
                                "content_layer": "PASSED" if (c1 and c2 and c3 and c4)
                                else "NOT PASSED",
                                "deployment_layer": ("SUPPORTED"
                                                     if q2["DEPENDENCY-24"]["mean"] > 0
                                                     else "NOT SUPPORTED")}},
                  fh, indent=2, default=str)
    print("\nwrote stage06_verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
