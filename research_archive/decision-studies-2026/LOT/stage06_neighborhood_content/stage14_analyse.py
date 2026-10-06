#!/usr/bin/env python
"""
Stage 14 analysis: does releasing the carry-overs at each machine's RECOVERY period keep most
of the benefit of releasing every short-term Z?

    .venv-hs/Scripts/python.exe stage14_analyse.py

Indicators
----------
    g(m) = mean over (state, seed) of (J_A - J_m) / max(1, |J_r|)      gain over A
    R    = g(RECOVERY) / g(AB)                                          coverage of AB's gain

R uses the ratio of MEANS, not the mean of per-state ratios.

Pre-agreed exploratory reading
------------------------------
* R >= 80%, RECOVERY better than END on at least 6/8 nominal instances, both rho groups positive
  -> supports continuing to study the recovery-period structural information
* RECOVERY close to END -> no extra value from the recovery position established; do not build
  learning features around it yet
* RECOVERY clearly loses AB's extra gain -> releasing only the recovery period is not enough;
  the short-term effect cannot be compressed into that one position
* g(AB) <= 0 or unstable -> do not interpret coverage; record as inconclusive

Even with high coverage, if the ACTUAL running time did not fall and same-budget quality did not
improve, this still does not establish an algorithmic advantage over AB.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

METHODS = ("A", "AB", "RECOVERY", "END")
SEEDS = (0, 1)
TOL = 1e-6


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    D = load("stage14_runs.json")
    runs = D["runs"]
    print("Stage 14 | %d MIPs | group-release rule comparison" % len(runs))
    print("scope: %s" % D["meta"]["scope"])
    print("rule: %s" % D["meta"]["rule"])
    print()

    jr = {r["state"]: float(r["repair_cost"]) for r in runs}
    cost: Dict[Tuple[str, str, int], float] = {}
    for r in runs:
        cost[(r["state"], r["method"], int(r["seed"]))] = float(r["final_cost"])
    states = sorted(jr)
    insts = sorted({s.split("|")[0] for s in states})

    # ---------------- release sizes
    print("=== release set sizes (Z slots) ===")
    for m in METHODS:
        v = sorted({r["n_rel_Z"] for r in runs if r["method"] == m})
        print("  %-9s n_rel_Z = %s" % (m, v))
    print()

    # ---------------- g(m)
    def g_of(m: str) -> float:
        vals = []
        for st in states:
            for sd in SEEDS:
                vals.append((cost[(st, "A", sd)] - cost[(st, m, sd)]) / max(1.0, abs(jr[st])))
        return sum(vals) / len(vals)

    g = {m: g_of(m) for m in METHODS}
    print("=== gain over A, g(m) = mean (J_A - J_m)/max(1,|J_r|) ===")
    for m in METHODS:
        print("  %-9s g = %+.4f" % (m, g[m]))
    print()

    R = None
    if g["AB"] > TOL:
        R = g["RECOVERY"] / g["AB"]
        print("=== coverage of AB's gain ===")
        print("  R = g(RECOVERY)/g(AB) = %+.4f / %+.4f = %.4f  (%.1f%%)"
              % (g["RECOVERY"], g["AB"], R, 100 * R))
        print("  END coverage for reference: %+.4f / %+.4f = %.4f  (%.1f%%)"
              % (g["END"], g["AB"], g["END"] / g["AB"], 100 * g["END"] / g["AB"]))
    else:
        print("=== coverage NOT computed: g(AB) = %+.4f <= 0 -> inconclusive ===" % g["AB"])
    print()

    # ---------------- RECOVERY vs END per nominal instance
    print("=== RECOVERY vs END, per nominal instance (positive => RECOVERY cheaper) ===")
    per = defaultdict(list)
    for st in states:
        for sd in SEEDS:
            per[st.split("|")[0]].append(
                (cost[(st, "END", sd)] - cost[(st, "RECOVERY", sd)]) / max(1.0, abs(jr[st])))
    rows = {k: sum(v) / len(v) for k, v in sorted(per.items())}
    for k, v in rows.items():
        print("  %-22s %+.4f" % (k, v))
    pos = sum(1 for v in rows.values() if v > TOL)
    neg = sum(1 for v in rows.values() if v < -TOL)
    print("  n=%d  mean=%+.4f  positive=%d  negative=%d  tie=%d"
          % (len(rows), sum(rows.values()) / len(rows), pos, neg,
             len(rows) - pos - neg))
    for rho in ("0.75", "1.10"):
        v = [x for k, x in rows.items() if ("rho%s" % rho) in k]
        print("    rho%-5s mean=%+.4f positive=%d/%d"
              % (rho, sum(v) / len(v), sum(1 for x in v if x > TOL), len(v)))
    print()

    # ---------------- RECOVERY / END vs A, per instance
    print("=== each rule vs A, per nominal instance ===")
    for m in ("AB", "RECOVERY", "END"):
        acc = defaultdict(list)
        for st in states:
            for sd in SEEDS:
                acc[st.split("|")[0]].append(
                    (cost[(st, "A", sd)] - cost[(st, m, sd)]) / max(1.0, abs(jr[st])))
        v = {k: sum(x) / len(x) for k, x in acc.items()}
        print("  %-9s mean=%+.4f  positive=%d/%d  min=%+.4f  max=%+.4f"
              % (m, sum(v.values()) / len(v), sum(1 for x in v.values() if x > TOL), len(v),
                 min(v.values()), max(v.values())))
    print()

    # ---------------- time and presolve: does the narrower rule actually solve faster?
    print("=== actual cost of the narrower rule (time and presolve size) ===")
    print("  %-9s %12s %12s %12s %12s" % ("method", "mean total s", "median s",
                                           "presolve cols", "presolve binary"))
    for m in METHODS:
        sub = [r for r in runs if r["method"] == m]
        ts = sorted(r["timing"]["total_s"] for r in sub)
        cols = [r["presolve"]["cols"] for r in sub if r["presolve"].get("cols")]
        bins = [r["presolve"]["binary"] for r in sub if r["presolve"].get("binary") is not None]
        print("  %-9s %12.3f %12.3f %12s %12s"
              % (m, sum(ts) / len(ts), ts[len(ts) // 2],
                 "MISSING" if not cols else "%.1f" % (sum(cols) / len(cols)),
                 "MISSING" if not bins else "%.1f" % (sum(bins) / len(bins))))
    n_opt = {m: sum(1 for r in runs if r["method"] == m and r["mip_proven_optimal"])
             for m in METHODS}
    print("  proven optimal early: %s" % {m: "%d/32" % n_opt[m] for m in METHODS})
    print()

    # ---------------- source vs improvement
    print("=== source vs strict improvement (kept apart) ===")
    for m in METHODS:
        sub = [r for r in runs if r["method"] == m]
        s = sum(1 for r in sub if r["strictly_below_lp"])
        print("  %-9s strictly<LP %2d/%-2d | final source: %s"
              % (m, s, len(sub),
                 {k: sum(1 for r in sub if r["final_source"] == k)
                  for k in ("solver", "lp", "repair", "none")}))
    print()

    # ---------------- pre-agreed reading
    print("=== pre-agreed exploratory criteria ===")
    if R is None:
        print("  g(AB) <= 0 -> record as INCONCLUSIVE; coverage is not interpreted.")
    else:
        c1 = R >= 0.80
        c2 = pos >= 6
        rho_pos = all(sum(1 for k, x in rows.items()
                          if ("rho%s" % r) in k and x > TOL) >= 1 for r in ("0.75", "1.10"))
        print("  R >= 80%%                                  : %.1f%% -> %s"
              % (100 * R, "yes" if c1 else "no"))
        print("  RECOVERY > END on >=6/8 nominal instances : %d/8 -> %s"
              % (pos, "yes" if c2 else "no"))
        print("  both rho groups positive                  : %s -> %s"
              % (rho_pos, "yes" if rho_pos else "no"))
        if c1 and c2 and rho_pos:
            print("  => supports continuing to study the RECOVERY-position structural information.")
        elif abs(sum(rows.values()) / len(rows)) <= 0.02 and pos < 6:
            print("  => RECOVERY is close to END: no extra value from the recovery position is")
            print("     established; do not build learning features around it yet.")
        elif R < 0.5:
            print("  => RECOVERY clearly loses AB's extra gain: releasing only the recovery")
            print("     period is NOT enough; the short-term effect cannot be compressed there.")
        else:
            print("  => inconclusive by the pre-agreed criteria.")
    print()
    print("  REMINDER: even with high coverage, an algorithmic advantage over AB is NOT")
    print("  established unless the actual running time fell or same-budget quality improved.")
    print("  2%% remains an engineering reference against J_r, not a publication standard.")

    with open("stage14_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"g": g, "coverage_R": R,
                   "recovery_vs_end_per_instance": rows,
                   "recovery_vs_end_positive": pos,
                   "proven_optimal": n_opt,
                   "meta": D["meta"]}, fh, indent=2, default=str)
    print("\nwrote stage14_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
