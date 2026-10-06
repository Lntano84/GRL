#!/usr/bin/env python
"""
Stage 12 analysis: quality-retention intervals.

    .venv-hs/Scripts/python.exe stage12_analyse.py

For every state and configuration, meld the old and new results into

    U_m = the LOWEST valid plan cost found for m          (a valid UPPER bound on J*_m)
    L_m = max(valid LP relaxation, valid MIP dual bounds, proven-optimal objective)
                                                          (a valid LOWER bound on J*_m)

All bounds used for one interval come from the SAME state and the SAME configuration definition.

Interval 1 -- AB vs MR-Z-1 optimal-value difference:

    (J*_MR - J*_AB)/D  in  [ (L_MR - U_AB)/D ,  (U_MR - L_AB)/D ] ,  D = max(1, |J_r|)

    lower end > 0        -> AB demonstrably retains the better attainable quality
    upper end < 0        -> the random configuration is demonstrably better
    interval inside +-2% -> the optimal-value difference is bounded small at the engineering scale
    otherwise            -> still undetermined

Interval 2 -- AB's restriction loss against ABC:

    0 <= J*_AB - J*_ABC

    (J*_AB - J*_ABC)/D in [ max(0, (L_AB - U_ABC)/D) , (U_AB - L_ABC)/D ]

Only relative to ABC, which still fixes the short-term Y; this must NOT be extended to FULL.

These are intervals from solving bounds, NOT statistical confidence intervals.  U is built from
two stage-11 seeds purely as an offline QUALITY CERTIFICATE; it is not a deployable score and
does not merge the two seeds' budgets into one run.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

METHODS = ("AB", "MR-Z-1", "ABC")
OLD_MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")
OLD_SEEDS = (0, 1)
TOL = 1e-9
SCALE = 0.02


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    S11 = load("stage11_runs.json")["runs"]
    S11B = load("stage11b_relax.json")
    S12 = load("stage12_runs.json")["runs"]
    old = {(r["state"], r["method"], int(r["seed"])): r for r in S11}
    new = {(r["state"], r["method"]): r for r in S12}
    jr = {r["state"]: float(r["repair_cost"]) for r in S11}
    states = sorted(jr)

    def U(st: str, m: str) -> Tuple[Optional[float], str]:
        cands = []
        for sd in OLD_SEEDS:
            r = old.get((st, m, sd))
            if r and r["final_cost"] is not None:
                cands.append((float(r["final_cost"]), "s11/s%d" % sd))
        r = new.get((st, m))
        if r and r["final_cost"] is not None:
            cands.append((float(r["final_cost"]), "s12"))
        if not cands:
            return None, "none"
        c, src = min(cands, key=lambda x: x[0])
        return c, src

    def L(st: str, m: str) -> Tuple[Optional[float], List[str]]:
        """Strongest valid lower bound from the same state and configuration."""
        cands: List[Tuple[float, str]] = []
        lpv = S11B.get(st, {}).get(m, {}).get("objective")
        if lpv is not None and S11B[st][m].get("proven_optimal"):
            cands.append((float(lpv), "lp-relax"))
        for sd in OLD_SEEDS:
            r = old.get((st, m, sd))
            if r and r.get("mip_dual_bound") is not None:
                cands.append((float(r["mip_dual_bound"]), "s11-dual/s%d" % sd))
        r = new.get((st, m))
        if r is not None:
            if r.get("mip_dual_bound") is not None:
                cands.append((float(r["mip_dual_bound"]), "s12-dual"))
            if r.get("mip_proven_optimal") and r.get("mip_objective") is not None:
                cands.append((float(r["mip_objective"]), "s12-optimal"))
        if not cands:
            return None, []
        v, _ = max(cands, key=lambda x: x[0])
        return v, [n for x, n in cands if abs(x - v) <= TOL * (1 + abs(v))]

    def classify(lo: float, hi: float) -> str:
        if lo > TOL:
            return "AB better CONFIRMED"
        if hi < -TOL:
            return "MR better CONFIRMED"
        if lo >= -SCALE and hi <= SCALE:
            return "bounded within +-2%"
        return "undetermined"

    print("Stage 12 | %d diagnostic MIPs | seed 0 | 60 s each" % len(S12))
    print("role: %s" % load("stage12_runs.json")["meta"]["role"])
    print()

    # ---------------------------------------------------------------- table
    print("=" * 138)
    print("PER-STATE TABLE")
    print("=" * 138)
    print("%-30s %12s %12s %12s | %12s %12s | %10s %10s | %-22s"
          % ("state", "U_AB", "U_MR", "U_ABC", "L_AB", "L_MR",
             "I1 low", "I1 high", "interval 1 (AB vs MR)"))
    print("-" * 138)
    rows = []
    for st in states:
        d = max(1.0, abs(jr[st]))
        ua, sa = U(st, "AB")
        um, sm = U(st, "MR-Z-1")
        uc, sc = U(st, "ABC")
        la, _ = L(st, "AB")
        lm, _ = L(st, "MR-Z-1")
        lc, _ = L(st, "ABC")
        i1lo = (lm - ua) / d if (lm is not None and ua is not None) else None
        i1hi = (um - la) / d if (um is not None and la is not None) else None
        verdict = classify(i1lo, i1hi) if (i1lo is not None and i1hi is not None) else "missing"
        rows.append((st, ua, um, uc, la, lm, lc, i1lo, i1hi, verdict))
        print("%-30s %12s %12s %12s | %12s %12s | %10s %10s | %-22s"
              % (st,
                 "-" if ua is None else "%.6g" % ua,
                 "-" if um is None else "%.6g" % um,
                 "-" if uc is None else "%.6g" % uc,
                 "-" if la is None else "%.6g" % la,
                 "-" if lm is None else "%.6g" % lm,
                 "-" if i1lo is None else "%+.4f" % i1lo,
                 "-" if i1hi is None else "%+.4f" % i1hi, verdict))
    print()

    # ---------------------------------------------------------------- interval 2
    print("=" * 138)
    print("INTERVAL 2: AB restriction loss vs ABC   (J*_AB - J*_ABC)/D  in  [lo, hi]")
    print("=" * 138)
    print("%-30s %10s %10s | %10s | %-24s"
          % ("state", "I2 low", "I2 high", "width", "verdict"))
    print("-" * 138)
    i2rows = []
    for st, ua, um, uc, la, lm, lc, i1lo, i1hi, v1 in rows:
        d = max(1.0, abs(jr[st]))
        if None in (ua, uc, la, lc):
            print("%-30s %12s %12s | %10s | missing" % (st, "-", "-", "-"))
            i2rows.append((st, None, None, "missing"))
            continue
        lo = max(0.0, (la - uc) / d)
        hi = (ua - lc) / d
        if hi < -TOL:
            verdict = "IMPOSSIBLE (check)"
        elif hi <= SCALE:
            verdict = "loss upper bound <= 2%"
        elif lo > SCALE:
            verdict = "loss lower bound > 2%"
        else:
            verdict = "loss interval spans 2%"
        i2rows.append((st, lo, hi, verdict))
        print("%-30s %+10.4f %+10.4f | %10.4f | %-24s"
              % (st, lo, hi, max(0.0, hi - lo), verdict))
    print()

    # ---------------------------------------------------------------- summary
    print("=" * 138)
    print("SUMMARY")
    print("=" * 138)
    cnt = defaultdict(int)
    for *_x, v in rows:
        cnt[v] += 1
    print("  interval 1 (AB vs MR-Z-1), 16 states: %s" % dict(cnt))
    c2 = defaultdict(int)
    for *_x, v in i2rows:
        c2[v] += 1
    print("  interval 2 (AB loss vs ABC), 16 states: %s" % dict(c2))
    conf = [r[0] for r in rows if r[9] == "AB better CONFIRMED"]
    print("  states where AB's attainable quality is CONFIRMED better than MR-Z-1: %d %s"
          % (len(conf), conf))
    print("  states where MR-Z-1 is CONFIRMED better: %d"
          % sum(1 for r in rows if r[9] == "MR better CONFIRMED"))
    print()
    # the strongest individual certificates across all three random sets
    print("  strongest certificates against ANY matched random set (all three, both seeds):")
    n_cert = 0
    cert_states = set()
    for st in states:
        ua, _ = U(st, "AB")
        for mr in OLD_MR:
            lm, _ = L(st, mr)
            if ua is not None and lm is not None and ua < lm - TOL * max(1, abs(ua)):
                n_cert += 1
                cert_states.add(st)
                print("    %-30s %-8s U_AB=%-11.6g L_MR=%-11.6g  gap=%+.4f"
                      % (st, mr, ua, lm, (lm - ua) / max(1.0, abs(jr[st]))))
    print("    total certificates=%d over %d states (%d nominal instances)"
          % (n_cert, len(cert_states), len({s.split('|')[0] for s in cert_states})))
    print()
    # what the diagnostic budget bought
    print("  new valid upper bounds from stage 12 (improved over stage 11):")
    n_impr = 0
    for st in states:
        for m in METHODS:
            r = new.get((st, m))
            if r is None or r["final_cost"] is None:
                continue
            prev = min([old[(st, m, sd)]["final_cost"] for sd in OLD_SEEDS
                        if (st, m, sd) in old] or [float("inf")])
            if float(r["final_cost"]) < prev - TOL:
                n_impr += 1
                print("    %-30s %-8s %.6g -> %.6g" % (st, m, prev, r["final_cost"]))
    print("    improved in %d of %d (state, config) runs" % (n_impr, len(states) * len(METHODS)))
    print()
    n_opt = sum(1 for st in states for m in METHODS
                if new.get((st, m)) and new[(st, m)]["mip_proven_optimal"])
    print("  stage 12 runs proven optimal: %d/%d" % (n_opt, len(states) * len(METHODS)))
    src = defaultdict(int)
    for st in states:
        for m in METHODS:
            r = new.get((st, m))
            if r:
                src[r["start_source"].split(":")[0]] += 1
    print("  start sources: %s" % dict(src))
    print()
    print("  NOTE: intervals come from solving bounds, NOT from a statistical model.")
    print("        U uses the better of two stage-11 seeds purely as an offline quality")
    print("        certificate; it is not a deployable result and does not combine budgets.")

    with open("stage12_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"per_state": [{"state": r[0], "U_AB": r[1], "U_MR": r[2], "U_ABC": r[3],
                                  "L_AB": r[4], "L_MR": r[5], "L_ABC": r[6],
                                  "interval1": [r[7], r[8]], "verdict1": r[9]} for r in rows],
                   "interval2": [{"state": r[0], "lo": r[1], "hi": r[2], "verdict": r[3]}
                                 for r in i2rows],
                   "certificates_vs_any_random": {"count": n_cert,
                                                  "states": sorted(cert_states)},
                   "new_upper_bounds_improved": n_impr,
                   "stage12_proven_optimal": n_opt},
                  fh, indent=2, default=str)
    print("\nwrote stage12_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
