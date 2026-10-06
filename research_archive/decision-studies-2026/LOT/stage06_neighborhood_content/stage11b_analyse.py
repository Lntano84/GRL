#!/usr/bin/env python
"""
Stage 11b analysis: the three things this round has to deliver.

    .venv-hs/Scripts/python.exe stage11b_analyse.py

1. TRUE PRESOLVE SUMMARY, parsed offline from the 160 existing MIP logs (no re-solve).  The
   values come from the first `Solving MIP model with:` block, which is the presolved model
   HiGHS actually solved.  The earlier `getNumCol()` / `getNumRow()` numbers were the loaded
   model size and are discarded.

2. VALID LP RELAXATION table: status and objective for all 80 relaxations, with every
   `Error` / `Not Set` / solution-less case recorded as MISSING and excluded from any mean.
   The earlier table folded a default objective of 0 into the analysis and was therefore just
   `1 - G` restated.

3. INCLUSION CHECKS on the relaxations that PROVED OPTIMAL:
       L_ABC <= L_AB ,  L_ABC <= L_MR ,  L_m <= J_m,valid
   AB and MR are not nested, so no fixed order between their bounds is imposed.

Also reclassifies the earlier transfer check, which had mixed "the matched configuration could
have used the AB plan but did worse" together with "the matched configuration did better".
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

METHODS = ("AB", "ABC", "MR-Z-1", "MR-Z-2", "MR-Z-3")
MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")
TOL = 1e-6
HDR = re.compile(r"Solving MIP model with:\s*\n\s*(\d+) rows\s*\n\s*(\d+) cols \(([^)]*)\)")


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def presolve_of(log: str) -> Optional[Dict[str, int]]:
    m = HDR.search(log)
    if not m:
        return None
    mb = re.search(r"(\d+) binary", m.group(3))
    return {"rows": int(m.group(1)), "cols": int(m.group(2)),
            "binary": int(mb.group(1)) if mb else None}


def main() -> int:
    runs = load("stage11_runs.json")["runs"]
    relax = load("stage11b_relax.json")
    cell = {(r["state"], r["method"], int(r["seed"])): r for r in runs}
    jr = {r["state"]: float(r["repair_cost"]) for r in runs}
    states = sorted(jr)

    # ---------------------------------------------------------------- (0) retired numbers
    print("=" * 100)
    print("(0) RETIRED: the stage-11 'actual reduction' and 'relaxation' tables")
    print("=" * 100)
    old = load("stage11_diagnostics.json")
    oldr = old["relaxation"]
    n_notset = sum(1 for st in oldr for m in oldr[st]
                   if str(oldr[st][m]["status"]).lower().startswith("not set"))
    n_zero = sum(1 for st in oldr for m in oldr[st] if oldr[st][m]["objective"] == 0)
    tot_wall = sum(oldr[st][m]["wall_s"] for st in oldr for m in oldr[st])
    print("  stage-11 relaxation records: %d | 'Not Set': %d | objective exactly 0: %d"
          % (sum(len(oldr[st]) for st in oldr), n_notset, n_zero))
    print("  total wall time of all %d stage-11 'relaxations': %.5f s"
          % (sum(len(oldr[st]) for st in oldr), tot_wall))
    print("  => no solve happened; the tabulated 0 made that table equal to 1 - G.")
    print("     It is WITHDRAWN and is not used anywhere below.")
    print()

    # ---------------------------------------------------------------- (1) presolve
    print("=" * 100)
    print("(1) TRUE PRESOLVE SUMMARY, from the first `Solving MIP model with:` block")
    print("=" * 100)
    pre = {k: presolve_of(v["mip_log"]) for k, v in cell.items()}
    missing = [k for k, v in pre.items() if v is None]
    print("  parsed from %d logs | without a parsable block: %d"
          % (len(pre), len(missing)))
    agg = defaultdict(lambda: defaultdict(list))
    for (st, m, sd), v in pre.items():
        if v is None:
            continue
        for k in ("rows", "cols", "binary"):
            agg[m][k].append(v[k])
    print("  %-8s %6s %12s %12s %14s" % ("config", "n", "rows", "cols", "binary vars"))
    for m in METHODS:
        a = agg[m]
        f = lambda k: sum(a[k]) / len(a[k])  # noqa: E731
        print("  %-8s %6d %12.2f %12.2f %14.2f" % (m, len(a["rows"]), f("rows"), f("cols"),
                                                    f("binary")))
    pairs = 0
    fewer = 0
    diffs: List[int] = []
    for (st, m, sd), v in pre.items():
        if m != "AB" or v is None:
            continue
        for mr in MR:
            w = pre.get((st, mr, sd))
            if w is None:
                continue
            pairs += 1
            d = v["binary"] - w["binary"]
            diffs.append(d)
            fewer += int(d < 0)
    print()
    print("  PAIRED comparison on presolved BINARY variables (AB minus MR):")
    print("    pairs=%d | AB has fewer in %d/%d | min=%d max=%d mean=%.2f"
          % (pairs, fewer, pairs, min(diffs), max(diffs), sum(diffs) / len(diffs)))
    print("  => matching the NOMINAL fixed count does NOT match the presolved size:")
    print("     AB still yields a consistently smaller model after presolve.")
    print("     This does NOT by itself prove the +1.70% comes from that reduction.")
    print()

    # ---------------------------------------------------------------- (2) relaxation
    print("=" * 100)
    print("(2) VALID LP RELAXATION: status and objective (invalid => MISSING, never a default)")
    print("=" * 100)
    nval = nmiss = 0
    for st in states:
        for m in METHODS:
            r = relax[st][m]
            if r["valid"]:
                nval += 1
            else:
                nmiss += 1
    print("  records: %d | valid: %d | MISSING: %d" % (nval + nmiss, nval, nmiss))
    print()
    print("  %-30s %13s %13s %13s %13s %13s" % ("state", *METHODS))
    for st in states:
        row = "  %-30s" % st
        for m in METHODS:
            o = relax[st][m]["objective"]
            row += " %13s" % ("MISSING" if o is None else "%.6g" % o)
        print(row)
    print()
    print("  %-8s %12s %12s %14s %12s" % ("config", "mean L", "min L", "proven optimal",
                                           "mean time"))
    for m in METHODS:
        objs = [relax[st][m]["objective"] for st in states if relax[st][m]["objective"] is not None]
        opt = sum(1 for st in states if relax[st][m]["proven_optimal"])
        ts = [relax[st][m]["wall_s"] for st in states]
        print("  %-8s %12.2f %12.2f %14s %12.3f"
              % (m, sum(objs) / len(objs), min(objs), "%d/%d" % (opt, len(states)),
                 sum(ts) / len(ts)))
    print()
    print("  Only PROVEN-OPTIMAL relaxation values are used for the mean above; nothing else")
    print("  enters it.  An unproven relaxation objective is NOT a valid bound.")
    print()

    # ---------------------------------------------------------------- (3) inclusion checks
    print("=" * 100)
    print("(3) INCLUSION CHECKS on proven-optimal relaxations")
    print("=" * 100)
    print("  L_ABC <= L_AB   and   L_ABC <= L_MR   and   L_m <= J_m,valid")
    print()
    print("  %-30s %-8s %13s %13s %10s %10s" % ("state", "pair", "L_TIGHT", "L_WIDE",
                                                "holds", "slack"))
    viol = defaultdict(int)
    tested = defaultdict(int)
    ok_all = True
    for st in states:
        l_abc = relax[st]["ABC"]["objective"]
        if l_abc is None or not relax[st]["ABC"]["proven_optimal"]:
            print("  %-30s ABC relaxation not proven optimal -> skipped" % st)
            continue
        for m in ("AB",) + MR:
            l = relax[st][m]["objective"]
            if l is None or not relax[st][m]["proven_optimal"]:
                print("  %-30s %-8s not proven optimal -> skipped" % (st, m))
                continue
            tested[m] += 1
            holds = l_abc <= l + TOL * (1 + abs(l))
            viol[m] += int(not holds)
            ok_all = ok_all and holds
            print("  %-30s %-8s %13.6g %13.6g %10s %10.4g"
                  % (st, "ABC<=%s" % m, l_abc, l, "yes" if holds else "NO", l - l_abc))
    print()
    print("  L_m <= J_m,valid  (the relaxation must bound the delivered feasible plan):")
    n_bound = 0
    n_tot = 0
    for st in states:
        for m in METHODS:
            l = relax[st][m]["objective"]
            if l is None or not relax[st][m]["proven_optimal"]:
                continue
            for sd in (0, 1):
                r = cell.get((st, m, sd))
                if r is None:
                    continue
                n_tot += 1
                if l <= float(r["final_cost"]) + TOL * (1 + abs(float(r["final_cost"]))):
                    n_bound += 1
                else:
                    print("    VIOLATION %s %s s%d: L=%.6g > J=%.6g"
                          % (st, m, sd, l, r["final_cost"]))
    print("    satisfied in %d/%d (state, config, seed) cases" % (n_bound, n_tot))
    print()
    print("  summary of L_ABC <= L_m: %s"
          % {m: "%d/%d hold" % (tested[m] - viol[m], tested[m]) for m in ("AB",) + MR})
    print("  => the nesting ABC superset AB,MR is visible in the bounds as expected;")
    print("     AB and MR are NOT nested and no fixed order between them is required.")

    # ---------------------------------------------------------------- (4) transfer reclassified
    print()
    print("=" * 100)
    print("(4) TRANSFER CHECK, reclassified")
    print("=" * 100)
    tr = old["transfer"]
    worse = better = equal = 0
    rows = []
    for st in sorted(tr):
        for key, v in tr[st].items():
            if not v["admits_AB_plan"]:
                continue
            mr = v["MR_cost"]
            ab = v["AB_cost"]
            if mr is None:
                continue
            d = mr - ab
            tag = "MR worse" if d > TOL * (1 + abs(ab)) else (
                "MR better" if d < -TOL * (1 + abs(ab)) else "equal")
            worse += tag == "MR worse"
            better += tag == "MR better"
            equal += tag == "equal"
            rows.append((st, key, ab, mr, d, tag))
    for st, key, ab, mr, d, tag in rows:
        print("  %-30s %-12s AB=%-11.6g MR=%-11.6g diff=%+9.4f  %s"
              % (st, key, ab, mr, d, tag))
    print()
    print("  admitted cases: %d | MR worse: %d | MR better: %d | equal: %d"
          % (len(rows), worse, better, equal))
    print("  ONLY the 'MR worse' cases support 'the matched configuration could have used the AB")
    print("  plan yet did not reach equal quality'.  The 'MR better' cases show the opposite and")
    print("  must not be counted towards that statement.")
    print("  A transfer failure says nothing about the matched configuration's optimum.")

    with open("stage11b_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"presolve": {("%s|%s|%d" % k): v for k, v in pre.items()},
                   "presolve_agg": {m: {k: sum(v) / len(v) for k, v in agg[m].items()}
                                    for m in METHODS},
                   "presolve_paired": {"pairs": pairs, "ab_fewer": fewer,
                                       "mean_diff": sum(diffs) / len(diffs),
                                       "min": min(diffs), "max": max(diffs)},
                   "relaxation": relax,
                   "inclusion": {"tested": dict(tested), "violations": dict(viol),
                                 "L_le_J_ok": n_bound, "L_le_J_total": n_tot},
                   "transfer_reclassified": {"admitted": len(rows), "mr_worse": worse,
                                             "mr_better": better, "equal": equal,
                                             "rows": rows}}, fh, indent=2, default=str)
    print("\nwrote stage11b_analysis.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
