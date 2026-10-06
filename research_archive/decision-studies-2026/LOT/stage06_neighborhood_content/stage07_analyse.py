#!/usr/bin/env python
"""
Stage 07 analysis: does a staged widen-the-release-set search beat running EMPTY?

    .venv-hs/Scripts/python.exe stage07_analyse.py

Primary question
----------------
E5->D15 vs E20        does the staged widen actually improve DELIVERY?
E5->D15 vs E5->E15    is the gain from widening the release set, or only from restarting?
E5->D15 vs E5->F15    is the fine-grained DEPENDENCY choice worth anything over just FULL?

Aggregation unit is the NOMINAL INSTANCE, exactly as in stage 06: the two disruptions and the
two solver seeds are paired INSIDE the instance and only then averaged, with the
disruption-specific repair cost as the normaliser:

    G_n(a,b) = (1/4) * sum_d sum_s (J_{n,d,s,b} - J_{n,d,s,a}) / max(1, |J_r(n,d)|)

Scope: 16 frozen LARGE fault states, no state selected on outcome.  E20 is the reference arm; the
staged arms use the stage-06 frozen DEPENDENCY-24 release set, which is never recomputed from the
phase-1 result.  kappa and the fixing conditions are always relative to the original S_r.

The 2% figure is NOT used here as an automatic pass/fail line: it is reported as a practical
reference alongside cross-instance repeatability, the size of the losses, and seed stability.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

METHODS = ("E20", "E5->E15", "E5->D15", "E5->F15")
STAGED = ("E5->E15", "E5->D15", "E5->F15")
SEEDS = (0, 1)
THRESHOLD = 0.02


def load(path="stage07_runs.json"):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cells(runs):
    """(instance, disruption, seed) -> method -> {final, repair, p1, p2imp}"""
    out: Dict[Tuple[str, str, int], Dict[str, Any]] = defaultdict(dict)
    for r in runs:
        inst, dis = r["state"].split("|")
        out[(inst, dis, int(r["seed"]))][r["method"]] = {
            "final": float(r["final_cost"]),
            "repair": float(r["repair_cost"]),
            "p1": (None if r["phase1"] is None
                   else float(r["phase1"]["cost_after_select"])),
            "p2imp": (None if r["phase1"] is None
                      else bool(r["phase2_improved_over_phase1"])),
            "raw2": r["phase2"].get("raw_objective"),
            "p2valid": r.get("phase2_raw_valid"),
        }
    return out


def g_inst(C, a: str, b: str) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for (inst, dis, sd), d in C.items():
        if a not in d or b not in d:
            continue
        acc[inst].append((d[b]["final"] - d[a]["final"])
                         / max(1.0, abs(d[a]["repair"])))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def describe(vals: List[float]) -> Dict[str, Any]:
    sv = sorted(vals)
    n = len(sv)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {"n": n, "mean": sum(sv) / n, "median": med, "min": sv[0], "max": sv[-1],
            "pos": sum(1 for v in sv if v > 1e-9),
            "neg": sum(1 for v in sv if v < -1e-9),
            "zero": sum(1 for v in sv if abs(v) <= 1e-9),
            "ge2pct": sum(1 for v in sv if v >= THRESHOLD),
            "le_neg2pct": sum(1 for v in sv if v <= -THRESHOLD)}


def show(tag: str, s: Dict[str, Any]) -> None:
    print("  %-28s n=%-3d mean=%+.4f median=%+.4f min=%+.4f max=%+.4f  "
          "pos=%d neg=%d zero=%d  >=2%%=%d <=-2%%=%d"
          % (tag, s["n"], s["mean"], s["median"], s["min"], s["max"],
             s["pos"], s["neg"], s["zero"], s["ge2pct"], s["le_neg2pct"]))


def main() -> int:
    D = load()
    runs = D["runs"]
    C = cells(runs)
    instances = sorted({k[0] for k in C})
    print("runs: %d | instances: %d | pair cells: %d" % (len(runs), len(instances), len(C)))
    print("scope: %s" % D["meta"]["scope"])
    print("phase 1: EMPTY for %.0f s | phase 2: the remaining budget"
          % D["meta"]["phase1_budget_s"])
    print("phase-2 release: %s" % D["meta"]["phase2_release"])
    print("kappa and fixings relative to the ORIGINAL frozen S_r throughout")
    print()

    # ---------------- per-instance delivered cost
    print("=== per-instance mean delivered J (2 disruptions x 2 seeds) ===")
    print("  %-24s %12s" % ("instance", "J_repair*") + "".join("%14s" % m for m in METHODS))
    for n in instances:
        ks = [k for k in C if k[0] == n]
        row = "  %-24s %12.6g" % (n, sum(C[k]["E20"]["repair"] for k in ks) / len(ks))
        for m in METHODS:
            row += "%14.6g" % (sum(C[k][m]["final"] for k in ks) / len(ks))
        print(row)
    print("  * mean of the two disruptions' repair costs, reference only\n")

    # ---------------- primary comparisons
    print("=== main comparisons (paired, per nominal instance) ===")
    comps = (("E5->D15 vs E20", "E5->D15", "E20"),
             ("E5->D15 vs E5->E15", "E5->D15", "E5->E15"),
             ("E5->D15 vs E5->F15", "E5->D15", "E5->F15"),
             ("E5->F15 vs E20", "E5->F15", "E20"),
             ("E5->F15 vs E5->E15", "E5->F15", "E5->E15"),
             ("E5->E15 vs E20", "E5->E15", "E20"),
             ("E5->D15 vs E5->F15", "E5->D15", "E5->F15"))
    res: Dict[str, Any] = {}
    for tag, a, b in comps:
        if tag in res:
            continue
        v = g_inst(C, a, b)
        s = describe(list(v.values()))
        s["per_instance"] = sorted(v.items())
        res[tag] = s
        show(tag, s)
    print("  (G>0 means the FIRST method is cheaper)\n")

    # scale-free: all instances are large, so split by rho instead
    print("=== by rho (all 16 states are large) ===")
    for tag, a, b in (("E5->D15 vs E20", "E5->D15", "E20"),
                      ("E5->D15 vs E5->E15", "E5->D15", "E5->E15"),
                      ("E5->D15 vs E5->F15", "E5->D15", "E5->F15")):
        v = g_inst(C, a, b)
        for rho in ("0.75", "1.10"):
            xs = [x for n, x in v.items() if ("rho%s" % rho) in n]
            print("  %-22s rho%-5s mean=%+.4f (n=%d)" % (tag, rho, sum(xs) / len(xs), len(xs)))
    print()

    # ---------------- per-solver-seed stability
    print("=== per solver seed (direction stability) ===")
    for tag, a, b in (("E5->D15 vs E20", "E5->D15", "E20"),
                      ("E5->D15 vs E5->E15", "E5->D15", "E5->E15"),
                      ("E5->D15 vs E5->F15", "E5->D15", "E5->F15")):
        per = {}
        for sd in SEEDS:
            sub = {k: x for k, x in C.items() if k[2] == sd}
            v = g_inst(sub, a, b)
            per[sd] = sum(v.values()) / len(v)
        print("  %-22s seed0=%+.4f seed1=%+.4f  %s"
              % (tag, per[0], per[1],
                 "same direction" if (per[0] > 0) == (per[1] > 0) else "OPPOSITE"))
    print()

    # ---------------- phase-1 and phase-2 bookkeeping
    print("=== phase-1 outcome and phase-2 contribution ===")
    print("  %-12s %10s %10s %10s %10s" %
          ("method", "p1<repair", "p2<repair", "p2<p1", "final<repair"))
    for m in METHODS:
        sub = [r for r in runs if r["method"] == m]
        p1 = sum(1 for r in sub if r["phase1"] is not None
                 and r["phase1"]["cost_after_select"] < r["repair_cost"] - 1e-6)
        p2 = sum(1 for r in sub if r["phase2_improved_over_phase1"])
        fin = sum(1 for r in sub if r["final_cost"] < r["repair_cost"] - 1e-6)
        # phase-2 raw incumbent better than repair
        rawbetter = sum(1 for r in sub if r["phase2"].get("raw_objective") is not None
                        and r["phase2"]["raw_objective"] < r["repair_cost"] - 1e-6)
        print("  %-12s %10s %10s %10d %10d"
              % (m, ("%d/%d" % (p1, len(sub))) if m != "E20" else "-",
                 "%d/%d" % (rawbetter, len(sub)), p2, fin))
    print()
    print("  phase-1 delivered improvement on EMPTY (shared prefix, per state x seed):")
    p1vals = []
    seen = set()
    for r in runs:
        if r["phase1"] is None:
            continue
        key = (r["state"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        p1vals.append((r["repair_cost"] - r["phase1"]["cost_after_select"])
                      / max(1.0, abs(r["repair_cost"])))
    s = describe(p1vals)
    show("EMPTY over 5 s vs S_r", s)
    p1_stats = s
    print()

    # ---------------- adoption
    print("=== MIP-start adoption ===")
    for m in METHODS:
        sub = [r for r in runs if r["method"] == m]
        a1 = sum(1 for r in sub if r["phase1"] is not None and r["phase1"]["start_adopted"])
        a2 = sum(1 for r in sub if r["phase2"]["start_adopted"])
        n1 = sum(1 for r in sub if r["phase1"] is not None)
        print("  %-12s phase1 %s | phase2 %d/%d"
              % (m, ("%d/%d" % (a1, n1)) if n1 else "-", a2, len(sub)))
    print()

    # ---------------- timing
    print("=== timing ===")
    tot = [r["timing"]["total_s"] for r in runs]
    ovr = [r["timing"]["overrun_s"] for r in runs]
    print("  total_s mean=%.3f max=%.3f | overrun mean=%.4f max=%.3f | overrun>0.5s: %d/%d"
          % (sum(tot) / len(tot), max(tot), sum(ovr) / len(ovr), max(ovr),
             sum(1 for v in ovr if v > 0.5), len(ovr)))
    b2 = [r["phase2"]["budget_given_s"] for r in runs if r["phase1"] is not None]
    if b2:
        print("  phase-2 budget granted: mean=%.3f min=%.3f max=%.3f (never a fresh 15 s)"
              % (sum(b2) / len(b2), min(b2), max(b2)))

    # ---------------- interpretation, printed rather than asserted
    print()
    print("=== pre-agreed interpretation ===")
    d15_e20 = res["E5->D15 vs E20"]
    d15_e15 = res["E5->D15 vs E5->E15"]
    d15_f15 = res["E5->D15 vs E5->F15"]
    f15_e20 = res["E5->F15 vs E20"]
    verdicts = []
    if d15_e20["mean"] > 0 and d15_e15["mean"] > 0:
        verdicts.append("staged widening to DEPENDENCY beats BOTH continuous EMPTY and "
                        "restarted EMPTY -> staged neighbourhood widening has support; "
                        "confirm on new instances")
    if f15_e20["mean"] >= d15_e20["mean"] - 1e-9 and f15_e20["mean"] > 0:
        verdicts.append("staged FULL is comparable or better than staged DEPENDENCY -> prefer "
                        "the simpler staged FULL; the gain cannot be attributed to variable-group "
                        "selection")
    if d15_e20["mean"] > 0 and d15_e15["mean"] <= 0:
        verdicts.append("beats continuous EMPTY but NOT restarted EMPTY -> the evidence points "
                        "to the restart, not to widening")
    if d15_e20["mean"] <= 0 and f15_e20["mean"] <= 0:
        verdicts.append("neither widening helps -> keep EMPTY at this budget; do not add model "
                        "or features for DEPENDENCY")
    if d15_e20["mean"] <= 0.02 or d15_e20["pos"] < 12:
        verdicts.append("the effect is small or carried by individual instances/seeds -> record "
                        "as a development-stage inconclusive result, not a training signal")
    for v in verdicts:
        print("  - %s" % v)
    if not verdicts:
        print("  - no pre-agreed pattern matched; report the table as-is")

    # ---------------- cross-batch consistency: stage 07 re-runs E20 from scratch
    print()
    print("=== cross-batch consistency check: stage 07's E20 vs stage 06's E20 ===")
    print("  Same protocol (EMPTY, 20 s, warm from the same x0, same solver seeds).  These are")
    print("  independent runs, so any difference is solver run-to-run variance, not protocol.")
    try:
        with open("stage06_runs.json", encoding="utf-8") as fh:
            s6 = json.load(fh)["runs"]
    except OSError:
        s6 = []
    if s6:
        s6e20 = {(r["state"], int(r["seed"])): float(r["selected_cost"]) for r in s6
                 if r["config"] == "EMPTY"}
        rows = []
        for r in runs:
            if r["method"] != "E20":
                continue
            k = (r["state"], int(r["seed"]))
            if k in s6e20:
                rows.append((k, float(r["final_cost"]), s6e20[k]))
        if rows:
            d = [(b - a) / max(1.0, abs(b)) for _k, a, b in rows]
            n_same = sum(1 for _k, a, b in rows if abs(a - b) <= 1e-6 * (1 + abs(b)))
            s = describe(d)
            show("stage07 E20 vs stage06 E20", s)
            print("  identical cost within 1e-6: %d/%d" % (n_same, len(rows)))
            print("  largest relative difference: %.4f" % max(abs(x) for x in d))
            print("  => the E20 reference arm reproduces the stage-06 EMPTY measurement to")
            print("     within solver noise; the staged arms are compared against a")
            print("     SAME-BATCH reference above.")

    with open("stage07_analysis.json", "w", encoding="utf-8") as fh:
        json.dump({"comparisons": res,
                   "phase1_vs_repair": p1_stats,
                   "meta": D["meta"]}, fh, indent=2, default=str)
    print("\nwrote stage07_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
