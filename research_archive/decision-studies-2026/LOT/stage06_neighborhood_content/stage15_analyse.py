"""Stage 15 analysis: disruption breadth x duration crossed grid.

Primary metric, per nominal instance n and cell (breadth b, duration d):

    D_{n,b,d} = (J_END - J_RECOVERY) / max(1, |J_r|)

Reported per cell as mean / direction over the 8 instances / the two rho groups, together with
    g(RECOVERY) = (J_A - J_RECOVERY) / max(1, |J_r|)
    g(AB)       = (J_A - J_AB)       / max(1, |J_r|)
so that "RECOVERY picked the wrong position" can be told apart from "this cell simply has almost
no short-term-Z improvement to capture".

The 32 states are NOT treated as independent samples: the nominal instance is the unit.
"""

import collections
import csv
import json
import statistics as st
from typing import Any, Dict, List

RUNS_FILE = "stage15_runs.json"
OUT_JSON = "stage15_analysis.json"
OUT_CSV = "stage15_per_instance.csv"

METHODS = ("A", "AB", "RECOVERY", "END")

# (breadth, duration) -> disruption name; printed in the user's requested order
CELLS = [
    ("single", "1p", "D1_m0_1p"),
    ("single", "2p", "D1_m0_2p"),
    ("all", "1p", "D2_all_1p"),
    ("all", "2p", "D2_all_2p"),
]
CELL_LABEL = {"D1_m0_1p": "single/1p", "D1_m0_2p": "single/2p",
              "D2_all_1p": "all/1p", "D2_all_2p": "all/2p"}
THRESHOLD = 0.02          # engineering reference against J_r, NOT a significance level


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def direction(xs):
    xs = list(xs)
    return {"n": len(xs), "positive": sum(1 for v in xs if v > 0),
            "negative": sum(1 for v in xs if v < 0),
            "tie": sum(1 for v in xs if v == 0),
            "mean": mean(xs), "median": st.median(xs) if xs else float("nan"),
            "min": min(xs) if xs else float("nan"), "max": max(xs) if xs else float("nan")}


def main() -> int:
    with open(RUNS_FILE, encoding="utf-8") as fh:
        payload = json.load(fh)
    runs = payload["runs"]
    print("Stage 15 | %d MIPs | breadth x duration crossed grid" % len(runs))

    by = collections.defaultdict(list)
    for r in runs:
        by[(r["state"], r["method"])].append(r)
    states = sorted({r["state"] for r in runs})
    insts = sorted({s.split("|")[0] for s in states})
    print("states: %d | nominal instances: %d | seeds: %s | budget: %.1f s"
          % (len(states), len(insts), sorted({r["seed"] for r in runs}),
             payload["meta"]["budget_s"]))

    def jr_of(state):
        return max(1.0, abs(by[(state, "A")][0]["repair_cost"]))

    def jm(state, method):
        return mean(r["final_cost"] for r in by[(state, method)])

    # ---------------- per-state table, keyed by (instance, disruption) ----------------
    cell_of_state = {r["state"]: r["disruption"] for r in runs}
    per: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
    for s in states:
        dis = cell_of_state[s]
        per[dis][s.split("|")[0]] = {
            "J_A": jm(s, "A"), "J_AB": jm(s, "AB"),
            "J_RECOVERY": jm(s, "RECOVERY"), "J_END": jm(s, "END"),
            "J_r": jr_of(s),
            "D": (jm(s, "END") - jm(s, "RECOVERY")) / jr_of(s),
            "g_RECOVERY": (jm(s, "A") - jm(s, "RECOVERY")) / jr_of(s),
            "g_AB": (jm(s, "A") - jm(s, "AB")) / jr_of(s),
        }

    print("\n=== release structure per cell ===")
    print("  %-10s %-8s %-4s | %-11s %-6s | %s"
          % ("cell", "breadth", "dur", "r_j", "relZ", "n_affected"))
    for bre, dur, dis in CELLS:
        ss = [s for s in states if cell_of_state[s] == dis]
        rec = by[(ss[0], "RECOVERY")][0]
        print("  %-10s %-8s %-4s | %-11s %-6d | %d"
              % (CELL_LABEL[dis], bre, dur,
                 ",".join(str(rec["recovery_periods"][k])
                          for k in sorted(rec["recovery_periods"], key=int)),
                 rec["n_rel_Z"], len([k for k, v in rec["recovery_periods"].items()
                                      if int(v) <= 6])))

    # ---------------- PRIMARY: D per cell ----------------
    print("\n=== PRIMARY  D = (J_END - J_RECOVERY)/max(1,|J_r|)  per cell ===")
    print("  %-10s %10s %10s | %-24s | %s"
          % ("cell", "mean D", "median D", "direction", "by rho group (mean, pos/n)"))
    primary = {}
    for bre, dur, dis in CELLS:
        d = {n: v["D"] for n, v in per[dis].items()}
        info = direction(d.values())
        groups = {}
        for g in ("0.75", "1.10"):
            vals = [v for n, v in d.items() if ("rho%s" % g) in n]
            groups[g] = {"mean": mean(vals),
                         "positive": sum(1 for v in vals if v > 0), "n": len(vals)}
        primary[dis] = {"label": CELL_LABEL[dis], "breadth": bre, "duration": dur,
                        "mean": info["mean"], "median": info["median"],
                        "positive": info["positive"], "negative": info["negative"],
                        "tie": info["tie"], "n": info["n"],
                        "min": info["min"], "max": info["max"], "by_rho": groups,
                        "above_threshold": int(info["mean"] >= THRESHOLD),
                        "per_instance": d}
        print("  %-10s %+10.5f %+10.5f | pos %d neg %d tie %d (n=%d)%s | "
              "rho0.75 %+.5f %d/%d ; rho1.10 %+.5f %d/%d"
              % (CELL_LABEL[dis], info["mean"], info["median"], info["positive"],
                 info["negative"], info["tie"], info["n"], " " * 0,
                 groups["0.75"]["mean"], groups["0.75"]["positive"], groups["0.75"]["n"],
                 groups["1.10"]["mean"], groups["1.10"]["positive"], groups["1.10"]["n"]))

    # ---------------- main effects on the 2x2 grid ----------------
    print("\n=== 2x2 structure: main effects on D (equal-weight over the 8 instances) ===")
    D = {CELL_LABEL[dis]: primary[dis]["mean"] for _, _, dis in CELLS}
    print("  %-14s %-12s %s" % ("", "1p", "2p"))
    for bre in ("single", "all"):
        print("  %-14s %+12.5f %+12.5f"
              % (bre, D["%s/1p" % bre], D["%s/2p" % bre]))
    breadth_eff = 0.5 * ((D["all/1p"] - D["single/1p"]) + (D["all/2p"] - D["single/2p"]))
    duration_eff = 0.5 * ((D["single/2p"] - D["single/1p"]) + (D["all/2p"] - D["all/1p"]))
    interaction = (D["all/2p"] - D["all/1p"]) - (D["single/2p"] - D["single/1p"])
    print("\n  main effect of BREADTH (all - single), averaged over durations : %+.5f" % breadth_eff)
    print("  main effect of DURATION (2p - 1p), averaged over breadths     : %+.5f" % duration_eff)
    print("  interaction ((all/2p - all/1p) - (single/2p - single/1p))    : %+.5f" % interaction)
    print("  cells above the 2%% reference: %d/4  %s"
          % (sum(1 for _, _, dis in CELLS if primary[dis]["mean"] >= THRESHOLD),
             [CELL_LABEL[dis] for _, _, dis in CELLS if primary[dis]["mean"] >= THRESHOLD]))

    # ---------------- g(RECOVERY) and g(AB) per cell ----------------
    print("\n=== g(RECOVERY) and g(AB), both relative to A, per cell ===")
    print("  %-10s %14s %14s %14s" % ("cell", "g(RECOVERY)", "g(AB)", "g(REC)/g(AB)"))
    gains = {}
    for bre, dur, dis in CELLS:
        gr = mean(v["g_RECOVERY"] for v in per[dis].values())
        ga = mean(v["g_AB"] for v in per[dis].values())
        rate = (gr / ga) if abs(ga) > 1e-12 else float("nan")
        gains[dis] = {"g_RECOVERY": gr, "g_AB": ga, "coverage": rate,
                      "g_RECOVERY_positive": sum(1 for v in per[dis].values()
                                                 if v["g_RECOVERY"] > 0),
                      "g_AB_positive": sum(1 for v in per[dis].values() if v["g_AB"] > 0),
                      "n": len(per[dis])}
        print("  %-10s %+14.5f %+14.5f %14s"
              % (CELL_LABEL[dis], gr, ga, "%.4f" % rate if rate == rate else "-"))
    print("\n  ==== pooled over all 32 states (equal-weight over instances within each cell) ====")
    gr_all = mean(v["g_RECOVERY"] for dis in per for v in per[dis].values())
    ga_all = mean(v["g_AB"] for dis in per for v in per[dis].values())
    print("  g(RECOVERY)=%+.5f   g(AB)=%+.5f   pooled R=%s"
          % (gr_all, ga_all, "%.4f" % (gr_all / ga_all) if abs(ga_all) > 1e-12 else "-"))

    # ---------------- per-instance table ----------------
    print("\n=== per-instance detail ===")
    print("  %-19s %-10s %10s %10s %10s %10s | %9s %9s %9s"
          % ("instance", "cell", "J_A", "J_AB", "J_REC", "J_END", "D", "g(REC)", "g(AB)"))
    for bre, dur, dis in CELLS:
        for n in insts:
            v = per[dis][n]
            print("  %-19s %-10s %10.1f %10.1f %10.1f %10.1f | %+9.5f %+9.5f %+9.5f"
                  % (n, CELL_LABEL[dis], v["J_A"], v["J_AB"], v["J_RECOVERY"], v["J_END"],
                     v["D"], v["g_RECOVERY"], v["g_AB"]))

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "cell", "breadth", "duration", "disruption", "J_A", "J_AB",
                    "J_RECOVERY", "J_END", "J_r", "D", "g_RECOVERY", "g_AB"])
        for bre, dur, dis in CELLS:
            for n in insts:
                v = per[dis][n]
                w.writerow([n, CELL_LABEL[dis], bre, dur, dis] +
                           ["%.4f" % v[k] for k in ("J_A", "J_AB", "J_RECOVERY", "J_END",
                                                    "J_r", "D", "g_RECOVERY", "g_AB")])

    # ---------------- efficiency ----------------
    print("\n=== efficiency: time and presolved size by method ===")
    eff = {}
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        pre = [r["presolve"] for r in v if not r["presolve"].get("missing")]
        eff[m] = {"total_s_mean": mean(r["timing"]["total_s"] for r in v),
                  "cols_mean": mean(p["cols"] for p in pre) if pre else None,
                  "binary_mean": mean(p["binary"] for p in pre if p.get("binary") is not None)
                  if pre else None,
                  "proven_optimal": sum(1 for r in v if r["mip_proven_optimal"]),
                  "n": len(v)}
        print("  %-9s total_s mean=%.3f  presolve cols mean=%.1f  binary mean=%.1f  "
              "proven optimal %d/%d"
              % (m, eff[m]["total_s_mean"], eff[m]["cols_mean"] or float("nan"),
                 eff[m]["binary_mean"] or float("nan"), eff[m]["proven_optimal"], eff[m]["n"]))

    # ---------------- integrity ----------------
    print("\n=== run integrity ===")
    integ = {
        "n_runs": len(runs),
        "start_adopted": sum(1 for r in runs if r["mip_start_adopted"]),
        "select_errors": sum(len(r["select_errors"]) if isinstance(r["select_errors"], list)
                             else r["select_errors"] for r in runs),
        "final_source_solver": sum(1 for r in runs if r["final_source"] == "solver"),
        "worse_than_repair": sum(1 for r in runs if r["final_cost"] is not None
                                 and r["final_cost"] > r["repair_cost"]
                                 + 1e-6 * (1 + abs(r["repair_cost"]))),
        "not_strictly_below_lp": sum(1 for r in runs if not r["strictly_below_lp"]),
        "proven_optimal": sum(1 for r in runs if r["mip_proven_optimal"]),
    }
    for k, v in integ.items():
        print("  %-24s %s" % (k, v))

    # ---------------- pre-registered reading ----------------
    print("\n=== pre-registered interpretation (2%% reference against J_r) ===")
    strong = {CELL_LABEL[dis]: primary[dis]["mean"] >= THRESHOLD for _, _, dis in CELLS}
    for k, v in strong.items():
        print("  %-10s mean D=%+.5f  %s" % (k, D[k], ">= 2%" if v else "< 2%"))
    verdict = []
    if strong["single/1p"] and not strong["all/2p"]:
        verdict.append("SUPPORTS duration / earlier-recovery reading "
                       "(new single/1p works, new all/2p weaker)")
    if strong["all/2p"] and not strong["single/1p"]:
        verdict.append("SUPPORTS breadth reading "
                       "(new all/2p works, new single/1p weaker)")
    if strong["all/1p"] and not strong["single/1p"] and not strong["all/2p"]:
        verdict.append("NARROW INTERACTION: only the original all/1p cell is strong")
    stable = sum(1 for _, _, dis in CELLS
                 if primary[dis]["positive"] >= 6 and primary[dis]["mean"] >= THRESHOLD)
    if stable == 0:
        verdict.append("no cell is both stable (>=6/8 positive) and >=2%: "
                       "position line NOT extended; keep AB and return to the whole-method "
                       "comparison against a strong baseline")
    if not verdict:
        verdict.append("no pre-registered pattern matched cleanly; report as inconclusive")
    for v in verdict:
        print("  => %s" % v)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": payload["meta"], "cells": {
            CELL_LABEL[dis]: {"primary_D": primary[dis], "gains": gains[dis]}
            for _, _, dis in CELLS},
            "D_grid": D, "breadth_effect": breadth_eff, "duration_effect": duration_eff,
            "interaction": interaction, "pooled": {"g_RECOVERY": gr_all, "g_AB": ga_all},
            "efficiency": eff, "integrity": integ, "verdict": verdict}, fh, indent=2,
            default=str)
    print("\nwrote %s and %s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
