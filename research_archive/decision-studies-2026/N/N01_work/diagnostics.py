"""Detailed diagnostics on the frozen run and the N01-S probe, for the report."""
from __future__ import annotations

import csv
import gzip
import json
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parent
OUT = {}


def rows(name):
    return list(csv.DictReader((WORK / name).open(encoding="utf-8-sig")))


def f(r, k):
    return float(r[k])


def analyse(label, rs, worlds):
    n = len(rs)
    dc = [f(r, "delta_c_mean") for r in rs]
    dd = [f(r, "delta_d_mean") for r in rs]
    # point estimates on opposite sides
    opp = [i for i in range(n) if dc[i] * dd[i] < 0]
    # cases where X_c == X_d in every confirm world
    ident = []
    for r, pc in zip(rs, worlds["per_case"]):
        xc, xd = pc["confirm"]["X_c"], pc["confirm"]["X_d"]
        if all(u == v for u, v in zip(xc, xd)):
            ident.append(int(r["index"]))
    # coupling magnitude: mean |X_c - X_d|
    coup = []
    for pc in worlds["per_case"]:
        d = [abs(u - v) for u, v in zip(pc["confirm"]["X_c"], pc["confirm"]["X_d"])]
        coup.append(sum(d) / len(d))
    # the frozen-condition identity term, spelled out from the raw spreads
    identity_term = []
    for pc in worlds["per_case"]:
        sp = pc["confirm"]["spreads"] if "spreads" in pc["confirm"] else None
        identity_term.append(sp is not None)
    out = {
        "n": n,
        "delta_c_range": [min(dc), max(dc)],
        "delta_d_range": [min(dd), max(dd)],
        "opposite_sign_point_estimates": opp,
        "n_opposite_sign_point_estimates": len(opp),
        "identical_all_worlds": ident,
        "max_abs_Xc_minus_Xd_mean": max(coup),
        "mean_abs_Xc_minus_Xd_mean": sum(coup) / len(coup),
        "median_m_over_n": statistics.median([f(r, "m_over_n") for r in rs]),
        "max_m_over_n": max(f(r, "m_over_n") for r in rs),
    }
    print(f"\n=== {label} ===")
    print(f"  delta_c range {min(dc):+.4f} .. {max(dc):+.4f}")
    print(f"  delta_d range {min(dd):+.4f} .. {max(dd):+.4f}")
    print(f"  opposite-sign point estimates: {len(opp)}/{n} -> cases {opp}")
    print(f"  X_c == X_d in ALL 2048 confirm worlds: {len(ident)}/{n} -> cases {ident}")
    print(f"  mean |X_c - X_d| over cases: max {max(coup):.5f}, mean {sum(coup)/len(coup):.5f}")
    print(f"  m/n: median {out['median_m_over_n']:.6f}, max {out['max_m_over_n']:.6f}")
    return out


main_rows = rows("N01_results.csv")
probe_rows = rows("N01S_results.csv")
main_worlds = json.loads(gzip.open(WORK / "N01_worlds.json.gz", "rt", encoding="utf-8").read())
probe_worlds = json.loads(gzip.open(WORK / "N01S_worlds.json.gz", "rt", encoding="utf-8").read())

# the main worlds payload holds both runs; the probe file is standalone
OUT["frozen"] = analyse("FROZEN (40 comparisons, condition forbids coupling)", main_rows, main_worlds)
OUT["probe"] = analyse("PROBE N01-S (40 comparisons, cross-side coupling allowed)", probe_rows,
                       probe_worlds)

# Identity check on the frozen run: is X_c == X_d exact, and what is the biggest exception?
worst = None
for r, pc in zip(main_rows, main_worlds["per_case"]):
    xc, xd = pc["confirm"]["X_c"], pc["confirm"]["X_d"]
    d = [abs(u - v) for u, v in zip(xc, xd)]
    if worst is None or max(d) > worst[1]:
        worst = (int(r["index"]), max(d), sum(d) / len(d))
print(f"\n  frozen: largest single-world |X_c - X_d| = {worst[1]} on case {worst[0]} "
      f"(mean over that case's worlds {worst[2]:.6f})")
OUT["frozen_worst_identity_violation"] = {"case": worst[0], "max_abs": worst[1],
                                          "mean_abs": worst[2]}

# how many frozen comparisons are exact-identity in every single world
exact_all = 0
for pc in main_worlds["per_case"]:
    ok = all(u == v for u, v in zip(pc["confirm"]["X_c"], pc["confirm"]["X_d"]))
    exact_all += 1 if ok else 0
print(f"  frozen: comparisons with X_c == X_d in every confirm world: {exact_all}/40")
OUT["frozen_exact_all_worlds"] = exact_all

# For the probe: the strongest case where the two sides disagree most in mean
gaps = []
for r, pc in zip(probe_rows, probe_worlds["per_case"]):
    xc, xd = pc["confirm"]["X_c"], pc["confirm"]["X_d"]
    gaps.append((f(r, "delta_c_mean") - f(r, "delta_d_mean"), int(r["index"]),
                 f(r, "delta_c_mean"), f(r, "delta_d_mean")))
gaps.sort(key=lambda t: -abs(t[0]))
print("\n  probe: largest |delta_c - delta_d| in mean:")
for g, i, a, b in gaps[:6]:
    print(f"    case {i:>2}: delta_c {a:+.4f}, delta_d {b:+.4f}, gap {g:+.4f}")

(WORK / "diagnostics.json").write_text(json.dumps(OUT, indent=2), encoding="utf-8")
print(f"\nwrote {WORK / 'diagnostics.json'}")
