"""Stage 16 analysis: AB vs FULL vs RINS-LP-ONE baseline screen.

Sole PRIMARY comparison, frozen in advance:

    D_s = (J_{RINS-LP-ONE,s} - J_{AB,s}) / max(1, |J_{r,s}|)

Positive => AB better.  Averaged EQUALLY over the four disruption cells within each nominal
instance first, then equally over the 8 nominal instances.  The 32 states are NOT 32
independent samples; the nominal instance is the unit.

Symmetric engineering criteria, frozen in advance:
    AB mean advantage >= 2%, >= 6/8 nominal instances positive, both rho groups positive, and
    the mean stays positive under leave-one-instance-out
        -> AB passes this screen; continue against in-domain fix-and-optimize opponents
    RINS-LP-ONE reaches the same conditions in the reverse direction
        -> a generic LP-information rule is more competitive; stop pushing fixed AB as the core
    anything else
        -> inconclusive; do NOT tune the threshold for a win, do NOT start training
"""

import collections
import csv
import json
import statistics as st
from typing import Any, Dict, List

RUNS_FILE = "stage16_runs.json"
OUT_JSON = "stage16_analysis.json"
OUT_CSV = "stage16_per_instance.csv"

METHODS = ("AB", "FULL", "RINS-LP-ONE")
CELLS = [("single", "1p", "D1_m0_1p"), ("single", "2p", "D1_m0_2p"),
         ("all", "1p", "D2_all_1p"), ("all", "2p", "D2_all_2p")]
CELL_LABEL = {c[2]: "%s/%s" % (c[0], c[1]) for c in CELLS}
THRESHOLD = 0.02


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def main() -> int:
    with open(RUNS_FILE, encoding="utf-8") as fh:
        payload = json.load(fh)
    runs = payload["runs"]
    print("Stage 16 | %d MIPs | AB vs FULL vs RINS-LP-ONE" % len(runs))
    print("scope: %s" % payload["meta"]["scope"])
    print("budget: %.1f s | LP budget: %.1f s | RINS tol: %g | seed %s"
          % (payload["meta"]["budget_s"], payload["meta"]["lp_budget_s"],
             payload["meta"]["rins_tol"], payload["meta"]["seed"]))

    by = collections.defaultdict(list)
    for r in runs:
        by[(r["state"], r["method"])].append(r)
    states = sorted({r["state"] for r in runs})
    insts = sorted({s.split("|")[0] for s in states})
    cell_of = {r["state"]: r["disruption"] for r in runs}

    def jr(s):
        return max(1.0, abs(by[(s, "AB")][0]["repair_cost"]))

    def jm(s, m):
        return mean(r["final_cost"] for r in by[(s, m)])

    # ---------------- release structure ----------------
    print("\n=== neighbourhood structure per method (over 32 states) ===")
    print("  %-13s %-16s %-16s %-16s" % ("method", "fixed (min..max)", "free (min..max)",
                                          "free binaries after"))
    nf = {}
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        fx = sorted({r["n_fixed_total"] for r in v})
        fr = sorted({r["n_free_binary_after"] for r in v})
        nf[m] = {"fixed": fx[0] if len(fx) == 1 else (fx[0], fx[-1]),
                 "free": fr[0] if len(fr) == 1 else (fr[0], fr[-1])}
        print("  %-13s %-16s %-16s %-16s"
              % (m, ("%d" % fx[0]) if len(fx) == 1 else "%d..%d" % (fx[0], fx[-1]),
                 ("%d" % fr[0]) if len(fr) == 1 else "%d..%d" % (fr[0], fr[-1]), ""))

    print("\n=== RINS-LP-ONE fixed counts by group (per state) ===")
    gsum = collections.defaultdict(list)
    for r in runs:
        if r["method"] == "RINS-LP-ONE" and r.get("n_fixed_by_group"):
            for k, v in r["n_fixed_by_group"].items():
                gsum[k].append(v)
    for k in ("A", "B", "C", "short_Y", "structural"):
        if gsum[k]:
            print("  %-11s min=%d max=%d mean=%.1f" % (k, min(gsum[k]), max(gsum[k]),
                                                       mean(gsum[k])))

    # ---------------- PRIMARY ----------------
    print("\n=== PRIMARY  D = (J_RINS-LP-ONE - J_AB)/max(1,|J_r|)   positive => AB better ===")
    per: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
    for s in states:
        per["RINS-LP-ONE"][s] = {
            "J_AB": jm(s, "AB"), "J_FULL": jm(s, "FULL"), "J_RINS": jm(s, "RINS-LP-ONE"),
            "J_r": jr(s),
            "D": (jm(s, "RINS-LP-ONE") - jm(s, "AB")) / jr(s),
            "D_full": (jm(s, "FULL") - jm(s, "AB")) / jr(s),
            "g_FULL_vs_RINS": (jm(s, "FULL") - jm(s, "RINS-LP-ONE")) / jr(s),
        }

    def inst_means(field, method="RINS-LP-ONE"):
        out = {}
        for n in insts:
            v = [per[method][s][field] for s in states if s.split("|")[0] == n]
            out[n] = mean(v)
        return out

    D = inst_means("D")
    print("  %-19s %12s" % ("instance", "D (primary)"))
    for n in insts:
        print("  %-19s %+12.5f" % (n, D[n]))
    pos = sum(1 for v in D.values() if v > 0)
    neg = sum(1 for v in D.values() if v < 0)
    tie = sum(1 for v in D.values() if v == 0)
    print("  n=%d mean=%+.5f median=%+.5f  positive=%d negative=%d tie=%d"
          % (len(D), mean(D.values()), st.median(D.values()), pos, neg, tie))

    groups = {}
    for g in ("0.75", "1.10"):
        vals = [v for n, v in D.items() if ("rho%s" % g) in n]
        groups[g] = {"mean": mean(vals), "positive": sum(1 for v in vals if v > 0),
                     "n": len(vals)}
        print("  rho%-6s mean=%+.5f positive=%d/%d" % (g, groups[g]["mean"],
                                                       groups[g]["positive"], groups[g]["n"]))

    print("\n  leave-one-instance-out mean range:")
    loo = {}
    for n in insts:
        rest = [v for k, v in D.items() if k != n]
        loo[n] = mean(rest)
    print("    min=%+.5f (drop %s)  max=%+.5f (drop %s)  all positive=%s"
          % (min(loo.values()), min(loo, key=loo.get), max(loo.values()),
             max(loo, key=loo.get), all(v > 0 for v in loo.values())))

    print("\n  === by disruption cell ===")
    print("  %-10s %12s %12s %12s" % ("cell", "mean D", "positive", "n"))
    cell_D = {}
    for bre, dur, dis in CELLS:
        v = [per["RINS-LP-ONE"][s]["D"] for s in states if cell_of[s] == dis]
        cell_D[CELL_LABEL[dis]] = {"mean": mean(v), "positive": sum(1 for x in v if x > 0),
                                   "n": len(v)}
        print("  %-10s %+12.5f %12s %12d"
              % (CELL_LABEL[dis], cell_D[CELL_LABEL[dis]]["mean"],
                 "%d/%d" % (cell_D[CELL_LABEL[dis]]["positive"], len(v)), len(v)))

    # ---------------- FULL as reference ----------------
    print("\n=== FULL reference: is either neighbourhood better than solving the full task? ===")
    Df = inst_means("D_full")
    print("  D_FULL = (J_FULL - J_AB)/D      mean=%+.5f positive=%d/%d"
          % (mean(Df.values()), sum(1 for v in Df.values() if v > 0), len(Df)))
    dr = inst_means("g_FULL_vs_RINS")
    print("  (J_FULL - J_RINS)/D             mean=%+.5f positive=%d/%d  "
          "(positive => RINS-LP-ONE better than FULL)"
          % (mean(dr.values()), sum(1 for v in dr.values() if v > 0), len(dr)))

    # ---------------- diagnostics ----------------
    print("\n=== diagnostics (explanatory only; no threshold is tuned on these) ===")
    rl = [r for r in runs if r["method"] == "RINS-LP-ONE"]
    print("  fallbacks: %d/%d" % (sum(1 for r in rl if r["fallback"]), len(rl)))
    print("  relax optimal: %d/%d ; relax <= J_LP: %d/%d"
          % (sum(1 for r in rl if r["relax_is_optimal"]), len(rl),
             sum(1 for r in rl if r["relax_le_j_lp"]), len(rl)))
    print("  relax time mean=%.3f s (max %.3f) | relax full-budget hits: %d"
          % (mean(r["timing"]["relax_s"] for r in rl), max(r["timing"]["relax_s"] for r in rl),
             sum(1 for r in rl if r["timing"]["relax_s"] >= 4.9)))
    print("  MIP start adopted: %d/%d ; proven optimal: %d/%d"
          % (sum(1 for r in rl if r["mip_start_adopted"]), len(rl),
             sum(1 for r in rl if r["mip_proven_optimal"]), len(rl)))
    print("\n  --- S_LP is NOT a lower bound for the full task (binaries are pinned in it) ---")
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        imp = sum(1 for r in v if r.get("mip_improves_over_s_lp"))
        print("  %-13s MIP strictly better than S_LP: %2d/%d | S_LP itself optimal (%s): %d/%d"
              % (m, imp, len(v), "lp_status", sum(1 for r in v if r.get("lp_is_optimal")),
                 len(v)))
    src = collections.Counter(r["final_source"] for r in rl)
    print("  final source (RINS): %s" % dict(src))
    print("  final source (AB):   %s"
          % dict(collections.Counter(r["final_source"] for r in runs if r["method"] == "AB")))
    print("  final source (FULL): %s"
          % dict(collections.Counter(r["final_source"] for r in runs if r["method"] == "FULL")))
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        pre = [r["presolve"] for r in v if not r["presolve"].get("missing")]
        print("  %-13s total_s mean=%.3f | presolve cols mean=%.1f binary mean=%.1f"
              % (m, mean(r["timing"]["total_s"] for r in v),
                 mean(p["cols"] for p in pre) if pre else float("nan"),
                 mean(p["binary"] for p in pre if p.get("binary") is not None)
                 if pre else float("nan")))

    # ---------------- integrity ----------------
    print("\n=== run integrity ===")
    integ = {
        "n_runs": len(runs),
        "start_adopted": sum(1 for r in runs if r["mip_start_adopted"]),
        "select_errors": sum(len(r["select_errors"]) if isinstance(r["select_errors"], list)
                             else r["select_errors"] for r in runs),
        "final_source_none": sum(1 for r in runs if r["final_source"] == "none"),
        "worse_than_repair": sum(1 for r in runs if r["final_cost"] is not None
                                 and r["final_cost"] > r["repair_cost"]
                                 + 1e-6 * (1 + abs(r["repair_cost"]))),
        "proven_optimal": sum(1 for r in runs if r["mip_proven_optimal"]),
        "runs_le_20s_plus_tol": sum(1 for r in runs if r["timing"]["total_s"] <= 20.25),
    }
    for k, v in integ.items():
        print("  %-24s %s" % (k, v))

    # ---------------- frozen verdict ----------------
    print("\n=== frozen symmetric verdict ===")
    ab_pass = (mean(D.values()) >= THRESHOLD and pos >= 6 and groups["0.75"]["mean"] > 0
               and groups["1.10"]["mean"] > 0 and all(v > 0 for v in loo.values()))
    rev = {n: -v for n, v in D.items()}
    rpos = sum(1 for v in rev.values() if v > 0)
    rloo = {n: mean([v for k, v in rev.items() if k != n]) for n in insts}
    rins_pass = (mean(rev.values()) >= THRESHOLD and rpos >= 6
                 and mean([v for n, v in rev.items() if "rho0.75" in n]) > 0
                 and mean([v for n, v in rev.items() if "rho1.10" in n]) > 0
                 and all(v > 0 for v in rloo.values()))
    print("  AB mean advantage            = %+.5f (>= 2%%? %s)" % (mean(D.values()),
                                                                   mean(D.values()) >= THRESHOLD))
    print("  AB positive instances        = %d/8 (>= 6? %s)" % (pos, pos >= 6))
    print("  both rho groups positive     = %s" % (groups["0.75"]["mean"] > 0
                                                   and groups["1.10"]["mean"] > 0))
    print("  leave-one-out all positive   = %s" % all(v > 0 for v in loo.values()))
    print("  => AB passes the screen      = %s" % ab_pass)
    print("  => RINS-LP-ONE passes reverse= %s" % rins_pass)
    if ab_pass:
        verdict = ("AB passes this baseline screen; continue against in-domain "
                   "fix-and-optimize opponents")
    elif rins_pass:
        verdict = ("RINS-LP-ONE reaches the criteria in reverse: a generic LP-information rule "
                   "is more competitive; stop pushing fixed AB as the core contribution")
    else:
        verdict = ("inconclusive: do not tune the threshold for a win and do not start training")
    print("  VERDICT: %s" % verdict)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "cell", "disruption", "J_AB", "J_FULL", "J_RINS",
                    "J_r", "D", "D_full", "g_FULL_vs_RINS"])
        for s in states:
            v = per["RINS-LP-ONE"][s]
            w.writerow([s.split("|")[0], CELL_LABEL[cell_of[s]], cell_of[s]] +
                       ["%.4f" % v[k] for k in ("J_AB", "J_FULL", "J_RINS", "J_r", "D",
                                                "D_full", "g_FULL_vs_RINS")])

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": payload["meta"], "primary_D": D, "mean_D": mean(D.values()),
                   "positive": pos, "negative": neg, "tie": tie,
                   "by_rho": groups, "by_cell": cell_D, "leave_one_out": loo,
                   "D_full": Df, "D_full_vs_rins": dr,
                   "neighbourhood_structure": nf,
                   "rins_fixed_by_group": {k: {"min": min(v), "max": max(v), "mean": mean(v)}
                                           for k, v in gsum.items()},
                   "integrity": integ, "ab_passes": ab_pass, "rins_passes_reverse": rins_pass,
                   "verdict": verdict}, fh, indent=2, default=str)
    print("\nwrote %s and %s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
