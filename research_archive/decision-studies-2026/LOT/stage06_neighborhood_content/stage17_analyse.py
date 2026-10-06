"""Stage 17 analysis: seed-1 confirmation + the targeted RINS-intersect-AB control.

Sole MAIN comparison (frozen):

    G_s = (J_{AB,s} - J_{RINS,s}) / max(1, |J_{r,s}|)          positive => RINS-LP-ONE better

Averaged EQUALLY over the four disruption cells within each nominal instance, then equally over
the 8 nominal instances.  Seed 1 is judged ALONE by the original four engineering criteria; the
two seeds' per-instance directions are listed side by side and NEVER merged into one average.

Sole secondary diagnostic (frozen):

    G_int_AB = (J_AB - J_{RINS-AB}) / D       positive => RINS-AB better than AB
    G_R_int  = (J_{RINS-AB} - J_{RINS}) / D   positive => plain RINS better than RINS-AB
    D = max(1, |J_r|)

The 32 states are NOT 32 independent samples: the nominal instance is the unit.
"""

import collections
import csv
import json
import statistics as st
from typing import Any, Dict, List

RUNS_FILE = "stage17_runs.json"
RUNS16_FILE = "stage16_runs.json"
OUT_JSON = "stage17_analysis.json"
OUT_CSV = "stage17_per_instance.csv"

METHODS = ("AB", "RINS-LP-ONE", "RINS-AB")
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
    print("Stage 17 | %d runs | seed %d | methods %s"
          % (len(runs), payload["meta"]["seed"], list(METHODS)))
    print("scope: %s" % payload["meta"]["scope"])
    print("timing: %s" % payload["meta"]["timing"])

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

    def inst_mean(field, mk):
        out = {}
        for n in insts:
            ss = [s for s in states if s.split("|")[0] == n]
            out[n] = mean(mk(s) for s in ss)
        return out

    # ---------------- comparison 1: seed-1 RINS vs AB ----------------
    G1 = inst_mean("G1", lambda s: (jm(s, "AB") - jm(s, "RINS-LP-ONE")) / jr(s))
    print("\n=== MAIN (seed 1):  G = (J_AB - J_RINS)/D   positive => RINS-LP-ONE better ===")
    print("  %-19s %12s" % ("instance", "G (seed 1)"))
    for n in insts:
        print("  %-19s %+12.5f" % (n, G1[n]))
    pos = sum(1 for v in G1.values() if v > 0)
    neg = sum(1 for v in G1.values() if v < 0)
    print("  n=%d mean=%+.5f median=%+.5f positive=%d negative=%d"
          % (len(G1), mean(G1.values()), st.median(G1.values()), pos, neg))
    groups = {}
    for g in ("0.75", "1.10"):
        vals = [v for n, v in G1.items() if ("rho%s" % g) in n]
        groups[g] = {"mean": mean(vals), "positive": sum(1 for v in vals if v > 0),
                     "n": len(vals)}
        print("  rho%-6s mean=%+.5f positive=%d/%d" % (g, groups[g]["mean"],
                                                       groups[g]["positive"], groups[g]["n"]))
    loo = {n: mean([v for k, v in G1.items() if k != n]) for n in insts}
    print("  leave-one-instance-out: min=%+.5f (drop %s) max=%+.5f (drop %s) all positive=%s"
          % (min(loo.values()), min(loo, key=loo.get), max(loo.values()),
             max(loo, key=loo.get), all(v > 0 for v in loo.values())))

    # ---------------- seed comparison (never merged) ----------------
    print("\n=== per-instance direction, seed 0 vs seed 1 (NOT merged) ===")
    G0 = None
    if RUNS16_FILE and __import__("os").path.exists(RUNS16_FILE):
        with open(RUNS16_FILE, encoding="utf-8") as fh:
            r16 = json.load(fh)["runs"]
        b16 = collections.defaultdict(list)
        for r in r16:
            b16[(r["state"], r["method"])].append(r)
        def jm16(s, m):
            return mean(x["final_cost"] for x in b16[(s, m)])
        G0 = {}
        for n in insts:
            ss = [s for s in states if s.split("|")[0] == n]
            G0[n] = mean((jm16(s, "AB") - jm16(s, "RINS-LP-ONE"))
                         / max(1.0, abs(b16[(s, "AB")][0]["repair_cost"])) for s in ss)
        print("  %-19s %12s %12s %s" % ("instance", "seed 0", "seed 1", "same sign?"))
        for n in insts:
            print("  %-19s %+12.5f %+12.5f %s"
                  % (n, G0[n], G1[n], "yes" if (G0[n] > 0) == (G1[n] > 0) else "NO"))
        print("  mean: seed0=%+.5f  seed1=%+.5f   same-sign instances: %d/8"
              % (mean(G0.values()), mean(G1.values()),
                 sum(1 for n in insts if (G0[n] > 0) == (G1[n] > 0))))

    # ---------------- secondary: RINS-AB ----------------
    print("\n=== SECONDARY: RINS-AB (F_R u F_AB) ===")
    G_int_AB = inst_mean("a", lambda s: (jm(s, "AB") - jm(s, "RINS-AB")) / jr(s))
    G_R_int = inst_mean("b", lambda s: (jm(s, "RINS-AB") - jm(s, "RINS-LP-ONE")) / jr(s))
    print("  G_int_AB = (J_AB - J_RINS-AB)/D   positive => RINS-AB better than AB")
    for n in insts:
        print("    %-19s %+12.5f" % (n, G_int_AB[n]))
    print("    mean=%+.5f positive=%d/%d  rho0.75=%+.5f  rho1.10=%+.5f"
          % (mean(G_int_AB.values()), sum(1 for v in G_int_AB.values() if v > 0), len(G_int_AB),
             mean([v for n, v in G_int_AB.items() if "rho0.75" in n]),
             mean([v for n, v in G_int_AB.items() if "rho1.10" in n])))
    print("  G_R_int  = (J_RINS-AB - J_RINS)/D  positive => plain RINS better than RINS-AB")
    for n in insts:
        print("    %-19s %+12.5f" % (n, G_R_int[n]))
    print("    mean=%+.5f positive=%d/%d" % (mean(G_R_int.values()),
                                             sum(1 for v in G_R_int.values() if v > 0),
                                             len(G_R_int)))

    # ---------------- structure & diagnostics ----------------
    print("\n=== neighbourhood structure ===")
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        fx = sorted({r["n_fixed_total"] for r in v})
        fr = sorted({r["n_free_binary_after"] for r in v})
        ov = sorted({r["n_overlap"] for r in v if r["n_overlap"] is not None})
        print("  %-13s fixed %d..%d | free %d..%d | overlap with F_R %s"
              % (m, fx[0], fx[-1], fr[0], fr[-1], ("%d..%d" % (ov[0], ov[-1])) if ov else "-"))

    print("\n=== diagnostics (explanatory only; no threshold is tuned on these) ===")
    for m in METHODS:
        v = [r for r in runs if r["method"] == m]
        print("  %-13s fallback %d/%d | start adopted %d/%d | proven optimal %d/%d"
              % (m, sum(1 for r in v if r["fallback"]), len(v),
                 sum(1 for r in v if r["mip_start_adopted"]), len(v),
                 sum(1 for r in v if r["mip_proven_optimal"]), len(v)))
    rel = [r for r in runs if r["method"] in ("RINS-LP-ONE", "RINS-AB")]
    print("  relax optimal %d/%d | relax_s mean=%.3f max=%.3f | relax hit 5s cap: %d"
          % (sum(1 for r in rel if r["relax_is_optimal"]), len(rel),
             mean(r["timing"]["relax_s"] for r in rel),
             max(r["timing"]["relax_s"] for r in rel),
             sum(1 for r in rel if r["timing"]["relax_s"] >= 4.9)))
    print("  final source: %s"
          % {m: dict(collections.Counter(r["final_source"] for r in runs if r["method"] == m))
             for m in METHODS})

    print("\n=== run integrity (timing now covers delivery selection) ===")
    integ = {
        "n_runs": len(runs),
        "start_adopted": sum(1 for r in runs if r["mip_start_adopted"]),
        "select_errors": sum(len(r["select_errors"]) if isinstance(r["select_errors"], list)
                             else r["select_errors"] for r in runs),
        "final_source_none": sum(1 for r in runs if r["final_source"] == "none"),
        "worse_than_repair": sum(1 for r in runs if r["final_cost"] is not None
                                 and r["final_cost"] > r["repair_cost"]
                                 + 1e-6 * (1 + abs(r["repair_cost"]))),
        "fixing_conflicts": sum(r["fixing_conflicts"] for r in runs),
        "max_total_s": max(r["timing"]["total_s"] for r in runs),
        "max_overrun_s": max(r["timing"]["overrun_s"] for r in runs),
        "runs_over_budget": sum(1 for r in runs if r["timing"]["overrun_s"] > 0),
    }
    for k, v in integ.items():
        print("  %-22s %s" % (k, v))

    # ---------------- frozen verdict ----------------
    print("\n=== frozen verdict (seed 1 judged ALONE) ===")
    passes = (mean(G1.values()) >= THRESHOLD and pos >= 6
              and groups["0.75"]["mean"] > 0 and groups["1.10"]["mean"] > 0
              and all(v > 0 for v in loo.values()))
    print("  RINS mean advantage over AB  = %+.5f (>= 2%%? %s)"
          % (mean(G1.values()), mean(G1.values()) >= THRESHOLD))
    print("  positive instances           = %d/8 (>= 6? %s)" % (pos, pos >= 6))
    print("  both rho groups positive     = %s"
          % (groups["0.75"]["mean"] > 0 and groups["1.10"]["mean"] > 0))
    print("  leave-one-out all positive   = %s" % all(v > 0 for v in loo.values()))
    print("  => RINS-LP-ONE again passes  = %s" % passes)
    if passes:
        verdict = ("seed 1 CONFIRMS the Stage 16 direction: RINS-LP-ONE still passes all four "
                   "criteria over AB")
    elif abs(mean(G1.values())) < THRESHOLD or pos in (4, 5):
        verdict = ("seed 1 gives a CLOSE or REVERSED result: record as a stability/diagnostic "
                   "issue; do NOT auto-add more seeds or budget")
    else:
        verdict = "inconclusive under the frozen criteria; do not tune for a win"
    print("  VERDICT: %s" % verdict)

    print("\n=== secondary interpretation boundaries ===")
    mi, mr = mean(G_int_AB.values()), mean(G_R_int.values())
    if mi >= THRESHOLD and sum(1 for v in G_int_AB.values() if v > 0) >= 6:
        print("  RINS-AB stably better than AB: the LP-guided fixings add value even while ALL")
        print("  of AB's restrictions are kept -> RINS's whole advantage cannot be explained")
        print("  merely by 'being allowed to change short-term Y and C'.")
    if mr >= THRESHOLD and sum(1 for v in G_R_int.values() if v > 0) >= 6:
        print("  plain RINS clearly better than RINS-AB: keeping AB's restrictions costs")
        print("  something under this algorithm and budget.  This still does NOT establish an")
        print("  optimality-value gap, nor separate the short-term-Y from the C contribution.")
    if mi <= -THRESHOLD and sum(1 for v in G_int_AB.values() if v < 0) >= 6:
        print("  RINS-AB clearly better than plain RINS: record as a stronger non-learning")
        print("  combination; do NOT name it a contribution yet and do NOT tune to enlarge it.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "cell", "disruption", "J_AB", "J_RINS", "J_RINS_AB", "J_r",
                    "G1", "G_int_AB", "G_R_int"])
        for s in states:
            j = jr(s)
            w.writerow([s.split("|")[0], CELL_LABEL[cell_of[s]], cell_of[s]] +
                       ["%.4f" % jm(s, m) for m in METHODS] + ["%.4f" % j,
                        "%.6f" % ((jm(s, "AB") - jm(s, "RINS-LP-ONE")) / j),
                        "%.6f" % ((jm(s, "AB") - jm(s, "RINS-AB")) / j),
                        "%.6f" % ((jm(s, "RINS-AB") - jm(s, "RINS-LP-ONE")) / j)])

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": payload["meta"], "seed1_G": G1, "mean_G_seed1": mean(G1.values()),
                   "positive_seed1": pos, "negative_seed1": neg, "by_rho_seed1": groups,
                   "leave_one_out_seed1": loo, "seed0_G": G0,
                   "G_int_AB": G_int_AB, "G_R_int": G_R_int,
                   "mean_G_int_AB": mi, "mean_G_R_int": mr,
                   "integrity": integ, "passes_seed1": passes, "verdict": verdict},
                  fh, indent=2, default=str)
    print("\nwrote %s and %s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
