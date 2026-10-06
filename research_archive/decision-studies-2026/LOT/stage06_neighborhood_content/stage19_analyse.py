"""Stage 19 analysis: start-quality sensitivity check on the frozen S_N+ plans.

Two state groups are reported SEPARATELY with the same fixed metric scale D_n = max(1,|J_{N,n}|),
where J_{N,n} is the OLD nominal cost:

    G_cont = (J_ref - J_LP)  / D_n
    G_bin  = (J_LP  - J_out) / D_n

PRIMARY metric: the fault arm's G_bin -- averaged equally over the four disruptions within each
nominal instance, then equally over the 8 instances.

The no-fault arm reports the same metrics on its own.

The engineering screening line is the same for both arms (NOT a significance test):
    mean >= 2%, at least 6/8 instances strictly improved, both rho groups positive.

The two arms must NOT be subtracted from each other: their feasible regions and reference
solutions differ even though the denominator is shared.
"""

import collections
import csv
import json
import statistics as st
from typing import Any, Dict, List

RUNS_FILE = "stage19_runs.json"
OUT_JSON = "stage19_analysis.json"
OUT_CSV = "stage19_per_instance.csv"
THRESHOLD = 0.02
# uniform cost tolerance: a gain this small is a TIE, not an improvement.  Without it,
# floating-point noise in the LP objective (e.g. 6.06e-16) is miscounted as "strictly
# improved" and the direction tally overstates the evidence.
TIE_TOL = 1e-9
CELLS = ["single/1p", "single/2p", "all/1p", "all/2p"]


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def screen(vals: Dict[str, float], label: str) -> Dict[str, Any]:
    pos = sum(1 for v in vals.values() if v > TIE_TOL)
    neg = sum(1 for v in vals.values() if v < -TIE_TOL)
    tie = sum(1 for v in vals.values() if abs(v) <= TIE_TOL)
    groups = {}
    for g in ("0.75", "1.10"):
        sub = [v for n, v in vals.items() if ("rho%s" % g) in n]
        groups[g] = {"mean": mean(sub), "positive": sum(1 for v in sub if v > TIE_TOL),
                     "n": len(sub)}
    loo = {n: mean([v for k, v in vals.items() if k != n]) for n in vals}
    ok = (mean(vals.values()) >= THRESHOLD and pos >= 6
          and groups["0.75"]["mean"] > 0 and groups["1.10"]["mean"] > 0)
    print("  [%s] n=%d mean=%+.5f median=%+.5f positive=%d negative=%d tie=%d"
          % (label, len(vals), mean(vals.values()), st.median(vals.values()), pos, neg, tie))
    print("       rho0.75 %+.5f (%d/%d) | rho1.10 %+.5f (%d/%d)"
          % (groups["0.75"]["mean"], groups["0.75"]["positive"], groups["0.75"]["n"],
             groups["1.10"]["mean"], groups["1.10"]["positive"], groups["1.10"]["n"]))
    print("       leave-one-out: %+.5f .. %+.5f (all positive: %s)"
          % (min(loo.values()), max(loo.values()), all(v > 0 for v in loo.values())))
    print("       screening line (mean>=2%%, >=6/8 strict, both rho>0): %s" % ("PASS" if ok else "FAIL"))
    return {"mean": mean(vals.values()), "positive": pos, "negative": neg, "tie": tie,
            "by_rho": groups, "leave_one_out": loo, "passes": ok, "per_instance": vals}


def main() -> int:
    with open(RUNS_FILE, encoding="utf-8") as fh:
        payload = json.load(fh)
    runs = payload["runs"]
    print("Stage 19 | %d runs | seed %d | metric scale D_n = max(1,|J_N|) (OLD nominal cost)"
          % (len(runs), payload["meta"]["seed"]))
    print("scale: %s" % payload["meta"]["metric_scale"])
    print("warning: %s" % payload["meta"]["warning"])

    insts = sorted({r["instance"] for r in runs})
    fault = [r for r in runs if r["arm"] == "fault"]
    nofault = [r for r in runs if r["arm"] == "no_fault"]
    print("\nstates: %d fault + %d no-fault | instances %d"
          % (len(fault), len(nofault), len(insts)))

    # ---------------- PRIMARY: fault arm G_bin ----------------
    print("\n=== PRIMARY (fault arm)  G_bin = (J_LP - J_out)/D_n ===")
    fbin = {}
    for n in insts:
        v = [r["G_bin"] for r in fault if r["instance"] == n]
        fbin[n] = mean(v)
        print("  %-19s %+9.5f   (4 disruptions: %s)"
              % (n, fbin[n], " ".join("%+.4f" % x for x in
                                      [r["G_bin"] for r in sorted(
                                          [r for r in fault if r["instance"] == n],
                                          key=lambda r: r["disruption"])])))
    res_fault = screen(fbin, "fault arm G_bin")

    print("\n=== fault arm  G_cont = (J_ref - J_LP)/D_n ===")
    fcont = {n: mean([r["G_cont"] for r in fault if r["instance"] == n]) for n in insts}
    for n in insts:
        print("  %-19s %+9.5f" % (n, fcont[n]))
    res_fault_cont = screen(fcont, "fault arm G_cont")

    print("\n=== fault arm by disruption cell (mean G_bin) ===")
    print("  %-10s %10s %10s" % ("cell", "mean G_bin", "positive"))
    cell_res = {}
    for cl in CELLS:
        bre, dur = cl.split("/")
        v = [r["G_bin"] for r in fault
             if r["cell"]["breadth"] == bre and r["cell"]["duration"] == dur]
        cell_res[cl] = {"mean": mean(v), "positive": sum(1 for x in v if x > 0), "n": len(v)}
        print("  %-10s %+10.5f %10s" % (cl, cell_res[cl]["mean"],
                                        "%d/%d" % (cell_res[cl]["positive"], len(v))))

    # ---------------- no-fault arm ----------------
    print("\n=== NO-FAULT arm (same fixed scale D_n) ===")
    print("  %-19s %10s %10s %10s" % ("instance", "G_bin", "G_cont", "G_total"))
    nbin, ncont = {}, {}
    for n in insts:
        rs = [r for r in nofault if r["instance"] == n]
        nbin[n] = mean([r["G_bin"] for r in rs])
        ncont[n] = mean([r["G_cont"] for r in rs])
        print("  %-19s %+10.5f %+10.5f %+10.5f"
              % (n, nbin[n], ncont[n], mean([r["G_total"] for r in rs])))
    res_nofault = screen(nbin, "no-fault arm G_bin")

    print("\n=== descriptive: repair cost, old nominal base vs S_N+ base ===")
    print("  %-19s %-11s %11s %11s %10s" % ("instance", "disruption", "J_r(old S_N)",
                                            "J_r(new S_N+)", "change"))
    pairs = []
    for n in insts:
        for r in sorted([x for x in fault if x["instance"] == n],
                        key=lambda x: x["disruption"]):
            pass
    with open("stage19_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    better = worse = same = 0
    for k in sorted(ST):
        rec = ST[k]
        if not rec["disruption"]["down"]:
            continue
        a = rec["repair"]["old_repair_objective"]
        b = rec["repair"]["objective"]
        if b < a - 1e-9:
            better += 1
        elif b > a + 1e-9:
            worse += 1
        else:
            same += 1
        pairs.append(b - a)
    for k in sorted(ST):
        rec = ST[k]
        if not rec["disruption"]["down"]:
            continue
        print("  %-19s %-11s %11.2f %11.2f %+10.2f"
              % (k.split("|")[0], rec["disruption"]["name"],
                 rec["repair"]["old_repair_objective"], rec["repair"]["objective"],
                 rec["repair"]["objective"] - rec["repair"]["old_repair_objective"]))
    print("  => new base cheaper in %d/32, worse in %d/32, equal in %d/32 (mean change %+.2f)"
          % (better, worse, same, mean(pairs)))
    print("  (A lower nominal cost does NOT guarantee an easier repair.)")

    # ---------------- diagnostics ----------------
    print("\n=== diagnostics ===")
    print("  fallbacks: %d/%d" % (sum(1 for r in runs if r["fallback"]), len(runs)))
    print("  lp_status: %s | relax_status: %s"
          % (dict(collections.Counter(r["lp_status"] for r in runs)),
             dict(collections.Counter(r["relax_status"] for r in runs))))
    print("  start adopted: %d/%d | proven optimal: %d/%d"
          % (sum(1 for r in runs if r["mip_start_adopted"]), len(runs),
             sum(1 for r in runs if r["mip_proven_optimal"]), len(runs)))
    print("  final source: %s"
          % dict(collections.Counter(r["final_source"] for r in runs)))
    for arm in ("fault", "no_fault"):
        v = [r for r in runs if r["arm"] == arm]
        print("  %-9s fixed %d..%d free %d..%d | short_y_changes %s"
              % (arm, min(r["n_fixed_total"] for r in v), max(r["n_fixed_total"] for r in v),
                 min(r["n_free_binary_after"] for r in v),
                 max(r["n_free_binary_after"] for r in v),
                 sorted({r["short_y_changes"] for r in v})))
    integ = {
        "n_runs": len(runs),
        "select_errors": sum(len(r["select_errors"]) if isinstance(r["select_errors"], list)
                             else r["select_errors"] for r in runs),
        "final_source_none": sum(1 for r in runs if r["final_source"] == "none"),
        "worse_than_reference": sum(1 for r in runs if r["final_cost"] is not None
                                    and r["final_cost"] > r["J_ref"]
                                    + 1e-6 * (1 + abs(r["J_ref"]))),
        "max_total_s": max(r["timing"]["total_s"] for r in runs),
        "max_overrun_s": max(r["timing"]["overrun_s"] for r in runs),
    }
    print("\n=== run integrity ===")
    for k, v in integ.items():
        print("  %-22s %s" % (k, v))

    # ---------------- frozen judgement ----------------
    print("\n=== frozen judgement (engineering screening line, both arms) ===")
    verdict = []
    if res_fault["passes"] and not res_nofault["passes"]:
        verdict.append("FAULT arm passes and the no-fault arm improves less -> supports "
                       "continuing to study post-disruption decisions from the improved "
                       "start.  BUT this only shows RINS can obtain gains; it does NOT show "
                       "that RINS has a learnable shortcoming.")
    if res_nofault["passes"]:
        verdict.append("NO-FAULT arm improves clearly AGAIN -> the once-polished nominal plan "
                       "still has cheap room; do NOT call it an adequately optimised start.  "
                       "Next step: define a nominal generation protocol with a fixed budget.  "
                       "Do NOT auto-iterate until the gain disappears.")
    if not res_fault["passes"]:
        verdict.append("FAULT arm does NOT reach the screening line -> on the improved start "
                       "this off-the-shelf method shows no stable, practically useful extra "
                       "binary improvement.  Pause the post-disruption learning module.  This "
                       "does NOT imply no other optimisation room exists.")
    if res_fault["passes"] and res_nofault["passes"]:
        verdict.append("BOTH arms pass: report the heterogeneity, do not pick a favourable "
                       "subset after the fact.")
    if not verdict:
        verdict.append("result concentrated in a few instances or cells: record the "
                       "heterogeneity; do not re-declare a pass on a favourable subset")
    for v in verdict:
        print("  => %s" % v)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "instance", "arm", "disruption", "cell", "rho", "D_n", "J_ref",
                    "J_LP", "J_out", "G_cont", "G_bin", "G_total", "short_y_changes"])
        for r in sorted(runs, key=lambda x: (x["instance"], x["arm"], x["disruption"])):
            w.writerow([r["state"], r["instance"], r["arm"], r["disruption"],
                        "%s/%s" % (r["cell"]["breadth"], r["cell"]["duration"]), r["rho"],
                        "%.4f" % r["D_n"], "%.4f" % r["J_ref"], "%.4f" % r["J_LP"],
                        "-" if r["final_cost"] is None else "%.4f" % r["final_cost"],
                        "%.6f" % r["G_cont"], "%.6f" % r["G_bin"], "%.6f" % r["G_total"],
                        r["short_y_changes"]])

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"meta": payload["meta"],
                   "fault_G_bin": res_fault, "fault_G_cont": res_fault_cont,
                   "fault_by_cell": cell_res,
                   "nofault_G_bin": res_nofault,
                   "nofault_G_cont": {n: ncont[n] for n in insts},
                   "repair_cost_change": {"better": better, "worse": worse, "equal": same,
                                          "mean_change": mean(pairs)},
                   "integrity": integ, "verdict": verdict}, fh, indent=2, default=str)
    print("\nwrote %s and %s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
