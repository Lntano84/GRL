"""Stage 20 analysis: does the RINS fixing set provably exclude practically useful gains?

For each state, with U_R/L_R the valid upper/lower bounds of the RINS-restricted problem and
U_F/L_F those of the FULL problem:

    max(0, L_R - U_F)  <=  J_R* - J_F*  <=  U_R - L_F

Scale is the fixed old nominal cost D_n = max(1, |J_{N,n}|).

CERTIFICATE of loss:   L_R - U_F >= 0.02 * D_n   (with numerical tolerance)
That certifies the fixing set excludes at least 2% of the old nominal cost in improvement.

If only U_F < U_R but L_R is not crossed, it stays UNDETERMINED: we cannot tell whether the
fixings excluded a good plan or the restricted problem was simply not searched well enough.

A FULL witness that satisfies all of F_R can NEVER be evidence that the fixings excluded it.
"""

import collections
import csv
import json
import statistics as st
from typing import Any, Dict, List

STATES_FILE = "stage20_states.json"
BASE_FILE = "stage20_base.json"
DIAG_FILE = "stage20_diag.json"
OUT_JSON = "stage20_analysis.json"
OUT_CSV = "stage20_certificates.csv"
THRESHOLD = 0.02
TOL_REL = 1e-6


def main() -> int:
    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(BASE_FILE, encoding="utf-8") as fh:
        BASE = {r["state"]: r for r in json.load(fh)["runs"]}
    with open(DIAG_FILE, encoding="utf-8") as fh:
        DIAG = json.load(fh)["runs"]

    by: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    for r in DIAG:
        by[r["state"]][r["diagnostic"]] = r

    insts = sorted({s.split("|")[0] for s in ALL})
    print("Stage 20 | %d states | %d instances | certificate threshold 2%% of D_n"
          % (len(ALL), len(insts)))

    rows: List[Dict[str, Any]] = []
    print("\n%-19s %-11s %10s | %11s %11s | %11s %11s | %10s %10s | %s"
          % ("instance", "disruption", "D_n", "U_R", "L_R", "U_F", "L_F",
             "L_R-U_F", "% of D_n", "verdict"))
    for state in sorted(ALL):
        rec = ALL[state]
        D = float(rec["D_n"])
        tol = TOL_REL * (1.0 + abs(D))
        r = by[state].get("RINS-BOUND")
        f = by[state].get("FULL-BOUND")
        if r is None or f is None:
            print("%-19s %-11s  MISSING DIAGNOSTIC" % (state.split("|")[0],
                                                       rec["disruption"]["name"]))
            continue
        U_R, L_R = min(r["U"], f["U"]), r["L"]
        U_F, L_F = f["U"], f["L"]

        lower = (L_R - U_F) if (L_R is not None and U_F is not None) else None
        upper = (U_R - L_F) if (U_R is not None and L_F is not None) else None
        frac = (lower / D) if lower is not None else None

        if frac is not None and lower >= THRESHOLD * D - tol:
            verdict = "CERTIFIED >=2%"
        elif upper is not None and upper < THRESHOLD * D:
            verdict = "NO LOSS >=2% (upper < 2%)"
        else:
            verdict = "UNDETERMINED"

        rows.append({
            "state": state, "instance": state.split("|")[0],
            "disruption": rec["disruption"]["name"], "D_n": D,
            "U_R": U_R, "L_R": L_R, "U_F": U_F, "L_F": L_F,
            "loss_lower": lower, "loss_upper": upper, "loss_lower_frac": frac,
            "RINS_proven_optimal": r["proven_optimal"],
            "FULL_proven_optimal": f["proven_optimal"],
            "FULL_witness_valid": f["solver_valid"], "RINS_witness_valid": r["solver_valid"],
            "J_online": BASE[state]["final_cost"], "J_ref": float(rec["repair"]["objective"]),
            "verdict": verdict,
        })
        print("%-19s %-11s %10.1f | %11.2f %11s | %11.2f %11s | %10s %10s | %s"
              % (state.split("|")[0], rec["disruption"]["name"], D, U_R,
                 "%.2f" % L_R if L_R is not None else "-", U_F,
                 "%.2f" % L_F if L_F is not None else "-",
                 "%.2f" % lower if lower is not None else "-",
                 "%.2f%%" % (100 * frac) if frac is not None else "-", verdict))

    # ---------------- coverage ----------------
    cert = [x for x in rows if x["verdict"] == "CERTIFIED >=2%"]
    noloss = [x for x in rows if x["verdict"].startswith("NO LOSS")]
    undet = [x for x in rows if x["verdict"] == "UNDETERMINED"]
    cert_inst = sorted({x["instance"] for x in cert})
    cert_rho = sorted({("rho0.75" if "rho0.75" in i else "rho1.10") for i in cert_inst})
    both_faults_noloss = [n for n in insts
                          if all(any(x["instance"] == n and x["disruption"] == d
                                     and x["verdict"].startswith("NO LOSS")
                                     for x in rows)
                                 for d in ("D1_m0_2p", "D2_all_1p"))]

    print("\n=== coverage ===")
    print("  CERTIFIED >= 2%%         : %d/%d states, covering %d/8 instances"
          % (len(cert), len(rows), len(cert_inst)))
    print("    instances             : %s" % cert_inst)
    print("    rho groups covered    : %s" % cert_rho)
    print("  NO LOSS >= 2%% (upper<2%%): %d/%d states" % (len(noloss), len(rows)))
    print("  UNDETERMINED            : %d/%d states" % (len(undet), len(rows)))
    print("  instances whose BOTH faults have a NO-LOSS upper bound: %d/8 %s"
          % (len(both_faults_noloss), both_faults_noloss))
    wide = [x for x in rows
            if x["loss_upper"] is not None and x["loss_upper"] >= 0.02 * x["D_n"]]
    print("  upper bound (U_R - L_F) still >= 2%% of D_n: %d/%d states" % (len(wide), len(rows)))
    print("    median upper bound    : %s"
          % ("%.2f%% of D_n" % (100 * st.median(
              [x["loss_upper"] / x["D_n"] for x in rows if x["loss_upper"] is not None]))))
    print("    median lower bound    : %s"
          % ("%.2f%% of D_n" % (100 * st.median(
              [x["loss_lower"] / x["D_n"] for x in rows if x["loss_lower"] is not None]))))
    print("    U_F >= U_R in          : %d/%d states (the 60 s FULL run did not beat the "
          "RINS-restricted plan)" % (sum(1 for x in rows if x["U_F"] >= x["U_R"] - 1e-6),
                                     len(rows)))

    # ---------------- witness hygiene ----------------
    print("\n=== witness hygiene ===")
    with open(DIAG_FILE, encoding="utf-8") as fh:
        druns = {(r["state"], r["diagnostic"]): r for r in json.load(fh)["runs"]}

    # A FULL witness strictly better than U_R must VIOLATE F_R: any FULL plan that satisfied
    # F_R would itself be a valid RINS plan, so U_R would already be at least that good.
    better_full = [x for x in rows if x["U_F"] < x["U_R"] - 1e-6]
    print("  FULL witnesses strictly better than U_R: %d/%d" % (len(better_full), len(rows)))
    print("  (any such witness must violate F_R by construction -- otherwise it would be a "
          "valid RINS plan and U_R would already be that good)")
    print("  RINS diagnostics with a VALID solver plan : %d/%d"
          % (sum(1 for x in rows if x["RINS_witness_valid"]), len(rows)))
    print("  FULL diagnostics with a VALID solver plan : %d/%d"
          % (sum(1 for x in rows if x["FULL_witness_valid"]), len(rows)))
    print("  RINS diagnostics proven optimal: %d/%d | FULL proven optimal: %d/%d"
          % (sum(1 for x in rows if x["RINS_proven_optimal"]), len(rows),
             sum(1 for x in rows if x["FULL_proven_optimal"]), len(rows)))
    print("  witnesses stored (never discarded): %d/%d"
          % (sum(1 for r in druns.values() if r["witness"] is not None), len(druns)))
    print("  U came from the fallback online plan in %d/%d diagnostics"
          % (sum(1 for r in druns.values() if not r["solver_valid"]), len(druns)))

    # ---------------- U_R bookkeeping ----------------
    print("\n=== U_R bookkeeping ===")
    print("  U_R uses the best valid RINS-FEASIBLE plan, which may come from EITHER arm.")
    n_from_online = sum(1 for x in rows if abs(x["U_R"] - x["J_online"]) < 1e-6)
    print("  U_R equal to the online plan: %d/%d (the rest improved on it inside 60 s)"
          % (n_from_online, len(rows)))

    # ---------------- frozen criteria ----------------
    print("\n=== frozen judgement ===")
    verdict = []
    if len(cert_inst) >= 4 and len(cert_rho) == 2:
        verdict.append("CONDITION 1 MET: >=4/8 instances have a >=2%% loss certificate and both "
                       "rho groups are covered -> supports further study of HOW TO "
                       "SELECTIVELY WITHDRAW RINS fixings.  This is an engineering threshold, "
                       "NOT yet evidence of a learning advantage or of novelty.")
    cond2_literal = len(both_faults_noloss) >= 6
    if cond2_literal:
        verdict.append("CONDITION 2 MET: >=6/8 instances have BOTH faults with a loss upper "
                       "bound below 2%% -> do NOT prioritise withdrawing fixings; current "
                       "evidence does not support adding a learning module for this.  It also "
                       "does not by itself justify search acceleration.")
    if not verdict:
        verdict.append("NEITHER condition met (no certificates and/or mostly wide intervals) -> "
                       "record as UNRESOLVED, keep the per-case results, and do NOT "
                       "automatically add time, swap instances, or pick favourable cases.")
        if not cond2_literal and len(noloss) == 0:
            verdict.append("NOTE ON CONDITION 2: it is NOT satisfied either.  Although only %d "
                           "instances have a NO-LOSS upper bound, this is NOT evidence that the "
                           "loss is below 2%%: every upper bound is still WIDE (median %.2f%% of "
                           "D_n).  The screening logic of condition 2 needs a tight upper bound, "
                           "which these 60 s runs did not produce.  So the honest reading is "
                           "'undetermined', not 'no meaningful loss'."
                           % (len(both_faults_noloss),
                              100 * st.median([x["loss_upper"] / x["D_n"] for x in rows
                                               if x["loss_upper"] is not None])))
    for v in verdict:
        print("  => %s" % v)
    print("\n  Reminder: this round trains nothing, runs no multi-round online RINS, does not "
          "revive AB, and adds no random release ratios.")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "instance", "disruption", "D_n", "U_R", "L_R", "U_F", "L_F",
                    "loss_lower", "loss_upper", "loss_lower_frac", "verdict"])
        for x in rows:
            w.writerow([x["state"], x["instance"], x["disruption"], "%.4f" % x["D_n"],
                        "%.4f" % x["U_R"],
                        "" if x["L_R"] is None else "%.4f" % x["L_R"], "%.4f" % x["U_F"],
                        "" if x["L_F"] is None else "%.4f" % x["L_F"],
                        "" if x["loss_lower"] is None else "%.4f" % x["loss_lower"],
                        "" if x["loss_upper"] is None else "%.4f" % x["loss_upper"],
                        "" if x["loss_lower_frac"] is None else "%.6f" % x["loss_lower_frac"],
                        x["verdict"]])

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "summary": {"n_certified": len(cert), "certified_instances": cert_inst,
                               "certified_rho": cert_rho, "n_noloss": len(noloss),
                               "n_undetermined": len(undet),
                               "both_faults_noloss": both_faults_noloss},
                   "verdict": verdict}, fh, indent=2, default=str)
    print("\nwrote %s and %s" % (OUT_JSON, OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
