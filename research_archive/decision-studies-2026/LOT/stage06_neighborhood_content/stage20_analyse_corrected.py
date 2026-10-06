"""Stage 20 analysis, CORRECTED (C47): candidate eligibility, bound sharing, hard checks.

The previous version (archived as `stage20_analyse_BUGGY_C47.py`) computed

    U_R, L_R = min(r["U"], f["U"]), r["L"]
    U_F, L_F = f["U"], f["L"]

which folded the FULL plan's cost into U_R WITHOUT checking F_R eligibility.  Two consequences:

  * a FULL plan violating F_R was treated as RINS-feasible;
  * U_F >= U_R held BY CONSTRUCTION, so "16/16" was a tautology, not a finding.

Correct sharing direction:
  * a RINS-feasible plan MAY update the FULL bound (RINS is a restriction of FULL);
  * a FULL plan may update the RINS bound ONLY IF it satisfies ALL of F_R.

This script rebuilds the bounds OFFLINE from the stored witnesses.  No MIP is re-solved.

For every candidate it records its source run, cost, FULL eligibility and RINS eligibility, then

    U_R = min J(x) over {original model, kappa, F_R}
    U_F = min J(x) over {original model, kappa}

and enforces L_R <= U_R, L_F <= U_F, U_F <= U_R and the ordering of the loss interval, all within
a uniform numerical tolerance.  A violation ABORTS the analysis instead of printing a verdict.
"""

import collections
import csv
import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance
from lsp_model import MODE_AUDITED, Solution, build_model
from lsp_select import evaluate_candidate, is_usable
from lsp_release import build_state_context
from stage06_run import KAPPA, solution_from_record
from stage17_run import fixings_of
from stage20_run_base import disruption_of

STATES_FILE = "stage20_states.json"
BASE_FILE = "stage20_base.json"
DIAG_FILE = "stage20_diag.json"
OUT_JSON = "stage20_analysis.json"
OUT_CSV = "stage20_certificates.csv"
CAND_CSV = "stage20_candidates.csv"
THRESHOLD = 0.02
TOL_REL = 1e-6


def _sol(inst, plan, cost) -> Solution:
    return Solution(status="cand", objective=float(cost),
                    X=np.array(plan["X"], float).reshape(inst.N, inst.M, inst.T),
                    Y=np.array(plan["Y"], float).reshape(inst.N, inst.M, inst.T),
                    Z=np.array(plan["Z"], float).reshape(inst.N, inst.M, inst.T),
                    I=np.array(plan["I"], float).reshape(inst.N, inst.T),
                    L=np.array(plan["L"], float).reshape(inst.N, inst.T),
                    mode=MODE_AUDITED)


def main() -> int:
    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(BASE_FILE, encoding="utf-8") as fh:
        BASE = {r["state"]: r for r in json.load(fh)["runs"]}
    with open(DIAG_FILE, encoding="utf-8") as fh:
        DIAG = json.load(fh)["runs"]
    druns = {(r["state"], r["diagnostic"]): r for r in DIAG}

    insts = sorted({s.split("|")[0] for s in ALL})
    print("Stage 20 (corrected analysis) | %d states | %d instances | threshold 2%% of D_n"
          % (len(ALL), len(insts)))
    print("candidates are re-validated OFFLINE from the stored witnesses; no MIP is re-solved\n")

    cand_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for state in sorted(ALL):
        rec = ALL[state]
        meta = rec["meta"]
        inst, _m = build_instance(meta["scale"], meta["rho"], meta["seed"])
        dis = disruption_of(rec)
        pert = apply_disruption(inst, dis)
        reference = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        D = float(rec["D_n"])
        tol = TOL_REL * (1.0 + abs(D))

        b = BASE[state]
        fixed_cols = {int(k): float(v) for k, v in b["fixed_cols"].items()}
        C = build_state_context(inst, pert, reference, bm, 6)   # base, disrupted, reference
        C.update({"bm": bm, "tau": 6, "meta": meta, "repair": reference, "inst": pert,
                  "dis": dis, "mr_sets": {}})
        fy, fz = fixings_of(C, fixed_cols)

        # ---- candidate pool: (label, plan, recorded cost, source run) ---------------------
        pool: List[Tuple[str, Any, Optional[float], str]] = [
            ("reference_S_r", rec["repair"]["plan"], float(rec["repair"]["objective"]),
             "stage20_states"),
            ("online_20s", b["final_solution"], float(b["final_cost"]), "stage20_run_base"),
        ]
        for dname in ("RINS-BOUND", "FULL-BOUND"):
            dr = druns.get((state, dname))
            if dr is not None and dr.get("witness") is not None:
                pool.append((dname, dr["witness"], float(dr["mip_objective"]), dname))

        best_R = (float("inf"), None)
        best_F = (float("inf"), None)
        for label, plan, recorded, src in pool:
            if plan is None:
                continue
            sol = _sol(inst, plan, recorded if recorded is not None else 0.0)
            # FULL eligibility: original model + kappa
            c_full = evaluate_candidate(label, pert, sol, reference, KAPPA, 6, None, None)
            full_ok = bool(is_usable(c_full) and c_full.cost is not None)
            cost = float(c_full.cost) if c_full.cost is not None else float("nan")
            # RINS eligibility: additionally satisfy the whole fixing set F_R
            c_rins = evaluate_candidate(label, pert, sol, reference, KAPPA, 6, fy, fz)
            rins_ok = bool(is_usable(c_rins) and c_rins.cost is not None)
            n_viol = _n_fix_violations(sol, fy, fz)

            if full_ok and cost < best_F[0]:
                best_F = (cost, label)
            if rins_ok and cost < best_R[0]:
                best_R = (cost, label)
            cand_rows.append({
                "state": state, "instance": state.split("|")[0],
                "disruption": rec["disruption"]["name"], "candidate": label,
                "source": src, "recorded_cost": recorded, "cost": cost,
                "full_eligible": full_ok, "rins_eligible": rins_ok,
                "n_F_R_violations": n_viol,
            })

        r = druns.get((state, "RINS-BOUND"))
        f = druns.get((state, "FULL-BOUND"))
        U_R, src_R = best_R[0], best_R[1]
        U_F, src_F = best_F[0], best_F[1]
        L_R = r["L"] if r else None
        L_F = f["L"] if f else None

        # ---- mandatory consistency checks -------------------------------------------------
        if L_R is not None and L_R > U_R + tol:
            raise SystemExit("ABORT %s: L_R=%.6f > U_R=%.6f (tol %.3g)"
                             % (state, L_R, U_R, tol))
        if L_F is not None and L_F > U_F + tol:
            raise SystemExit("ABORT %s: L_F=%.6f > U_F=%.6f (tol %.3g)"
                             % (state, L_F, U_F, tol))
        if U_F > U_R + tol:
            raise SystemExit("ABORT %s: U_F=%.6f > U_R=%.6f (tol %.3g) -- RINS is a "
                             "restriction of FULL, so this is impossible" % (state, U_F, U_R, tol))

        raw_lower = (L_R - U_F) if (L_R is not None and U_F is not None) else None
        # the SPECIFIED non-negative lower end of the loss interval
        lower = max(0.0, raw_lower) if raw_lower is not None else None
        upper = (U_R - L_F) if (U_R is not None and L_F is not None) else None
        if lower is not None and upper is not None and lower > upper + tol:
            raise SystemExit("ABORT %s: loss interval inverted (%.6f > %.6f)"
                             % (state, lower, upper))

        # per-run own results, kept SEPARATE from the shared-pool bounds
        own_R = r["U"] if r else None
        own_F = f["U"] if f else None
        frac_raw = (raw_lower / D) if raw_lower is not None else None
        frac = (lower / D) if lower is not None else None

        if lower is not None and lower >= THRESHOLD * D - tol:
            verdict = "CERTIFIED >=2%"
        elif upper is not None and upper < THRESHOLD * D:
            verdict = "NO LOSS >=2% (upper < 2%)"
        else:
            verdict = "UNDETERMINED"

        rows.append({
            "state": state, "instance": state.split("|")[0],
            "disruption": rec["disruption"]["name"], "D_n": D,
            "U_R": U_R, "U_R_source": src_R, "L_R": L_R,
            "U_F": U_F, "U_F_source": src_F, "L_F": L_F,
            "own_R_run": own_R, "own_F_run": own_F, "J_online": b["final_cost"],
            "loss_lower": lower, "loss_lower_raw": raw_lower, "loss_upper": upper,
            "loss_lower_frac": frac, "loss_lower_raw_frac": frac_raw,
            "verdict": verdict,
            # eligibility of the two diagnostic witnesses, as the user requested
            "RINS_witness_rins_eligible":
                _lookup(cand_rows, state, "RINS-BOUND", "rins_eligible"),
            "FULL_witness_rins_eligible":
                _lookup(cand_rows, state, "FULL-BOUND", "rins_eligible"),
            "FULL_witness_F_R_violations":
                _lookup(cand_rows, state, "FULL-BOUND", "n_F_R_violations"),
            "FULL_better_than_own_RINS":
                (own_F is not None and own_R is not None and own_F < own_R - tol),
        })

    # ---------------- per-state table ----------------
    print("%-19s %-10s %8s | %-9s %10s %11s | %-9s %10s %11s | %9s %9s | %s"
          % ("instance", "disrupt.", "D_n", "U_R(src)", "L_R", "own_R_run",
             "U_F(src)", "L_F", "own_F_run", "L_R-U_F", "max(0,.)", "verdict"))
    for x in rows:
        print("%-19s %-10s %8.1f | %-9s %10.2f %11.2f | %-9s %10.2f %11.2f | %+9.2f %9.2f | %s"
              % (x["instance"], x["disruption"], x["D_n"],
                 "%s" % x["U_R_source"], x["U_R"], x["L_R"],
                 "%s" % x["U_F_source"], x["U_F"], x["L_F"],
                 x["loss_lower_raw"], x["loss_lower"], x["verdict"]))

    # ---------------- summary ----------------
    cert = [x for x in rows if x["verdict"] == "CERTIFIED >=2%"]
    noloss = [x for x in rows if x["verdict"].startswith("NO LOSS")]
    undet = [x for x in rows if x["verdict"] == "UNDETERMINED"]
    pos_raw = [x for x in rows if x["loss_lower_raw"] is not None and x["loss_lower_raw"] > 1e-9]
    cert_inst = sorted({x["instance"] for x in cert})
    cert_rho = sorted({("rho0.75" if "rho0.75" in i else "rho1.10") for i in cert_inst})
    both_noloss = [n for n in insts
                   if all(any(x["instance"] == n and x["disruption"] == d
                              and x["verdict"].startswith("NO LOSS") for x in rows)
                          for d in ("D1_m0_2p", "D2_all_1p"))]

    print("\n=== summary (corrected) ===")
    print("  CERTIFIED >= 2%%                  : %d/%d states, %d/8 instances %s"
          % (len(cert), len(rows), len(cert_inst), cert_inst))
    print("  NO LOSS (upper < 2%%)             : %d/%d" % (len(noloss), len(rows)))
    print("  UNDETERMINED                     : %d/%d" % (len(undet), len(rows)))
    print("  states with a POSITIVE raw margin (L_R - U_F > 0): %d/%d" % (len(pos_raw), len(rows)))
    for x in pos_raw:
        print("     %-26s L_R-U_F = %+8.4f = %+.4f%% of D_n  -> %s"
              % (x["state"], x["loss_lower_raw"], 100 * x["loss_lower_raw_frac"],
                 x["verdict"]))
    print("  loss upper bound: median %.2f%% of D_n, min %.2f%%, max %.2f%%"
          % (100 * st.median([x["loss_upper"] / x["D_n"] for x in rows]),
             100 * min(x["loss_upper"] / x["D_n"] for x in rows),
             100 * max(x["loss_upper"] / x["D_n"] for x in rows)))
    print("  loss lower bound (specified max(0,.) ): median %.4f%% of D_n"
          % (100 * st.median([x["loss_lower"] / x["D_n"] for x in rows])))
    print("  instances whose BOTH faults have a NO-LOSS upper bound: %d/8 %s"
          % (len(both_noloss), both_noloss))

    # ---------------- candidate eligibility ----------------
    print("\n=== candidate eligibility (offline re-validation) ===")
    byk = collections.Counter((c["candidate"], c["full_eligible"], c["rins_eligible"])
                              for c in cand_rows)
    for k in sorted(byk, key=str):
        print("  %-14s FULL=%s RINS=%s : %d" % (k[0], k[1], k[2], byk[k]))
    fr_ok = sum(1 for c in cand_rows
                if c["candidate"] == "FULL-BOUND" and c["rins_eligible"])
    print("  FULL diagnostic witnesses satisfying all of F_R : %d/16" % fr_ok)
    print("  RINS diagnostic witnesses violating F_R         : %d/16"
          % sum(1 for c in cand_rows
                if c["candidate"] == "RINS-BOUND" and not c["rins_eligible"]))
    better = sum(1 for x in rows if x["FULL_better_than_own_RINS"])
    print("  FULL's own valid plan better than RINS's own plan: %d/16" % better)
    print("  RINS's own plan better / equal                    : %d/16 / %d/16"
          % (sum(1 for x in rows if x["own_R_run"] is not None and x["own_F_run"] is not None
                 and x["own_R_run"] < x["own_F_run"] - 1e-6),
             sum(1 for x in rows if x["own_R_run"] is not None and x["own_F_run"] is not None
                 and abs(x["own_R_run"] - x["own_F_run"]) <= 1e-6)))
    own_improve = sum(1 for x in rows
                      if x["own_R_run"] is not None
                      and x["own_R_run"] < x["J_online"] - 1e-6)
    print("  RINS diagnostics improving on their own online plan: %d/16" % own_improve)
    print("  shared-pool U_R better than the online plan        : %d/16"
          % sum(1 for x in rows if x["U_R"] < x["J_online"] - 1e-6))
    print("  U_F uses a plan from: %s"
          % dict(collections.Counter(x["U_F_source"] for x in rows)))
    print("  U_R uses a plan from: %s"
          % dict(collections.Counter(x["U_R_source"] for x in rows)))

    # ---------------- frozen criteria ----------------
    print("\n=== frozen judgement (corrected) ===")
    verdict_txt = []
    if len(cert_inst) >= 4 and len(cert_rho) == 2:
        verdict_txt.append("CONDITION 1 MET: >=4/8 instances have a >=2%% loss certificate with "
                           "both rho groups covered -> supports studying selective withdrawal "
                           "of RINS fixings (engineering threshold, not evidence of novelty).")
    if len(both_noloss) >= 6:
        verdict_txt.append("CONDITION 2 MET: >=6/8 instances have BOTH faults with a loss upper "
                           "bound below 2%%.")
    if not verdict_txt:
        verdict_txt.append("NEITHER condition met -> UNRESOLVED: keep the per-case results and do "
                           "NOT automatically add time, swap instances or pick favourable cases.")
    verdict_txt.append(
        "EVIDENCE IS NOT VACUOUS: %d/%d states have a POSITIVE raw margin L_R - U_F (i.e. a "
        "genuine optimality-loss certificate, %s), and FULL's own valid plan beats RINS's own "
        "plan in %d/16 states.  The correct statement is NOT 'FULL never won and nothing was "
        "seen'." % (len(pos_raw), len(rows),
                    "; ".join("%s = %+.4f%% of D_n" % (x["state"], 100 * x["loss_lower_raw_frac"])
                              for x in pos_raw), better))
    for v in verdict_txt:
        print("  => %s" % v)
    print("\n  Note: lowering U_F or raising L_R is what proves a fixing-induced loss; raising "
          "L_F mainly proves the loss is SMALL.")

    # ---------------- write ----------------
    with open(CAND_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        for c in cand_rows:
            w.writerow(c)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "instance", "disruption", "D_n", "U_R", "U_R_source", "L_R",
                    "U_F", "U_F_source", "L_F", "loss_lower", "loss_lower_raw", "loss_upper",
                    "verdict"])
        for x in rows:
            w.writerow([x["state"], x["instance"], x["disruption"], "%.4f" % x["D_n"],
                        "%.4f" % x["U_R"], x["U_R_source"], "%.4f" % x["L_R"],
                        "%.4f" % x["U_F"], x["U_F_source"], "%.4f" % x["L_F"],
                        "%.4f" % x["loss_lower"],
                        "" if x["loss_lower_raw"] is None else "%.4f" % x["loss_lower_raw"],
                        "%.4f" % x["loss_upper"], x["verdict"]])
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "candidates": cand_rows,
                   "summary": {"n_certified": len(cert), "certified_instances": cert_inst,
                               "certified_rho": cert_rho, "n_noloss": len(noloss),
                               "n_undetermined": len(undet),
                               "n_positive_raw_margin": len(pos_raw),
                               "both_faults_noloss": both_noloss,
                               "full_better_than_rins": better,
                               "full_witnesses_satisfying_F_R": fr_ok,
                               "rins_witnesses_violating_F_R":
                                   sum(1 for c in cand_rows
                                       if c["candidate"] == "RINS-BOUND"
                                       and not c["rins_eligible"])},
                   "checks": "L_R<=U_R, L_F<=U_F, U_F<=U_R and loss-interval ordering all "
                             "enforced; a violation aborts instead of printing a verdict",
                   "verdict": verdict_txt}, fh, indent=2, default=str)
    print("\nwrote %s, %s and %s" % (OUT_JSON, OUT_CSV, CAND_CSV))
    return 0


def _n_fix_violations(sol, fy, fz) -> int:
    n = 0
    for (i, j, t), want in fy.items():
        if abs(float(sol.Y[i, j, t - 1]) - want) > 0.5:
            n += 1
    for (i, j, t), want in fz.items():
        if abs(float(sol.Z[i, j, t - 1]) - want) > 0.5:
            n += 1
    return n


def _lookup(cand_rows, state, label, field):
    for c in cand_rows:
        if c["state"] == state and c["candidate"] == label:
            return c[field]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
