#!/usr/bin/env python
"""
Stage 13: how many short-term-Z fixings must be lifted to reach a blocked quality level?

    .venv-hs/Scripts/python.exe stage13_run.py [--smoke]
    .venv-hs/Scripts/python.exe stage13_analyse.py

The question
------------
> To reach a quality level that the original matched-random configuration provably excludes, how
> few originally-FIXED short-term Z (F_B) must change, and in which periods and directions?

That "3-9 positions suffice" is already witnessed directly by the AB plan, so this round does not
re-prove sufficiency.  It measures the MINIMUM.

Frozen scope
------------
* the 11 certificate cases (U_AB,witness < L_MR), taken from stage 11 records
* one solver seed, 0
* at most 60 s per case, at most 11 minutes of solve time in total
* the AB witness is the start; no new instances, no redrawn random sets, no automatic extension

Quality gate
------------
    Q = (U_AB,witness + L_MR) / 2      so that   U_AB,witness < Q < L_MR
The AB witness can reach Q; the original MR provably cannot.

Kept strictly apart
-------------------
  original cost J  -> the constraint J(x) <= Q and the verification of the returned plan
  diagnostic H     -> the solver objective, i.e. how many original F_B fixings were changed
The solver's reported objective is H and is NEVER passed to the checker as a production cost.

Everything is re-checked per returned plan: the original model, the short-term Y, F_C, the
quality gate, and a hand-recomputed H.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import highspy

from lsp_checker import check_solution
from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_minlift import build_min_lift_mip
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_release import build_state_context, stage11_columns
from lsp_repair3 import binary_change_count
from stage06_run import KAPPA, solution_from_record

BUDGET = 60.0
SEED = 0
MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")
STATES_FILE = "stage06_states.json"
MATCHED_FILE = "stage11_matched_random.json"
S11_RUNS = "stage11_runs.json"
S11B = "stage11b_relax.json"
OUT_FILE = "stage13_runs.json"


def _carry_replacement(changed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect the pattern 'one carry-over object replaced by another'.

    A carry-over Z_ijt is a single object.  Moving it from product a to product b on the same
    machine and period is encoded as TWO binary flips (a: 1->0, b: 0->1) but is ONE replacement,
    so counting flips overstates how many distinct structural changes are needed.  This helper
    reports that pairing; it does NOT claim the two flips are indispensable jointly.
    """
    by_cell: Dict[Any, List[Dict[str, Any]]] = {}
    for c in changed:
        by_cell.setdefault((c["j"], c["t"]), []).append(c)
    pairs = []
    leftover = []
    for (j, t), cs in by_cell.items():
        outs = [c for c in cs if c["from"] == 1 and c["to"] == 0]
        ins = [c for c in cs if c["from"] == 0 and c["to"] == 1]
        while outs and ins:
            a, b = outs.pop(), ins.pop()
            pairs.append({"j": j, "t": t, "out_product": a["i"], "in_product": b["i"]})
        leftover.extend(outs)
        leftover.extend(ins)
    return {"n_flips": len(changed), "n_replacements": len(pairs),
            "replacements": pairs, "unpaired_flips": len(leftover)}


def vector_of(sol: Solution, bm) -> np.ndarray:
    """Full variable vector of a Solution, in the model's own column order."""
    x = np.zeros(bm.idx.n)
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, bm.inst.T + 1):
                x[bm.idx.X(i, j, t)] = float(sol.X[i, j, t - 1])
                x[bm.idx.Y(i, j, t)] = float(sol.Y[i, j, t - 1])
                x[bm.idx.Z(i, j, t)] = float(sol.Z[i, j, t - 1])
            x[bm.idx.Z(i, j, 0)] = 0.0
        for t in range(1, bm.inst.T + 1):
            x[bm.idx.I(i, t)] = float(sol.I[i, t - 1])
            x[bm.idx.L(i, t)] = float(sol.L[i, t - 1])
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    with open(STATES_FILE, encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open(MATCHED_FILE, encoding="utf-8") as fh:
        MATCH = json.load(fh)
    with open(S11_RUNS, encoding="utf-8") as fh:
        S11 = json.load(fh)["runs"]
    with open(S11B, encoding="utf-8") as fh:
        RELAX = json.load(fh)
    cell = {(r["state"], r["method"], int(r["seed"])): r for r in S11}
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")

    def lower_bound(st: str, m: str) -> float:
        c = [RELAX[st][m]["objective"]]
        for sd in (0, 1):
            r = cell.get((st, m, sd))
            if r and r.get("mip_dual_bound") is not None:
                c.append(float(r["mip_dual_bound"]))
        return max(c)

    # ---- the certificate cases, each with its AB witness (a fully saved, re-verified plan)
    cases: List[Dict[str, Any]] = []
    for st in states:
        wit = min((cell[(st, "AB", sd)] for sd in (0, 1) if (st, "AB", sd) in cell),
                  key=lambda r: float(r["final_cost"]))
        uab = float(wit["final_cost"])
        for mr in MR:
            lmr = lower_bound(st, mr)
            if uab < lmr - 1e-9 * max(1.0, abs(uab)):
                cases.append({"state": st, "method": mr, "U_AB": uab, "L_MR": lmr,
                              "Q": 0.5 * (uab + lmr), "witness_seed": int(wit["seed"]),
                              "witness_plan": wit["final_solution"]})
    print("certificate cases: %d" % len(cases))
    if args.smoke:
        cases = cases[:1]
        print("SMOKE: first case only -> %s / %s" % (cases[0]["state"], cases[0]["method"]))

    os.makedirs("stage13_logs", exist_ok=True)
    out: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    for ci, cs in enumerate(cases):
        st, method = cs["state"], cs["method"]
        rec = ALL[st]
        tau = rec["meta"]["tau"]
        inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        bm = build_model(pert, MODE_AUDITED)
        C = build_state_context(inst, pert, repair, bm, tau)
        C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert})
        C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                        for n in MR}

        # the AB witness as a full vector and as a Solution
        wp = cs["witness_plan"]
        wit = Solution(status="witness", objective=cs["U_AB"],
                       X=np.array(wp["X"], float).reshape(pert.N, pert.M, pert.T),
                       Y=np.array(wp["Y"], float).reshape(pert.N, pert.M, pert.T),
                       Z=np.array(wp["Z"], float).reshape(pert.N, pert.M, pert.T),
                       I=np.array(wp["I"], float).reshape(pert.N, pert.T),
                       L=np.array(wp["L"], float).reshape(pert.N, pert.T),
                       mode=MODE_AUDITED)
        chk_w = check_solution(pert, wit, MODE_AUDITED)
        C["_witness_vector"] = vector_of(wit, bm)

        Q = cs["Q"]
        lp, h_vec, x_wit, info = build_min_lift_mip(bm, pert, repair, C, method, Q)
        # the AB witness must itself satisfy every fixing except the F_B ones it changes
        rel, _f, _z = stage11_columns(C, method)
        rel_set = set(rel)
        relB = {tuple(u[1:]) for u in rel if u[0] == "Z"}
        wit_H = info["H_from_x"](x_wit)

        tag = "%s__%s" % (st.replace("|", "_"), method)
        log_path = os.path.join("stage13_logs", tag + ".log")
        if os.path.exists(log_path):
            os.remove(log_path)
        h = highspy.Highs()
        h.setOptionValue("output_flag", True)
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", log_path)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", BUDGET)
        h.setOptionValue("random_seed", SEED)
        # ---- solve, with the AB witness submitted as the start.
        #
        # The earlier version never called setSolution, so this path started from nothing and
        # simply failed to find any feasible incumbent in 60 s (the logs showed
        # `Primal bound inf` / `Solution status -`).  Submission is restored, and the return
        # codes plus the adoption line are recorded.
        h = highspy.Highs()
        h.setOptionValue("output_flag", True)
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("log_file", log_path)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", BUDGET)
        h.setOptionValue("random_seed", SEED)
        sp = h.passModel(lp)
        ss = h.setSolution(bm.idx.n, np.arange(bm.idx.n, dtype=np.int32),
                           np.asarray(x_wit, dtype=np.float64))
        start_ok = str(ss).endswith("kOk")
        t0 = time.perf_counter()
        sr = h.run()
        wall = time.perf_counter() - t0
        status = h.modelStatusToString(h.getModelStatus())
        i = h.getInfo()
        sol = h.getSolution()
        obj_solver = None
        try:
            v = float(i.objective_function_value)
            obj_solver = v if np.isfinite(v) else None
        except Exception:                                    # noqa: BLE001
            obj_solver = None
        # HighsInfo has no `mip_primal_bound` on this build; the primal side is
        # `objective_function_value` (already read as obj_solver).
        primal = obj_solver
        dual = float(i.mip_dual_bound) if np.isfinite(i.mip_dual_bound) else None
        proven = "optimal" in status.lower()
        # H = solver objective + c1, so the bounds convert the same way
        H_upper_from_solver = (None if obj_solver is None
                               else info["H_of_objective"](obj_solver))
        H_lower_from_solver = (None if dual is None else info["H_of_objective"](dual))

        # ---- re-cost and re-verify the returned plan with the ORIGINAL cost
        #
        # The solver objective is h.x = h_const - H, so a negative reported objective is normal
        # and does NOT mean H is negative.  H is therefore always taken from the solution vector
        # via H_from_x; the raw reported scalar is stored separately and never used as H.
        rec_out: Dict[str, Any] = {
            "state": st, "method": method, "Q": Q, "U_AB": cs["U_AB"], "L_MR": cs["L_MR"],
            "witness_seed": cs["witness_seed"], "witness_H": wit_H,
            "witness_feasible": bool(chk_w.ok),
            "witness_cost_recomputed": chk_w.cost_recomputed["total"],
            "n_F_B": info["n_F_B"], "n_F_C": info["n_F_C"], "c1": info["c1"],
            "rel_size": info["rel_size"], "mr_rel_size": info["mr_rel_size"],
            "status": status, "status_pass": str(sp), "status_run": str(sr),
            "status_set_solution": str(ss), "start_submitted_ok": bool(start_ok),
            "proven_optimal": bool(proven),
            "mip_primal_bound": primal,
            "H_solver_objective": obj_solver, "H_solver_objective_lb": dual,
            "H_upper_from_solver": H_upper_from_solver,
            "H_lower_from_solver": H_lower_from_solver,
            "mip_gap": (float(i.mip_gap) if np.isfinite(i.mip_gap) else None),
        }
        # the VERIFIED witness always provides the upper bound H* <= |V| = witness_H
        rec_out["H_upper_bound"] = int(wit_H)
        rec_out["H_lower_bound"] = None
        # solution validity flag: an array of the right length is NOT a valid solution
        try:
            rec_out["solution_value_valid"] = bool(sol.value_valid)
        except Exception:                                    # noqa: BLE001
            rec_out["solution_value_valid"] = None
        if sol.col_value is not None and len(sol.col_value) == bm.idx.n:
            x = np.asarray(sol.col_value, float)
            # A time-limited HiGHS run can hand back a point that does NOT satisfy the model.
            # Never believe an incumbent without checking the rows.
            A = csr_matrix((np.asarray(lp.a_matrix_.value_),
                            np.asarray(lp.a_matrix_.index_),
                            np.asarray(lp.a_matrix_.start_)),
                           shape=(lp.num_row_, lp.num_col_))
            ax = A @ x
            rl = np.asarray(lp.row_lower_)
            ru = np.asarray(lp.row_upper_)
            nviol = int(np.sum((ax < rl - 1e-6) | (ax > ru + 1e-6)))
            if nviol > 0 or not np.all(np.isfinite(x)):
                rec_out["incumbent_valid"] = False
                rec_out["incumbent_row_violations"] = nviol
            else:
                rec_out["incumbent_valid"] = True
                rec_out["incumbent_row_violations"] = 0
                J = info["J_from_x"](x)
                H_manual = info["H_from_x"](x)
                X, Y, Z, I2, L = unpack(x, bm.idx)
                got = Solution(status=status, objective=J, X=X, Y=Y, Z=Z, I=I2, L=L,
                               mode=MODE_AUDITED)
                chk = check_solution(pert, got, MODE_AUDITED)
                # which originally-fixed B positions moved, and in which direction
                changed = []
                F_B_set = {tuple(v) for v in info["F_B"]}
                for (i2, j2, t2) in F_B_set:
                    want = float(C["assign"][("Z", i2, j2, t2)])
                    g = float(Z[i2, j2, t2 - 1])
                    if abs(g - want) > 0.5:
                        changed.append({"i": i2, "j": j2, "t": t2,
                                        "from": int(round(want)), "to": int(round(g))})
                # fixing conditions of the ORIGINAL MR: everything outside F_B stays at S_r
                viol_C = viol_Y = 0
                for (kind, i2, j2, t2) in C["all_slots"]:
                    if kind != "Z" or (i2, j2, t2) in relB or (i2, j2, t2) in F_B_set:
                        continue
                    if abs(float(Z[i2, j2, t2 - 1]) - float(repair.Z[i2, j2, t2 - 1])) > 0.5:
                        viol_C += 1
                for i2 in range(pert.N):
                    for j2 in range(pert.M):
                        for t2 in range(1, tau + 1):
                            if abs(float(Y[i2, j2, t2 - 1])
                                   - float(repair.Y[i2, j2, t2 - 1])) > 0.5:
                                viol_Y += 1
                rec_out.update({
                    "J_recomputed": J, "H_manual": H_manual,
                    "H_solver_consistent": (obj_solver is None
                                            or abs(H_manual - info["H_of_objective"](obj_solver))
                                            <= 1e-6),
                    "quality_gate_ok": bool(J <= Q + 1e-6 * (1 + abs(Q))),
                    "model_feasible": bool(chk.ok),
                    "max_violation": float(chk.max_violation),
                    "short_Y_violations": viol_Y, "FC_violations": viol_C,
                    "carry_replacement": _carry_replacement(changed),
                    "n_changed": len(changed), "changed": changed,
                    "solution": {k: np.asarray(getattr(got, k)).tolist()
                                 for k in ("X", "Y", "Z", "I", "L")},
                })
                # tighten the bounds with what this run established
                rec_out["H_upper_bound"] = min(int(wit_H), int(H_manual))
                if proven and H_upper_from_solver is not None:
                    rec_out["H_lower_bound"] = int(round(H_lower_from_solver))
                elif H_lower_from_solver is not None:
                    rec_out["H_lower_bound"] = int(np.ceil(H_lower_from_solver - 1e-6))
        out.append(rec_out)
        print("  [%2d/%2d] %-30s %-8s Q=%-11.6g H_wit=%-2d H=%-5s %-18s J=%-11s gate=%-5s t=%.1f"
              % (ci + 1, len(cases), st, method, Q, wit_H,
                 "MISSING" if rec_out.get("H_manual") is None else "%d" % rec_out["H_manual"],
                 status[:18],
                 "-" if rec_out.get("J_recomputed") is None
                 else "%.6g" % rec_out["J_recomputed"],
                 rec_out.get("quality_gate_ok"), wall), flush=True)

    print("\ncases: %d | wall %.1f s" % (len(out), time.perf_counter() - t_all))
    if args.smoke:
        ok = all(r.get("quality_gate_ok") and r.get("model_feasible") for r in out)
        print("SMOKE %s" % ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"budget_s": BUDGET, "seed": SEED, "n_cases": len(out),
                            "role": "OFFLINE MECHANISM DIAGNOSTIC on saved certificates; not a "
                                    "deployable selector and no evidence for GRL",
                            "Q_rule": "Q = (U_AB,witness + L_MR)/2",
                            "objective_split": "solver objective = H (fixings changed); the "
                                               "original J is a row constraint and is "
                                               "recomputed for verification"},
                   "cases": out}, fh, indent=2, default=str)
    n_prov = sum(1 for r in out if r["proven_optimal"])
    n_gate = sum(1 for r in out if r.get("quality_gate_ok"))
    print("proven optimal: %d/%d | quality gate satisfied: %d/%d"
          % (n_prov, len(out), n_gate, len(out)))
    print("wrote %s" % OUT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
