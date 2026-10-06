#!/usr/bin/env python
"""
Stage 13 probe: is the diagnostic feasible at all, and at which H levels?

    .venv-hs/Scripts/python.exe stage13_probe.py [case_index] [seconds_per_level]

Builds the SAME diagnostic model as `stage13_run.py` but adds an explicit H cap row instead of
relying on the objective, then asks pure FEASIBILITY at a ladder of caps.  This distinguishes
"the model has no qualifying plan at all" from "60 s is not enough to find one", which the smoke
test could not tell apart.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import coo_matrix, vstack as sp_vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import highspy

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_minlift import build_min_lift_mip
from lsp_model import MODE_AUDITED, Solution, build_model, unpack
from lsp_release import build_state_context
from stage06_run import solution_from_record

BUDGET_PER_LEVEL = 30.0
MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")


def main() -> int:
    ci = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else BUDGET_PER_LEVEL

    with open("stage06_states.json", encoding="utf-8") as fh:
        ALL = json.load(fh)
    with open("stage11_matched_random.json", encoding="utf-8") as fh:
        MATCH = json.load(fh)
    with open("stage11_runs.json", encoding="utf-8") as fh:
        S11 = json.load(fh)["runs"]
    with open("stage11b_relax.json", encoding="utf-8") as fh:
        RELAX = json.load(fh)
    cell = {(r["state"], r["method"], int(r["seed"])): r for r in S11}
    states = sorted(s for s in ALL if ALL[s]["meta"]["scale"] == "large")

    def lower_bound(st, m):
        c = [RELAX[st][m]["objective"]]
        for sd in (0, 1):
            r = cell.get((st, m, sd))
            if r and r.get("mip_dual_bound") is not None:
                c.append(float(r["mip_dual_bound"]))
        return max(c)

    cases = []
    for st in states:
        wit = min((cell[(st, "AB", sd)] for sd in (0, 1) if (st, "AB", sd) in cell),
                  key=lambda r: float(r["final_cost"]))
        for mr in MR:
            lmr = lower_bound(st, mr)
            if float(wit["final_cost"]) < lmr - 1e-9 * max(1.0, abs(float(wit["final_cost"]))):
                cases.append((st, mr, float(wit["final_cost"]), lmr, wit))
    st, method, uab, lmr, witrec = cases[ci]
    Q = 0.5 * (uab + lmr)
    print("case %d: %s / %s" % (ci, st, method))
    print("  U_AB=%.6g  L_MR=%.6g  Q=%.6g" % (uab, lmr, Q))

    rec = ALL[st]
    tau = rec["meta"]["tau"]
    inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
    dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
    pert = apply_disruption(inst, dis)
    repair = solution_from_record(pert, rec["repair"])
    bm = build_model(pert, MODE_AUDITED)
    C = build_state_context(inst, pert, repair, bm, tau)
    C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair, "inst": pert})
    C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]} for n in MR}
    wp = witrec["final_solution"]
    wit = Solution(status="w", objective=uab,
                   X=np.array(wp["X"], float).reshape(pert.N, pert.M, pert.T),
                   Y=np.array(wp["Y"], float).reshape(pert.N, pert.M, pert.T),
                   Z=np.array(wp["Z"], float).reshape(pert.N, pert.M, pert.T),
                   I=np.array(wp["I"], float).reshape(pert.N, pert.T),
                   L=np.array(wp["L"], float).reshape(pert.N, pert.T), mode=MODE_AUDITED)
    xv = np.zeros(bm.idx.n)
    for i in range(pert.N):
        for j in range(pert.M):
            for t in range(1, pert.T + 1):
                xv[bm.idx.X(i, j, t)] = float(wit.X[i, j, t - 1])
                xv[bm.idx.Y(i, j, t)] = float(wit.Y[i, j, t - 1])
                xv[bm.idx.Z(i, j, t)] = float(wit.Z[i, j, t - 1])
            xv[bm.idx.Z(i, j, 0)] = 0.0
        for t in range(1, pert.T + 1):
            xv[bm.idx.I(i, t)] = float(wit.I[i, t - 1])
            xv[bm.idx.L(i, t)] = float(wit.L[i, t - 1])
    C["_witness_vector"] = xv

    lp, h, x0, info = build_min_lift_mip(bm, pert, repair, C, method, Q)
    n = lp.num_col_
    hconst = info["h_const"]
    # h.x = hconst - H, so H <= cap  <=>  h.x >= hconst - cap
    for cap in (1, 2, 3, 4, 6, 8, info["n_F_B"]):
        lpk = highspy.HighsLp()
        lpk.num_col_ = lp.num_col_
        lpk.num_row_ = lp.num_row_ + 1
        lpk.col_cost_ = np.zeros(n)                            # pure feasibility
        lpk.col_lower_ = np.array(lp.col_lower_)
        lpk.col_upper_ = np.array(lp.col_upper_)
        lpk.row_lower_ = np.concatenate([np.array(lp.row_lower_), [hconst - float(cap)]])
        lpk.row_upper_ = np.concatenate([np.array(lp.row_upper_), [np.inf]])
        A = sp_vstack([__csr(lp), coo_matrix((h, (np.zeros(n, np.int32),
                                                  np.arange(n, dtype=np.int32))),
                                             shape=(1, n)).tocsr()], format="csr")
        lpk.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lpk.a_matrix_.start_ = A.indptr.astype(np.int32)
        lpk.a_matrix_.index_ = A.indices.astype(np.int32)
        lpk.a_matrix_.value_ = A.data.astype(np.float64)
        lpk.integrality_ = list(lp.integrality_)

        g = highspy.Highs()
        g.setOptionValue("output_flag", False)
        g.setOptionValue("threads", 1)
        g.setOptionValue("time_limit", budget)
        g.setOptionValue("random_seed", 0)
        sp = g.passModel(lpk)
        ss = g.setSolution(n, np.arange(n, dtype=np.int32), np.asarray(x0, float))
        t0 = time.perf_counter()
        sr = g.run()
        w = time.perf_counter() - t0
        sm = g.modelStatusToString(g.getModelStatus())
        sol = g.getSolution()
        found = sol.col_value is not None and len(sol.col_value) == n
        msg = ""
        if found:
            x = np.asarray(sol.col_value, float)
            J = info["J_from_x"](x)
            Hm = info["H_from_x"](x)
            msg = "J=%.6g (<=Q %s)  H=%d" % (J, J <= Q + 1e-6 * (1 + abs(Q)), Hm)
        print("  cap H<=%-3d setSolution=%-20s status=%-20s found=%-5s %-9s %s"
              % (cap, str(ss).replace("HighsStatus.", ""), sm, found, "%.1fs" % w, msg))

    return 0


def __csr(lp):
    from scipy.sparse import csr_matrix
    return csr_matrix((np.asarray(lp.a_matrix_.value_), np.asarray(lp.a_matrix_.index_),
                       np.asarray(lp.a_matrix_.start_)), shape=(lp.num_row_, lp.num_col_))


if __name__ == "__main__":
    sys.exit(main())
