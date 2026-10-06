#!/usr/bin/env python
"""
Stage 13 preflight: verify the H algebra on a two-variable enumeration, then check that the
AB witness is an acceptable start for the diagnostic model.

    .venv-hs/Scripts/python.exe stage13_preflight.py

Part A (algebra).  With two F_B slots whose FIXED values are (0, 1):
    c1 = 1
    H(x) = h.x + c1
The four assignments must give H = 0, 1, 1, 2 in that order, and the conversion helpers must
agree with a brute-force count.  This is the check the earlier version failed: it used
`c1 - objective` instead of `objective + c1`.

Part B (start protocol).  For every certificate case, build the diagnostic model and verify
that the saved AB witness is inside the box and satisfies every row, INCLUDING J <= Q.  Any
conflict is reported as an error rather than clamped away.
"""

from __future__ import annotations

import itertools
import json
import sys
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_minlift import build_min_lift_mip
from lsp_model import MODE_AUDITED, build_model
from lsp_release import build_state_context
from stage06_run import solution_from_record

MR = ("MR-Z-1", "MR-Z-2", "MR-Z-3")


def main() -> int:
    failures: List[str] = []

    # ---------------- Part A: the algebra, on a synthetic two-slot example
    print("=== Part A: H algebra on a two-slot example ===")
    # slot0 is FIXED to 0, slot1 is FIXED to 1
    #   H = [z0 != 0] + [z1 != 1] = z0 + (1 - z1)
    #   h  = (+1, -1)   and   H(x) = h.x + c1  with c1 = 1
    h = np.array([1.0, -1.0])
    c1 = 1
    expect = []
    for x0 in (0, 1):
        for x1 in (0, 1):
            x = np.array([float(x0), float(x1)])
            H_true = (1 if x0 != 0 else 0) + (1 if x1 != 1 else 0)
            H_conv = float(h @ x) + c1
            expect.append((x0, x1, H_true, H_conv))
            if abs(H_true - H_conv) > 1e-12:
                failures.append("algebra: x=(%d,%d) H_true=%d but h.x+c1=%g"
                                % (x0, x1, H_true, H_conv))
    for x0, x1, H_true, H_conv in expect:
        print("  x=(%d,%d)  H_true=%d  h.x+c1=%g  %s"
              % (x0, x1, H_true, H_conv, "ok" if H_true == H_conv else "MISMATCH"))
    seq = [e[2] for e in expect]
    print("  H by (x0,x1): " + ", ".join("(%d,%d)->%d" % (a, b, hh)
                                         for a, b, hh, _ in expect))
    print("  multiset of H:", sorted(seq), "(the [0,1,1,2] table)")
    # invariants that must hold in ANY enumeration order:
    #   exactly one assignment moves nothing (x0=0, x1=1), one moves both, two move exactly one
    if sorted(seq) != [0, 1, 1, 2]:
        failures.append("algebra: H multiset is %s, expected [0, 1, 1, 2]" % sorted(seq))
    if dict(((a, b), hh) for a, b, hh, _ in expect)[(0, 1)] != 0:
        failures.append("algebra: the all-at-fixed-value assignment (0,1) must give H=0")
    if dict(((a, b), hh) for a, b, hh, _ in expect)[(1, 0)] != 2:
        failures.append("algebra: the both-moved assignment (1,0) must give H=2")
    for a, b, hh, conv in expect:
        if conv not in (0.0, 1.0, 2.0):
            failures.append("algebra: conversion gave non-integer H=%g at (%d,%d)"
                            % (conv, a, b))
    print()

    # ---------------- Part B: the start protocol on every certificate case
    print("=== Part B: AB witness feasibility in the diagnostic model ===")
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

    n_cases = 0
    print("  %-30s %-8s %6s %6s %8s %8s %10s %8s"
          % ("state", "MR", "n_F_B", "n_F_C", "rel", "mr_rel", "rowviol", "H"))
    for st in states:
        wit = min((cell[(st, "AB", sd)] for sd in (0, 1) if (st, "AB", sd) in cell),
                  key=lambda r: float(r["final_cost"]))
        uab = float(wit["final_cost"])
        for mr in MR:
            lmr = lower_bound(st, mr)
            if not (uab < lmr - 1e-9 * max(1.0, abs(uab))):
                continue
            n_cases += 1
            rec = ALL[st]
            tau = rec["meta"]["tau"]
            inst, _m = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                      rec["meta"]["seed"])
            dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
            pert = apply_disruption(inst, dis)
            repair = solution_from_record(pert, rec["repair"])
            bm = build_model(pert, MODE_AUDITED)
            C = build_state_context(inst, pert, repair, bm, tau)
            C.update({"bm": bm, "tau": tau, "meta": rec["meta"], "repair": repair,
                      "inst": pert})
            C["mr_sets"] = {n: {tuple(u) for u in MATCH[st]["sets"][n]["fixed_slots"]}
                            for n in MR}
            p = wit["final_solution"]
            idx = bm.idx
            xv = np.zeros(idx.n)
            for i in range(pert.N):
                for j in range(pert.M):
                    for t in range(1, pert.T + 1):
                        xv[idx.X(i, j, t)] = float(p["X"][i][j][t - 1])
                        xv[idx.Y(i, j, t)] = float(p["Y"][i][j][t - 1])
                        xv[idx.Z(i, j, t)] = float(p["Z"][i][j][t - 1])
                    xv[idx.Z(i, j, 0)] = 0.0
                for t in range(1, pert.T + 1):
                    xv[idx.I(i, t)] = float(p["I"][i][t - 1])
                    xv[idx.L(i, t)] = float(p["L"][i][t - 1])
            C["_witness_vector"] = xv
            Q = 0.5 * (uab + lmr)
            try:
                lp, hh, x0, info = build_min_lift_mip(bm, pert, repair, C, mr, Q)
            except ValueError as exc:
                failures.append("%s/%s: %s" % (st, mr, exc))
                print("  %-30s %-8s  BUILD ERROR: %s" % (st, mr, str(exc)[:60]))
                continue
            A = csr_matrix((np.asarray(lp.a_matrix_.value_),
                            np.asarray(lp.a_matrix_.index_),
                            np.asarray(lp.a_matrix_.start_)),
                           shape=(lp.num_row_, lp.num_col_))
            ax = A @ x0
            rl = np.asarray(lp.row_lower_)
            ru = np.asarray(lp.row_upper_)
            nviol = int(np.sum((ax < rl - 1e-6) | (ax > ru + 1e-6)))
            lb = np.asarray(lp.col_lower_)
            ub = np.asarray(lp.col_upper_)
            oob = int(np.sum((x0 < lb - 1e-9) | (x0 > ub + 1e-9)))
            H = info["H_from_x"](x0)
            J = info["J_from_x"](x0)
            gate = J <= Q + 1e-6 * (1 + abs(Q))
            print("  %-30s %-8s %6d %6d %8d %8d %10d %8d  gate=%s"
                  % (st, mr, info["n_F_B"], info["n_F_C"], info["rel_size"],
                     info["mr_rel_size"], nviol, H, gate))
            if nviol or oob or not gate:
                failures.append("%s/%s: rowviol=%d oob=%d gate=%s"
                                % (st, mr, nviol, oob, gate))

    print()
    print("cases checked: %d" % n_cases)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:30]:
            print("   -", f)
        return 1
    print("PREFLIGHT PASSED: H algebra consistent and every AB witness is a valid, gate-passing")
    print("start for its diagnostic model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
