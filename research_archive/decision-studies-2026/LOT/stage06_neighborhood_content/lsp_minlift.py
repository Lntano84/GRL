#!/usr/bin/env python
"""
Stage 13 diagnostic MIP: minimum number of short-term-Z fixings that must be lifted.

    from lsp_minlift import build_min_lift_mip

For one certificate case:

    Q = (U_AB,witness + L_MR) / 2          so that  U_AB,witness < Q < L_MR

    min  H(x) = sum_{z in F_B, z^r = 0} z  +  sum_{z in F_B, z^r = 1} (1 - z)
    s.t. the ORIGINAL audited model,
         short-term Y fixed at S_r,
         F_C fixed at S_r (the matched-random set's C fixings),
         every B slot OUTSIDE F_B still fixed at S_r,
         F_B released to [0, 1],
         J(x) <= Q                            <- the ORIGINAL cost expression

H counts how many originally-FIXED B positions deviate from their fixed value, so H* is exactly
"how many of those fixings must be lifted".  No extra "is this released?" binary variables are
needed: for any qualifying plan, only the positions that actually change need to be lifted and
the rest can be re-fixed, so minimising H really does minimise the number lifted.

TWO OBJECTIVES, KEPT STRICTLY APART
-----------------------------------
Highs holds exactly one objective, so this builder sets the SOLVER objective to H and returns
`J_from_x`, which recomputes the ORIGINAL cost from a solution vector.  The solver's reported
objective is H and must NEVER be handed to the checker as a production cost.  Every solution is
re-costed with `J_from_x` before it is used anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, vstack as sp_vstack

import highspy

from lsp_release import stage11_columns


def build_min_lift_mip(bm, pert, repair, C: Dict[str, Any], method: str, Q: float):
    """Return (lp, h_vec, x_witness, info).

    `lp` is a HighsLp whose objective is H.  `info` carries `J_from_x`, `H_from_x` and the
    bookkeeping the analysis needs.  `x_witness` is the AB witness as a full variable vector.
    """
    idx = bm.idx
    tau = C["tau"]

    Z_all_B = {tuple(u[1:]) for u in C["groups"]["B"]}
    Z_all_C = {tuple(u[1:]) for u in C["groups"]["C"]}
    F_B = {tuple(u) for u in C["mr_sets"][method]} & Z_all_B
    F_C = {tuple(u) for u in C["mr_sets"][method]} & Z_all_C

    # The diagnostic is the ORIGINAL MR configuration with F_B additionally released, so its
    # fixing set is
    #     short-term Y  +  F_C  +  (every other non-pinned Z)
    # i.e. every Z slot is FIXED except F_B and the C slots that this MR set leaves free.
    # Earlier this anchored on AB's release set instead, which pinned the wrong Z slots: the
    # AB released set is A u B, so ALL of C ended up pinned while 286 B slots were freed, and a
    # legal AB plan then came out bound-infeasible.
    Z_released = (Z_all_B - F_B) | (Z_all_C - F_C)
    rel = sorted(C["groups"]["A"]) + [("Z",) + tuple(u) for u in sorted(Z_released)]

    # ---- bounds: pin every binary to S_r, then release exactly the two sets above
    lb = bm.var_lb.copy()
    ub = bm.var_ub.copy()
    for slot, val in C["assign"].items():
        kind, i, j, t = slot
        col = idx.Y(i, j, t) if kind == "Y" else idx.Z(i, j, t)
        lb[col] = float(val)
        ub[col] = float(val)
    for (kind, i, j, t) in rel:
        col = idx.Y(i, j, t) if kind == "Y" else idx.Z(i, j, t)
        lb[col] = float(bm.var_lb[col])
        ub[col] = float(bm.var_ub[col])
    # release F_B explicitly as well (it is inside Z_all_B, so the line above already covers it,
    # but be explicit so a later edit cannot silently re-pin it)
    for (i, j, t) in F_B:
        lb[idx.Z(i, j, t)] = 0.0
        ub[idx.Z(i, j, t)] = 1.0
    # pin F_C exactly at S_r (it is in Z_released only if it is also free in this MR set)
    for (i, j, t) in F_C:
        v = float(C["assign"][("Z", i, j, t)])
        lb[idx.Z(i, j, t)] = v
        ub[idx.Z(i, j, t)] = v

    # ---- rows: base A, then J(x) <= Q, then H(x) >= 1
    A_blocks = [bm.A.tocsr()]
    rl = [bm.row_lb.copy()]
    ru = [bm.row_ub.copy()]

    A_blocks.append(coo_matrix((np.asarray(bm.c, float),
                                (np.zeros(idx.n, np.int32), np.arange(idx.n, dtype=np.int32))),
                               shape=(1, idx.n)).tocsr())
    rl.append(np.array([-np.inf]))
    ru.append(np.array([float(Q)]))

    h = np.zeros(idx.n)
    c1 = 0                      # c1 = number of F_B slots whose FIXED value is 1
    for (i, j, t) in F_B:
        r = float(C["assign"][("Z", i, j, t)])
        h[idx.Z(i, j, t)] = 1.0 if r == 0 else -1.0
        c1 += int(r == 1)

    # ALGEBRA (this is the part the earlier version got wrong)
    #   h.x = (#ones AFTER) - (#ones BEFORE that are still 1) ... concretely
    #   H(x) = h.x + c1
    # so minimising h.x minimises H -- the objective coefficients are CORRECT and must NOT be
    # flipped.  Every conversion must add c1:
    #     H          = solver objective + c1
    #     H lower bd = solver objective lower bound + c1
    #     H >= 1     <=>  h.x >= 1 - c1
    #     H <= k     <=>  h.x <= k - c1
    # The H >= 1 row IS valid (any qualifying plan must move at least one fixing, by the
    # certificate), and it gives a genuine lower bound on H* instead of only an upper bound.
    A_blocks.append(coo_matrix((h, (np.zeros(idx.n, np.int32), np.arange(idx.n, dtype=np.int32))),
                               shape=(1, idx.n)).tocsr())
    rl.append(np.array([1.0 - float(c1)]))
    ru.append(np.array([np.inf]))

    A = sp_vstack(A_blocks, format="csr")
    row_lb = np.concatenate(rl)
    row_ub = np.concatenate(ru)

    lp = highspy.HighsLp()
    lp.num_col_ = idx.n
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = h.astype(np.float64)
    lp.col_lower_ = lb.astype(np.float64)
    lp.col_upper_ = ub.astype(np.float64)
    lp.row_lower_ = row_lb.astype(np.float64)
    lp.row_upper_ = row_ub.astype(np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.integrality_ = [highspy.HighsVarType.kInteger if v == 1
                       else highspy.HighsVarType.kContinuous
                       for v in bm.integrality]

    c_orig = np.asarray(bm.c, float)
    F_B_list = sorted(F_B)
    assign = C["assign"]

    def J_from_x(x: np.ndarray) -> float:
        """The ORIGINAL audited cost of a full variable vector."""
        return float(np.dot(c_orig, np.asarray(x, float)))

    def H_from_x(x: np.ndarray) -> int:
        """How many originally-fixed B positions this plan moved."""
        tot = 0
        for (i, j, t) in F_B_list:
            want = float(assign[("Z", i, j, t)])
            got = float(x[idx.Z(i, j, t)])
            tot += int(abs(got - want) > 0.5)
        return tot

    def H_of_objective(obj: float) -> float:
        """Convert a solver objective value (h.x) into H.  H = h.x + c1."""
        return float(obj) + float(c1)

    # The diagnostic releases Y (group A) plus Z_released = (B - F_B) u (C - F_C).
    #     |diagnostic Z release| = |B - F_B| + |C - F_C|
    # and equivalently = |MR's own short-term-Z release| + |F_B|, because the MR configuration
    # releases B - F_B out of B and C - F_C out of C.
    n_rel_B = len(Z_all_B - F_B)
    n_rel_C = len(Z_all_C - F_C)
    info = {"n_F_B": len(F_B_list), "F_B": [list(u) for u in F_B_list],
            "n_F_C": len(F_C), "c1": int(c1), "h_const": int(c1),
            "Q": float(Q),
            "n_rel_Y": len(C["groups"]["A"]),
            "n_rel_B": n_rel_B, "n_rel_C": n_rel_C,
            "n_rel_Z": n_rel_B + n_rel_C,
            "rel_size": len(rel),
            "mr_stz_release_only": (n_rel_B + n_rel_C) - len(F_B),
            "J_from_x": J_from_x, "H_from_x": H_from_x, "H_of_objective": H_of_objective}

    # ---- start vector: ONLY tolerance-level binary rounding, never force a real conflict
    #
    # The earlier version clamped values into the box, which can hide a genuine conflict between
    # the saved plan and the diagnostic fixings.  Now a value that is outside its bounds by more
    # than the binary tolerance is a hard ERROR, reported with its column, rather than silently
    # corrected.
    x0 = C.get("_witness_vector")
    if x0 is not None:
        x0 = np.asarray(x0, float).copy()
        for c in range(idx.n):
            if bm.integrality[c] == 1:
                x0[c] = float(round(x0[c]))
        pinned = np.where(lb == ub)[0]
        bad = pinned[np.abs(x0[pinned] - lb[pinned]) > 0.5]
        free = np.where(lb != ub)[0]
        oob = free[(x0[free] < lb[free] - 1e-6) | (x0[free] > ub[free] + 1e-6)]
        if len(bad) or len(oob):
            raise ValueError(
                "start vector conflicts with the diagnostic model: %d pinned-column mismatches "
                "(e.g. %s) and %d out-of-box values (e.g. %s)"
                % (len(bad), bad[:4].tolist(), len(oob), oob[:4].tolist()))
    return lp, h, x0, info
