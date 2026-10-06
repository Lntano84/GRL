"""
Stage 01b — genuinely independent continuous LP path.

Why this file exists
--------------------
The stage-01 enumeration path (`lsp_model.solve_lp_with_binaries_fixed`) converts the
model's own constraint matrix `bm.A`, its `row_lb`/`row_ub` and its variable bounds into an
`linprog` call. That is a real check on the MILP *solver* path, but it cannot detect a
mistake in how the constraints were assembled: any wrong coefficient is reproduced
identically on both sides.

This module rebuilds the continuous problem from the raw instance parameters only.

Guarantees
----------
  * It never imports a BuiltModel, never reads bm.A / bm.c / bm.row_lb / bm.row_ub /
    bm.var_lb / bm.var_ub, and never calls lsp_model.activation_upper_bounds or
    lsp_model.cost_breakdown.
  * The upper bound U_ijt for (1.8) is recomputed here, from the mode definition and the
    raw parameters, by a second implementation.
  * The binary logic (1.5), (1.6), (1.7), (1.11) is checked explicitly on the given Y, Z
    before any LP is built, so an infeasible binary pattern is reported as such rather than
    being handed to the LP solver.
  * The LP itself is built here with its own variable ordering (X, I, L only), its own
    rows, its own objective, and its own cost recomputation.

Allowed to be shared: the LP/MILP solver library (HiGHS through SciPy). What has to be
independent is the mathematical model construction, not the numerical solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from lsp_model import MODE_AUDITED, MODE_PAPER, Instance, MODES

TOL = 1e-6

# The independent LP uses its own variable ordering, unrelated to lsp_model.Indexer:
#   columns 0 .. N*M*T-1                  X[i,j,t]
#   then     N*T                          I[i,t]
#   then     N*T                          L[i,t]
# Binary values Y, Z are given as data and never become columns.


def independent_upper_bound(inst: Instance, mode: str, i: int, j: int, t: int) -> float:
    """Second, independent implementation of U_ijt for constraint (1.8).

    paper_literal : min( sum_{t'>=t} d_it' , (c_jt - s_i)/b_i ), negative values preserved.
    audited_v1    : c_jt / b_i.
    """
    if mode == MODE_PAPER:
        future = 0.0
        for tt in range(t, inst.T + 1):
            future += inst.d[i][tt - 1]
        return min(future, (inst.c[j][t - 1] - inst.s[i]) / inst.b[i])
    if mode == MODE_AUDITED:
        return inst.c[j][t - 1] / inst.b[i]
    raise ValueError(f"unknown mode {mode!r}")


def independent_cost_breakdown(inst: Instance, X, Y, Z, I, L) -> Dict[str, float]:
    """Second, independent implementation of the four cost components of objective (1.1)."""
    setup = 0.0
    production = 0.0
    inventory = 0.0
    lost = 0.0
    for i in range(inst.N):
        for t in range(1, inst.T + 1):
            for j in range(inst.M):
                setup += inst.f[i] * float(Y[i][j][t - 1])
                production += inst.p[i] * float(X[i][j][t - 1])
            inventory += inst.h[i] * float(I[i][t - 1])
            lost += inst.l[i] * float(L[i][t - 1])
    return {"setup": setup, "production": production, "inventory": inventory,
            "lost_sales": lost, "total": setup + production + inventory + lost}


@dataclass
class BinaryCheck:
    ok: bool
    violations: List[str]


def check_binary_logic(inst: Instance, mode: str,
                       Y: Sequence[Sequence[Sequence[float]]],
                       Z: Sequence[Sequence[Sequence[float]]]) -> BinaryCheck:
    """Explicitly verify the purely binary constraints on a given (Y, Z) pattern.

    (1.5) Y_ijt <= w_ij
    (1.6) Z_ijt <= Y_ijt
    (1.7) Z_ij(t-1) + Z_ijt <= 1   with Z_ij0 = 0 from (1.13)
    (1.11) sum_i Z_ijt <= 1
    plus: in audited_v1, Z_ijT = 0.
    Returns the list of violated rules; empty means the pattern is admissible.
    """
    bad: List[str] = []

    def zval(i, j, t):
        # t == 0 is the fixed initial carry-over from (1.13)
        return 0.0 if t == 0 else float(Z[i][j][t - 1])

    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                y = float(Y[i][j][t - 1])
                if y not in (0.0, 1.0):
                    bad.append(f"(1.12) Y[{i},{j},{t}]={y} is not binary")
                if zval(i, j, t) not in (0.0, 1.0):
                    bad.append(f"(1.12) Z[{i},{j},{t}]={zval(i, j, t)} is not binary")
                if y > 0.5 and inst.w[i][j] == 0:
                    bad.append(f"(1.5) Y[{i},{j},{t}]=1 but item {i} is incompatible with "
                               f"machine {j}")
                if zval(i, j, t) > y + TOL:
                    bad.append(f"(1.6) Z[{i},{j},{t}]=1 while Y[{i},{j},{t}]=0")
                if zval(i, j, t - 1) + zval(i, j, t) > 1 + TOL:
                    bad.append(f"(1.7) carry-over of item {i} on machine {j} continues "
                               f"through t={t}")
            if mode == MODE_AUDITED and zval(i, j, inst.T) > TOL:
                bad.append(f"(D2) Z[{i},{j},{inst.T}]=1 but the terminal carry-over is "
                           f"fixed to 0 in {mode}")
    for j in range(inst.M):
        for t in range(1, inst.T + 1):
            tot = sum(zval(i, j, t) for i in range(inst.N))
            if tot > 1 + TOL:
                bad.append(f"(1.11) machine {j} period {t} has {tot:g} carry-overs")
    return BinaryCheck(ok=not bad, violations=bad)


def solve_independent_lp(inst: Instance, mode: str,
                         Y: Sequence[Sequence[Sequence[float]]],
                         Z: Sequence[Sequence[Sequence[float]]],
                         ) -> Tuple[str, float, Optional[Dict[str, Any]], List[str]]:
    """Solve the continuous problem for a FIXED binary pattern, built from scratch.

    Returns (status, objective, values, binary_violations).
    status is one of "optimal", "infeasible", "binary_infeasible", "status=N".
    """
    N, M, T = inst.N, inst.M, inst.T

    # ---- step 1: binary logic first, without any LP
    bcheck = check_binary_logic(inst, mode, Y, Z)
    if not bcheck.ok:
        return "binary_infeasible", float("nan"), None, bcheck.violations

    # ---- step 2: our own column layout, X/I/L only
    n_x = N * M * T
    n_i = N * T
    n_l = N * T
    n = n_x + n_i + n_l

    def cx(i, j, t):
        return (i * M + j) * T + (t - 1)

    def ci(i, t):
        return n_x + i * T + (t - 1)

    def cl(i, t):
        return n_x + n_i + i * T + (t - 1)

    cost = np.zeros(n)
    for i in range(N):
        for t in range(1, T + 1):
            for j in range(M):
                cost[cx(i, j, t)] = inst.p[i]
            cost[ci(i, t)] = inst.h[i]
            cost[cl(i, t)] = inst.l[i]
    # setup cost is a constant here: Y is data, not a variable
    setup_constant = 0.0
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                setup_constant += inst.f[i] * float(Y[i][j][t - 1])

    # ---- step 3: constraints
    rows_eq: List[Dict[int, float]] = []
    rhs_eq: List[float] = []
    rows_ub: List[Dict[int, float]] = []
    rhs_ub: List[float] = []

    # (1.2) I_i(t-1) + sum_j X_ijt + L_it = d_it + I_it
    for i in range(N):
        for t in range(1, T + 1):
            row: Dict[int, float] = {}
            for j in range(M):
                row[cx(i, j, t)] = 1.0
            row[cl(i, t)] = 1.0
            row[ci(i, t)] = -1.0
            left_const = inst.I0[i] if t == 1 else 0.0
            if t > 1:
                row[ci(i, t - 1)] = 1.0
            rhs = inst.d[i][t - 1] - left_const
            rows_eq.append(row)
            rhs_eq.append(rhs)

    # (1.3) L_it <= d_it
    for i in range(N):
        for t in range(1, T + 1):
            rows_ub.append({cl(i, t): 1.0})
            rhs_ub.append(inst.d[i][t - 1])

    # (1.4) sum_i (s_i Y_ijt + b_i X_ijt) <= c_jt   with Y constant
    for j in range(M):
        for t in range(1, T + 1):
            row = {}
            const = 0.0
            for i in range(N):
                row[cx(i, j, t)] = inst.b[i]
                const += inst.s[i] * float(Y[i][j][t - 1])
            rows_ub.append(row)
            rhs_ub.append(inst.c[j][t - 1] - const)

    # (1.8) X_ijt <= U_ijt (Y_ijt + Z_ij(t-1))   with both binaries constant
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                u = independent_upper_bound(inst, mode, i, j, t)
                z_prev = 0.0 if t == 1 else float(Z[i][j][t - 2])
                cap = u * (float(Y[i][j][t - 1]) + z_prev)
                rows_ub.append({cx(i, j, t): 1.0})
                rhs_ub.append(cap)

    # (1.9) X_ijt >= m_i (Y_ijt - Z_ijt)   ->  -X_ijt <= -m_i (Y_ijt - Z_ijt)
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                rhs = -inst.m[i] * (float(Y[i][j][t - 1]) - float(Z[i][j][t - 1]))
                rows_ub.append({cx(i, j, t): -1.0})
                rhs_ub.append(rhs)

    # (1.10) X_ijt + X_ij(t+1) >= m_i Z_ijt   ->  -X_ijt - X_ij(t+1) <= -m_i Z_ijt
    for i in range(N):
        for j in range(M):
            for t in range(1, T):
                rows_ub.append({cx(i, j, t): -1.0, cx(i, j, t + 1): -1.0})
                rhs_ub.append(-inst.m[i] * float(Z[i][j][t - 1]))

    def to_matrix(rows: List[Dict[int, float]]) -> Optional[coo_matrix]:
        if not rows:
            return None
        r_idx: List[int] = []
        c_idx: List[int] = []
        vals: List[float] = []
        for r, row in enumerate(rows):
            for c, v in row.items():
                if v != 0.0:
                    r_idx.append(r)
                    c_idx.append(c)
                    vals.append(float(v))
        return coo_matrix((vals, (r_idx, c_idx)), shape=(len(rows), n)).tocsr()

    A_eq = to_matrix(rows_eq)
    b_eq = np.array(rhs_eq)
    A_ub = to_matrix(rows_ub)
    b_ub = np.array(rhs_ub)

    bounds = [(0.0, None)] * n
    res = linprog(c=cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status == 2:
        return "infeasible", float("nan"), None, []
    if res.status != 0:
        return f"status={res.status}", float("nan"), None, []

    x = res.x
    X = [[[x[cx(i, j, t)] for t in range(1, T + 1)] for j in range(M)] for i in range(N)]
    I = [[x[ci(i, t)] for t in range(1, T + 1)] for i in range(N)]
    L = [[x[cl(i, t)] for t in range(1, T + 1)] for i in range(N)]
    return "optimal", float(res.fun) + setup_constant, {"X": X, "I": I, "L": L}, []


def enumerate_independent(inst: Instance, mode: str, max_configs: int = 4096
                          ) -> Dict[str, Any]:
    """Exhaustive enumeration of the binary space using ONLY the independent LP path.

    Binary slots follow the same definition as everywhere else: Z_ij0 is fixed by (1.13),
    and Z_ijT is fixed by D2 under audited_v1.
    """
    import itertools

    slots: List[Tuple[str, int, int, int]] = []
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                slots.append(("Y", i, j, t))
                if mode == MODE_PAPER or t != inst.T:
                    slots.append(("Z", i, j, t))
    total = 2 ** len(slots)
    if total > max_configs:
        raise ValueError(f"enumeration needs {total} configurations (> cap {max_configs})")

    best = float("inf")
    best_assign = None
    n_optimal = 0
    n_infeasible = 0
    n_binary_infeasible = 0
    for bits in itertools.product((0, 1), repeat=len(slots)):
        assign = {slots[k]: int(bits[k]) for k in range(len(slots))}
        Y = [[[0.0] * inst.T for _ in range(inst.M)] for _ in range(inst.N)]
        Z = [[[0.0] * inst.T for _ in range(inst.M)] for _ in range(inst.N)]
        for (kind, i, j, t), v in assign.items():
            if kind == "Y":
                Y[i][j][t - 1] = float(v)
            else:
                Z[i][j][t - 1] = float(v)
        status, obj, _, viol = solve_independent_lp(inst, mode, Y, Z)
        if status == "optimal":
            n_optimal += 1
            if obj < best - 1e-12:
                best = obj
                best_assign = dict(assign)
        elif status == "infeasible":
            n_infeasible += 1
        elif status == "binary_infeasible":
            n_binary_infeasible += 1
    return {"n_slots": len(slots), "n_configs": total, "n_optimal": n_optimal,
            "n_infeasible": n_infeasible, "n_binary_infeasible": n_binary_infeasible,
            "best_objective": best, "best_assignment": best_assign, "slots": slots}


def evaluate_from_model_solution(inst: Instance, mode: str, sol) -> Dict[str, Any]:
    """Take Y and Z from an already-solved plan and re-evaluate it through the independent
    path. This is the key cross-check: the MILP's own binaries are re-examined by code that
    shares no constraint construction with it.
    """
    Y = [[[float(sol.Y[i, j, t]) for t in range(inst.T)] for j in range(inst.M)]
         for i in range(inst.N)]
    Z = [[[float(sol.Z[i, j, t]) for t in range(inst.T)] for j in range(inst.M)]
         for i in range(inst.N)]
    status, obj, vals, viol = solve_independent_lp(inst, mode, Y, Z)
    out: Dict[str, Any] = {"status": status, "objective": obj, "binary_violations": viol}
    if status == "optimal":
        out["cost"] = independent_cost_breakdown(
            inst, vals["X"], Y, Z, vals["I"], vals["L"])
        # also recompute the cost of the plan's OWN X/I/L, which must match the solver's
        out["cost_of_solver_continuous"] = independent_cost_breakdown(
            inst,
            [[[float(sol.X[i, j, t]) for t in range(inst.T)] for j in range(inst.M)]
             for i in range(inst.N)],
            Y, Z,
            [[float(sol.I[i, t]) for t in range(inst.T)] for i in range(inst.N)],
            [[float(sol.L[i, t]) for t in range(inst.T)] for i in range(inst.N)])
    return out
