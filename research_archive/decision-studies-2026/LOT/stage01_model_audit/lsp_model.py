"""
Stage 01 model core: multi-item, multi-machine, multi-period capacitated lot sizing
with setups, setup carry-over, lost sales, minimum per-setup production quantity and
machine-item incompatibilities.

Two explicitly named modes are provided (see MODEL_SPEC.md):

  paper_literal : transcription of the published formulation (1.1)-(1.15) of
                  Lerouge, Lodi, Malaguti, Monaci, Focacci (arXiv:2605.27339v1, Sec. 2.1).
                  Nothing is added, nothing is silently repaired.

  audited_v1    : disclosed variant used as the candidate basis for later experiments.
                  Differences vs paper_literal:
                    D1. the production activation upper bound is U_ijt = c_jt / b_i
                        (the setup time s_i is still charged through capacity constraint (1.4));
                    D2. Z_ijT is fixed to 0 for every (i, j): no carry-over may be declared
                        out of the last period, where it would have no successor.

This module is an INDEPENDENT implementation from the published formulas. It is not a
reproduction of the authors' code, which was not available.

Array conventions (identical for every array, so that index arithmetic never shifts):
    X[i, j, t-1], Y[i, j, t-1], Z[i, j, t-1]   for t = 1..T
    I[i, t-1],    L[i, t-1]                    for t = 1..T
The internal flat vector additionally carries Z[i, j, 0], the pre-fixed initial carry-over.

Flat variable layout (fixed block order):
    X[i, j, t]  continuous >= 0
    Y[i, j, t]  binary
    Z[i, j, t]  binary, t = 0..T ; Z[i, j, 0] is pinned to 0 by (1.13)
    I[i, t]     continuous >= 0
    L[i, t]     continuous >= 0
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack as sp_vstack

MODE_PAPER = "paper_literal"
MODE_AUDITED = "audited_v1"
MODES = (MODE_PAPER, MODE_AUDITED)

TOL = 1e-6

# A binary slot is identified by (kind, i, j, t) with kind in {"Y", "Z"}.
Slot = Tuple[str, int, int, int]


# --------------------------------------------------------------------------------------
# Instance / solution containers
# --------------------------------------------------------------------------------------


@dataclass
class Instance:
    """A lot sizing instance. The paper indexes items 1..N; arrays here are 0-indexed."""

    name: str
    N: int
    M: int
    T: int
    f: List[float]  # setup cost per item
    p: List[float]  # unit production cost per item
    h: List[float]  # unit inventory cost per item
    l: List[float]  # lost sale cost per unit of unsatisfied demand, per item
    s: List[float]  # setup time per item
    b: List[float]  # unit production time per item
    m: List[float]  # minimum per-setup production quantity per item
    d: List[List[float]]  # d[i][t-1]
    c: List[List[float]]  # c[j][t-1]
    w: List[List[int]]  # w[i][j]
    I0: List[float]  # initial inventory
    note: str = ""

    def validate(self) -> None:
        assert self.N > 0 and self.M > 0 and self.T > 0
        for arr, nm in ((self.f, "f"), (self.p, "p"), (self.h, "h"), (self.l, "l"),
                        (self.s, "s"), (self.b, "b"), (self.m, "m"), (self.I0, "I0")):
            assert len(arr) == self.N, f"{nm} must have length N"
        assert len(self.d) == self.N and all(len(row) == self.T for row in self.d)
        assert len(self.c) == self.M and all(len(row) == self.T for row in self.c)
        assert len(self.w) == self.N and all(len(row) == self.M for row in self.w)
        for i in range(self.N):
            assert all(v >= 0 for v in self.d[i]), "demands must be non-negative"
            assert self.b[i] > 0, "unit production time must be positive"
        for j in range(self.M):
            assert all(v >= 0 for v in self.c[j]), "capacities must be non-negative"


@dataclass
class Solution:
    status: str
    objective: float
    X: np.ndarray  # (N, M, T)
    Y: np.ndarray  # (N, M, T)
    Z: np.ndarray  # (N, M, T)
    I: np.ndarray  # (N, T)
    L: np.ndarray  # (N, T)
    solver_message: str = ""
    mip_gap: Optional[float] = None
    solve_time_s: float = 0.0
    mode: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "optimal"


# --------------------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------------------


class Indexer:
    def __init__(self, inst: Instance):
        self.N, self.M, self.T = inst.N, inst.M, inst.T
        n_x = inst.N * inst.M * inst.T
        self.X0 = 0
        self.Y0 = self.X0 + n_x
        self.Z0 = self.Y0 + n_x
        self.I0 = self.Z0 + inst.N * inst.M * (inst.T + 1)
        self.L0 = self.I0 + inst.N * inst.T
        self.n = self.L0 + inst.N * inst.T

    def X(self, i: int, j: int, t: int) -> int:
        return self.X0 + (i * self.M + j) * self.T + (t - 1)

    def Y(self, i: int, j: int, t: int) -> int:
        return self.Y0 + (i * self.M + j) * self.T + (t - 1)

    def Z(self, i: int, j: int, t: int) -> int:
        return self.Z0 + (i * self.M + j) * (self.T + 1) + t

    def I(self, i: int, t: int) -> int:
        return self.I0 + i * self.T + (t - 1)

    def L(self, i: int, t: int) -> int:
        return self.L0 + i * self.T + (t - 1)


# --------------------------------------------------------------------------------------
# Row accumulator
# --------------------------------------------------------------------------------------


class _Rows:
    def __init__(self):
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []
        self.names: List[str] = []

    def add(self, coeffs: Dict[int, float], lb: float, ub: float, name: str) -> None:
        r = len(self.lb)
        for col, val in coeffs.items():
            if val != 0.0:
                self.rows.append(r)
                self.cols.append(col)
                self.vals.append(float(val))
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.names.append(name)

    def matrix(self, n_cols: int) -> csr_matrix:
        return coo_matrix((self.vals, (self.rows, self.cols)),
                          shape=(len(self.lb), n_cols)).tocsr()


@dataclass
class BuiltModel:
    inst: Instance
    mode: str
    idx: Indexer
    c: np.ndarray
    A: csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    var_lb: np.ndarray
    var_ub: np.ndarray
    integrality: np.ndarray
    row_names: List[str]
    upper_bounds: Dict[Tuple[int, int, int], float]
    notes: List[str]

    def row_index(self, name: str) -> int:
        return self.row_names.index(name)


# --------------------------------------------------------------------------------------
# Constraint (1.8) upper bounds
# --------------------------------------------------------------------------------------


def activation_upper_bounds(inst: Instance, mode: str) -> Tuple[Dict[Tuple[int, int, int], float], List[str]]:
    """U_ijt used by constraint (1.8), plus notes on how it was computed."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    notes: List[str] = []
    upper: Dict[Tuple[int, int, int], float] = {}
    clipped: List[Tuple[int, int, int, float]] = []
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                future_demand = sum(inst.d[i][t - 1:])
                if mode == MODE_PAPER:
                    cap_term = (inst.c[j][t - 1] - inst.s[i]) / inst.b[i]
                    val = min(future_demand, cap_term)
                else:
                    cap_term = inst.c[j][t - 1] / inst.b[i]
                    val = min(future_demand, cap_term)
                if val < 0.0:
                    clipped.append((i, j, t, val))
                    val = 0.0
                upper[(i, j, t)] = val
    if clipped:
        notes.append(
            "activation upper bound was negative for %d (i,j,t) combination(s) because "
            "c_jt < s_i; clipped to 0 since the right-hand side of (1.8) cannot be negative. "
            "Affected (i,j,t,raw value): %s" % (len(clipped), clipped[:8])
        )
    return upper, notes


# --------------------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------------------


def build_model(inst: Instance, mode: str) -> BuiltModel:
    inst.validate()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    idx = Indexer(inst)
    upper, notes = activation_upper_bounds(inst, mode)
    N, M, T = inst.N, inst.M, inst.T
    r = _Rows()

    # ---- objective (1.1)
    c = np.zeros(idx.n)
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                c[idx.Y(i, j, t)] += inst.f[i]
                c[idx.X(i, j, t)] += inst.p[i]
        for t in range(1, T + 1):
            c[idx.I(i, t)] += inst.h[i]
            c[idx.L(i, t)] += inst.l[i]

    # All rows are stored in the form  row_lb <= sum_k a_k x_k <= row_ub, i.e. the standard
    # form  lb <= A x <= ub. Every constraint below is therefore first moved into that form
    # (`lhs <= rhs` becomes `lhs - rhs <= 0`); the sign of each coefficient is dictated by
    # which side of the original equation the variable sits on.

    # ---- (1.2) inventory flow conservation  I_i(t-1) + sum_j X_ijt + L_it = d_it + I_it
    #      ->  I_it - I_i(t-1) - sum_j X_ijt - L_it = -d_it.
    #      Terms on the left of the original equation keep a positive sign (I_it, X_ijt,
    #      L_it); terms moved from the right get a negative sign (I_it on the right, d_it).
    #      For t = 1 the "previous" inventory is the constant parameter I_i0, so the
    #      rearranged row is   I_i1 - sum_j X_ij1 - L_i1 = I_i0 - d_i1.
    for i in range(N):
        for t in range(1, T + 1):
            co: Dict[int, float] = {}
            co[idx.I(i, t)] = 1.0                     # + I_it
            co[idx.L(i, t)] = -1.0                    # - L_it
            for j in range(M):
                co[idx.X(i, j, t)] = -1.0             # - sum_j X_ijt
            if t == 1:
                rhs = inst.I0[i] - inst.d[i][t - 1]   # + I_i0 - d_i1
            else:
                co[idx.I(i, t - 1)] = -1.0            # - I_i(t-1)
                rhs = -inst.d[i][t - 1]               # - d_it
            r.add(co, rhs, rhs, f"inv_balance[i{i},t{t}]")

    # ---- (1.3) lost sales bounded by demand:  L_it <= d_it
    for i in range(N):
        for t in range(1, T + 1):
            r.add({idx.L(i, t): 1.0}, -np.inf, inst.d[i][t - 1], f"lost_ub[i{i},t{t}]")

    # ---- (1.4) machine capacity:  sum_i (s_i Y_ijt + b_i X_ijt) <= c_jt
    for j in range(M):
        for t in range(1, T + 1):
            co = {}
            for i in range(N):
                co[idx.Y(i, j, t)] = inst.s[i]
                co[idx.X(i, j, t)] = inst.b[i]
            r.add(co, -np.inf, inst.c[j][t - 1], f"capacity[j{j},t{t}]")

    # ---- (1.5) machine-item compatibility:  w_ij = 0  =>  Y_ijt = 0
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                if inst.w[i][j] == 0:
                    r.add({idx.Y(i, j, t): 1.0}, -np.inf, 0.0, f"compat[i{i},j{j},t{t}]")

    # ---- (1.6) a carry-over exists only after a setup:  Z_ijt <= Y_ijt
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                r.add({idx.Z(i, j, t): 1.0, idx.Y(i, j, t): -1.0}, -np.inf, 0.0,
                      f"carry_implies_setup[i{i},j{j},t{t}]")

    # ---- (1.7) no two consecutive carry-overs:  Z_ij(t-1) + Z_ijt <= 1
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                r.add({idx.Z(i, j, t - 1): 1.0, idx.Z(i, j, t): 1.0}, -np.inf, 1.0,
                      f"no_two_carry[i{i},j{j},t{t}]")

    # ---- (1.8) production only with a setup in t or a carry-over from t-1:
    #           X_ijt <= U_ijt (Y_ijt + Z_ij(t-1))   ->   X - U Y - U Z_prev <= 0
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                u = upper[(i, j, t)]
                r.add({idx.X(i, j, t): 1.0,
                       idx.Y(i, j, t): -u,
                       idx.Z(i, j, t - 1): -u},
                      -np.inf, 0.0, f"prod_activation[i{i},j{j},t{t}]")

    # ---- (1.9) minimum production for a setup that is not carried over:
    #           X_ijt >= m_i (Y_ijt - Z_ijt)   ->   X_ijt + m_i Z_ijt - m_i Y_ijt >= 0
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                mi = inst.m[i]
                r.add({idx.X(i, j, t): 1.0,
                       idx.Z(i, j, t): mi,
                       idx.Y(i, j, t): -mi},
                      0.0, np.inf, f"minlot[i{i},j{j},t{t}]")

    # ---- (1.10) cross-period minimum production under carry-over:
    #            X_ijt + X_ij(t+1) >= m_i Z_ijt
    for i in range(N):
        for j in range(M):
            for t in range(1, T):
                r.add({idx.X(i, j, t): 1.0,
                       idx.X(i, j, t + 1): 1.0,
                       idx.Z(i, j, t): -inst.m[i]},
                      0.0, np.inf, f"minlot_carry[i{i},j{j},t{t}]")

    # ---- (1.11) at most one carry-over per machine and period:  sum_i Z_ijt <= 1
    for j in range(M):
        for t in range(1, T + 1):
            r.add({idx.Z(i, j, t): 1.0 for i in range(N)}, -np.inf, 1.0,
                  f"one_carry[j{j},t{t}]")

    # ---- domains (1.12)-(1.15)
    var_lb = np.zeros(idx.n)
    var_ub = np.full(idx.n, np.inf)
    integrality = np.zeros(idx.n, dtype=int)
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                var_ub[idx.Y(i, j, t)] = 1.0
                var_ub[idx.Z(i, j, t)] = 1.0
                integrality[idx.Y(i, j, t)] = 1
                integrality[idx.Z(i, j, t)] = 1
            var_ub[idx.Z(i, j, 0)] = 0.0  # (1.13)
            integrality[idx.Z(i, j, 0)] = 1

    if mode == MODE_AUDITED:
        for i in range(N):
            for j in range(M):
                var_ub[idx.Z(i, j, T)] = 0.0
        notes.append(
            "D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as "
            "published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, "
            "so the final setup can no longer be used to exit below the minimum quantity."
        )

    built = BuiltModel(inst=inst, mode=mode, idx=idx, c=c, A=r.matrix(idx.n),
                       row_lb=np.array(r.lb), row_ub=np.array(r.ub),
                       var_lb=var_lb, var_ub=var_ub, integrality=integrality,
                       row_names=list(r.names), upper_bounds=upper, notes=notes)

    # Structural guards: the indexer, the objective, the bounds and the matrix must all
    # agree on the number of variables, and every indexer method must stay in range. These
    # caught a real bug once, so they stay.
    assert built.A.shape == (len(built.row_lb), idx.n), \
        f"matrix {built.A.shape} disagrees with {len(built.row_lb)} rows / {idx.n} columns"
    assert len(built.c) == idx.n and len(built.var_lb) == idx.n \
        and len(built.var_ub) == idx.n and len(built.integrality) == idx.n
    max_col = 0
    for i in range(N):
        for j in range(M):
            max_col = max(max_col, idx.Z(i, j, 0))
            for t in range(1, T + 1):
                max_col = max(max_col, idx.X(i, j, t), idx.Y(i, j, t), idx.Z(i, j, t))
        for t in range(1, T + 1):
            max_col = max(max_col, idx.I(i, t), idx.L(i, t))
    assert max_col == idx.n - 1, f"indexer produced column {max_col} for {idx.n} variables"
    return built


# --------------------------------------------------------------------------------------
# Unpacking
# --------------------------------------------------------------------------------------


def unpack(x: np.ndarray, idx: Indexer) -> Tuple[np.ndarray, ...]:
    N, M, T = idx.N, idx.M, idx.T
    X = np.zeros((N, M, T))
    Y = np.zeros((N, M, T))
    Z = np.zeros((N, M, T))
    I = np.zeros((N, T))
    L = np.zeros((N, T))
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                X[i, j, t - 1] = x[idx.X(i, j, t)]
                Y[i, j, t - 1] = x[idx.Y(i, j, t)]
                Z[i, j, t - 1] = x[idx.Z(i, j, t)]
        for t in range(1, T + 1):
            I[i, t - 1] = x[idx.I(i, t)]
            L[i, t - 1] = x[idx.L(i, t)]
    return X, Y, Z, I, L


def _empty_solution(inst: Instance, status: str, message: str, mode: str,
                    elapsed: float) -> Solution:
    return Solution(status=status, objective=float("nan"),
                    X=np.zeros((inst.N, inst.M, inst.T)),
                    Y=np.zeros((inst.N, inst.M, inst.T)),
                    Z=np.zeros((inst.N, inst.M, inst.T)),
                    I=np.zeros((inst.N, inst.T)), L=np.zeros((inst.N, inst.T)),
                    solver_message=message, solve_time_s=elapsed, mode=mode)


# --------------------------------------------------------------------------------------
# MILP solve with optional fixing / free-Y neighbourhood
# --------------------------------------------------------------------------------------


def solve_milp(
    bm: BuiltModel,
    time_limit: Optional[float] = None,
    fix: Optional[Dict[int, float]] = None,
    free_Y_subset: Optional[Iterable[Tuple[int, int, int]]] = None,
    Y_reference: Optional[np.ndarray] = None,
    extra_ub: Optional[Dict[int, float]] = None,
) -> Solution:
    """Solve the MILP, optionally with hard fixings.

    fix            : {flat variable index: value}, hard equalities.
    free_Y_subset  : with Y_reference, every Y variable outside this subset is fixed to its
                     reference value. This is the fix-and-optimize interface of Sec. 3.1.
    extra_ub       : {flat variable index: upper bound} applied on top of the model bounds.
    """
    n = bm.idx.n
    lb = bm.var_lb.copy()
    ub = bm.var_ub.copy()
    for col, val in (extra_ub or {}).items():
        ub[col] = min(ub[col], float(val))

    fixed: Dict[int, float] = {}
    if free_Y_subset is not None:
        if Y_reference is None:
            raise ValueError("free_Y_subset requires Y_reference")
        allowed = set(free_Y_subset)
        for i in range(bm.inst.N):
            for j in range(bm.inst.M):
                for t in range(1, bm.inst.T + 1):
                    if (i, j, t) not in allowed:
                        fixed[bm.idx.Y(i, j, t)] = float(Y_reference[i, j, t - 1])
    for col, val in (fix or {}).items():
        fixed[col] = float(val)

    A = bm.A
    row_lb = bm.row_lb
    row_ub = bm.row_ub
    if fixed:
        k = list(fixed.keys())
        rows = np.arange(len(k))
        A_fix = coo_matrix((np.ones(len(k)), (rows, np.array(k))), shape=(len(k), n)).tocsr()
        A = sp_vstack([A, A_fix], format="csr")
        vals = np.array([fixed[col] for col in k], dtype=float)
        row_lb = np.concatenate([row_lb, vals])
        row_ub = np.concatenate([row_ub, vals])
        for col in k:
            lb[col] = fixed[col]
            ub[col] = fixed[col]

    options = {"time_limit": float(time_limit)} if time_limit else None
    t0 = time.perf_counter()
    res = milp(c=bm.c, constraints=LinearConstraint(A, lb=row_lb, ub=row_ub),
               integrality=bm.integrality, bounds=Bounds(lb=lb, ub=ub), options=options)
    elapsed = time.perf_counter() - t0

    if res.x is None:
        return _empty_solution(bm.inst, f"FAILED(status={res.status})", str(res.message),
                               bm.mode, elapsed)
    X, Y, Z, I, L = unpack(res.x, bm.idx)
    return Solution(
        status="optimal" if res.status == 0 else f"suboptimal(status={res.status})",
        objective=float(res.fun), X=X, Y=Y, Z=Z, I=I, L=L,
        solver_message=str(res.message),
        mip_gap=float(res.mip_gap) if res.mip_gap is not None else None,
        solve_time_s=elapsed, mode=bm.mode, meta={"n_fixed": len(fixed)})


# --------------------------------------------------------------------------------------
# Independent enumeration path (fix all binaries -> LP)
# --------------------------------------------------------------------------------------


def enumeration_space(inst: Instance, mode: str) -> List[Slot]:
    """Binary slots that are genuinely free.

    Z_ij0 is excluded because (1.13) pins it to 0. In audited_v1, Z_ijT is pinned to 0 too
    and is excluded for the same reason.
    """
    slots: List[Slot] = []
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                slots.append(("Y", i, j, t))
                if mode == MODE_PAPER or t != inst.T:
                    slots.append(("Z", i, j, t))
    return slots


def _slot_col(idx: Indexer, slot: Slot) -> int:
    kind, i, j, t = slot
    return idx.Y(i, j, t) if kind == "Y" else idx.Z(i, j, t)


def solve_lp_with_binaries_fixed(bm: BuiltModel,
                                 assignment: Dict[Slot, int]) -> Tuple[str, float, Optional[np.ndarray]]:
    """Independent continuous solve: pin every binary slot, then call plain `linprog`.

    Deliberately does not route through solve_milp, and rebuilds A_ub/A_eq/b from the
    assembled matrices with integrality dropped, so a mistake in the MILP path cannot
    cancel out against a mistake in this path.
    """
    idx = bm.idx
    lb = bm.var_lb.copy()
    ub = bm.var_ub.copy()
    for slot in enumeration_space(bm.inst, bm.mode):
        val = float(assignment.get(slot, 0))
        col = _slot_col(idx, slot)
        lb[col] = val
        ub[col] = val
    # structural pins (Z_ij0, and Z_ijT under audited_v1) keep their model bounds

    A = bm.A.tocsr()
    eq_mask = bm.row_lb == bm.row_ub
    if eq_mask.any():
        A_eq = A[eq_mask]
        b_eq = bm.row_lb[eq_mask]
    else:
        A_eq, b_eq = None, None

    # Every remaining row is `lo <= a.x <= hi` with at least one finite side. Convert each
    # finite side into its own row of the `A_ub x <= b_ub` form by slicing the original row
    # out of A (so column indices stay correct by construction, never rebuilt by hand):
    #     a.x <= hi   ->   (+a).x <= +hi
    #     a.x >= lo   ->   (-a).x <= -lo
    ub_blocks: List[csr_matrix] = []
    ub_rhs: List[float] = []
    for ridx in np.where(~eq_mask)[0]:
        lo, hi = bm.row_lb[ridx], bm.row_ub[ridx]
        row = A[ridx]
        if np.isfinite(hi):
            ub_blocks.append(row)
            ub_rhs.append(float(hi))
        if np.isfinite(lo):
            ub_blocks.append(-row)
            ub_rhs.append(float(-lo))
    if ub_blocks:
        A_ub = sp_vstack(ub_blocks, format="csr")
        b_ub = np.array(ub_rhs)
    else:
        A_ub, b_ub = None, None

    res = linprog(c=bm.c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs")
    if res.status == 0:
        return "optimal", float(res.fun), res.x
    if res.status == 2:
        return "infeasible", float("nan"), None
    return f"status={res.status}", float("nan"), None


def assignment_of_solution(sol: Solution, inst: Instance, mode: str) -> Dict[Slot, int]:
    """Binary assignment implied by a solved model, restricted to free slots."""
    assign: Dict[Slot, int] = {}
    for slot in enumeration_space(inst, mode):
        kind, i, j, t = slot
        val = sol.Y[i, j, t - 1] if kind == "Y" else sol.Z[i, j, t - 1]
        assign[slot] = int(round(val))
    return assign


def enumerate_optimal(bm: BuiltModel, max_configs: int = 4096) -> Dict[str, Any]:
    """Exhaustive enumeration of the free binary space, one LP solve per configuration."""
    slots = enumeration_space(bm.inst, bm.mode)
    n = len(slots)
    total = 2 ** n
    if total > max_configs:
        raise ValueError(f"enumeration needs {total} configurations (> cap {max_configs})")
    best = float("inf")
    best_assignment: Optional[Dict[Slot, int]] = None
    n_feasible = 0
    for bits in itertools.product((0, 1), repeat=n):
        assign = {slots[k]: int(bits[k]) for k in range(n)}
        status, val, _ = solve_lp_with_binaries_fixed(bm, assign)
        if status == "optimal":
            n_feasible += 1
            if val < best - 1e-12:
                best = val
                best_assignment = dict(assign)
    return {"n_binary_slots": n, "n_configs": total, "n_feasible": n_feasible,
            "best_objective": best, "best_assignment": best_assignment, "slots": slots}


# --------------------------------------------------------------------------------------
# Direct cost recomputation
# --------------------------------------------------------------------------------------


def cost_breakdown(inst: Instance, X, Y, Z, I, L) -> Dict[str, float]:
    setup = prod = inv = lost = 0.0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                setup += inst.f[i] * Y[i, j, t - 1]
                prod += inst.p[i] * X[i, j, t - 1]
        for t in range(1, inst.T + 1):
            inv += inst.h[i] * I[i, t - 1]
            lost += inst.l[i] * L[i, t - 1]
    return {"setup": setup, "production": prod, "inventory": inv, "lost_sales": lost,
            "total": setup + prod + inv + lost}
