"""
Independent feasibility / optimality checker for stage 01 / 01b.

This module deliberately does NOT import or reuse the constraint matrix built in
`lsp_model.build_model`. Every business rule is restated from scratch, in scalar Python,
directly from the paper's equations (1.2)-(1.13). The point of a second implementation is
that a mistake in the assembler cannot hide behind the same mistake here.

Stage 01b change
----------------
In stage 01 this checker imported `activation_upper_bounds` and `cost_breakdown` from
`lsp_model`, so the (1.8) upper bound and the four cost components were NOT independently
checked: a wrong U or a wrong cost coefficient would have been reproduced here. Both are now
recomputed locally by the functions `_independent_upper_bound` and
`_independent_cost_breakdown` below, and the module no longer imports either helper.

Checks performed
  1.  domain            : X, Y, Z, I, L non-negative; Y, Z binary; Z_ij0 == 0; Z_ijT == 0
                          when required by the mode.
  2.  balance (1.2)     : I_{t-1} + sum_j X_ijt + L_it == d_it + I_it, restated and with
                          the residual reported.
  3.  lost-sales ub (1.3)
  4.  capacity (1.4)
  5.  compatibility (1.5)
  6.  carry-over logic (1.6), (1.7), (1.11)
  7.  activation (1.8)
  8.  minimum lot (1.9), (1.10)
  9.  cost identity     : the four cost components recomputed from the variable values must
                          match the objective the solver reported.
  10. diagnostic only   : least-lost-sales / least-inventory attribution of the balance.

The "max violation" is the largest absolute residual over all checks, in cost or quantity
units as appropriate; per-rule detail is always returned so a failure can be localised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsp_model import MODE_AUDITED, MODE_PAPER, Instance, Solution

TOL = 1e-6


# --------------------------------------------------------------------------------------
# Locally reimplemented helpers (NOT imported from lsp_model)
# --------------------------------------------------------------------------------------


def _independent_upper_bound(inst: Instance, mode: str, i: int, j: int, t: int) -> float:
    """Local restatement of U_ijt for constraint (1.8).

    paper_literal : min( sum_{t'>=t} d_it' , (c_jt - s_i)/b_i ), negative values kept.
    audited_v1    : c_jt / b_i.
    Intentionally written out again here rather than imported, so that the checker does not
    inherit a mistake made in the model's own implementation of the same formula.
    """
    if mode == MODE_PAPER:
        future = 0.0
        for tt in range(t, inst.T + 1):
            future += inst.d[i][tt - 1]
        return min(future, (inst.c[j][t - 1] - inst.s[i]) / inst.b[i])
    if mode == MODE_AUDITED:
        return inst.c[j][t - 1] / inst.b[i]
    raise ValueError(f"unknown mode {mode!r}")


def _independent_cost_breakdown(inst: Instance, X, Y, Z, I, L) -> Dict[str, float]:
    """Local restatement of objective (1.1), written as four separate accumulators."""
    setup = 0.0
    production = 0.0
    inventory = 0.0
    lost = 0.0
    for i in range(inst.N):
        for t in range(1, inst.T + 1):
            for j in range(inst.M):
                setup += inst.f[i] * float(Y[i, j, t - 1])
                production += inst.p[i] * float(X[i, j, t - 1])
            inventory += inst.h[i] * float(I[i, t - 1])
            lost += inst.l[i] * float(L[i, t - 1])
    return {"setup": setup, "production": production, "inventory": inventory,
            "lost_sales": lost, "total": setup + production + inventory + lost}


@dataclass
class CheckReport:
    ok: bool
    max_violation: float
    checks: Dict[str, Dict[str, Any]]
    cost_recomputed: Dict[str, float]
    cost_reported: float
    cost_delta: float
    least_lost_attribution: Dict[str, float]
    messages: List[str] = field(default_factory=list)

    def failed_checks(self) -> List[str]:
        return [k for k, v in self.checks.items() if not v["ok"]]

    def worst(self, k: int = 3) -> List[Tuple[str, float]]:
        pairs = [(name, info["max_violation"]) for name, info in self.checks.items()]
        pairs.sort(key=lambda kv: -kv[1])
        return pairs[:k]


def _scalar(x: np.ndarray, i: int, j: int, t: int) -> float:
    return float(x[i, j, t])


def check_solution(inst: Instance, sol: Solution, mode: str,
                   require_zT_zero: bool = True) -> CheckReport:
    """Restate every constraint from scratch and measure residuals."""
    N, M, T = inst.N, inst.M, inst.T
    # U_ijt is recomputed locally, not taken from the model
    upper = {(i, j, t): _independent_upper_bound(inst, mode, i, j, t)
             for i in range(inst.N) for j in range(inst.M) for t in range(1, inst.T + 1)}

    checks: Dict[str, Dict[str, Any]] = {}

    def record(name: str, viols: List[Tuple[float, str]]) -> None:
        viols = [(v, w) for v, w in viols if v > TOL]
        mx = max((v for v, _ in viols), default=0.0)
        checks[name] = {"ok": not viols, "max_violation": mx,
                        "n_violations": len(viols),
                        "worst": sorted(viols, key=lambda p: -p[0])[:5]}

    # ---- 1. domain
    viols = []
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                if sol.X[i, j, t - 1] < -TOL:
                    viols.append((-sol.X[i, j, t - 1], f"X[{i},{j},{t}]<0"))
                for nm, arr in (("Y", sol.Y), ("Z", sol.Z)):
                    v = arr[i, j, t - 1]
                    if v < -TOL or v > 1 + TOL:
                        viols.append((max(-v, v - 1), f"{nm}[{i},{j},{t}] outside [0,1]"))
                    elif abs(v - round(v)) > TOL:
                        viols.append((abs(v - round(v)), f"{nm}[{i},{j},{t}] not integral"))
            # (1.13) Z_ij0 == 0 is structural (the flat Z[.,.,0] is pinned); the mode may
            # additionally forbid declaring a carry-over out of the final period.
            if require_zT_zero and mode == MODE_AUDITED and sol.Z[i, j, T - 1] > TOL:
                viols.append((sol.Z[i, j, T - 1], f"Z[{i},{j},{T}]=1 but mode pins Z_ijT=0"))
        for t in range(1, T + 1):
            if sol.I[i, t - 1] < -TOL:
                viols.append((-sol.I[i, t - 1], f"I[{i},{t}]<0"))
            if sol.L[i, t - 1] < -TOL:
                viols.append((-sol.L[i, t - 1], f"L[{i},{t}]<0"))
    record("domain", viols)

    # ---- 2. inventory balance (1.2)
    viols = []
    for i in range(N):
        for t in range(1, T + 1):
            prev = inst.I0[i] if t == 1 else sol.I[i, t - 2]
            lhs = prev + sum(sol.X[i, j, t - 1] for j in range(M)) + sol.L[i, t - 1]
            rhs = inst.d[i][t - 1] + sol.I[i, t - 1]
            res = abs(lhs - rhs)
            if res > TOL:
                viols.append((res, f"i{i},t{t}: lhs={lhs:.10g} rhs={rhs:.10g}"))
    record("inv_balance", viols)

    # ---- 3. lost sales upper bound (1.3)
    viols = []
    for i in range(N):
        for t in range(1, T + 1):
            excess = sol.L[i, t - 1] - inst.d[i][t - 1]
            if excess > TOL:
                viols.append((excess, f"L[{i},{t}]={sol.L[i, t - 1]:.10g} > d={inst.d[i][t - 1]}"))
    record("lost_sales_ub", viols)

    # ---- 4. machine capacity (1.4)
    viols = []
    for j in range(M):
        for t in range(1, T + 1):
            used = sum(inst.s[i] * sol.Y[i, j, t - 1] + inst.b[i] * sol.X[i, j, t - 1]
                       for i in range(N))
            excess = used - inst.c[j][t - 1]
            if excess > TOL:
                viols.append((excess, f"j{j},t{t}: used={used:.10g} > c={inst.c[j][t - 1]}"))
    record("capacity", viols)

    # ---- 5. compatibility (1.5)
    viols = []
    for i in range(N):
        for j in range(M):
            if inst.w[i][j] == 0:
                for t in range(1, T + 1):
                    if sol.Y[i, j, t - 1] > TOL:
                        viols.append((sol.Y[i, j, t - 1], f"Y[{i},{j},{t}]=1 but w=0"))
    record("compatibility", viols)

    # ---- 6. carry-over logic (1.6), (1.7), (1.11)
    viols = []
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                # (1.6) Z_ijt <= Y_ijt : a carry-over requires a setup in the same period
                if sol.Z[i, j, t - 1] - sol.Y[i, j, t - 1] > TOL:
                    viols.append((sol.Z[i, j, t - 1] - sol.Y[i, j, t - 1],
                                  f"(1.6) Z[{i},{j},{t}]=1 with Y=0"))
                # (1.7) Z_ij(t-1) + Z_ijt <= 1 : no two consecutive carry-overs
                prev_z = 0.0 if t == 1 else sol.Z[i, j, t - 2]
                if prev_z + sol.Z[i, j, t - 1] - 1.0 > TOL:
                    viols.append((prev_z + sol.Z[i, j, t - 1] - 1.0,
                                  f"(1.7) consecutive carry-over i{i},j{j} at t{t}"))
    for j in range(M):
        for t in range(1, T + 1):
            tot = sum(sol.Z[i, j, t - 1] for i in range(N))
            if tot - 1.0 > TOL:
                viols.append((tot - 1.0, f"(1.11) j{j},t{t}: {tot:.0f} carry-overs"))
    record("carry_over_logic", viols)

    # ---- 7. activation (1.8)
    viols = []
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                u = upper[(i, j, t)]
                prev_z = 0.0 if t == 1 else sol.Z[i, j, t - 2]
                allowed = u * (sol.Y[i, j, t - 1] + prev_z)
                excess = sol.X[i, j, t - 1] - allowed
                if excess > TOL:
                    viols.append((excess,
                                  f"X[{i},{j},{t}]={sol.X[i, j, t - 1]:.10g} > U*(Y+Zprev)="
                                  f"{allowed:.10g} (U={u:.10g})"))
    record("activation", viols)

    # ---- 8. minimum production (1.9), (1.10)
    viols = []
    for i in range(N):
        for j in range(M):
            for t in range(1, T + 1):
                rhs = inst.m[i] * (sol.Y[i, j, t - 1] - sol.Z[i, j, t - 1])
                short = rhs - sol.X[i, j, t - 1]
                if short > TOL:
                    viols.append((short, f"(1.9) i{i},j{j},t{t}: X={sol.X[i, j, t - 1]:.10g} "
                                         f"< m(Y-Z)={rhs:.10g}"))
            for t in range(1, T):
                rhs = inst.m[i] * sol.Z[i, j, t - 1]
                short = rhs - (sol.X[i, j, t - 1] + sol.X[i, j, t])
                if short > TOL:
                    viols.append((short, f"(1.10) i{i},j{j},t{t}: X_t+X_t+1="
                                         f"{sol.X[i, j, t - 1] + sol.X[i, j, t]:.10g} "
                                         f"< m*Z={rhs:.10g}"))
    record("min_production", viols)

    # ---- 9. cost identity
    br = _independent_cost_breakdown(inst, sol.X, sol.Y, sol.Z, sol.I, sol.L)
    cost_delta = abs(br["total"] - sol.objective)
    if not np.isnan(sol.objective):
        checks["cost_identity"] = {"ok": cost_delta <= TOL * (1.0 + abs(sol.objective)),
                                   "max_violation": cost_delta, "n_violations": int(cost_delta > TOL),
                                   "worst": [(cost_delta, "recomputed vs reported objective")]}

    # ---- 10. diagnostic: least-lost-sales / least-inventory attribution
    inv_attr = 0.0
    lost_attr = 0.0
    for i in range(N):
        stock = inst.I0[i]
        for t in range(1, T + 1):
            avail = stock + sum(sol.X[i, j, t - 1] for j in range(M))
            served = min(avail, inst.d[i][t - 1])
            lost_attr += inst.l[i] * (inst.d[i][t - 1] - served)
            stock = avail - served
            inv_attr += inst.h[i] * stock
    setup_attr = sum(inst.f[i] * sol.Y[i, j, t - 1]
                     for i in range(N) for j in range(M) for t in range(1, T + 1))
    prod_attr = sum(inst.p[i] * sol.X[i, j, t - 1]
                    for i in range(N) for j in range(M) for t in range(1, T + 1))
    attr = {"setup": setup_attr, "production": prod_attr, "inventory": inv_attr,
            "lost_sales": lost_attr, "total": setup_attr + prod_attr + inv_attr + lost_attr}

    max_viol = max((info["max_violation"] for info in checks.values()), default=0.0)
    messages = []
    for name, info in checks.items():
        if not info["ok"]:
            messages.append(f"{name}: {info['n_violations']} violation(s), "
                            f"max {info['max_violation']:.3e}, worst {info['worst'][:2]}")
    if abs(attr["total"] - br["total"]) > 1e-6:
        messages.append(
            "diagnostic: least-lost-sales attribution differs from the plan's own I/L by "
            f"{attr['total'] - br['total']:.6g}; the plan deliberately holds extra inventory "
            "and/or declares extra lost sales relative to the least-cost attribution."
        )
    return CheckReport(ok=all(info["ok"] for info in checks.values()),
                       max_violation=max_viol, checks=checks,
                       cost_recomputed=br, cost_reported=sol.objective,
                       cost_delta=cost_delta, least_lost_attribution=attr,
                       messages=messages)
