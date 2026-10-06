"""
`repair_minbatch_v2`: deterministic repair of a nominal plan after a machine breakdown.

Replaces `repair_minbatch_v1` (stage 02/03). Stage 02/03 results are history and are not
recomputed; the differences are recorded per test in `stage04_report.md`.

What changed and why
--------------------
v1 decided each machine's recovery in a loop over machines and screened every trial against
the WHOLE instance. A trial for machine A could therefore be rejected because machine B was
still in a half-repaired state, and vice versa. v1 also cleared Y and Z at any position with
X = 0, which can delete a legal arrangement.

v2 uses two phases:

  Phase 1  cancel ALL production of the disrupted machines during their down periods, then
           resolve every recovery-period production that has lost its activation, on every
           machine, WITHOUT deciding anything yet. Any production that cannot be activated is
           cancelled. The result is a complete, feasible base plan - the current state against
           which every later candidate is judged.

  Phase 2  from that feasible base plan, consider each affected (item, machine, recovery
           period) and compare exactly two candidates:
               restore : Y = 1, X = m_i
               keep    : leave the cancellation
           Each candidate is evaluated against the CURRENT plan, which is feasible at every
           step. A candidate is accepted only if it is feasible AND not worse in cost.

Two further corrections:

  * a candidate is never rejected because an unrelated machine is mid-repair (phase 1
    guarantees the current plan is always complete and feasible);
  * the "X = 0 implies clear Y and Z" rule is GONE. Y = 1 with X = 0 can be legal: the
    minimum-lot rule (1.9) reads X >= m_i (Y - Z), which X = 0 satisfies whenever Z = 1, and
    (1.10) then requires production across the pair of periods. Deleting such a position can
    destroy a legal "set up now, produce next period" arrangement.

Still true
----------
* no disruption -> the nominal plan is returned unchanged;
* a disruption covering the whole horizon skips phase 2;
* no MILP is called: every evaluation is arithmetic;
* every cancellation, restoration and refusal is logged with its location and reason;
* the returned plan is screened, and an infeasible result RAISES rather than being silently
  replaced by an all-lost plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lsp_model import Instance, Solution, cost_breakdown

REPAIR_NAME = "repair_minbatch_v2"


class RepairFailure(RuntimeError):
    """Raised when the repaired plan is not feasible."""


@dataclass
class RepairEvent:
    kind: str          # cancelled | restored | kept | note
    phase: int = 0
    machine: Optional[int] = None
    period: Optional[int] = None
    item: Optional[int] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "phase": self.phase, "machine": self.machine,
                "period": self.period, "item": self.item, "detail": self.detail}


@dataclass
class RepairResult:
    solution: Solution
    events: List[RepairEvent] = field(default_factory=list)
    unchanged: bool = False
    changes_vs_v1: List[str] = field(default_factory=list)

    def log_text(self) -> str:
        if not self.events:
            return "(no events)"
        return " | ".join(
            "p%d %s%s%s: %s" % (
                e.phase, e.kind,
                "" if e.machine is None else " m%d" % e.machine,
                "" if e.period is None else " t%d" % e.period,
                e.detail)
            for e in self.events)


# --------------------------------------------------------------------------------------
# Arithmetic helpers
# --------------------------------------------------------------------------------------


def capacity_use(inst: Instance, X, Y, j: int, t: int,
                 exclude_item: Optional[int] = None) -> float:
    total = 0.0
    for i in range(inst.N):
        if i == exclude_item:
            continue
        total += inst.s[i] * float(Y[i][j][t - 1]) + inst.b[i] * float(X[i][j][t - 1])
    return total


def is_activated(inst: Instance, X, Y, Z, i: int, j: int, t: int) -> bool:
    if float(Y[i][j][t - 1]) > 0.5:
        return True
    if t > 1 and float(Z[i][j][t - 2]) > 0.5:
        return True
    return False


def _roll_forward(inst: Instance, X, Y, Z) -> Tuple[np.ndarray, np.ndarray]:
    I = np.zeros((inst.N, inst.T))
    L = np.zeros((inst.N, inst.T))
    for i in range(inst.N):
        for t in range(1, inst.T + 1):
            prev = inst.I0[i] if t == 1 else I[i, t - 2]
            avail = prev + sum(float(X[i, j, t - 1]) for j in range(inst.M))
            I[i, t - 1] = max(avail - inst.d[i][t - 1], 0.0)
            L[i, t - 1] = max(inst.d[i][t - 1] - avail, 0.0)
    return I, L


def cost_of(inst: Instance, X, Y, Z) -> float:
    I, L = _roll_forward(inst, X, Y, Z)
    return cost_breakdown(inst, X, Y, Z, I, L)["total"]


def screen_feasible(inst: Instance, X, Y, Z) -> Tuple[bool, str]:
    """Structural screen on the WHOLE instance. In v2 this is only ever called on a plan
    that is already complete and feasible, so it can no longer veto a single machine's
    recovery because of another machine's transient state."""
    for j in range(inst.M):
        for t in range(1, inst.T + 1):
            used = capacity_use(inst, X, Y, j, t)
            if used > inst.c[j][t - 1] + 1e-9:
                return False, "capacity j%d,t%d: %g > %g" % (j, t, used, inst.c[j][t - 1])
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                u = inst.c[j][t - 1] / inst.b[i]
                zprev = 0.0 if t == 1 else float(Z[i][j][t - 2])
                x = float(X[i][j][t - 1])
                y = float(Y[i][j][t - 1])
                z = float(Z[i][j][t - 1])
                if x > u * (y + zprev) + 1e-9:
                    return False, "activation i%d,j%d,t%d" % (i, j, t)
                if x < inst.m[i] * (y - z) - 1e-9:
                    return False, "minlot i%d,j%d,t%d" % (i, j, t)
                if z > y + 1e-9:
                    return False, "carry_implies_setup i%d,j%d,t%d" % (i, j, t)
                if inst.w[i][j] == 0 and y > 1e-9:
                    return False, "compatibility i%d,j%d,t%d" % (i, j, t)
            for t in range(1, inst.T):
                if (float(X[i][j][t - 1]) + float(X[i][j][t])
                        < inst.m[i] * float(Z[i][j][t - 1]) - 1e-9):
                    return False, "minlot_carry i%d,j%d,t%d" % (i, j, t)
    return True, ""


# --------------------------------------------------------------------------------------
# The repair
# --------------------------------------------------------------------------------------


def repair_minbatch_v2(inst: Instance, nominal: Solution, dis,
                       mode: str = "audited_v1", strict: bool = True
                       ) -> RepairResult:
    """Repair `nominal` against `inst` (already carrying c'_jt = 0 in the downed periods)."""
    events: List[RepairEvent] = []
    changes: List[str] = []
    N, M, T = inst.N, inst.M, inst.T

    if not dis.down:
        events.append(RepairEvent("note", phase=0,
                                  detail="no disruption: nominal plan returned as is"))
        out = Solution(status="nominal", objective=float(nominal.objective),
                       X=nominal.X.copy(), Y=nominal.Y.copy(), Z=nominal.Z.copy(),
                       I=nominal.I.copy(), L=nominal.L.copy(), mode=mode)
        return RepairResult(solution=out, events=events, unchanged=True)

    X = np.array(nominal.X, dtype=float)
    Y = np.array(nominal.Y, dtype=float)
    Z = np.array(nominal.Z, dtype=float)

    # ==================================================================================
    # PHASE 1a: cancel production of the disrupted machines during their down periods
    # ==================================================================================
    for j in sorted(dis.down.keys()):
        for t in sorted(dis.down[j]):
            for i in range(N):
                removed = (X[i, j, t - 1], Y[i, j, t - 1], Z[i, j, t - 1])
                if any(v != 0.0 for v in removed):
                    events.append(RepairEvent(
                        "cancelled", phase=1, machine=j, period=t, item=i,
                        detail="down period: removed X=%g Y=%g Z=%g" % removed))
                X[i, j, t - 1] = 0.0
                Y[i, j, t - 1] = 0.0
                Z[i, j, t - 1] = 0.0

    # ==================================================================================
    # PHASE 1b: reach a CONSISTENT, feasible base plan.
    #
    # Cancelling a down period removes production but leaves two kinds of dangling
    # structure, and neither can be resolved by the "is it activated" test alone:
    #
    #   (a) an OUT-FLOW carry-over with nothing left to feed. If Z_ij(t-1) was justified by
    #       production in periods t and t+1, cancelling the production in t can leave the
    #       carry-over declaring a run that produces nothing. It must go.
    #
    #   (b) an IN-FLOW carry-over that RE-ACTIVATES production which never nominally had a
    #       setup. Nominal production may ride on the in-flow carry-over with Y = 0; once
    #       (a) removes that carry-over, (1.9) reads X >= m_i*Y = 0 and Y is a free binary,
    #       so a solver could set Y = 1 and reinstate production whose setup time the plan
    #       never paid for. Restoring (1.4) then needs s_i + b_i*X <= c_jt, which can fail.
    #       Production whose NOMINAL setup and carry-over are both absent is therefore
    #       cancelled, with its whole chain.
    #
    # The rules below are applied repeatedly until nothing changes, i.e. to a fixpoint. This
    # is what makes the phase-1 result a genuine, complete, feasible plan.
    # ----------------------------------------------------------------------------------
    def _cancel_production(i, j, t, why):
        if X[i, j, t - 1] == 0.0 and Y[i, j, t - 1] == 0.0:
            return False
        events.append(RepairEvent(
            "cancelled", phase=1, machine=j, period=t, item=i,
            detail="%s: removed X=%g Y=%g" % (why, X[i, j, t - 1], Y[i, j, t - 1])))
        X[i, j, t - 1] = 0.0
        Y[i, j, t - 1] = 0.0
        return True

    for _round in range(T + 3):
        changed = False
        # (a) carry-overs whose minimum-lot requirement cannot be met
        for i in range(N):
            for j in range(M):
                for t in range(1, T + 1):
                    if Z[i, j, t - 1] == 0.0:
                        continue
                    rhs = 0.0
                    if Y[i, j, t - 1] > 0.5:
                        rhs += inst.m[i]
                    if t < T and Y[i, j, t] > 0.5:
                        rhs += inst.m[i]
                    if X[i, j, t - 1] + (X[i, j, t] if t < T else 0.0) < rhs - 1e-9:
                        events.append(RepairEvent(
                            "cancelled", phase=1, machine=j, period=t, item=i,
                            detail="carry-over no longer supportable (X_t + X_t+1 = %g < %g): "
                                   "removed Z=1" % (X[i, j, t - 1]
                                                    + (X[i, j, t] if t < T else 0.0), rhs)))
                        Z[i, j, t - 1] = 0.0
                        changed = True
        # (b) production that can no longer be activated without charging a setup the plan
        #     never paid for.
        #     Two situations reach this point:
        #       - nominal production whose coinciding setup was cancelled and whose in-flow
        #         carry-over is gone as well: nothing activates it;
        #       - nominal production riding on an in-flow carry-over that rule (a) removed,
        #         and with no coinciding setup: the carry-over that justified it is gone.
        #     In both cases the CURRENT setup is 0, so a solver could set Y = 1 for free and
        #     reinstate production whose setup time was never budgeted.
        for i in range(N):
            for j in range(M):
                for t in range(1, T + 1):
                    if X[i, j, t - 1] <= 0.0:
                        continue
                    # keep production that is still activated by a setup paid for NOW
                    if Y[i, j, t - 1] > 0.5:
                        continue
                    in_flow = t > 1 and Z[i, j, t - 2] > 0.5
                    if in_flow:
                        continue
                    if _cancel_production(i, j, t,
                                          "no current setup and no in-flow carry-over: "
                                          "reinstating it would charge a setup the plan "
                                          "never paid for"):
                        changed = True
        # (c) capacity: an over-loaded period is resolved by cancelling production that has
        #     no current activation, i.e. the same positions rule (b) targets
        for j in range(M):
            for t in range(1, T + 1):
                if capacity_use(inst, X, Y, j, t) <= inst.c[j][t - 1] + 1e-9:
                    continue
                for i in range(N):
                    if capacity_use(inst, X, Y, j, t) <= inst.c[j][t - 1] + 1e-9:
                        break
                    if X[i, j, t - 1] <= 0.0 or Y[i, j, t - 1] > 0.5:
                        continue
                    if t > 1 and Z[i, j, t - 2] > 0.5:
                        continue
                    if _cancel_production(i, j, t,
                                          "capacity overload in this period and no current "
                                          "activation here"):
                        changed = True
        if not changed:
            break

    ok, why = screen_feasible(inst, X, Y, Z)
    if not ok:
        msg = ("phase 1 of %s did not reach a feasible base plan (%s); this is an internal "
               "inconsistency, not a recovery decision" % (REPAIR_NAME, why))
        if strict:
            raise RepairFailure(msg)
        events.append(RepairEvent("note", phase=1, detail=msg))
    base_cost = cost_of(inst, X, Y, Z)
    events.append(RepairEvent("note", phase=1,
                              detail="feasible base plan established, cost=%g" % base_cost))

    # which (machine, recovery period, item) positions may be offered a restoration?
    # Every position whose NOMINAL production in the recovery period is now cancelled.
    recovery_candidates: List[Tuple[int, int, int]] = []
    for j in sorted(dis.down.keys()):
        delta = dis.delta_for(j)
        if delta >= T:
            events.append(RepairEvent("note", phase=1, machine=j,
                                      detail="disruption covers the whole horizon "
                                             "(delta=%d=T): phase 2 skipped" % delta))
            continue
        rec = delta + 1
        events.append(RepairEvent("note", phase=1, machine=j, period=rec,
                                  detail="recovery period (delta=%d)" % delta))
        for i in range(N):
            if nominal.X[i, j, rec - 1] > 0.0 and X[i, j, rec - 1] <= 0.0:
                recovery_candidates.append((j, rec, i))

    # ==================================================================================
    # PHASE 2: from the feasible base plan, restore or keep, one candidate at a time
    # ==================================================================================
    # deterministic priority: larger lost-sale exposure first, ties by (j, item)
    recovery_candidates.sort(key=lambda c: (-(inst.l[c[2]] * sum(inst.d[c[2]])),
                                            c[0], c[2]))
    for (j, rec, i) in recovery_candidates:
        cur_cost = cost_of(inst, X, Y, Z)

        Xr, Yr = X.copy(), Y.copy()
        Xr[i, j, rec - 1] = inst.m[i]
        Yr[i, j, rec - 1] = 1.0
        ok_r, why_r = screen_feasible(inst, Xr, Yr, Z)
        cost_r = cost_of(inst, Xr, Yr, Z) if ok_r else float("inf")

        keep_cost = cur_cost
        events.append(RepairEvent(
            "note", phase=2, machine=j, period=rec, item=i,
            detail="restore (feasible=%s%s, cost=%s) vs keep (cost=%s)"
                   % (ok_r, "" if ok_r else " " + why_r, _fmt(cost_r), _fmt(keep_cost))))

        if ok_r and cost_r <= keep_cost + 1e-9:
            X, Y = Xr, Yr
            events.append(RepairEvent("restored", phase=2, machine=j, period=rec, item=i,
                                      detail="set Y=1, X=m_i=%g" % inst.m[i]))
        else:
            events.append(RepairEvent(
                "kept", phase=2, machine=j, period=rec, item=i,
                detail="kept the cancellation (%s)"
                       % ("restoring is infeasible" if not ok_r
                          else "restoring would cost %s vs %s" % (_fmt(cost_r),
                                                                  _fmt(keep_cost)))))

    # ==================================================================================
    # finish
    # ==================================================================================
    sol = Solution(status="repaired", objective=float("nan"),
                   X=X, Y=Y, Z=Z, I=np.zeros((N, T)), L=np.zeros((N, T)), mode=mode)
    sol.I, sol.L = _roll_forward(inst, X, Y, Z)
    br = cost_breakdown(inst, sol.X, sol.Y, sol.Z, sol.I, sol.L)
    sol.objective = br["total"]
    sol.meta = {"cost_breakdown": br, "repair": REPAIR_NAME}

    ok, why = screen_feasible(inst, X, Y, Z)
    if not ok:
        msg = ("%s produced an infeasible plan (%s); refusing to return it" % (REPAIR_NAME, why))
        if strict:
            raise RepairFailure(msg)
        sol.status = "INFEASIBLE_REPAIR"
        sol.meta["infeasible_reason"] = why
        events.append(RepairEvent("note", phase=2, detail=msg))

    return RepairResult(solution=sol, events=events, unchanged=False, changes_vs_v1=changes)


def _fmt(v: float) -> str:
    return "inf" if not np.isfinite(v) else "%g" % v


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def binary_change_count(inst: Instance, Y_a, Y_b, tau: int) -> int:
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                if abs(float(Y_a[i, j, t - 1]) - float(Y_b[i, j, t - 1])) > 0.5:
                    n += 1
    return n


def binary_change_count_full(inst: Instance, Y_a, Y_b) -> int:
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                if abs(float(Y_a[i, j, t - 1]) - float(Y_b[i, j, t - 1])) > 0.5:
                    n += 1
    return n
