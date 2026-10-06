"""
`repair_minbatch_v1`: deterministic repair of a nominal plan after a machine breakdown.

Structure of Algorithm 1 in the paper's Appendix A, stated explicitly for our model variant
(`audited_v1`). It is a DETERMINISTIC STARTING POINT, not a strong baseline.

Two root causes found while building this, both of which shape the implementation
--------------------------------------------------------------------------------
(1) Cancelling the production of a period DOWNSTREAM of a broken carry-over. The nominal
    plan may serve a recovery period purely by riding on the setup of period delta, i.e. with
    Y = 0 and Z_{delta} = 1. Cancelling that carry-over leaves the recovery production with
    no activation at all, and (1.8) then forces it to 0. The repair must therefore treat
    every recovery-period production separately, not only the ones that nominally produced.

(2) A trial restoration can be infeasible even when the ledger arithmetic fits. Restoring a
    fresh setup charges s_i + b_i*X, so a plan whose recovery production was riding on a
    carry-over may fit at b_i*X but not at s_i + b_i*X. Conversely a plan can fit and still
    be pointless: with positive initial inventory, producing an item that the stock already
    covers only adds holding cost. Both cases are resolved by evaluating a concrete trial
    plan instead of trusting an arithmetic bound.

Rules implemented
-----------------
1. copy the nominal plan; cancel X, Y, Z of the disrupted machine(s) during 1..delta;
2. if delta < T, inspect the carry-over from period delta into delta+1;
3. for every recovery-period production that is no longer activated - either because its
   carry-over was broken or because it never had a setup - test restoring it as
   (Y = 1, X = m_i) while keeping the other products' arrangements in that period;
4. fits and is not worse than cancelling -> set Y = 1, X = m_i; otherwise cancel that
   product's production and setup in the recovery period. A follow-on carry-over out of the
   recovery period must be 0 whenever its production is cancelled; this is asserted and
   recorded.
5. recompute inventory and lost sales from the repaired production quantities:
       a_it = I^r_i(t-1) + sum_j X^r_ijt ,  I^r_it = max(a_it - d_it, 0) ,
       L^r_it = max(d_it - a_it, 0)

Guarantees
----------
* no disruption -> the nominal plan is returned unchanged;
* a disruption covering the whole horizon skips the recovery step;
* no MILP is called anywhere in this module (the trial evaluation is arithmetic);
* every cancellation, restoration and refusal is logged with its location and reason;
* the returned plan is screened for feasibility; if it fails, the repair RAISES instead of
  handing back an infeasible plan, because silently substituting an all-lost plan would
  corrupt every downstream statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsp_instances2 import Disruption
from lsp_model import Instance, Solution, cost_breakdown


class RepairFailure(RuntimeError):
    """Raised when the repaired plan is not feasible. Never silently swallowed."""


@dataclass
class RepairEvent:
    kind: str          # cancelled | restored | refused | note
    machine: Optional[int] = None
    period: Optional[int] = None
    item: Optional[int] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "machine": self.machine, "period": self.period,
                "item": self.item, "detail": self.detail}


@dataclass
class RepairResult:
    solution: Solution
    events: List[RepairEvent] = field(default_factory=list)
    unchanged: bool = False

    def log_text(self) -> str:
        if not self.events:
            return "(no events)"
        return " | ".join(
            "%s%s%s: %s" % (
                e.kind,
                "" if e.machine is None else " m%d" % e.machine,
                "" if e.period is None else " t%d" % e.period,
                e.detail)
            for e in self.events)


# --------------------------------------------------------------------------------------
# Arithmetic helpers (no MILP, no LP)
# --------------------------------------------------------------------------------------


def capacity_use(inst: Instance, X, Y, j: int, t: int,
                 exclude_item: Optional[int] = None) -> float:
    """Setup + production time consumed on machine j in period t (constraint (1.4))."""
    total = 0.0
    for i in range(inst.N):
        if i == exclude_item:
            continue
        total += inst.s[i] * float(Y[i][j][t - 1]) + inst.b[i] * float(X[i][j][t - 1])
    return total


def capacity_use_conservative(inst: Instance, X, Y, j: int, t: int,
                              exclude_item: Optional[int] = None) -> float:
    """Same, but charges s_i to every item with X > 0 in that period, whether or not it
    nominally has a setup. Used for the ledger report, because whether such an item really
    needs a new setup depends on a carry-over the repair does not control."""
    total = 0.0
    for i in range(inst.N):
        if i == exclude_item:
            continue
        y = float(Y[i][j][t - 1])
        x = float(X[i][j][t - 1])
        total += inst.s[i] * max(y, 1.0 if x > 0 else 0.0) + inst.b[i] * x
    return total


def is_activated(inst: Instance, X, Y, Z, i: int, j: int, t: int) -> bool:
    """True if production of (i,j,t) has an activation, i.e. Y_ijt = 1 or Z_ij(t-1) = 1."""
    if float(Y[i][j][t - 1]) > 0.5:
        return True
    if t > 1 and float(Z[i][j][t - 2]) > 0.5:
        return True
    return False


def _roll_forward(inst: Instance, X, Y, Z) -> Tuple[np.ndarray, np.ndarray]:
    """Rule 5 on its own: recompute I and L from the production quantities."""
    I = np.zeros((inst.N, inst.T))
    L = np.zeros((inst.N, inst.T))
    for i in range(inst.N):
        for t in range(1, inst.T + 1):
            prev = inst.I0[i] if t == 1 else I[i, t - 2]
            avail = prev + sum(float(X[i, j, t - 1]) for j in range(inst.M))
            I[i, t - 1] = max(avail - inst.d[i][t - 1], 0.0)
            L[i, t - 1] = max(inst.d[i][t - 1] - avail, 0.0)
    return I, L


def _cost_of(inst: Instance, X, Y, Z) -> float:
    I, L = _roll_forward(inst, X, Y, Z)
    return cost_breakdown(inst, X, Y, Z, I, L)["total"]


def _screen_feasible(inst: Instance, X, Y, Z) -> Tuple[bool, str]:
    """Internal feasibility screen mirroring the rules a candidate can break: capacity,
    activation, the two minimum-production rules and the carry-over logic. The independent
    checker stays authoritative for acceptance; this only steers the repair."""
    for j in range(inst.M):
        for t in range(1, inst.T + 1):
            used = capacity_use(inst, X, Y, j, t)
            if used > inst.c[j][t - 1] + 1e-9:
                return False, "capacity j%d,t%d: %g > %g" % (j, t, used, inst.c[j][t - 1])
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                u = inst.c[j][t - 1] / inst.b[i]           # audited_v1
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


def repair_minbatch_v1(inst: Instance, nominal: Solution, dis: Disruption,
                       mode: str = "audited_v1", strict: bool = True) -> RepairResult:
    """Repair `nominal` against the perturbed instance `inst` (already carrying c'_jt = 0 in
    the downed periods). The returned plan is complete and screened for feasibility."""
    events: List[RepairEvent] = []
    N, M, T = inst.N, inst.M, inst.T

    if not dis.down:
        events.append(RepairEvent("note", detail="no disruption: nominal plan returned as is"))
        out = Solution(status="nominal", objective=float(nominal.objective),
                       X=nominal.X.copy(), Y=nominal.Y.copy(), Z=nominal.Z.copy(),
                       I=nominal.I.copy(), L=nominal.L.copy(), mode=mode)
        return RepairResult(solution=out, events=events, unchanged=True)

    X = np.array(nominal.X, dtype=float)
    Y = np.array(nominal.Y, dtype=float)
    Z = np.array(nominal.Z, dtype=float)

    # ---- rule 1: cancel the disrupted machine(s) during their down periods
    for j in sorted(dis.down.keys()):
        for t in sorted(dis.down[j]):
            for i in range(N):
                removed = (X[i, j, t - 1], Y[i, j, t - 1], Z[i, j, t - 1])
                if any(v != 0.0 for v in removed):
                    events.append(RepairEvent(
                        "cancelled", machine=j, period=t, item=i,
                        detail="removed X=%g Y=%g Z=%g from the nominal plan" % removed))
                X[i, j, t - 1] = 0.0
                Y[i, j, t - 1] = 0.0
                Z[i, j, t - 1] = 0.0

    # ---- rules 2-4: recovery period of each machine
    for j in sorted(dis.down.keys()):
        delta = dis.delta_for(j)
        if delta >= T:
            events.append(RepairEvent("note", machine=j,
                                      detail="disruption covers the whole horizon "
                                             "(delta=%d=T): recovery step skipped" % delta))
            continue
        rec = delta + 1
        events.append(RepairEvent("note", machine=j, period=rec,
                                  detail="recovery period for machine %d (delta=%d)" % (j, delta)))

        # who nominally ran on this machine at the recovery period, and was the run carried
        # over from the disrupted period?
        affected: List[Tuple[int, str]] = []
        for i in range(N):
            had_prod = nominal.X[i, j, rec - 1] > 0.0
            carried = nominal.Z[i, j, delta - 1] != 0.0
            if not (had_prod or carried):
                continue
            if carried and not is_activated(inst, X, Y, Z, i, j, rec):
                affected.append((i, "carry-over from t=%d was broken" % delta))
            elif had_prod and not is_activated(inst, X, Y, Z, i, j, rec):
                affected.append((i, "production had no activation after the cancellation"))
        if not affected:
            events.append(RepairEvent("note", machine=j, period=rec,
                                      detail="no broken carry-over and no unactivated "
                                             "production: nothing to restore"))
            continue

        # deterministic priority: larger lost-sale exposure first, ties broken by item index
        affected.sort(key=lambda t: (-inst.l[t[0]] * sum(inst.d[t[0]]), t[0]))
        for i, reason in affected:
            need = inst.s[i] + inst.b[i] * inst.m[i]
            used_cons = capacity_use_conservative(inst, X, Y, j, rec, exclude_item=i)
            room = inst.c[j][rec - 1] - used_cons
            events.append(RepairEvent(
                "note", machine=j, period=rec, item=i,
                detail="%s; ledger: need s+b*m=%g, other items use %g, room=%g"
                       % (reason, need, used_cons, room)))

            # candidate A: restore exactly the minimum lot with a fresh setup
            Xa, Ya = X.copy(), Y.copy()
            Xa[i, j, rec - 1] = inst.m[i]
            Ya[i, j, rec - 1] = 1.0
            ok_a, why_a = _screen_feasible(inst, Xa, Ya, Z)
            cost_a = _cost_of(inst, Xa, Ya, Z) if ok_a else float("inf")

            # candidate B: cancel the production and the setup for this product
            Xb, Yb = X.copy(), Y.copy()
            Xb[i, j, rec - 1] = 0.0
            Yb[i, j, rec - 1] = 0.0
            ok_b, why_b = _screen_feasible(inst, Xb, Yb, Z)
            cost_b = _cost_of(inst, Xb, Yb, Z) if ok_b else float("inf")

            events.append(RepairEvent(
                "note", machine=j, period=rec, item=i,
                detail="trial restore=(feasible=%s%s, cost=%s) vs cancel=(feasible=%s%s, "
                       "cost=%s)" % (ok_a, "" if ok_a else " " + why_a,
                                     _fmt(cost_a), ok_b, "" if ok_b else " " + why_b,
                                     _fmt(cost_b))))

            if ok_a and cost_a <= cost_b + 1e-9:
                X, Y = Xa, Ya
                events.append(RepairEvent(
                    "restored", machine=j, period=rec, item=i,
                    detail="set Y=1, X=m_i=%g" % inst.m[i]))
            else:
                X, Y = Xb, Yb
                events.append(RepairEvent(
                    "refused", machine=j, period=rec, item=i,
                    detail="cancelled the recovery-period production: %s"
                           % ("restoring would not fit" if not ok_a
                              else "restoring would cost %s vs %s"
                                   % (_fmt(cost_a), _fmt(cost_b)))))

        # rule 4, second half: a cancelled recovery period must not declare a carry-over out
        for i in range(N):
            if X[i, j, rec - 1] == 0.0 and Z[i, j, rec - 1] != 0.0:
                events.append(RepairEvent(
                    "note", machine=j, period=rec, item=i,
                    detail="cleared Z=%g: the recovery period produces nothing, so it cannot "
                           "carry a run forward" % Z[i, j, rec - 1]))
                Z[i, j, rec - 1] = 0.0
            if X[i, j, rec - 1] == 0.0 and Y[i, j, rec - 1] != 0.0:
                events.append(RepairEvent(
                    "note", machine=j, period=rec, item=i,
                    detail="cleared Y=%g: no production to activate" % Y[i, j, rec - 1]))
                Y[i, j, rec - 1] = 0.0

    # ---- rule 5
    sol = Solution(status="repaired", objective=float("nan"),
                   X=X, Y=Y, Z=Z, I=np.zeros((N, T)), L=np.zeros((N, T)), mode=mode)
    sol.I, sol.L = _roll_forward(inst, X, Y, Z)
    br = cost_breakdown(inst, sol.X, sol.Y, sol.Z, sol.I, sol.L)
    sol.objective = br["total"]
    sol.meta = {"cost_breakdown": br, "repair": "repair_minbatch_v1"}

    # ---- guarantee: never return an infeasible repaired plan
    ok, why = _screen_feasible(inst, X, Y, Z)
    if not ok:
        msg = ("repair_minbatch_v1 produced an infeasible plan (%s); refusing to return it "
               "rather than silently substituting an all-lost plan" % why)
        if strict:
            raise RepairFailure(msg)
        sol.status = "INFEASIBLE_REPAIR"
        sol.meta["infeasible_reason"] = why
        events.append(RepairEvent("note", detail=msg))
    return RepairResult(solution=sol, events=events, unchanged=False)


def _fmt(v: float) -> str:
    return "inf" if not np.isfinite(v) else "%g" % v


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def binary_change_count(inst: Instance, Y_a, Y_b, tau: int) -> int:
    """Setup variables that flip inside the short-term horizon t <= tau, i.e. the quantity
    constrained by the stability budget."""
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, tau + 1):
                if abs(float(Y_a[i, j, t - 1]) - float(Y_b[i, j, t - 1])) > 0.5:
                    n += 1
    return n


def binary_change_count_full(inst: Instance, Y_a, Y_b) -> int:
    """Same count over the whole horizon. Deviations outside the short-term horizon must be
    visible, but they are not charged to the stability budget."""
    n = 0
    for i in range(inst.N):
        for j in range(inst.M):
            for t in range(1, inst.T + 1):
                if abs(float(Y_a[i, j, t - 1]) - float(Y_b[i, j, t - 1])) > 0.5:
                    n += 1
    return n
