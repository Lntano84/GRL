"""
Stage 02 instances: the mechanism tests with hand-computed expectations, and the four
seeded small instances that produce twelve disruption states.

Two families
------------
mechanism_instances()
    Tiny instances used to pin down individual behaviour of the repair heuristic. Their
    nominal plans are written out by hand (`mechanism_nominal_plans`), because the point is
    to exercise the repair logic on a known starting point, not to optimise.

small_instances()
    Four fixed-seed instances (2 items, 2 machines, 3 periods) whose nominal plans MUST come
    from an optimisation run. Parameters are frozen here and are not re-drawn if a result
    turns out to be uninteresting.

Disruption states
-----------------
Every small instance receives three disruptions:
    1. machine 0 down for 1 period
    2. machine 1 down for 2 periods
    3. both machines down for 1 period
Applied to the four instances this yields 12 disruption states. Each state has a small
instance behind it; the same 12 states are used for every release-set experiment.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from lsp_model import Instance

# --------------------------------------------------------------------------------------
# Disruption description
# --------------------------------------------------------------------------------------


class Disruption:
    """c'_jt = 0 for the affected (machine, period) pairs; derived bounds follow."""

    def __init__(self, name: str, down: Dict[int, List[int]], note: str = ""):
        self.name = name
        self.down = {int(j): sorted(int(t) for t in ts) for j, ts in down.items()}
        self.note = note

    def delta_for(self, j: int) -> int:
        """Number of leading periods during which machine j is unavailable."""
        ts = self.down.get(j, [])
        d = 0
        while (d + 1) in ts:
            d += 1
        return d

    def is_down(self, j: int, t: int) -> bool:
        return t in self.down.get(j, [])

    def describe(self) -> str:
        parts = ["machine %d down during %s" % (j, ts) for j, ts in sorted(self.down.items())]
        return "; ".join(parts) if parts else "no disruption"

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "down": {str(j): ts for j, ts in self.down.items()},
                "note": self.note, "description": self.describe()}


def apply_disruption(inst: Instance, dis: Disruption) -> Instance:
    """Build the perturbed instance: only the affected capacities change.

    Demands, costs, setup/production times, minimum lots, compatibility and initial
    inventory are carried over unchanged. The derived activation bound for audited_v1 is
    c'_jt / b_i, so a downed period automatically gets U' = 0; no separate rule is needed.
    """
    c = [[float(inst.c[j][t - 1]) for t in range(1, inst.T + 1)] for j in range(inst.M)]
    for j, ts in dis.down.items():
        for t in ts:
            if 1 <= t <= inst.T:
                c[j][t - 1] = 0.0
    return Instance(name="%s|%s" % (inst.name, dis.name), N=inst.N, M=inst.M, T=inst.T,
                    f=list(inst.f), p=list(inst.p), h=list(inst.h), l=list(inst.l),
                    s=list(inst.s), b=list(inst.b), m=list(inst.m),
                    d=[list(row) for row in inst.d], c=c,
                    w=[list(row) for row in inst.w], I0=list(inst.I0),
                    note="%s :: %s" % (inst.note, dis.describe()))


DISRUPTIONS = [
    Disruption("D1_m0_1p", {0: [1]}, "machine 0 down for period 1 only"),
    Disruption("D2_m1_2p", {1: [1, 2]}, "machine 1 down for periods 1-2"),
    Disruption("D3_both_1p", {0: [1], 1: [1]}, "both machines down for period 1"),
]


# --------------------------------------------------------------------------------------
# Mechanism instances (hand-written nominal plans)
# --------------------------------------------------------------------------------------


def _single(name: str, T: int, d, c, *, f, p, h, l, s, b, m, i0, note) -> Instance:
    """One item, one machine."""
    return Instance(name=name, N=1, M=1, T=T, f=[f], p=[p], h=[h], l=[l],
                    s=[s], b=[b], m=[m], d=[d], c=[c], w=[[1]], I0=[i0], note=note)


# Instance A and F parameters are the same as in stages 01/01b, so the disruption costs
# stated there carry over.
BASE_A = dict(f=2.0, p=1.0, h=1.0, l=10.0, s=1.0, b=1.0, m=1.0, i0=0.0)
BASE_F = dict(f=1.0, p=0.0, h=1.0, l=10.0, s=1.0, b=1.0, m=1.0, i0=0.0)


def mechanism_instances() -> Dict[str, Instance]:
    out: Dict[str, Instance] = {}

    # no disruption: the repair must return the nominal plan untouched
    out["M0_nodis"] = _single("M0_nodis", 1, [3.0], [5.0], note="instance A, no disruption", **BASE_A)

    # ordinary shutdown: instance A, the only period is lost
    out["M1_shutdown"] = _single("M1_shutdown", 1, [3.0], [5.0],
                                 note="instance A, machine down for the only period", **BASE_A)

    # carry-over broken, minimum lot fits in the recovery period.
    # F with demand (1,2), capacity (2,2): recovery needs s + b*m = 1 + 1 = 2 <= 2.
    out["M2_repair_fits"] = _single("M2_repair_fits", 2, [1.0, 2.0], [2.0, 2.0],
                                    note="instance F, machine down for period 1", **BASE_F)

    # carry-over broken, minimum lot does NOT fit in the recovery period.
    # demand (1,1), capacity (2,1): recovery needs 1 + 1 = 2 > 1, so production is cancelled.
    out["M3_repair_fails"] = _single("M3_repair_fails", 2, [1.0, 1.0], [2.0, 1.0],
                                     note="F with d=(1,1), c=(2,1); recovery cannot fit",
                                     **BASE_F)

    # the minimum-lot repair leaves production quantity on the table.
    # F with capacity (2,4): repair produces exactly m = 1 in period 2, but 2 are needed.
    out["M4_repair_headroom"] = _single("M4_repair_headroom", 2, [1.0, 2.0], [2.0, 4.0],
                                        note="F with c=(2,4); repair leaves X headroom",
                                        **BASE_F)

    # structural: another item already occupies the recovery period's capacity.
    # item 0 needs s0 + b0*m0 = 1 + 2 = 3 to be restored, but item 1 holds 2 of the 4 units.
    out["M5_capacity_occupied"] = Instance(
        name="M5_capacity_occupied", N=2, M=1, T=2,
        f=[1.0, 1.0], p=[0.0, 0.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[2.0, 1.0],
        d=[[0.0, 2.0], [0.0, 2.0]], c=[[2.0, 4.0]], w=[[1], [1]], I0=[0.0, 0.0],
        note="recovery period has only 4 units, item 1 keeps 2, item 0's m=2 needs 3")

    # structural: positive initial inventory survives a full-horizon shutdown.
    out["M6_initial_stock"] = _single("M6_initial_stock", 2, [2.0, 2.0], [1.0, 1.0],
                                      f=2.0, p=1.0, h=1.0, l=10.0, s=1.0, b=1.0,
                                      m=1.0, i0=3.0,
                                      note="initial stock 3 covers period 1 and part of 2")

    return out


def mechanism_nominal_plans() -> Dict[str, Dict[str, object]]:
    """Hand-written FEASIBLE nominal plans for the mechanism instances.

    Each entry holds X / Y / Z as nested lists indexed [i][j][t-1] plus the period the
    disruption hits. They are checked for feasibility by the independent checker before
    being used; a hand-written plan that is not feasible is a bug in this table, not in the
    repair heuristic.
    """
    return {
        # A optimal: produce 3
        "M0_nodis": {"X": [[[3.0]]], "Y": [[[1.0]]], "Z": [[[0.0]]]},
        "M1_shutdown": {"X": [[[3.0]]], "Y": [[[1.0]]], "Z": [[[0.0]]]},
        # F optimal under audited_v1: set up in period 1, carry over, produce 1 then 2
        "M2_repair_fits": {"X": [[[1.0, 2.0]]], "Y": [[[1.0, 0.0]]], "Z": [[[1.0, 0.0]]]},
        # F-like: set up in period 1, carry over, produce 1 then 1
        "M3_repair_fails": {"X": [[[1.0, 1.0]]], "Y": [[[1.0, 0.0]]], "Z": [[[1.0, 0.0]]]},
        # same shape as M2 so the repair leaves headroom
        "M4_repair_headroom": {"X": [[[1.0, 2.0]]], "Y": [[[1.0, 0.0]]], "Z": [[[1.0, 0.0]]]},
        # item 0 set up in period 1 with a carry-over; item 1 produces in period 2 using 2 units
        "M5_capacity_occupied": {
            "X": [[[0.0, 2.0]], [[0.0, 2.0]]],
            "Y": [[[1.0, 0.0]], [[0.0, 1.0]]],
            "Z": [[[1.0, 0.0]], [[0.0, 0.0]]],
        },
        # no production at all: initial stock 3 serves the demand
        "M6_initial_stock": {"X": [[[0.0, 0.0]]], "Y": [[[0.0, 0.0]]], "Z": [[[0.0, 0.0]]]},
    }


MECHANISM_DISRUPTIONS: Dict[str, Disruption] = {
    "M0_nodis": Disruption("none", {}, "no disruption"),
    "M1_shutdown": Disruption("m0_1p", {0: [1]}),
    "M2_repair_fits": Disruption("m0_1p", {0: [1]}),
    "M3_repair_fails": Disruption("m0_1p", {0: [1]}),
    "M4_repair_headroom": Disruption("m0_1p", {0: [1]}),
    "M5_capacity_occupied": Disruption("m0_1p", {0: [1]}),
    "M6_initial_stock": Disruption("all_T", {0: [1, 2]}),
}

# Hand-computed expectations for the mechanism tests.
MECHANISM_EXPECTATIONS: Dict[str, Dict[str, object]] = {
    "M0_nodis": {"repair_cost": 5.0, "unchanged": True,
                 "note": "no disruption -> every variable identical to the nominal plan"},
    "M1_shutdown": {"repair_cost": 30.0,
                    "note": "instance A with its only period lost: all demand lost"},
    "M2_repair_fits": {"repair_cost": 21.0, "recovery_Y": 1.0, "recovery_X": 1.0,
                       "note": "carry-over broken, recovery fits: Y=1, X=m=1 in period 2"},
    "M3_repair_fails": {"repair_cost": 20.0,
                        "note": "recovery needs 2 but only 1 is available: production cancelled"},
    "M4_repair_headroom": {"repair_cost": 21.0, "fixed_binary_lp_cost": 11.0,
                           "note": "repair stops at the minimum lot; fixing all Y,Z and "
                                   "re-optimising X,I,L recovers 10"},
}


# --------------------------------------------------------------------------------------
# Small seeded instances
# --------------------------------------------------------------------------------------

SMALL_SPEC = dict(N=2, M=2, T=3,
                  f=(2.0, 3.0), p=(0.0, 0.0), h=(1.0, 1.0), l=(10.0, 20.0),
                  s=(1.0, 1.0), b=(1.0, 1.0), m=(1.0, 2.0), I0=(1.0, 0.0),
                  d_low=0, d_high=3, c_low=4, c_high=6, seeds=(0, 1, 2, 3))


def small_instances() -> Dict[str, Instance]:
    """Four frozen instances. Parameters are generated once here and then fixed; they are
    NOT re-drawn if a disruption turns out to leave little room for improvement."""
    import random

    spec = SMALL_SPEC
    out: Dict[str, Instance] = {}
    for seed in spec["seeds"]:
        rng = random.Random(seed)
        N, M, T = spec["N"], spec["M"], spec["T"]
        d = [[float(rng.randint(spec["d_low"], spec["d_high"])) for _ in range(T)]
             for _ in range(N)]
        c = [[float(rng.randint(spec["c_low"], spec["c_high"])) for _ in range(T)]
             for _ in range(M)]
        out["S%d" % seed] = Instance(
            name="S%d" % seed, N=N, M=M, T=T,
            f=list(spec["f"]), p=list(spec["p"]), h=list(spec["h"]), l=list(spec["l"]),
            s=list(spec["s"]), b=list(spec["b"]), m=list(spec["m"]),
            d=d, c=c, w=[[1] * M for _ in range(N)], I0=list(spec["I0"]),
            note="fixed-seed small instance, seed=%d" % seed)
    return out


def disruption_states() -> List[Tuple[str, Instance, Disruption]]:
    """The 12 disruption states: (state_name, perturbed_instance, disruption)."""
    states: List[Tuple[str, Instance, Disruption]] = []
    for name, inst in small_instances().items():
        for dis in DISRUPTIONS:
            states.append(("%s|%s" % (name, dis.name), apply_disruption(inst, dis), dis))
    return states


TAU = 2
KAPPAS = (0, 1, 2)
