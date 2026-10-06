"""
Stage 01 instances.

  * hand_instances(): the seven hand-checkable instances A-G requested for this round.
  * micro_instances(): eight fixed-seed micro instances (2 items, 1 machine, 2 periods,
    small integers) covering tight capacity, incompatibility, zero demand and zero capacity.

All parameters are small integers so that the optimal cost can be checked by hand or by
exhaustive enumeration.
"""

from __future__ import annotations

from typing import Dict, List

from lsp_model import Instance


def hand_instances() -> Dict[str, Instance]:
    """Instances A-G. See stage01_report.md for the hand-computed expectations."""
    out: Dict[str, Instance] = {}

    def one(prefix: str, t: int, d, c, w, f, p, h, l, s, b, m, i0, note: str) -> Instance:
        """Single-item, single-machine instance (N = M = 1)."""
        return Instance(name=prefix, N=1, M=1, T=t, f=[f], p=[p], h=[h], l=[l],
                        s=[s], b=[b], m=[m], d=[d], c=[c], w=[w], I0=[i0], note=note)

    # Common parameters for A-E: one item, one machine, empty initial inventory,
    # setup cost 2, production cost 1, holding cost 1, lost sale cost 10,
    # setup time 1, unit production time 1, minimum lot 1.
    BASE = dict(f=2.0, p=1.0, h=1.0, l=10.0, s=1.0, b=1.0, m=1.0, i0=0.0)

    # A: ample capacity. Hand expectation 5 = produce 3 (setup 2 + production 3).
    out["A_ample"] = one("A_ample", 1, [3.0], [5.0], [1], note="one period, demand 3, capacity 5", **BASE)

    # B: capacity short. Hand expectation 14 = produce 2, lose 1 (2 + 2 + 10).
    out["B_short"] = one("B_short", 1, [3.0], [3.0], [1], note="one period, demand 3, capacity 3", **BASE)

    # C: item incompatible with the only machine. Hand expectation 30 = all lost.
    out["C_incompat"] = one("C_incompat", 1, [3.0], [5.0], [0], note="same as A but w=0", **BASE)

    # D: machine down. Hand expectation 30 = all lost.
    out["D_down"] = one("D_down", 1, [3.0], [0.0], [1], note="one period, demand 3, capacity 0", **BASE)

    # E: produce early and carry inventory. Hand expectation 8 = 2 + 3 production + 3 holding.
    out["E_early"] = one("E_early", 2, [0.0, 3.0], [5.0, 0.0], [1],
                         note="two periods, demand (0,3), capacity (5,0)", **BASE)

    # F: does a carried-over setup still pay the setup time in the next period?
    # Hand expectation recorded for this round: audited_v1 = 1, paper_literal = 11.
    out["F_carryover"] = one("F_carryover", 2, [1.0, 2.0], [2.0, 2.0], [1],
                             f=1.0, p=0.0, h=1.0, l=10.0, s=1.0, b=1.0, m=1.0, i0=0.0,
                             note="two periods, demand (1,2), capacity (2,2)")

    # G: can a terminal carry-over variable be used to escape the minimum lot size?
    # Hand expectation recorded for this round: audited_v1 = 10, paper_literal = 1.
    out["G_terminal_Z"] = one("G_terminal_Z", 1, [1.0], [1.0], [1],
                              f=1.0, p=0.0, h=1.0, l=10.0, s=0.0, b=1.0, m=2.0, i0=0.0,
                              note="one period, demand 1, capacity 1, m=2, s=0")
    return out


def stage01b_instances() -> Dict[str, Instance]:
    """Instances H and I added in stage 01b, plus the negative-control instance.

    These are designed to be DISCRIMINATING: each one separates the corrected model from a
    plausible wrong implementation, so passing them is evidence rather than a formality.

      H  separates U = c/b from the stage-01 audited_v1 bound min(sum of future demand, c/b).
         With m = 2 and demand 1, the legal plan must produce 2 and carry 1 unit of
         end-of-horizon inventory; min(., sum_t d) = 1 forbids exactly that plan.
      I  separates a correct tau-aware fix-and-optimize interface from one that freezes every
         period, including t > tau. Period 1 has zero capacity, so all production must happen
         in period 2.
      D_fix_X   negative control: the zero-capacity instance with X hard-fixed to 1, which
         must be reported infeasible.
    """
    out: Dict[str, Instance] = {}

    # H: minimum lot size larger than remaining demand -> end-of-horizon inventory.
    # Main model expectation: X = 2, I = 1, Y = 1, cost = f*1 + h*1 = 1 + 1 = 2.
    # A bound truncated by future demand (min(1, 3/1) = 1) would force X <= 1, so no setup
    # can respect (1.9), and the instance would cost 10 via total lost sales.
    out["H_minlot_stock"] = Instance(
        name="H_minlot_stock", N=1, M=1, T=1,
        f=[1.0], p=[0.0], h=[1.0], l=[10.0], s=[1.0], b=[1.0], m=[2.0],
        d=[[1.0]], c=[[3.0]], w=[[1]], I0=[0.0],
        note="one period, demand 1, capacity 3, m=2: must produce 2 and leave inventory 1")

    # I: outside the short-term horizon, Y must stay free.
    # Reference plan: produce nothing anywhere -> all demand lost -> cost 10.
    # With tau = 1 and Y empty, only Y_ij1 is frozen (to 0); Y_ij2 is free, so period 2 may
    # set up and produce 1: cost = f = 1.
    out["I_tau_freedom"] = Instance(
        name="I_tau_freedom", N=1, M=1, T=2,
        f=[1.0], p=[0.0], h=[1.0], l=[10.0], s=[1.0], b=[1.0], m=[1.0],
        d=[[0.0, 1.0]], c=[[0.0, 2.0]], w=[[1]], I0=[0.0],
        note="period 1 has zero capacity, so period 2 must be reachable")

    # J: releasing a variable must be able to change the answer.
    # Same parameters as instance A, but the reference plan produces nothing (cost 30).
    # tau = 1, Y empty -> Y_ij1 frozen to 0 -> cost 30; releasing that single Y -> cost 5.
    out["J_release_matters"] = Instance(
        name="J_release_matters", N=1, M=1, T=1,
        f=[2.0], p=[1.0], h=[1.0], l=[10.0], s=[1.0], b=[1.0], m=[1.0],
        d=[[3.0]], c=[[5.0]], w=[[1]], I0=[0.0],
        note="instance A parameters with an all-zero reference plan")

    # Negative control: instance D (zero capacity) with X hard-fixed to 1.
    out["D_fix_X"] = Instance(
        name="D_fix_X", N=1, M=1, T=1,
        f=[2.0], p=[1.0], h=[1.0], l=[10.0], s=[1.0], b=[1.0], m=[1.0],
        d=[[3.0]], c=[[0.0]], w=[[1]], I0=[0.0],
        note="zero capacity; with X fixed to 1 the fixture must be infeasible")

    return out


def _micro(name: str, params: dict, note: str) -> Instance:
    """Build a 2-item, 1-machine, 2-period instance.

    `params` holds cost/time/item vectors only. Demand, capacity and compatibility are
    named arguments, so a mismatch between them and the item vectors is caught here
    instead of silently producing a different instance than intended.
    """
    return Instance(name=name, N=2, M=1, T=2, note=note, **params)


def micro_instances() -> Dict[str, Instance]:
    """Eight fixed-seed micro instances: 2 items, 1 machine, 2 periods, small integers."""
    out: Dict[str, Instance] = {}

    # 0. textbook trade-off: two setups versus one setup plus inventory
    out["MI0_basic"] = _micro("MI0_basic", dict(
        f=[5.0, 4.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[20.0, 20.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 2.0], [2.0, 2.0]], c=[[6.0, 6.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "compatible; capacity 6 forces a real trade-off between setups and inventory")

    # 1. item 0 cannot be produced at all
    out["MI1_incompat"] = _micro("MI1_incompat", dict(
        f=[5.0, 5.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 1.0], [2.0, 2.0]], c=[[4.0, 5.0]], w=[[0], [1]], I0=[0.0, 0.0]),
        "item 0 incompatible with the only machine, so its demand must be lost")

    # 2. zero demand everywhere
    out["MI2_zero_demand"] = _micro("MI2_zero_demand", dict(
        f=[5.0, 5.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[0.0, 0.0], [0.0, 0.0]], c=[[4.0, 4.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "zero demand; the optimum must cost 0")

    # 3. initial inventory covers everything.
    #    Note on the choice of numbers: the balance (1.2) requires, for each item and period,
    #    I_it = I_i(t-1) + sum_j X_ijt + L_it - d_it with L_it allowed only up to d_it. Large
    #    initial inventory with no production is therefore NOT automatically feasible: stock
    #    that no demand can absorb has nowhere to go once L is capped. Here I0 covers every
    #    period's demand without ever needing production, so the optimum is 0.
    out["MI3_initial_inv"] = _micro("MI3_initial_inv", dict(
        f=[5.0, 5.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[10.0, 10.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 2.0], [1.0, 1.0]], c=[[4.0, 4.0]], w=[[1], [1]], I0=[4.0, 2.0]),
        "initial inventory exactly covers all demand; the optimum must cost 0")

    # 4. tight capacity: production must be concentrated in period 2
    out["MI4_tight"] = _micro("MI4_tight", dict(
        f=[3.0, 3.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[15.0, 15.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[2.0, 2.0],
        d=[[3.0, 3.0], [1.0, 2.0]], c=[[2.0, 8.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "capacity is tight in period 1 and loose in period 2; m=2 is active")

    # 5. zero capacity: everything is lost
    out["MI5_zero_cap"] = _micro("MI5_zero_cap", dict(
        f=[5.0, 5.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[7.0, 7.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 2.0], [1.0, 1.0]], c=[[0.0, 0.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "capacity 0 in both periods; all demand becomes lost sales")

    # 6. cheap lost sales, plenty of capacity
    out["MI6_cheap_lost"] = _micro("MI6_cheap_lost", dict(
        f=[2.0, 2.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[3.0, 3.0],
        s=[1.0, 1.0], b=[1.0, 1.0], m=[1.0, 1.0],
        d=[[2.0, 2.0], [1.0, 1.0]], c=[[5.0, 5.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "lost sales are cheap relative to setups; partial service may be optimal")

    # 7. larger setup time so that the capacity row really uses s_i*Y + b_i*X
    out["MI7_setup_time"] = _micro("MI7_setup_time", dict(
        f=[4.0, 4.0], p=[1.0, 1.0], h=[1.0, 1.0], l=[25.0, 25.0],
        s=[2.0, 1.0], b=[1.0, 2.0], m=[1.0, 1.0],
        d=[[2.0, 2.0], [3.0, 1.0]], c=[[3.0, 9.0]], w=[[1], [1]], I0=[0.0, 0.0]),
        "asymmetric setup times and production times exercise the capacity row")

    for name, inst in out.items():
        inst.validate()
        assert len(inst.d) == 2 and len(inst.c) == 1 and len(inst.w) == 2, \
            f"{name}: d/c/w were not attached to the instance correctly"
    return out
