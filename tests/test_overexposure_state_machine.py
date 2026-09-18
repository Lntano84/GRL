"""Regression tests for the two P0 model defects found in review.

Both were reproduced exactly before being fixed, and both are pinned here so they cannot return.
The reproduction script is ``scripts/audit/verify_p0_counterexamples.py``.

P0-1  A non-seed node must be able to go positive -> negative, and ``spread`` must count nodes
      that are positive at the END.  The old implementation marked a node ``settled`` on its
      first transition, so an already-positive node was never re-evaluated and could never be
      overexposed later.  That systematically under-counted overexposure: every seed contributes
      +1 that could never be cancelled by the damage it caused.

P0-2  The ``delta < 1`` guard made ``delta = 1`` negative even when the drawn window was
      ``[0, 1]``.  The rule is the literal ``kappa <= delta <= tau``.

A third behaviour is pinned too, because it is the reason P0-1 was subtle: exposure keeps
accumulating from nodes that have since turned negative.  The paper is explicit that
``A^in_t(v)`` retains previously activated neighbours, so ``ever_positive`` -- not
``positive`` -- is what drives ``delta``.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion import overexposure as oe


def test_p0_1_positive_node_can_later_be_overexposed():
    """s->a=0.4, s->b=0.5, b->a=0.4; a=[0.2,0.6], b=[0.2,0.9]; seed {s}.

    Round 1 promotes both a and b.  Round 2 lets b push a to delta = 0.8 > tau_a = 0.6, so a
    must turn negative and must stop counting towards the spread.
    """
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}

    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))

    assert run.delta["a"] == pytest.approx(0.8)
    assert run.negative == {"a"}
    assert run.positive == {"s", "b"}
    assert run.spread == 2
    # a was positive at some point, so it still feeds its neighbours' exposure
    assert run.ever_positive == {"s", "a", "b"}


def test_p0_1_ever_positive_is_a_superset_of_positive():
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    assert run.positive <= run.ever_positive
    assert run.negative.isdisjoint(run.positive)


def test_p0_2_delta_one_with_tau_one_activates():
    """s->v=1, v=[0.2, 1.0].  delta = 1 is inside the window, so v activates."""
    graph = nx.DiGraph()
    graph.add_edge("s", "v", weight=1.0)
    windows = {"s": (0.0, 1.0), "v": (0.2, 1.0)}

    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))

    assert run.delta["v"] == pytest.approx(1.0)
    assert run.positive == {"s", "v"}
    assert run.negative == set()
    assert run.spread == 2


def test_p0_2_boundary_is_literal_on_both_sides():
    """delta == kappa activates; delta > tau does not."""
    graph = nx.DiGraph()
    graph.add_edge("s", "v", weight=0.5)

    at_kappa = {"s": (0.0, 1.0), "v": (0.5, 1.0)}
    run = oe.run_overexposure(graph, ["s"], at_kappa, random.Random(0))
    assert "v" in run.positive, "delta == kappa is inside the window"

    above_tau = {"s": (0.0, 1.0), "v": (0.0, 0.4)}
    run = oe.run_overexposure(graph, ["s"], above_tau, random.Random(0))
    assert "v" in run.negative, "delta > tau is overexposure"


def test_seeds_are_never_overexposed():
    """A seed is positive by definition and stays so, even with a tiny window."""
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    windows = {"a": (0.9, 0.95), "b": (0.0, 1.0)}
    run = oe.run_overexposure(graph, ["a"], windows, random.Random(0))
    assert "a" in run.positive
    assert "a" not in run.negative


def test_ever_positive_is_the_set_that_drives_exposure():
    """The exact contract, stated directly.

    ``ever_positive`` is not "every node evaluated".  It is the set of nodes that were positively
    activated at some point, including those later turned negative, and it is that set -- not
    ``positive`` -- whose edges feed ``delta``.  Two consequences, both pinned below:

      * a node that goes straight from inactive to negative contributes nothing, because it was
        never in the set;
      * a node that was positive and later turned negative keeps contributing.
    """
    # Part 1: inactive -> negative, so it never promotes.
    graph = nx.DiGraph()
    graph.add_edge("s", "m", weight=0.8)
    graph.add_edge("m", "t", weight=0.8)
    windows = {"s": (0.0, 1.0), "m": (0.1, 0.5), "t": (0.1, 0.9)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    assert "m" in run.negative
    assert "m" not in run.ever_positive
    assert run.delta["t"] == pytest.approx(0.0), "a never-positive node promotes nothing"
    assert "t" not in run.positive


def test_two_seeds_acting_in_the_same_round_overexpose_immediately():
    """Both seeds fire in round 1, so m sees the full 0.6 at once and is negative from the start.

    This is the behaviour that made an earlier version of this test file wrong: the intent was to
    build a "positive first, negative later" case, but putting both contributors in the seed set
    means there is no "later".  Pinned because it is easy to get wrong again.
    """
    graph = nx.DiGraph()
    graph.add_edge("s1", "m", weight=0.3)
    graph.add_edge("s2", "m", weight=0.3)
    graph.add_edge("m", "t", weight=0.8)
    windows = {"s1": (0.0, 1.0), "s2": (0.0, 1.0), "m": (0.1, 0.5), "t": (0.1, 0.9)}

    run = oe.run_overexposure(graph, ["s1", "s2"], windows, random.Random(0))

    assert run.delta["m"] == pytest.approx(0.6)
    assert "m" in run.negative
    assert "m" not in run.ever_positive
    assert run.delta["t"] == pytest.approx(0.0)


def test_positive_then_negative_keeps_promoting_its_neighbours():
    """A staged construction where m is positive in one round and overexposed in a later one.

    The two exposure contributions to m must arrive in DIFFERENT rounds, so the first must travel
    one hop further than the second:

        s -> q (0.2)                       q promotes m in round 2
        s -> r (0.2) -> p (0.35)           p reaches m one round later, in round 3
        m window (0.2, 0.6), m -> t (0.8)

    Round 2: delta_m = 0.35, inside the window -> m is positive and promotes t (delta_t = 0.8).
    Round 3: delta_m = 0.35 + 0.35 = 0.7 > tau_m -> m turns negative, and t must keep the 0.8 it
    already received.

    An earlier version of this test put both contributors at distance one from m, so they fired in
    the same round and m was overexposed before ever being positive.  The extra hop is the whole
    point of the construction.
    """
    graph = nx.DiGraph()
    graph.add_edge("s", "q", weight=0.2)
    graph.add_edge("s", "r", weight=0.2)
    graph.add_edge("r", "p", weight=0.35)
    graph.add_edge("q", "m", weight=0.35)
    graph.add_edge("p", "m", weight=0.35)
    graph.add_edge("m", "t", weight=0.8)
    windows = {
        "s": (0.0, 1.0), "q": (0.1, 1.0), "r": (0.1, 1.0), "p": (0.1, 1.0),
        "m": (0.2, 0.6), "t": (0.1, 0.9),
    }

    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))

    assert run.delta["m"] == pytest.approx(0.7)
    assert "m" in run.ever_positive, "m must have been positive in round 2"
    assert "m" in run.negative, "m must be overexposed by the end"
    assert "m" not in run.positive
    assert run.delta["t"] == pytest.approx(0.8), (
        "m's influence on t must persist after m turned negative"
    )
    assert "t" in run.positive


def test_spread_counts_final_state_not_history():
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    assert run.spread == len(run.positive)
    assert run.spread != len(run.ever_positive), (
        "this construction is chosen so that the historical and final counts differ"
    )
