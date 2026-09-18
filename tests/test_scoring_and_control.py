"""Tests for the analytic scorers and the random-pruning control.

These two modules were extracted from scripts so that the Go/No-Go runner depends on the library
rather than on a script, and because both carry a semantic claim that has to be pinned:

* ``exposure_scores_delta`` is **signed**.  The model's activation probability ``2d(1-d)`` peaks at
  ``d = 0.5``, so a candidate can make a neighbour *less* likely to activate.  A scorer that clipped
  or abs()-ed that away would silently become a different method.
* ``random_pruning_greedy`` is a **matched control**, not a second treatment: it must cost the same
  as ``selective_greedy`` at the same shortlist size, and it must consult no score at all.  If it
  ever became cheaper or smarter it would stop controlling for anything.
"""

from __future__ import annotations

import networkx as nx
import pytest

from grl.algorithms.sequential_im import FILL_BUDGET, STOP_ON_NON_POSITIVE, selective_greedy
from grl.baselines import random_pruning_greedy
from grl.scoring import (
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
    out_edges,
    random_scores,
    rank_by_score,
)


def hub_graph() -> nx.DiGraph:
    """A node with two contributors, so the peak of 2d(1-d) is reachable."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.3)
    graph.add_edge("b", "t", weight=0.3)
    graph.add_edge("t", "z", weight=0.5)
    graph.add_edge("c", "t", weight=0.3)
    return graph


class CountingOracle:
    """An 'exact' oracle that returns a fixed score and counts evaluations."""

    def __init__(self, scores: dict) -> None:
        self.scores = scores
        self.evaluations = 0
        self.calls = 0

    def score(self, seeds, candidates, step=0):
        del seeds, step
        self.calls += 1
        self.evaluations += len(candidates)
        return {v: self.scores.get(v, 0.0) for v in candidates}


# ---------------------------------------------------------------------------------------------
# exposure_scores_delta
# ---------------------------------------------------------------------------------------------
def test_seeds_score_negative_infinity_so_they_cannot_be_reselected():
    graph = hub_graph()
    delta = {v: 0.0 for v in graph.nodes()}
    scores = exposure_scores_delta(graph, ["a", "b"], delta, seeds={"a"})
    assert scores[0] == float("-inf")
    assert scores[1] != float("-inf")


def test_the_score_is_signed():
    """The whole point: pushing a neighbour past its upper threshold must count against a seed."""
    graph = nx.DiGraph()
    graph.add_edge("w", "t", weight=0.4)
    # t is already near its peak exposure, where 2d(1-d) is falling
    delta = {"t": 0.5, "w": 0.0}
    scores = exposure_scores_delta(graph, ["w"], delta, seeds=set())
    assert scores[0] < 0.0, (
        "a candidate that over-exposes a neighbour must score negatively; clipping the sign would "
        "turn this into a different method"
    )


def test_the_score_is_positive_when_a_neighbour_is_under_exposed():
    graph = nx.DiGraph()
    graph.add_edge("w", "t", weight=0.2)
    delta = {"t": 0.0, "w": 0.0}
    scores = exposure_scores_delta(graph, ["w"], delta, seeds=set())
    assert scores[0] > 0.0


def test_two_hop_propagation_reaches_a_grandchild():
    """hops=2 must be able to score a candidate whose only effect is on a grandchild."""
    graph = nx.DiGraph()
    graph.add_edge("w", "m", weight=0.4)
    graph.add_edge("m", "g", weight=0.4)
    delta = {v: 0.0 for v in graph.nodes()}
    one_hop = exposure_scores_delta(graph, ["w"], delta, seeds=set(), hops=1)
    two_hop = exposure_scores_delta(graph, ["w"], delta, seeds=set(), hops=2)
    assert two_hop[0] > one_hop[0]


def test_score_is_deterministic():
    graph = hub_graph()
    delta = {"a": 0.1, "b": 0.2, "t": 0.35, "z": 0.0, "c": 0.0}
    first = exposure_scores_delta(graph, ["a", "c"], delta, seeds=set())
    second = exposure_scores_delta(graph, ["a", "c"], delta, seeds=set())
    assert first == second


def test_degree_scores_reads_out_degree():
    graph = hub_graph()
    assert degree_scores(graph, ["t", "z"]) == [0.0 + 1, 0.0]  # t has one out-edge, z none


def test_out_edges_handles_both_directions():
    digraph = nx.DiGraph()
    digraph.add_edge(1, 2, weight=0.5)
    assert list(out_edges(digraph, 1)) == [(2, 0.5)]
    assert list(out_edges(digraph, 2)) == []

    undirected = nx.Graph()
    undirected.add_edge(1, 2, weight=0.5)
    assert list(out_edges(undirected, 1)) == [(2, 0.5)]
    assert list(out_edges(undirected, 2)) == [(1, 0.5)]


def test_mean_exposure_state_is_a_mean_over_trials():
    graph = hub_graph()
    state = mean_exposure_state(graph, ["a"], trials=8, seed=1)
    assert set(state) == set(graph.nodes())
    assert all(v >= 0.0 for v in state.values())
    assert state["t"] > 0.0, "t receives a's weight, so its exposure is positive"


def test_rank_by_score_breaks_ties_deterministically():
    candidates = [5, 3, 9]
    scores = [1.0, 1.0, 0.5]
    assert rank_by_score(candidates, scores) == [3, 5, 9]


def test_random_scores_are_reproducible():
    assert random_scores([1, 2, 3], 7) == random_scores([1, 2, 3], 7)
    assert random_scores([1, 2, 3], 7) != random_scores([1, 2, 3], 8)


# ---------------------------------------------------------------------------------------------
# random_pruning_greedy -- the matched control
# ---------------------------------------------------------------------------------------------
def test_random_pruning_evaluates_exactly_one_shortlist_per_step():
    pool = list(range(20))
    exact = CountingOracle({v: float(v) for v in pool})
    result = random_pruning_greedy(pool, 3, exact, shortlist_size=5, random_seed=1)
    assert len(result.selected_seeds) == 3
    assert exact.evaluations == 15, "3 steps x 5 candidates"
    assert all(s["verified"] == 5 for s in result.steps)


def test_random_pruning_reports_no_predicted_score():
    """It has no predictor, and inventing one would hide that it is a control."""
    pool = list(range(10))
    exact = CountingOracle({v: float(v) for v in pool})
    result = random_pruning_greedy(pool, 2, exact, shortlist_size=4, random_seed=1)
    assert all(s["predicted_score"] is None for s in result.steps)
    assert all(s["control"] == "random_pruning" for s in result.steps)


def test_random_pruning_costs_the_same_as_selective_greedy_at_equal_m():
    """This is what makes it a control: equal cost, only the ranking differs."""
    pool = list(range(24))
    scores = {v: float(24 - v) for v in pool}

    control_oracle = CountingOracle(scores)
    random_pruning_greedy(pool, 3, control_oracle, shortlist_size=6, random_seed=3)

    class FixedLearned:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: scores[v] for v in candidates}

    exact = CountingOracle(scores)
    selective_greedy(pool, 3, FixedLearned(), exact, top_m=6)
    assert exact.evaluations == control_oracle.evaluations


def test_random_pruning_picks_the_best_of_its_own_shortlist():
    pool = list(range(12))
    exact = CountingOracle({v: float(v) for v in pool})
    result = random_pruning_greedy(pool, 1, exact, shortlist_size=4, random_seed=5)
    shortlist = result.steps[0]["shortlist"]
    assert result.selected_seeds[0] == max(shortlist), (
        "the control still refines exactly within its shortlist; only the shortlist is random"
    )


def test_random_pruning_obeys_a_declared_stopping_rule():
    pool = list(range(10))
    scores = {v: 5.0 - v for v in pool}
    exact = CountingOracle(scores)
    result = random_pruning_greedy(pool, 4, exact, shortlist_size=8, random_seed=2,
                                   stopping=STOP_ON_NON_POSITIVE)
    assert all(s["stopping_rule"] == "stop_on_non_positive" for s in result.steps)
    assert len(result.selected_seeds) <= 4


def test_random_pruning_rejects_a_zero_shortlist():
    with pytest.raises(ValueError, match="shortlist_size must be >= 1"):
        random_pruning_greedy(list(range(5)), 1, CountingOracle({}), shortlist_size=0)


def test_random_pruning_stops_when_the_pool_is_exhausted():
    pool = [1, 2]
    exact = CountingOracle({v: float(v) for v in pool})
    result = random_pruning_greedy(pool, 5, exact, shortlist_size=4, random_seed=1)
    assert len(result.selected_seeds) == 2


def test_the_control_is_actually_worse_than_a_ranking_on_a_rankable_instance():
    """Guard the guard: if the control were not handicapped it would prove nothing."""
    pool = list(range(30))
    strength = {v: float(v) for v in pool}

    class TrueOracle:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: strength[v] for v in candidates}

    class TrueLearned:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: strength[v] for v in candidates}

    ranked = selective_greedy(pool, 3, TrueLearned(), TrueOracle(), top_m=6)
    control = random_pruning_greedy(pool, 3, TrueOracle(), shortlist_size=6, random_seed=11)
    assert sum(strength[v] for v in ranked.selected_seeds) > \
        sum(strength[v] for v in control.selected_seeds)
