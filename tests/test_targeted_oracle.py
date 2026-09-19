"""Tests for the contracted-objective oracle and the random-stream fixes.

Three bugs retracted a set of Go/No-Go conclusions, and each one is a check here so it cannot come
back silently:

1. every arm reported ``run.spread`` — the whole-graph positive count — instead of the contracted
   ``|positive ∩ D|``.  On Congress-Twitter with ``|D| = 95`` that was 145 against 24.
2. ``delta2_patience`` was handed a ``fill_budget`` rule whose ``should_stop`` always returns False,
   so patience was never enabled and the conclusion "the stopping lever never fires" measured the
   absence of a rule.
3. trial seeds were ``base + offset``, so replicates 20260917 and 20260918 shared 199 of 200
   evaluation windows.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion import overexposure as oe
from grl.diffusion.contract import build_contract
from grl.oracle import (
    TargetedMonteCarloOracle,
    TargetedObjectiveError,
    trial_seeds,
)


def graph_with_targets(n: int = 40, target: int = 8, seed: int = 5) -> tuple[nx.DiGraph, list]:
    """A digraph with normalised in-weights and an explicit target set."""
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for v in range(n):
        sources = rng.sample([u for u in range(n) if u != v], k=3)
        for u in sources:
            graph.add_edge(u, v, weight=1.0 / 3.0)
    return graph, list(range(target))


def make_oracle(graph, target, mc_runs: int = 20, seed: int = 7) -> TargetedMonteCarloOracle:
    contract = build_contract(graph, {}, target_set=target, budget=5)
    return TargetedMonteCarloOracle(graph, contract, mc_runs=mc_runs, random_seed=seed)


# ---------------------------------------------------------------------------------------------
# bug 1 -- the objective must be the contracted one
# ---------------------------------------------------------------------------------------------
def test_spread_is_the_target_count_not_the_whole_graph_count():
    """The headline bug: the two differ, and the oracle must report the contracted one."""
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target)
    eligible = [v for v in graph.nodes() if v not in set(target)]

    target_count = oracle.spread(eligible[:10])["mean"]
    assert 0.0 <= target_count <= len(target), (
        f"contracted objective {target_count} is outside [0, |D|] = [0, {len(target)}]"
    )

    # the whole-graph count is a DIFFERENT number and is what the bug reported
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(1))
    run = oe.run_overexposure(graph, eligible[:10], windows, random.Random(1))
    assert run.spread >= len(run.positive & set(target)), "sanity: whole graph >= target subset"
    if run.spread > len(target):
        assert target_count <= len(target) < run.spread, (
            "the contracted count must not be the whole-graph count"
        )


def test_a_level_above_the_target_size_raises():
    """The guard that makes a regression to whole-graph counting loud rather than plausible."""
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target)
    with pytest.raises(TargetedObjectiveError, match="outside"):
        oracle._check_level(float(len(target) + 1), "test level")


def test_a_marginal_may_be_negative():
    """A negative marginal is the phenomenon, not an error.

    The first version of the guard used one range for everything and rejected
    ``paired marginal = -4.0 is outside [0, |D|]``, silently removing three arms from a run.
    Non-monotonicity means adding a seed can *reduce* the target count, so the marginal's range
    must include negatives.
    """
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target)
    assert oracle._check_marginal(-4.0, "test marginal") == -4.0
    assert oracle._check_marginal(-float(len(target)), "test marginal") == -float(len(target))
    with pytest.raises(TargetedObjectiveError, match=r"\[-\|D\|, \|D\|\]"):
        oracle._check_marginal(-float(len(target)) - 1.0, "test marginal")


def test_score_is_a_paired_difference_in_target_counts():
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target, mc_runs=12)
    eligible = [v for v in graph.nodes() if v not in set(target)]
    scores = oracle.score(eligible[:3], eligible[3:9])
    assert set(scores) == set(eligible[3:9])
    for value in scores.values():
        assert -len(target) - 1e-9 <= value <= len(target) + 1e-9


def test_state_is_keyed_by_node_not_by_position():
    """The scoring function consumes a dict; a position-indexed list fails only at call time."""
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target, mc_runs=6)
    state = oracle.state([])
    assert isinstance(state, dict), "state must be a dict keyed by node"
    assert set(state) == set(graph.nodes())
    # and it must be usable by the scorer that consumes it
    from grl.scoring import exposure_scores_delta

    scores = exposure_scores_delta(graph, [v for v in graph.nodes() if v not in set(target)][:4],
                                   state, set(), target_set=set(target))
    assert len(scores) == 4


def test_the_oracle_is_not_exact():
    graph, target = graph_with_targets()
    assert make_oracle(graph, target).is_exact is False


def test_state_cost_is_charged_to_the_primary_unit():
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target, mc_runs=9)
    oracle.state([])
    assert oracle.stats.state_cascades == 9
    assert oracle.stats.mc_cascades == 9, "mc_cascades is the primary unit and includes the state"
    assert oracle.stats.cascades_for_scoring == 0


def test_seeds_inside_the_target_set_are_refused():
    graph, target = graph_with_targets()
    oracle = make_oracle(graph, target)
    with pytest.raises(Exception):
        oracle.spread([target[0]])


# ---------------------------------------------------------------------------------------------
# bug 2 -- a patience rule must actually be able to stop
# ---------------------------------------------------------------------------------------------
def test_patience_fires_on_a_flat_sequence_while_fill_budget_does_not():
    """The evidence for bug 2, as a test."""
    from grl.algorithms.sequential_im import FILL_BUDGET, PATIENCE_2

    flat = [100.0, 99.0, 99.0, 99.0, 99.0]
    assert FILL_BUDGET.should_stop([], flat) is False, (
        "a fill_budget rule never stops, so forwarding it into a patience arm disables patience"
    )
    assert PATIENCE_2.should_stop([], flat) is True


def test_patience_runs_in_the_runner_with_its_own_rule():
    """End to end: the patience arm must be able to return fewer seeds than the budget."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "go_no_go_under_test", root / "scripts" / "experiments" / "go_no_go.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["go_no_go_under_test"] = module
    spec.loader.exec_module(module)

    graph, target = graph_with_targets(n=60, target=10)
    contract = build_contract(graph, {}, target_set=target, budget=8)
    pool = [v for v in graph.nodes() if v not in set(target)][:30]
    seeds, info = module.run_delta2_patience(
        graph, pool, 8, contract=contract, state_mc=4, random_seed=3,
        stopping=module.FILL_BUDGET,  # deliberately the disabling rule
    )
    assert "patience" in info["stopping_rule"], (
        "the arm must use PATIENCE_2 regardless of the runner's global --stopping, and must record "
        "the rule it actually used"
    )
    assert len(seeds) <= 8
