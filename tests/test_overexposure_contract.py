"""Contract tests for the overexposure diffusion model.

These cover the formal requirements in the task brief that the existing
``tests/test_overexposure.py`` does not: parameter bounds, config-driven control, reproducibility
contract, and the structural edge cases.

They deliberately do NOT re-assert the model's behavioural claims (those live in
``test_overexposure.py`` and are about activation rates and state transitions).
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion import overexposure as oe
from grl.diffusion.params import (
    DEFAULT_MC_RUNS,
    OverexposureParams,
    resolve_overexposure_params,
)


# --------------------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------------------
def test_empty_seed_set_gives_empty_positive_set():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(0))
    run = oe.run_overexposure(graph, [], windows, random.Random(0))
    assert run.spread == 0
    assert run.positive == set()
    assert run.negative == set()
    assert run.rounds == 0


def test_budget_zero_is_an_empty_seed_set():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(1))
    run = oe.run_overexposure(graph, [], windows, random.Random(1))
    assert run.spread == 0


def test_isolated_nodes_never_activate_unless_seeded():
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])  # no edges at all
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(2))
    run = oe.run_overexposure(graph, [0], windows, random.Random(2))
    # only the seed itself; delta stays 0 for the others and P(positive | 0) = 0
    assert run.positive == {0}
    assert run.spread == 1


def test_single_node_graph_with_itself_as_seed():
    graph = nx.DiGraph()
    graph.add_node(0)
    windows = oe.sample_threshold_windows([0], random.Random(3))
    run = oe.run_overexposure(graph, [0], windows, random.Random(3))
    assert run.spread == 1


def test_duplicate_seeds_are_counted_once():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(4))
    once = oe.run_overexposure(graph, [0], windows, random.Random(4)).spread
    twice = oe.run_overexposure(graph, [0, 0], windows, random.Random(4)).spread
    assert once == twice


def test_unknown_seed_node_is_rejected_loudly():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = oe.sample_threshold_windows(list(graph.nodes()), random.Random(5))
    with pytest.raises(KeyError):
        oe.run_overexposure(graph, [99], windows, random.Random(5))


# --------------------------------------------------------------------------------------
# probability bounds
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("delta", [-1.0, -1e-12, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0 + 1e-12, 2.0])
def test_activation_probability_stays_in_unit_interval(delta):
    value = oe.positive_activation_probability(delta)
    assert 0.0 <= value <= 1.0, f"P(positive | delta={delta}) = {value}"


def test_activation_probability_is_symmetric_and_peaks_at_one_half():
    assert oe.positive_activation_probability(0.5) == pytest.approx(0.5)
    for d in (0.1, 0.2, 0.3, 0.4):
        assert oe.positive_activation_probability(d) == pytest.approx(
            oe.positive_activation_probability(1.0 - d)
        )


def test_marginal_probability_is_an_alias_of_the_window_probability():
    for d in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert oe.marginal_positive_probability(d) == pytest.approx(
            oe.positive_activation_probability(d)
        )


@pytest.mark.parametrize("window_lo", [0.0, 0.25, 0.5, 0.9])
def test_sampled_windows_are_ordered_and_in_unit_interval(window_lo):
    nodes = list(range(500))
    windows = oe.sample_threshold_windows(nodes, random.Random(6), window_lo=window_lo)
    for node, (kappa, tau) in windows.items():
        assert 0.0 <= kappa <= 1.0, (node, kappa)
        assert 0.0 <= tau <= 1.0, (node, tau)
        assert kappa <= tau, (node, kappa, tau)


def test_overexposure_free_clamps_tau_to_one():
    nodes = list(range(200))
    windows = oe.sample_threshold_windows(nodes, random.Random(7), overexposure_free=True)
    assert all(tau == 1.0 for _, tau in windows.values())


def test_window_lo_raises_the_lower_end_of_the_tau_support():
    nodes = list(range(400))
    plain = oe.sample_threshold_windows(nodes, random.Random(8))
    raised = oe.sample_threshold_windows(nodes, random.Random(8), window_lo=0.5)
    plain_min = min(tau for _, tau in plain.values())
    raised_min = min(tau for _, tau in raised.values())
    assert plain_min < 0.5 <= raised_min, (plain_min, raised_min)


# --------------------------------------------------------------------------------------
# reproducibility contract
# --------------------------------------------------------------------------------------
def test_same_seed_and_params_reproduce_exactly():
    graph = nx.DiGraph()
    for i in range(30):
        graph.add_edge(i, (i * 7 + 3) % 30, weight=0.25)
    nodes = list(graph.nodes())

    def once(seed: int) -> tuple[int, frozenset[int]]:
        rng = random.Random(seed)
        windows = oe.sample_threshold_windows(nodes, rng)
        run = oe.run_overexposure(graph, [0, 5], windows, rng)
        return run.spread, frozenset(run.positive)

    assert once(11) == once(11)


def test_different_seeds_generally_differ():
    graph = nx.DiGraph()
    for i in range(40):
        graph.add_edge(i, (i * 3 + 1) % 40, weight=0.4)
    nodes = list(graph.nodes())
    outs = set()
    for seed in range(12):
        rng = random.Random(seed)
        windows = oe.sample_threshold_windows(nodes, rng)
        outs.add(oe.run_overexposure(graph, [0], windows, rng).spread)
    assert len(outs) > 1, "the process looks deterministic across seeds"


def test_diffusion_does_not_mutate_the_graph():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.5)
    graph.add_edge(1, 2, weight=0.5)
    before = sorted((u, v, d["weight"]) for u, v, d in graph.edges(data=True))
    nodes = list(graph.nodes())
    for seed in range(5):
        rng = random.Random(seed)
        windows = oe.sample_threshold_windows(nodes, rng)
        oe.run_overexposure(graph, [0], windows, rng)
    after = sorted((u, v, d["weight"]) for u, v, d in graph.edges(data=True))
    assert before == after


def test_result_is_independent_of_node_iteration_order():
    """Relabelling the graph consistently must not change the spread."""
    base = nx.DiGraph()
    for i in range(10):
        base.add_edge(i, (i + 1) % 10, weight=0.5)
    mapping = {i: 100 - i for i in base.nodes()}
    relabelled = nx.relabel_nodes(base, mapping)

    nodes_a = list(base.nodes())
    rng_a = random.Random(21)
    win_a = oe.sample_threshold_windows(nodes_a, rng_a)
    spread_a = oe.run_overexposure(base, [0], win_a, rng_a).spread

    nodes_b = list(relabelled.nodes())
    rng_b = random.Random(21)
    win_b = oe.sample_threshold_windows(nodes_b, rng_b)
    spread_b = oe.run_overexposure(relabelled, [mapping[0]], win_b, rng_b).spread

    assert spread_a == spread_b


# --------------------------------------------------------------------------------------
# config-driven control
# --------------------------------------------------------------------------------------
def test_defaults_match_the_source_model():
    params = resolve_overexposure_params(None)
    assert params.activation_mode == oe.DETERMINISTIC
    assert params.overexposure_free is False
    assert params.window_lo == 0.0
    assert params.mc_runs == DEFAULT_MC_RUNS


def test_config_block_is_read():
    config = {
        "overexposure": {
            "activation_mode": oe.STOCHASTIC,
            "overexposure_free": True,
            "mc_runs": 7,
            "random_seed": 123,
        },
        "experiment": {"random_seed": 999},
    }
    params = resolve_overexposure_params(config)
    assert params.activation_mode == oe.STOCHASTIC
    assert params.overexposure_free is True
    assert params.mc_runs == 7
    assert params.random_seed == 123


def test_missing_block_falls_back_to_diffusion_mc_runs():
    config = {"diffusion": {"mc_runs_eval": 321}, "experiment": {"random_seed": 5}}
    params = resolve_overexposure_params(config)
    assert params.mc_runs == 321
    assert params.random_seed == 5


@pytest.mark.parametrize(
    "block",
    [
        {"activation_mode": "bogus"},
        {"window_lo": 1.0},
        {"window_lo": -0.1},
        {"mc_runs": 0},
        {"mc_runs": -3},
        {"overexposure_free": True, "window_lo": 0.5},
    ],
)
def test_invalid_config_is_rejected(block):
    with pytest.raises((ValueError, TypeError)):
        resolve_overexposure_params({"overexposure": block})


def test_non_mapping_block_is_rejected():
    with pytest.raises(TypeError):
        resolve_overexposure_params({"overexposure": [1, 2, 3]})


def test_params_are_frozen():
    params = OverexposureParams()
    with pytest.raises(Exception):
        params.mc_runs = 5  # type: ignore[misc]


# --------------------------------------------------------------------------------------
# degeneracy: the no-overexposure limit must actually remove overexposure
# --------------------------------------------------------------------------------------
def test_overexposure_free_never_produces_negative_nodes():
    """With tau = 1 a node's delta can never exceed tau, so nothing turns negative."""
    graph = nx.DiGraph()
    rng = random.Random(31)
    for i in range(60):
        graph.add_edge(i, rng.randrange(60), weight=0.3)
    nodes = list(graph.nodes())
    windows = oe.sample_threshold_windows(nodes, random.Random(32), overexposure_free=True)
    run = oe.run_overexposure(graph, [0, 1, 2], windows, random.Random(32))
    assert run.negative == set()


def test_overexposure_free_is_monotone_in_seed_set_size_on_average():
    """Removing overexposure should restore the usual 'more seeds, more spread' trend.

    This is the meaningful degeneracy check for this model.  It is NOT a claim that the
    no-overexposure limit reproduces independent cascade: the audit showed the two processes
    percolate at different scales because activation here still requires delta >= theta_kappa.
    """
    graph = nx.DiGraph()
    rng = random.Random(41)
    for i in range(120):
        for _ in range(3):
            graph.add_edge(i, rng.randrange(120), weight=0.2)
    nodes = list(graph.nodes())

    def mean_spread(size: int, trials: int = 40) -> float:
        total = 0.0
        for t in range(trials):
            r = random.Random(1000 + t)
            seeds = r.sample(nodes, size)
            windows = oe.sample_threshold_windows(nodes, r, overexposure_free=True)
            total += oe.run_overexposure(graph, seeds, windows, r).spread
        return total / trials

    small, large = mean_spread(2), mean_spread(12)
    assert large > small, (small, large)


def test_negative_state_is_permanent():
    """A node that turns negative must never return to positive."""
    graph = nx.DiGraph()
    # hub 0 feeds a node 1 with weight 1.0, so node 1's delta jumps to 1.0 in one step
    graph.add_edge(0, 1, weight=1.0)
    windows = {0: (0.0, 1.0), 1: (0.0, 0.5)}  # node 1 overexposes at delta = 1.0
    run = oe.run_overexposure(graph, [0], windows, random.Random(0))
    assert 1 in run.negative
    assert 1 not in run.positive
