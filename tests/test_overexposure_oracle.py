"""Tests for the state-tracking overexposure Monte-Carlo oracle.

The critical properties are:
  * ``score`` really returns a *marginal gain* over the given seed set, checked against a
    brute-force estimate computed a different way;
  * the paired construction removes window-draw noise from the differences;
  * cost is counted in MC cascades and the count matches the work actually done;
  * the same (seeds, step) reproduces exactly.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion import overexposure as oe
from grl.diffusion.params import OverexposureParams
from grl.oracle import OverexposureMonteCarloOracle


def make_graph(n: int = 40, seed: int = 0, weight: float = 0.3) -> nx.DiGraph:
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u in range(n):
        for _ in range(3):
            graph.add_edge(u, rng.randrange(n), weight=weight)
    return graph


# --------------------------------------------------------------------------------------
# basic contract
# --------------------------------------------------------------------------------------
def test_empty_candidates_returns_empty_and_costs_nothing():
    oracle = OverexposureMonteCarloOracle(make_graph(), mc_runs=10)
    assert oracle.score([], []) == {}
    assert oracle.stats.mc_cascades == 0
    assert oracle.stats.candidate_evaluations == 0


def test_score_returns_one_entry_per_candidate():
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=10)
    candidates = list(graph.nodes())[:5]
    scores = oracle.score([0], candidates)
    assert set(scores) == set(candidates)
    assert all(isinstance(v, float) for v in scores.values())


def test_seed_candidates_score_near_zero_marginal_gain():
    """A node already in the seed set adds nothing, so its marginal gain must be 0."""
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=20)
    seeds = [0, 1, 2]
    scores = oracle.score(seeds, seeds)
    for node, value in scores.items():
        assert value == pytest.approx(0.0), (node, value)


def test_marginal_gain_is_non_negative_when_windows_cannot_overexpose():
    """With tau = 1 no node can be pushed into the negative state, so adding a seed cannot hurt.

    This is the monotone limit: marginal gains must be >= 0, which is a strong internal
    consistency check on both the oracle and the diffusion model.
    """
    graph = make_graph()
    params = OverexposureParams(threshold_law="simplex_tau_clamped_to_one")
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=30, params=params, random_seed=5)
    seeds = [0]
    candidates = [v for v in graph.nodes() if v not in seeds][:8]
    scores = oracle.score(seeds, candidates)
    for node, value in scores.items():
        assert value >= 0.0, (node, value)


def test_negative_marginal_gain_is_possible_when_overexposure_is_enabled():
    """The overexposure model must be able to produce a harmful seed; otherwise it is not the model.

    Construction (all weights and windows chosen so the outcome is exact, no Monte Carlo):

        A -> B   weight 0.5     A alone puts B at delta = 0.5, inside B's window [0.2, 0.9]
        B -> d0..d3 weight 0.3  B promotes four downstream nodes (windows [0.1, 0.8])
        C -> B   weight 0.6     C pushes B to delta = 1.1 > tau = 0.9, so B turns negative
        C -> SINK weight 0.3    the candidate's own leaf

    Seeding C therefore adds itself and SINK (+2) but destroys B and its four children (-5),
    for a net marginal gain of -3.  Note a candidate always gains +1 for itself, so a negative
    marginal requires it to destroy at least two already-positive nodes.
    """
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=0.5)
    graph.add_edge("C", "B", weight=0.6)
    graph.add_edge("C", "SINK", weight=0.3)
    for i in range(4):
        graph.add_edge("B", f"d{i}", weight=0.3)

    windows = {
        "A": (0.0, 1.0),
        "B": (0.2, 0.9),
        "C": (0.0, 1.0),
        "SINK": (0.0, 1.0),
        **{f"d{i}": (0.1, 0.8) for i in range(4)},
    }

    alone = oe.run_overexposure(graph, ["A"], windows, random.Random(0))
    with_candidate = oe.run_overexposure(graph, ["A", "C"], windows, random.Random(0))

    assert alone.spread == 6, sorted(alone.positive)
    assert with_candidate.spread == 3, sorted(with_candidate.positive)
    assert with_candidate.spread - alone.spread == -3
    assert "B" in with_candidate.negative


# --------------------------------------------------------------------------------------
# paired construction and noise
# --------------------------------------------------------------------------------------
def test_pairing_removes_window_noise_from_differences():
    """Two identical candidate calls must agree exactly, because windows are shared per trial."""
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=15, random_seed=17)
    candidates = list(graph.nodes())[:6]
    first = oracle.score([0], candidates, step=1)
    second = oracle.score([0], candidates, step=1)
    assert first == second


def test_different_steps_use_independent_draws():
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=15, random_seed=17)
    candidates = list(graph.nodes())[:6]
    a = oracle.score([0], candidates, step=1)
    b = oracle.score([0], candidates, step=2)
    assert a != b


def test_more_mc_runs_reduces_the_spread_of_the_estimate():
    graph = make_graph()
    candidates = list(graph.nodes())[:4]
    estimates_low, estimates_high = [], []
    for rep in range(6):
        low = OverexposureMonteCarloOracle(graph, mc_runs=5, random_seed=100 + rep)
        high = OverexposureMonteCarloOracle(graph, mc_runs=60, random_seed=100 + rep)
        estimates_low.append(low.score([0], candidates)[candidates[0]])
        estimates_high.append(high.score([0], candidates)[candidates[0]])
    spread_low = max(estimates_low) - min(estimates_low)
    spread_high = max(estimates_high) - min(estimates_high)
    assert spread_high <= spread_low, (spread_low, spread_high)


# --------------------------------------------------------------------------------------
# cost accounting
# --------------------------------------------------------------------------------------
def test_cascade_count_matches_the_work_done():
    graph = make_graph()
    mc = 7
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc)
    candidates = list(graph.nodes())[:5]
    oracle.score([0], candidates)
    # one base cascade plus one per candidate, per trial
    assert oracle.stats.mc_cascades == mc * (1 + len(candidates))
    assert oracle.stats.candidate_evaluations == len(candidates)


def test_spread_query_count_and_cost():
    graph = make_graph()
    mc = 6
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc)
    result = oracle.spread([0, 1])
    assert result["n"] == mc
    assert oracle.stats.mc_cascades == mc
    assert oracle.stats.spread_queries == 1


def test_stats_reset():
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=5)
    oracle.score([0], list(graph.nodes())[:3])
    oracle.state([0])
    assert oracle.stats.mc_cascades > 0
    assert oracle.stats.state_cascades > 0
    oracle.stats.reset()
    # Every counter must be zero after a reset; the set is pinned in full so that a new cost
    # channel cannot be added without this test noticing.
    assert oracle.stats.as_dict() == {
        "mc_cascades": 0,
        "candidate_evaluations": 0,
        "spread_queries": 0,
        "state_reads": 0,
        "state_cascades": 0,
        "cascades_for_scoring": 0,
    }


def test_state_acquisition_is_priced():
    """Confound P1-3.2: the state a state-conditioned policy reads costs cascades.

    Reporting that policy at zero cost next to policies charged for every cascade makes the cost
    column meaningless, so the acquisition must be visible in the same accounting --- and it must
    be a *breakdown* of the primary unit, so that a caller charging ``mc_cascades`` cannot
    accidentally give a state-conditioned policy free state.
    """
    graph = make_graph()
    mc = 7
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc)
    assert oracle.stats.mc_cascades == 0
    oracle.state([0])
    assert oracle.stats.state_reads == 1
    assert oracle.stats.state_cascades == mc
    assert oracle.stats.mc_cascades == mc, "the state must be inside the primary cost unit"
    assert oracle.stats.cascades_for_scoring == 0

    # scoring adds to the same unit, and the two channels partition it rather than double-count
    oracle.score([0], list(graph.nodes())[:2])
    assert oracle.stats.state_cascades == mc
    assert oracle.stats.mc_cascades > mc
    assert oracle.stats.cascades_for_scoring == oracle.stats.mc_cascades - mc
    assert (oracle.stats.cascades_for_scoring + oracle.stats.state_cascades
            == oracle.stats.mc_cascades), "the breakdown must sum to the total, not exceed it"


def test_score_with_uncertainty_reports_a_paired_standard_error():
    """Confound P1-3.7: a candidate difference must be comparable to its own estimation error."""
    graph = make_graph(n=30, seed=11)
    mc = 60
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc, random_seed=5)
    candidates = list(graph.nodes())[:4]

    rows = oracle.score_with_uncertainty([0], candidates)
    assert set(rows) == set(candidates)
    for v, row in rows.items():
        assert row["n"] == mc
        assert row["stderr"] >= 0.0
    # the point estimates must agree exactly with the plain score, which averages the same trials
    oracle.stats.reset()
    plain = oracle.score([0], candidates)
    for v in candidates:
        assert plain[v] == rows[v]["mean"]

    # a standard error is not a standard deviation: with enough trials it must be smaller
    wide = OverexposureMonteCarloOracle(graph, mc_runs=4 * mc, random_seed=5)
    wide_rows = wide.score_with_uncertainty([0], candidates)
    mean_se = sum(r["stderr"] for r in rows.values()) / len(rows)
    wide_se = sum(r["stderr"] for r in wide_rows.values()) / len(wide_rows)
    assert wide_se < mean_se, "quadrupling the trials must shrink the standard error"


def test_spread_mean_is_consistent_with_brute_force():
    """Independent re-implementation of the same estimator must agree within Monte-Carlo error."""
    graph = make_graph(n=30, seed=9)
    mc = 120
    seeds = [0, 1]
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc, random_seed=2024)
    got = oracle.spread(seeds)["mean"]

    nodes = list(graph.nodes())
    manual = 0.0
    for offset in range(mc):
        rng = random.Random(2024 + offset)
        windows = oe.sample_threshold_windows(nodes, rng)
        manual += oe.run_overexposure(graph, seeds, windows, rng).spread
    manual /= mc
    assert got == pytest.approx(manual)


# --------------------------------------------------------------------------------------
# parameter validation and config plumbing
# --------------------------------------------------------------------------------------
def test_invalid_mc_runs_rejected():
    with pytest.raises(ValueError):
        OverexposureMonteCarloOracle(make_graph(), mc_runs=0)


def test_invalid_window_lo_rejected():
    with pytest.raises(ValueError):
        OverexposureMonteCarloOracle(make_graph(), mc_runs=5, window_lo=1.0)


def test_window_lo_override_is_used():
    graph = make_graph()
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=5, window_lo=0.4)
    assert oracle.window_lo == pytest.approx(0.4)


def test_stochastic_activation_mode_is_honoured():
    graph = make_graph()
    params = OverexposureParams(activation_mode=oe.STOCHASTIC)
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=10, params=params)
    assert oracle.params.activation_mode == oe.STOCHASTIC
    # must run without error and produce a finite estimate
    value = oracle.score([0], list(graph.nodes())[1:4])
    assert all(v == v for v in value.values())


def test_oracle_does_not_mutate_the_graph():
    graph = make_graph()
    before = sorted((u, v, d["weight"]) for u, v, d in graph.edges(data=True))
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=8)
    oracle.score([0], list(graph.nodes())[:4])
    oracle.spread([1, 2])
    after = sorted((u, v, d["weight"]) for u, v, d in graph.edges(data=True))
    assert before == after
