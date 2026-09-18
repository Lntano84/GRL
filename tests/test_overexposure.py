"""Sanity, degeneracy and correctness checks for the overexposure model."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from grl.diffusion.overexposure import (
    DETERMINISTIC,
    NEGATIVE,
    POSITIVE,
    STOCHASTIC,
    estimate_marginal_gains,
    estimate_spread,
    estimate_overexposure_spread_over_configs,
    marginal_positive_probability,
    positive_activation_probability,
    run_overexposure,
    sample_threshold_windows,
)


# --------------------------------------------------------------------------------------
# The two probabilities: conditional (Lemma 1 factor) vs unconditional (observable)
# --------------------------------------------------------------------------------------

def test_conditional_probability_is_two_delta_one_minus_delta():
    """Lemma 1's factor, used as the per-evaluation probability in stochastic mode."""
    for delta in (0.05, 0.2, 0.5, 0.75, 0.9):
        assert positive_activation_probability(delta) == pytest.approx(2 * delta * (1 - delta))


def test_marginal_probability_matches_lemma_one():
    """In deterministic mode the observable positive rate *is* 2*delta*(1-delta).

    The window ``[theta_kappa, theta_tau]`` contains ``delta`` with probability
    ``2*delta*(1-delta)``, and in deterministic mode that event activates the node
    outright.  So Lemma 1 is exact here.
    """
    for delta in (0.05, 0.2, 0.5, 0.75, 0.9):
        assert marginal_positive_probability(delta) == pytest.approx(2 * delta * (1 - delta))
        assert marginal_positive_probability(delta) == pytest.approx(
            positive_activation_probability(delta)
        )


def test_probability_peaks_at_one_half():
    for fn in (positive_activation_probability, marginal_positive_probability):
        assert fn(0.5) == pytest.approx(0.5)
        assert fn(0.5) > fn(0.4)
        assert fn(0.5) > fn(0.6)


def test_probability_is_not_monotone():
    """The core reason RR/RIS loses its premise: p(delta) rises then falls."""
    for fn in (positive_activation_probability, marginal_positive_probability):
        rising = fn(0.30) < fn(0.45)
        falling = fn(0.60) > fn(0.75)
        assert rising and falling


def test_probability_bounds():
    for fn in (positive_activation_probability, marginal_positive_probability):
        assert fn(0.0) == 0.0
        assert fn(-0.1) == 0.0
        assert fn(1.0) == 0.0
        assert fn(1.5) == 0.0


# --------------------------------------------------------------------------------------
# Threshold sampling
# --------------------------------------------------------------------------------------

def test_windows_lie_in_simplex():
    rng = random.Random(7)
    windows = sample_threshold_windows(list(range(2000)), rng)
    assert len(windows) == 2000
    for kappa, tau in windows.values():
        assert 0.0 <= kappa <= tau <= 1.0


def test_overexposure_free_clamps_tau_to_one():
    rng = random.Random(7)
    windows = sample_threshold_windows(list(range(500)), rng, overexposure_free=True)
    assert all(tau == 1.0 for _, tau in windows.values())


# --------------------------------------------------------------------------------------
# Cascade mechanics on hand-built graphs
# --------------------------------------------------------------------------------------

def _chain(length: int, weight: float = 0.5) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in range(length):
        graph.add_node(node)
    for node in range(length - 1):
        graph.add_edge(node, node + 1, weight=weight)
    return graph


def test_no_overexposure_means_no_node_ever_goes_negative():
    """Degeneracy: with tau = 1 the overexposure branch is unreachable."""
    graph = _chain(8, weight=0.9)
    nodes = list(graph.nodes())
    for trial in range(50):
        rng = random.Random(trial)
        windows = sample_threshold_windows(nodes, rng, overexposure_free=True)
        run = run_overexposure(graph, [0], windows, rng)
        assert not run.negative
        assert 0 in run.positive


def test_delta_one_inside_window_activates():
    """The boundary rule is the literal ``kappa <= delta <= tau``.

    This test previously asserted the opposite -- that delta = 1 can never turn a node positive
    -- justified by ``2*delta*(1-delta) = 0``.  That reasoning was wrong and is corrected here.
    ``2*delta*(1-delta)`` is the probability of the window event under the *sampled* window
    distribution; the state rule is conditional on a window that has already been drawn.  Mixing
    them made ``delta = 1`` negative even when the drawn window was ``[0, 1]``, which is outside
    the rule.  See ``scripts/audit/verify_p0_counterexamples.py`` (P0-2).

    With window ``[0, 1]`` the endpoints are inside it, so the node activates.
    """
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = {0: (0.0, 1.0), 1: (0.0, 1.0)}
    run = run_overexposure(graph, [0], windows, random.Random(0))

    assert run.delta[1] == pytest.approx(1.0)
    assert 1 in run.positive
    assert run.spread == 2


def test_delta_exceeding_tau_turns_negative():
    """The other side of the boundary: delta strictly above tau is overexposure."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    windows = {0: (0.0, 1.0), 1: (0.0, 0.5)}
    run = run_overexposure(graph, [0], windows, random.Random(0))

    assert run.delta[1] == pytest.approx(1.0)
    assert 1 in run.negative
    assert 1 not in run.positive


def test_high_threshold_leaves_node_inactive():
    """A node whose delta never reaches kappa must stay inactive."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.3)
    windows = {0: (0.0, 1.0), 1: (0.9, 1.0)}
    run = run_overexposure(graph, [0], windows, random.Random(0))

    assert run.spread == 1
    assert run.positive == {0}
    assert run.delta[1] == pytest.approx(0.3)


def test_overexposure_turns_node_negative():
    """delta above tau => permanently negative, and it does not promote further."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.6)
    graph.add_edge(1, 2, weight=1.0)
    windows = {0: (0.0, 1.0), 1: (0.1, 0.5), 2: (0.0, 1.0)}

    run = run_overexposure(graph, [0], windows, random.Random(0))

    assert 1 in run.negative
    assert 1 not in run.positive
    # 1 never turned positive, so it never propagated: 2 receives nothing.
    assert run.delta[2] == pytest.approx(0.0)
    assert 2 not in run.positive
    assert run.spread == 1


def test_node_becomes_positive_when_influence_accumulates():
    """delta below kappa first, then into the window, must still be able to activate."""
    graph = nx.DiGraph()
    graph.add_edge(0, 2, weight=0.25)
    graph.add_edge(1, 2, weight=0.25)
    windows = {0: (0.0, 1.0), 1: (0.0, 1.0), 2: (0.4, 1.0)}

    run = run_overexposure(graph, [0, 1], windows, random.Random(0))

    assert run.delta[2] == pytest.approx(0.5)
    assert run.rounds >= 1
    assert 2 not in run.negative


def test_activation_rate_follows_the_marginal_probability():
    """Empirically verify the observable per-node positive rate 2*delta*(1-delta).

    This is the strongest end-to-end check of the model: it exercises simplex sampling,
    the window test, the negative branch and the deterministic resolution together.
    At delta = 0.5 the expected rate is 0.5.
    """
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.5)
    nodes = list(graph.nodes())

    activated = negative = 0
    trials = 4000
    for trial in range(trials):
        rng = random.Random(trial)
        windows = sample_threshold_windows(nodes, rng)
        run = run_overexposure(graph, [0], windows, rng)
        activated += int(1 in run.positive)
        negative += int(1 in run.negative)

    rate = activated / trials
    expected = marginal_positive_probability(0.5)
    assert abs(rate - expected) < 0.03, (rate, expected)
    # The negative branch is exactly delta^2 = 0.25 here, so the two must be comparable.
    assert abs(negative / trials - 0.25) < 0.03, negative / trials


def test_positive_and_negative_rates_partition_the_window_probability():
    """P(positive) + P(negative) must equal P(kappa <= delta), not P(in window).

    ``delta > theta_tau`` and ``kappa <= delta <= theta_tau`` are disjoint, and together
    they cover ``kappa <= delta``, whose probability is ``1 - (1-delta)^2``.
    """
    delta = 0.4
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=delta)
    nodes = list(graph.nodes())

    positive = negative = 0
    trials = 6000
    for trial in range(trials):
        rng = random.Random(40000 + trial)
        windows = sample_threshold_windows(nodes, rng)
        run = run_overexposure(graph, [0], windows, rng)
        positive += int(1 in run.positive)
        negative += int(1 in run.negative)

    covered = (positive + negative) / trials
    expected_covered = 1 - (1 - delta) ** 2
    assert abs(covered - expected_covered) < 0.03, (covered, expected_covered)
    assert abs(positive / trials - marginal_positive_probability(delta)) < 0.03
    assert abs(negative / trials - delta ** 2) < 0.03


def test_stochastic_mode_applies_an_extra_bernoulli_layer():
    """In stochastic mode the observable rate is the square of the deterministic one."""
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.5)
    nodes = list(graph.nodes())

    activated = 0
    trials = 4000
    for trial in range(trials):
        rng = random.Random(trial)
        windows = sample_threshold_windows(nodes, rng)
        run = run_overexposure(graph, [0], windows, rng, activation_mode=STOCHASTIC)
        activated += int(1 in run.positive)

    rate = activated / trials
    expected = positive_activation_probability(0.5) ** 2  # 0.25 at delta = 0.5
    assert abs(rate - expected) < 0.03, (rate, expected)


def test_invalid_activation_mode_is_rejected():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.5)
    windows = {0: (0.0, 1.0), 1: (0.0, 1.0)}
    with pytest.raises(ValueError):
        run_overexposure(graph, [0], windows, random.Random(0), activation_mode="nope")


def test_undirected_graph_sums_both_directions():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.4)
    run = run_overexposure(graph, [0], {0: (0.0, 1.0), 1: (0.1, 1.0)}, random.Random(0))

    assert run.delta[1] == pytest.approx(0.4)
    assert 1 not in run.negative


def test_deterministic_given_windows_and_rng():
    graph = _chain(6, weight=0.5)
    windows = {node: (0.2, 1.0) for node in graph.nodes()}
    first = run_overexposure(graph, [0], windows, random.Random(11))
    second = run_overexposure(graph, [0], windows, random.Random(11))
    assert first.spread == second.spread
    assert first.positive == second.positive


def test_seed_alone_is_always_positive():
    graph = _chain(3)
    run = run_overexposure(graph, [1], {n: (0.0, 1.0) for n in graph.nodes()}, random.Random(0))
    assert 1 in run.positive
    assert run.spread >= 1


def test_positive_and_negative_are_disjoint():
    graph = _chain(6)
    rng = random.Random(3)
    windows = sample_threshold_windows(list(graph.nodes()), rng)
    run = run_overexposure(graph, [0], windows, rng)
    assert not (run.positive & run.negative)


# --------------------------------------------------------------------------------------
# Estimating with common random numbers
# --------------------------------------------------------------------------------------

def test_paired_estimates_share_thresholds():
    graph = _chain(5, weight=0.5)
    estimates = estimate_overexposure_spread_over_configs(graph, [[0], [0, 1]], 60, 123)
    assert len(estimates) == 2
    # Adding a seed can never reduce the spread: it is itself positive.
    assert estimates[1]["mean"] >= estimates[0]["mean"]


def test_marginal_gain_can_be_negative_under_overexposure():
    """Overexposure makes sigma non-monotone, so Delta(v|S) can legitimately be < 0.

    Base S = {0}: node 2 receives delta = 0.6, inside its window [0, 0.6], and turns
    positive with probability 2*0.6*0.4 = 0.48.  Its four children then receive
    delta = 0.6 each and are themselves independently activatable.

    Extended S = {0, 1}: node 2 now receives delta = 1.6 > tau = 0.6, so it turns
    permanently negative instead of positive.  Adding seed 1 gains the seed itself (+1)
    but loses node 2 together with every one of its four children (about 1 + 4*0.48).

    Net expected marginal gain is therefore negative: a "helpful-looking" extra seed
    destroys more spread than it creates.
    """
    child_weight = 0.6
    n_children = 4
    graph = nx.DiGraph()
    graph.add_edge(0, 2, weight=0.6)
    graph.add_edge(1, 2, weight=1.0)
    for child in range(3, 3 + n_children):
        graph.add_edge(2, child, weight=child_weight)

    windows = {0: (0.0, 1.0), 1: (0.0, 1.0), 2: (0.0, 0.6)}
    for child in range(3, 3 + n_children):
        windows[child] = (0.0, 1.0)

    base = run_overexposure(graph, [0], windows, random.Random(0))
    extended = run_overexposure(graph, [0, 1], windows, random.Random(0))
    assert base.delta[2] == pytest.approx(0.6)
    assert extended.delta[2] == pytest.approx(1.6)
    assert 2 in extended.negative
    assert 2 not in extended.positive

    base_est = estimate_spread(graph, [0], 300, 909)
    ext_est = estimate_spread(graph, [0, 1], 300, 909)

    # Extended is exactly {0, 1}: node 2 never promotes, so no child is ever reached.
    assert ext_est["mean"] == pytest.approx(2.0)
    assert base_est["mean"] > ext_est["mean"], (base_est, ext_est)

    gain = estimate_marginal_gains(graph, [0], [1], 300, 909)[1]
    assert gain["mean"] < 0.0, gain


def test_estimate_marginal_gains_rejects_seed_candidate_overlap():
    graph = _chain(3)
    with pytest.raises(ValueError):
        estimate_marginal_gains(graph, [0], [0], 5, 1)


def test_estimate_overexposure_spread_over_configs_validates_mc_runs():
    graph = _chain(3)
    with pytest.raises(ValueError):
        estimate_overexposure_spread_over_configs(graph, [[0]], 0, 1)
    with pytest.raises(ValueError):
        estimate_overexposure_spread_over_configs(graph, [], 5, 1)
