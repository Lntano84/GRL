"""Overexposure-aware threshold-window diffusion model.

This module implements the *threshold-window* propagation model in which a node is
positively activated only while the cumulative influence it receives stays inside a
per-node interval ``[theta_kappa, theta_tau]``.  Exceeding ``theta_tau`` means the node
has been *overexposed*: it turns permanently negative and stops promoting the product.

Reference model
---------------
Each node ``v`` samples ``(theta_kappa_v, theta_tau_v)`` uniformly from the 2-D simplex

    {(x, y) in [0, 1]^2 : x <= y}

so the density over ``0 <= x <= y <= 1`` is 2.  Writing the cumulative influence from
positively activated in-neighbours as::

    delta(v, t) = sum_{u in A_in(v)} w_uv

**Lemma 1** of the reference paper evaluates the resulting probability::

    P(theta_kappa_v <= delta <= theta_tau_v) = 2 * delta * (1 - delta).

In the **deterministic** activation mode (the default) that window-membership event
activates the node outright, exactly as in the paper's Example 1, so the observable
per-node positive rate equals Lemma 1's expression and is returned by
``marginal_positive_probability``.  The **stochastic** mode instead applies a further
Bernoulli trial with probability ``2 * delta * (1 - delta)``, as the paper's wording
"the probability ... is denoted by 2*delta*(1-delta)" can also be read; the observable
rate then becomes ``4 * delta^2 * (1 - delta)^2``.  Both violate monotonicity, which is
the property that matters here.

A deliberate consistency guard: the positive branch additionally requires
``delta < 1``.  The literal rule ``delta in [kappa, tau]`` would otherwise activate a
node whenever ``tau == 1``, even though ``2 * delta * (1 - delta)`` vanishes at
``delta = 1``.  Taking the derived probability as authoritative, maximal exposure is
treated as maximal rejection.

Three states are possible and transitions are one-way only
(``inactive`` -> ``positive``/``negative``, ``positive`` -> ``negative``):

    delta <  theta_kappa                  : stays inactive
    theta_kappa <= delta < 1, <= theta_tau: becomes positive
    delta >  theta_tau   (or delta >= 1)  : becomes negative (permanent)

Why this breaks reverse-influence sampling
------------------------------------------
``delta`` is monotone in the set of positively activated in-neighbours, but
``2 * delta * (1 - delta)`` is **not**: it peaks at ``delta = 0.5`` and decreases beyond
it.  Adding an active neighbour can therefore *lower* a node's activation probability,
which destroys the premise of the coverage identity ``sigma(S) = n * Pr[S ∩ R != ∅]``
that RR/RIS methods rely on.

Monotonicity of ``sigma`` itself fails too: see
``test_marginal_gain_can_be_negative_under_overexposure``, where an extra seed drives a
relay node past its overexposure threshold and destroys more spread than it creates.
The empirical RR-invalidity check is intended for
``scripts/experiments/evaluate_overexposure_diagnostics.py``.

Degeneracy
----------
Setting ``theta_tau_v = 1`` for every node removes overexposure entirely: the negative
branch becomes unreachable, every node with ``theta_kappa <= delta < 1`` activates, and
the process reduces to the standard linear threshold (LT) model.
``sample_threshold_windows(..., overexposure_free=True)`` builds that limiting
configuration and the test-suite uses it as a sanity check.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import pstdev

import networkx as nx

INACTIVE = 0
POSITIVE = 1
NEGATIVE = 2


def positive_activation_probability(delta: float) -> float:
    """Conditional probability of a positive transition given ``delta`` is in the window.

    Lemma 1 of the reference paper gives the window-membership probability
    ``P(theta_kappa <= delta <= theta_tau) = 2 * delta * (1 - delta)``.  Under the rule
    ``delta > theta_tau => negative``, only half of that mass yields a positive
    transition, so the *unconditional* positive rate is ``delta * (1 - delta)`` — see
    :func:`marginal_positive_probability`.

    This function returns the value applied once a node is known to be inside its
    window, which is the ``2 * delta * (1 - delta)`` factor.  It peaks at
    ``delta = 0.5``; that it decreases beyond the peak is exactly why coverage-based
    sampling loses its validity under overexposure.
    """
    if delta <= 0.0:
        return 0.0
    if delta >= 1.0:
        # delta > theta_tau with probability 1, so no positive activation is possible.
        return 0.0
    return 2.0 * delta * (1.0 - delta)


def marginal_positive_probability(delta: float) -> float:
    """Unconditional probability that a node becomes positive given ``delta``.

    In deterministic mode (the default) the window event *is* the positive event, so
    this equals ``2 * delta * (1 - delta)`` — Lemma 1 exactly.  The cascade simulation
    reproduces this rate; the test-suite checks it empirically.
    """
    return positive_activation_probability(delta)


def sample_threshold_windows(
    nodes: list[int],
    rng: random.Random,
    overexposure_free: bool = False,
) -> dict[int, tuple[float, float]]:
    """Sample one ``(theta_kappa, theta_tau)`` window per node from the 2-D simplex.

    With ``overexposure_free=True`` every ``theta_tau`` is clamped to 1, which is the
    degenerate no-overexposure limit (equivalent to the linear threshold model).
    """
    windows: dict[int, tuple[float, float]] = {}
    for node in nodes:
        kappa = rng.random()
        tau = rng.random()
        if kappa > tau:
            kappa, tau = tau, kappa
        if overexposure_free:
            tau = 1.0
        windows[node] = (kappa, tau)
    return windows


#: ``delta`` inside the window activates unconditionally.  This reproduces the paper's
#: Example 1 state rule and makes ``P(positive | delta) = 2*delta*(1-delta)`` exactly.
DETERMINISTIC = "deterministic"
#: ``delta`` inside the window activates with probability ``2*delta*(1-delta)``, i.e. a
#: second layer of randomness on top of the threshold draw.  The unconditional positive
#: rate then becomes ``4*delta^2*(1-delta)^2``, which is still non-monotone.
STOCHASTIC = "stochastic"

ACTIVATION_MODES = (DETERMINISTIC, STOCHASTIC)


@dataclass
class OverexposureRun:
    """Result of a single deterministic realization."""

    spread: int
    positive: set[int] = field(default_factory=set)
    negative: set[int] = field(default_factory=set)
    delta: dict[int, float] = field(default_factory=dict)
    rounds: int = 0

    @property
    def ever_positive(self) -> set[int]:
        """Nodes that were positively activated at some point (they promote)."""
        return self.positive


def run_overexposure(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    windows: dict[int, tuple[float, float]],
    rng: random.Random,
    *,
    max_rounds: int | None = None,
    activation_mode: str = DETERMINISTIC,
) -> OverexposureRun:
    """Run one realization of the threshold-window cascade.

    ``windows`` supplies the sampled thresholds.  ``activation_mode`` selects how a node
    whose cumulative influence lands inside its window is resolved:

    ``"deterministic"`` (default)
        ``delta`` in ``[theta_kappa, theta_tau]`` activates the node outright.  This
        matches the paper's Example 1 state rule and makes the per-node positive rate
        exactly ``2 * delta * (1 - delta)``.
    ``"stochastic"``
        An extra Bernoulli trial with probability ``2 * delta * (1 - delta)`` is applied.
        The unconditional positive rate then becomes ``4 * delta^2 * (1 - delta)^2``.

    Both modes are non-monotone in ``delta``, which is the property that invalidates
    coverage-based sampling.  ``"deterministic"`` is the default because it is the
    interpretation under which the paper's Lemma 1 is exact.

    A node that is still below its activation threshold is **not** settled: it is
    re-evaluated in later rounds as more influence arrives.  Only reaching
    ``delta > theta_tau`` (negative, permanent) or falling inside the window (resolved
    by ``activation_mode``) ends a node's evaluation.
    """
    if activation_mode not in ACTIVATION_MODES:
        raise ValueError(f"activation_mode must be one of {ACTIVATION_MODES}")
    stochastic = activation_mode == STOCHASTIC

    state = {node: INACTIVE for node in graph.nodes()}
    delta = {node: 0.0 for node in graph.nodes()}
    settled: set[int] = set()
    positive: set[int] = set()
    negative: set[int] = set()

    directed = graph.is_directed()

    def incident_edges(node: int):
        """Edges through which ``node`` exerts influence on its neighbours."""
        return graph.out_edges(node, data=True) if directed else graph.edges(node, data=True)

    frontier: list[int] = []
    for seed in seeds:
        if state[seed] != POSITIVE:
            state[seed] = POSITIVE
            positive.add(seed)
            frontier.append(seed)

    rounds = 0
    limit = max_rounds if max_rounds is not None else graph.number_of_nodes() + 1

    while frontier and rounds < limit:
        rounds += 1
        # Phase 1: every node that turned positive at the end of the previous round
        # exerts its influence on its neighbours.
        for node in frontier:
            for u, v, data in incident_edges(node):
                target = v if directed else (v if u == node else u)
                if target == node:
                    continue
                delta[target] += float(data.get("weight", 0.0))

        # Phase 2: evaluate nodes against the freshly updated influence.  At most one
        # transition per node happens in a round, so a chain advances one hop per round.
        newly_positive: list[int] = []
        for node in graph.nodes():
            if state[node] != INACTIVE or node in settled:
                continue
            current = delta[node]
            if current <= 0.0:
                continue
            kappa, tau = windows[node]
            if current > tau:
                # Overexposed: permanently negative, never promotes.
                state[node] = NEGATIVE
                negative.add(node)
                settled.add(node)
            elif current >= kappa and current < 1.0:
                # Inside the window.  The strict ``current < 1`` guard keeps the discrete
                # state rule consistent with Lemma 1: 2*delta*(1-delta) vanishes at
                # delta = 1, so maximal exposure must not yield a positive transition.
                # Without the guard the literal rule "delta in [kappa, tau]" would
                # activate a node whenever tau == 1, contradicting the very probability
                # the model was derived from.
                inside = True
                if stochastic:
                    inside = rng.random() < positive_activation_probability(current)
                if inside:
                    state[node] = POSITIVE
                    positive.add(node)
                    newly_positive.append(node)
                settled.add(node)
            elif current >= 1.0:
                # delta = 1 exhausts the influence budget: Lemma 1 gives probability 0.
                state[node] = NEGATIVE
                negative.add(node)
                settled.add(node)
            # else current < kappa: stays inactive and stays re-evaluable.

        frontier = newly_positive

    return OverexposureRun(
        spread=len(positive),
        positive=positive,
        negative=negative,
        delta=delta,
        rounds=rounds,
    )


def estimate_spread(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    mc_runs: int,
    random_seed: int,
    *,
    overexposure_free: bool = False,
    activation_mode: str = DETERMINISTIC,
) -> dict[str, float]:
    """Monte-Carlo estimate of the expected *positive* spread of ``seeds``."""
    return estimate_overexposure_spread_over_configs(
        graph,
        [seeds],
        mc_runs,
        random_seed,
        overexposure_free=overexposure_free,
        activation_mode=activation_mode,
    )[0]


def estimate_overexposure_spread_over_configs(
    graph: nx.Graph | nx.DiGraph,
    seed_sets: list[list[int]],
    mc_runs: int,
    random_seed: int,
    *,
    overexposure_free: bool = False,
    activation_mode: str = DETERMINISTIC,
) -> list[dict[str, float]]:
    """Paired Monte-Carlo estimate of several seed sets under common random numbers.

    All seed sets in ``seed_sets`` share the same sampled threshold windows in each
    trial, so differences between them are not contaminated by threshold noise.  This
    is the overexposure analogue of the common live-edge sampling used for IC, and it is
    what makes ``Delta(v | S)`` labels usable.

    .. note::
       Deliberately *not* named ``estimate_spread_over_configs``.  That name belongs to
       the independent-cascade estimator in :mod:`grl.diffusion.independent_cascade`, and
       re-exporting an overexposure function under the same name silently shadowed it —
       which made model comparisons evaluate overexposure against itself.
    """
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")
    if not seed_sets:
        raise ValueError("seed_sets must not be empty")

    nodes = list(graph.nodes())
    samples: list[list[float]] = [[] for _ in seed_sets]

    for offset in range(mc_runs):
        rng = random.Random(random_seed + offset)
        windows = sample_threshold_windows(nodes, rng, overexposure_free=overexposure_free)
        for index, seeds in enumerate(seed_sets):
            run = run_overexposure(
                graph, list(seeds), windows, rng, activation_mode=activation_mode
            )
            samples[index].append(float(run.spread))

    results: list[dict[str, float]] = []
    for values in samples:
        mean = sum(values) / len(values)
        results.append({
            "mean": float(mean),
            "std": float(pstdev(values) if len(values) > 1 else 0.0),
        })
    return results


def estimate_marginal_gains(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidates: list[int],
    mc_runs: int,
    random_seed: int,
    *,
    overexposure_free: bool = False,
    activation_mode: str = DETERMINISTIC,
) -> dict[int, dict[str, float]]:
    """Paired estimate of ``Delta(candidate | seeds)`` for several candidates.

    Base and extended configurations share sampled thresholds within each trial, so the
    reported marginal gain is ``sigma(S ∪ {v}) - sigma(S)`` rather than a difference of
    two independently noisy estimates.

    Note that under overexposure the marginal gain **may legitimately be negative**;
    unlike the IC implementation this function does not clamp it.
    """
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if set(seeds) & set(candidates):
        raise ValueError("candidates must not be in the seed set")

    configurations = [list(seeds)] + [[*seeds, candidate] for candidate in candidates]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc_runs, random_seed,
        overexposure_free=overexposure_free,
        activation_mode=activation_mode,
    )
    base = estimates[0]

    result: dict[int, dict[str, float]] = {}
    for index, candidate in enumerate(candidates, start=1):
        extended = estimates[index]
        result[candidate] = {
            "base_spread": float(base["mean"]),
            "extended_spread": float(extended["mean"]),
            "mean": float(extended["mean"] - base["mean"]),
            "std": float(extended["std"]),
        }
    return result


def estimate_marginal_gain(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidate: int,
    mc_runs: int,
    random_seed: int,
    *,
    overexposure_free: bool = False,
    activation_mode: str = DETERMINISTIC,
) -> dict[str, float]:
    """Single-candidate convenience wrapper around :func:`estimate_marginal_gains`."""
    return estimate_marginal_gains(
        graph, seeds, [candidate], mc_runs, random_seed,
        overexposure_free=overexposure_free,
        activation_mode=activation_mode,
    )[candidate]
