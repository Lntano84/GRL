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

The state rule is evaluated literally against the sampled window.  In particular,
``delta == theta_tau == 1`` is inside the window and therefore activates the node;
the marginal probability formula must not be used as an extra guard on a window that
has already been sampled.  Three states are possible and transitions are one-way only
(``inactive`` -> ``positive``/``negative``, ``positive`` -> ``negative``):

    delta <  theta_kappa                  : stays inactive
    theta_kappa <= delta <= theta_tau      : becomes positive
    delta >  theta_tau                     : becomes negative (permanent)

An already-positive non-seed node is re-evaluated when exposure grows, so it may
later turn negative.  Its historical positive activation still contributes to the
exposure of downstream nodes through ``ever_positive``.

Why reverse-influence sampling does not transfer
------------------------------------------------
``delta`` is monotone in the set of positively activated in-neighbours, but
``2 * delta * (1 - delta)`` is **not**: it peaks at ``delta = 0.5`` and decreases beyond
it.  Adding an active neighbour can therefore *lower* a node's activation probability, and
the objective ``F_D`` is non-monotone.  That is enough to rule out the standard non-negative
seed-independent coverage form ``c * Pr[S ∩ R != ∅]``, which is necessarily non-decreasing
and submodular; ``sections/coverage.tex`` proves it on a one-hop instance by exact
arithmetic, and ``test_marginal_gain_can_be_negative_under_overexposure`` exhibits the same
effect on a small graph.

Note what that argument does *not* say, because an earlier version of this docstring
overclaimed it:

* It does **not** say RR sets are empty.  A probe that reported all-empty RR sets was
  mis-specified --- it required a predecessor's window to contain zero, excluded the root,
  and tested whether the returned set was non-empty rather than whether it *intersected the
  seed set*.  That claim is withdrawn; see ``docs/WITHDRAWN_RESULTS.md``.
* It does **not** say ``sigma^kappa`` and ``sigma^tau`` are unusable.  Each fixes one
  threshold per node, so each *is* monotone submodular and the coverage identity applies to
  each individually.  What the identity does not reach is their difference, and the
  objective ``sigma`` itself.

Degeneracy, and three paths that must not be conflated
------------------------------------------------------
Setting ``theta_tau_v = 1`` for every node removes overexposure entirely: the negative
branch becomes unreachable and every node with ``theta_kappa <= delta`` activates.  That
limit is a linear-threshold model, but its threshold distribution depends on how the
window was drawn, and the three resulting processes are genuinely different:

``threshold_law="simplex_tau_clamped_to_one"``
    the simplex draw with ``tau`` forced to 1.  ``kappa`` keeps the marginal
    ``F_kappa(x) = 2x - x^2``, so ``E[kappa] = 1/3``.  This is the *correct* "overexposure
    off" ablation for the source model.
``threshold_law="uniform_lt_tau_one"``
    ``kappa ~ U[0, 1]``, so ``E[kappa] = 1/2``.  This is genuine uniform-threshold LT, a
    different process that is also *monotone* where the source law is not.
``overexposure_free=True``
    the legacy flag.  It maps onto one of the two above and is retained only so that older
    callers keep working; it is no longer a field of ``OverexposureParams`` (it is derived
    from ``threshold_law``) precisely because it used to conflate them.

None of the simplex-derived paths may be described as "standard uniform-threshold LT":
only ``uniform_lt_tau_one`` has uniform thresholds.  ``sample_threshold_windows`` exposes
all four laws by name.
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
    """Simplex-window probability used by the deterministic model.

    For a window sampled uniformly from ``0 <= theta_kappa <= theta_tau <= 1``,
    Lemma 1 gives
    ``P(theta_kappa <= delta <= theta_tau) = 2 * delta * (1 - delta)``.
    In deterministic mode this is the probability of the window event itself.

    The optional stochastic mode uses the same value as an additional Bernoulli
    factor after a node is in its window; that mode therefore has a different
    observable positive rate and is not the default paper protocol.  The function
    peaks at ``delta = 0.5``; its non-monotonicity is why coverage-based sampling
    loses its validity under overexposure.
    """
    if delta <= 0.0:
        return 0.0
    if delta >= 1.0:
        # The simplex window event has zero measure at delta=1; this helper is not
        # the conditional state rule applied after a concrete window is sampled.
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
    window_lo: float = 0.0,
    threshold_law: str | None = None,
) -> dict[int, tuple[float, float]]:
    """Sample one ``(theta_kappa, theta_tau)`` window per node.

    Four threshold laws are supported and they are NOT interchangeable.  The audit required the
    degeneracy paths to be kept apart, because sampling two uniforms and clamping tau to 1 does not
    give a uniform lower-threshold marginal:

    ``simplex`` (default)
        ``(kappa, tau)`` uniform on ``{0 <= kappa <= tau <= 1}``, i.e. the order statistics of two
        independent uniforms.  Marginals ``F_kappa(x) = 2x - x^2``, ``F_tau(x) = x^2``.  This is the
        source model's law.
    ``simplex_tau_clamped_to_one``
        the same draw with ``tau`` forced to 1.  Overexposure is impossible, but kappa keeps its
        ``2x - x^2`` marginal.  This is the "overexposure off" ablation.
    ``uniform_lt_tau_one``
        ``kappa ~ U[0, 1]`` and ``tau = 1``.  This is genuine uniform-threshold linear threshold and
        is a DIFFERENT process from the clamped-simplex law.  It exists as an explicit control, and
        in particular it is *monotone* where the simplex law is not.
    ``simplex_tau_support_raised``
        the simplex draw with tau's support raised to ``[max(kappa, window_lo), 1]``, which makes an
        activation attempt likelier to land inside the window.  An intervention knob, not a
        degeneracy path.

    ``overexposure_free`` and ``window_lo`` are retained for callers that predate
    ``threshold_law``; they map onto the laws above and raise if both are given inconsistently.
    """
    if threshold_law is None:
        if overexposure_free and window_lo != 0.0:
            raise ValueError(
                "overexposure_free=True clamps theta_tau to 1, so window_lo must be 0.0"
            )
        if overexposure_free:
            threshold_law = "simplex_tau_clamped_to_one"
        elif window_lo > 0.0:
            threshold_law = "simplex_tau_support_raised"
        else:
            threshold_law = "simplex"

    if threshold_law not in ("simplex", "simplex_tau_clamped_to_one",
                             "uniform_lt_tau_one", "simplex_tau_support_raised"):
        raise ValueError(f"unknown threshold_law {threshold_law!r}")
    if not 0.0 <= window_lo < 1.0:
        raise ValueError(f"window_lo must lie in [0, 1), got {window_lo}")
    if threshold_law == "simplex_tau_support_raised" and window_lo <= 0.0:
        raise ValueError("simplex_tau_support_raised needs a positive window_lo")
    if threshold_law != "simplex_tau_support_raised" and window_lo != 0.0:
        raise ValueError(
            f"window_lo only applies to simplex_tau_support_raised, not {threshold_law}"
        )

    windows: dict[int, tuple[float, float]] = {}
    for node in nodes:
        if threshold_law == "uniform_lt_tau_one":
            # kappa uniform on [0, 1] -- deliberately NOT min of two uniforms
            windows[node] = (rng.random(), 1.0)
            continue

        kappa = rng.random()
        tau = rng.random()
        if kappa > tau:
            kappa, tau = tau, kappa
        if threshold_law == "simplex_tau_clamped_to_one":
            tau = 1.0
        elif threshold_law == "simplex_tau_support_raised":
            lo = max(kappa, window_lo)
            tau = lo + tau * (1.0 - lo)
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
    """Result of a single deterministic realization.

    Three node sets are reported and they are NOT interchangeable:

    ``positive``
        Nodes that are positive at the END of the process.  This is what ``spread`` counts, and
        it is what the positive influence of a seed set means.  A seed is always here.
    ``negative``
        Nodes that became negatively active (overexposed).  A node here is not in ``positive``.
    ``ever_positive``
        Every node that was positive at some point, including nodes later turned negative.  This
        set drives the exposure accumulation ``delta``: the model keeps the influence of
        previously activated neighbours, so a node that has since turned negative still
        contributes to its neighbours' exposure.
    """

    spread: int
    positive: set[int] = field(default_factory=set)
    negative: set[int] = field(default_factory=set)
    delta: dict[int, float] = field(default_factory=dict)
    rounds: int = 0
    ever_positive: set[int] = field(default_factory=set)

    @property
    def ever_positive_or_positive(self) -> set[int]:
        """Union kept for callers that only want 'was ever activated'."""
        return set(self.positive) | set(self.ever_positive)

    @property
    def final_positive(self) -> set[int]:
        return set(self.positive)


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
    positive: set[int] = set()      # currently positive (seeds always)
    negative: set[int] = set()
    ever_positive: set[int] = set()  # every node that was positive at some point
    seeds_set = set(seeds)

    directed = graph.is_directed()

    def incident_edges(node: int):
        """Edges through which ``node`` exerts influence on its neighbours."""
        return graph.out_edges(node, data=True) if directed else graph.edges(node, data=True)

    frontier: list[int] = []
    for seed in seeds:
        if state[seed] != POSITIVE:
            state[seed] = POSITIVE
            positive.add(seed)
            ever_positive.add(seed)
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

        # Phase 2: evaluate nodes against the freshly updated influence.
        #
        # Two rules matter here and both were wrong before (see
        # scripts/audit/verify_p0_counterexamples.py):
        #
        #   1. A node that is already POSITIVE is NOT settled.  Exposure keeps accumulating from
        #      the ever-activated set, so a positive node can later cross its own tau and turn
        #      negative.  Freezing it at its first transition systematically under-counts
        #      overexposure and inflates spread.
        #   2. The state rule is the literal ``kappa <= delta <= tau``.  An earlier version added
        #      a ``delta < 1`` guard to force a negative transition at delta = 1, on the theory
        #      that the marginal probability 2*delta*(1-delta) vanishes there.  That conflated two
        #      different things: 2*delta*(1-delta) is the probability of the window event under
        #      the *sampled* window distribution, whereas the state rule is conditional on a
        #      window that has already been drawn.  With tau = 1 the literal rule puts delta = 1
        #      inside the window, and the guard wrongly made it negative.
        #
        # Seeds are exempt: they are positive by definition and never stop being so.
        newly_positive: list[int] = []
        for node in graph.nodes():
            if node in seeds_set:
                continue
            if state[node] == NEGATIVE:
                continue                      # one-way transition, permanently negative
            current = delta[node]
            if current <= 0.0:
                continue
            kappa, tau = windows[node]
            if current > tau:
                if state[node] == POSITIVE:
                    positive.discard(node)
                state[node] = NEGATIVE
                negative.add(node)
            elif current >= kappa:
                activates = True
                if stochastic:
                    # Extra Bernoulli layer on top of the window event.  Applied only on the
                    # first transition into the window; a node already positive is not re-rolled.
                    if state[node] != POSITIVE:
                        activates = rng.random() < positive_activation_probability(current)
                    else:
                        activates = True
                if activates and state[node] != POSITIVE:
                    state[node] = POSITIVE
                    positive.add(node)
                    ever_positive.add(node)
                    newly_positive.append(node)
                # already positive and still inside the window: no transition, no promotion
                # (its influence was exerted when it first turned positive)
            # else current < kappa: stays inactive and stays re-evaluable.

        frontier = newly_positive

    return OverexposureRun(
        spread=len(positive),
        positive=positive,
        negative=negative,
        delta=delta,
        rounds=rounds,
        ever_positive=ever_positive,
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
