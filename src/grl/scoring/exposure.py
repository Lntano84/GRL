"""Analytic, training-free scorers for the overexposure objective.

These lived in ``scripts/experiments/evaluate_overexposure_pool_ranking.py``, which made every
experiment that needed them depend on a script rather than on the library.  The Go/No-Go runner
needs them as first-class components, so they are moved here with tests.

``delta2`` is the state-conditioned closed form the paper's regime table compares against
out-degree.  It is **not** an estimate of ``sigma``: it propagates the *increase in activation
probability* that seeding a candidate causes, and scores the candidate by the resulting increase in
the expected number of positive out-neighbours.  Because it is a closed form it costs no cascades,
which is exactly why it is the thing a sample-efficient method has to beat on quality rather than on
cost.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator

import networkx as nx

from grl.diffusion import overexposure as oe

#: Propagation depth used by :func:`exposure_scores_delta`.  Two hops is what the paper reports.
DEFAULT_HOPS = 2


def out_edges(graph: nx.Graph | nx.DiGraph, node) -> Iterator[tuple]:
    """Yield ``(target, weight)`` for a node's out-edges under either graph type."""
    directed = graph.is_directed()
    edges = graph.out_edges(node, data=True) if directed else graph.edges(node, data=True)
    for u, v, data in edges:
        target = v if directed else (v if u == node else u)
        yield target, float(data.get("weight", 0.0))


def degree_scores(graph: nx.Graph | nx.DiGraph, candidates: Iterable) -> list[float]:
    """Out-degree (or degree, for an undirected graph) as a score.  No cascades."""
    source = graph.out_degree if graph.is_directed() else graph.degree
    return [float(source[v]) for v in candidates]


def random_scores(candidates: Iterable, seed: int) -> list[float]:
    """A deterministic pseudo-random score, so random arms are reproducible."""
    rng = random.Random(seed)
    return [rng.random() for _ in candidates]


def mean_exposure_state(
    graph: nx.Graph | nx.DiGraph,
    seeds: list,
    trials: int,
    seed: int,
) -> dict:
    """Mean per-node exposure ``delta`` under ``seeds``, over ``trials`` threshold draws.

    This is the observation a state-conditioned policy is allowed to see.  It is **not free**: it
    costs ``trials`` cascades, which is why the Go/No-Go runner charges it (audit item P1-3.2).
    """
    nodes = list(graph.nodes())
    accum = {node: 0.0 for node in nodes}
    for offset in range(trials):
        rng = random.Random(seed + offset)
        windows = oe.sample_threshold_windows(nodes, rng)
        run = oe.run_overexposure(graph, list(seeds), windows, rng)
        for node, value in run.delta.items():
            accum[node] += value
    return {node: value / trials for node, value in accum.items()}


def exposure_scores_delta(
    graph: nx.Graph | nx.DiGraph,
    candidates: list,
    delta: dict,
    seeds: set,
    hops: int = DEFAULT_HOPS,
) -> list[float]:
    """The two-hop state-conditioned closed form.

    For each candidate ``w``:

        dE(v)    <- dE(parent) * weight(parent, v) * (1 - E_S(v))     (Bellman over ``hops``)
        score(w) = sum over out-neighbours v of
                   [ g(E_S(v) + dE(v)) - g(E_S(v)) ]

    where ``E_S`` is the realised mean exposure of ``v`` under the current seed set and
    ``g`` is the model's positive-activation probability ``2 * delta * (1 - delta)``.

    A candidate already in ``seeds`` scores ``-inf`` so it can never be re-selected.  The score is
    signed: ``g`` is not monotone, so a candidate whose influence pushes a neighbour past its upper
    threshold contributes *negatively*, which is the whole point of a state-conditioned score.

    Cost: **zero cascades**.  It reads ``delta``, and the caller is responsible for having paid for
    that observation (see :func:`mean_exposure_state`).
    """
    scores: list[float] = []
    for candidate in candidates:
        if candidate in seeds:
            scores.append(float("-inf"))
            continue

        d_e: dict = {}
        for target, weight in out_edges(graph, candidate):
            if target == candidate or target in seeds or weight <= 0.0:
                continue
            gain = weight * (1.0 - delta.get(target, 0.0))
            if gain > d_e.get(target, 0.0):
                d_e[target] = gain

        if hops >= 2:
            layer = sorted(d_e.items(), key=lambda kv: -kv[1])
            for node, node_delta in layer:
                if node_delta <= 1e-9:
                    continue
                for target, weight in out_edges(graph, node):
                    if target == candidate or target in seeds or weight <= 0.0:
                        continue
                    propagated = node_delta * weight * (1.0 - delta.get(target, 0.0))
                    if propagated > d_e.get(target, 0.0):
                        d_e[target] = propagated

        total = 0.0
        for target, change in d_e.items():
            current = delta.get(target, 0.0)
            before = oe.positive_activation_probability(current)
            after = oe.positive_activation_probability(min(1.0, current + change))
            total += after - before
        scores.append(total)
    return scores


def rank_by_score(candidates: list, scores: list[float]) -> list:
    """Rank candidates by descending score, breaking ties by ascending label.

    The tie-break matters: without a deterministic rule the arms are not reproducible, and a
    reproducible arm is a precondition for a paired comparison.
    """
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
    return [candidates[i] for i in order]
