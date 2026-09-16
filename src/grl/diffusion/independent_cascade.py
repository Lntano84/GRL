from __future__ import annotations

import random
from statistics import pstdev

import networkx as nx


def _reachable_nodes(graph: nx.Graph | nx.DiGraph, sources: set[int]) -> set[int]:
    reached = set(sources)
    frontier = list(sources)
    while frontier:
        current = frontier.pop()
        for neighbor in graph.neighbors(current):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached


def _sample_live_graph(
    graph: nx.Graph | nx.DiGraph,
    rng: random.Random,
) -> nx.Graph | nx.DiGraph:
    live_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    live_graph.add_nodes_from(graph.nodes())
    live_graph.add_edges_from(
        (u, v)
        for u, v, data in graph.edges(data=True)
        if rng.random() < float(data.get("weight", 0.0))
    )
    return live_graph


def run_independent_cascade(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    rng: random.Random,
) -> int:
    activated = set(seeds)
    frontier = list(seeds)

    while frontier:
        current = frontier.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor in activated:
                continue
            probability = float(graph[current][neighbor].get("weight", 0.0))
            if rng.random() < probability:
                activated.add(neighbor)
                frontier.append(neighbor)

    return len(activated)


def estimate_spread(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    mc_runs: int,
    random_seed: int,
) -> dict[str, float]:
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")

    spreads = []
    for offset in range(mc_runs):
        rng = random.Random(random_seed + offset)
        spreads.append(run_independent_cascade(graph, seeds, rng))

    mean = sum(spreads) / len(spreads)
    std = pstdev(spreads) if len(spreads) > 1 else 0.0
    return {"mean": float(mean), "std": float(std)}


def estimate_marginal_gain(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidate: int,
    mc_runs: int,
    random_seed: int,
) -> dict[str, float]:
    """Estimate Delta(candidate | seeds) with common live-edge samples.

    Sampling each edge once per MC run makes the base and extended seed sets
    use exactly the same random cascade realization, so the marginal label is
    monotone up to floating-point noise.
    """
    result = estimate_marginal_gains(graph, seeds, [candidate], mc_runs, random_seed)[candidate]
    return {"mean": result["mean"], "std": result["std"]}


def estimate_marginal_gains(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidates: list[int],
    mc_runs: int,
    random_seed: int,
) -> dict[int, dict[str, float]]:
    """Estimate candidates under one seed context using paired live-edge samples."""
    if mc_runs <= 0:
        raise ValueError("mc_runs must be positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if set(seeds) & set(candidates):
        raise ValueError("candidates must not be in the seed set")

    base_spreads: list[float] = []
    extended_spreads = {candidate: [] for candidate in candidates}
    gains = {candidate: [] for candidate in candidates}
    seed_set = set(seeds)

    for offset in range(mc_runs):
        live_graph = _sample_live_graph(graph, random.Random(random_seed + offset))
        base_reached = _reachable_nodes(live_graph, seed_set)
        base_spread = float(len(base_reached))
        base_spreads.append(base_spread)
        for candidate in candidates:
            extended_reached = base_reached | _reachable_nodes(live_graph, {candidate})
            extended_spread = float(len(extended_reached))
            extended_spreads[candidate].append(extended_spread)
            gains[candidate].append(extended_spread - base_spread)

    base_mean = sum(base_spreads) / mc_runs
    result: dict[int, dict[str, float]] = {}
    for candidate in candidates:
        gain_values = gains[candidate]
        extended_values = extended_spreads[candidate]
        result[candidate] = {
            "base_spread": float(base_mean),
            "extended_spread": float(sum(extended_values) / mc_runs),
            "mean": float(sum(gain_values) / mc_runs),
            "std": float(pstdev(gain_values) if len(gain_values) > 1 else 0.0),
        }
    return result
