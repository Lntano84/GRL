from __future__ import annotations

import networkx as nx

from .degree import validate_budget


def select_degree_discount_nodes(
    graph: nx.Graph | nx.DiGraph,
    budget: int,
    probability: float,
) -> list[int]:
    validate_budget(graph, budget)
    degree_fn = graph.out_degree if graph.is_directed() else graph.degree
    d = {node: degree for node, degree in degree_fn()}
    dd = d.copy()
    t = {node: 0 for node in graph.nodes()}
    selected: list[int] = []

    for _ in range(budget):
        node = max(dd, key=lambda candidate: (dd[candidate], -candidate))
        selected.append(node)
        dd.pop(node)
        for neighbor in graph.neighbors(node):
            if neighbor in dd:
                t[neighbor] += 1
                dd[neighbor] = d[neighbor] - 2 * t[neighbor] - (d[neighbor] - t[neighbor]) * t[neighbor] * probability

    return selected


def rank_degree_discount_candidates(
    graph: nx.Graph | nx.DiGraph,
    selected_seeds: list[int],
    candidates: list[int],
    probability: float,
) -> list[int]:
    """Rank legal candidates after applying discounts from existing seeds."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("selected_seeds must be unique")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if set(selected_seeds) & set(candidates):
        raise ValueError("candidates must not contain selected seeds")
    if any(node not in graph for node in selected_seeds + candidates):
        raise ValueError("all selected seeds and candidates must be graph nodes")
    if not candidates:
        return []

    degree_fn = graph.out_degree if graph.is_directed() else graph.degree
    degrees = {node: degree for node, degree in degree_fn()}
    selected_set = set(selected_seeds)
    selected_neighbors = {
        node: sum(1 for seed in selected_set if graph.has_edge(seed, node))
        for node in candidates
    }
    discounts = {
        node: degrees[node]
        - 2 * selected_neighbors[node]
        - (degrees[node] - selected_neighbors[node]) * selected_neighbors[node] * probability
        for node in candidates
    }
    ranked: list[int] = []

    while discounts:
        node = max(discounts, key=lambda candidate: (discounts[candidate], -candidate))
        ranked.append(node)
        discounts.pop(node)
        for neighbor in graph.neighbors(node):
            if neighbor in discounts:
                selected_neighbors[neighbor] += 1
                count = selected_neighbors[neighbor]
                discounts[neighbor] = (
                    degrees[neighbor]
                    - 2 * count
                    - (degrees[neighbor] - count) * count * probability
                )

    return ranked
