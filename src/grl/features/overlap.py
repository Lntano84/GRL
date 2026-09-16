from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import networkx as nx
import numpy as np


OVERLAP_FEATURE_NAMES = (
    "nearest_seed_distance_norm",
    "unreachable_flag",
    "empty_seed_set_flag",
    "neighborhood_overlap_1hop",
    "neighborhood_overlap_2hop",
    "neighborhood_jaccard_2hop",
    "candidate_2hop_seed_fraction",
    "seed_union_2hop_graph_coverage",
    "direct_seed_neighbor_fraction",
)


class OverlapFeatureExtractor:
    """Compute deterministic, topology-only features for a pair (S, v)."""

    def __init__(self, graph: nx.Graph | nx.DiGraph, distance_cap: int = 6):
        if distance_cap <= 0:
            raise ValueError("distance_cap must be positive")
        self.topology = graph.to_undirected(as_view=False) if graph.is_directed() else graph
        self.distance_cap = int(distance_cap)

    def _validate(self, seed_set: Iterable[int], candidate: int) -> tuple[list[int], int]:
        seeds = list(seed_set)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seed_set must not contain duplicate nodes")
        if any(seed not in self.topology for seed in seeds):
            raise ValueError("all seed nodes must be graph nodes")
        if candidate not in self.topology:
            raise ValueError("candidate must be a graph node")
        if candidate in seeds:
            raise ValueError("candidate must not be in seed_set")
        return seeds, int(candidate)

    @lru_cache(maxsize=None)
    def _neighborhood(self, node: int, hops: int) -> frozenset[int]:
        return frozenset(nx.single_source_shortest_path_length(self.topology, node, cutoff=hops))

    @lru_cache(maxsize=None)
    def _seed_distances(self, seeds: tuple[int, ...]) -> dict[int, int]:
        return dict(nx.multi_source_dijkstra_path_length(self.topology, list(seeds), weight=None))

    def transform_one(self, seed_set: Iterable[int], candidate: int) -> np.ndarray:
        seeds, candidate = self._validate(seed_set, candidate)
        if not seeds:
            return np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        nearest = self._seed_distances(tuple(sorted(seeds))).get(candidate)
        nearest_norm = min(float(nearest), self.distance_cap) / self.distance_cap if nearest is not None else 1.0
        unreachable = 1.0 if nearest is None else 0.0
        candidate_1 = self._neighborhood(candidate, 1)
        candidate_2 = self._neighborhood(candidate, 2)
        seed_1 = set().union(*(self._neighborhood(seed, 1) for seed in seeds))
        seed_2 = set().union(*(self._neighborhood(seed, 2) for seed in seeds))
        overlap_1 = len(candidate_1 & seed_1) / max(len(candidate_1), 1)
        overlap_2 = len(candidate_2 & seed_2) / max(len(candidate_2), 1)
        jaccard_2 = len(candidate_2 & seed_2) / max(len(candidate_2 | seed_2), 1)
        candidate_2hop_seed_fraction = sum(1 for seed in seeds if seed in candidate_2) / len(seeds)
        seed_union_2hop_graph_coverage = len(seed_2) / max(self.topology.number_of_nodes(), 1)
        direct_seed_neighbor_fraction = len(candidate_1 & set(seeds)) / max(len(candidate_1), 1)
        return np.asarray(
            [nearest_norm, unreachable, 0.0, overlap_1, overlap_2, jaccard_2, candidate_2hop_seed_fraction, seed_union_2hop_graph_coverage, direct_seed_neighbor_fraction],
            dtype=np.float32,
        )

    def transform(self, seed_sets: Iterable[Iterable[int]], candidates: Iterable[int]) -> np.ndarray:
        seed_rows = list(seed_sets)
        candidate_rows = list(candidates)
        if len(seed_rows) != len(candidate_rows):
            raise ValueError("seed_sets and candidates must have the same length")
        rows = [self.transform_one(seeds, candidate) for seeds, candidate in zip(seed_rows, candidate_rows)]
        return np.stack(rows) if rows else np.empty((0, len(OVERLAP_FEATURE_NAMES)), dtype=np.float32)


def build_overlap_features(graph: nx.Graph | nx.DiGraph, seed_set: Iterable[int], candidate: int, distance_cap: int = 6) -> np.ndarray:
    return OverlapFeatureExtractor(graph, distance_cap=distance_cap).transform_one(seed_set, candidate)
