from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grl.baselines import select_degree_discount_nodes, select_high_degree_nodes
from grl.diffusion import estimate_marginal_gains


@dataclass(frozen=True)
class MarginalGainSample:
    context_id: str
    seed_set: list[int]
    candidate: int
    seed_set_size: int
    base_spread: float
    extended_spread: float
    marginal_gain: float
    label_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _partial_seed_sequences(graph_data, budget: int, probability: float) -> list[list[int]]:
    degree = select_high_degree_nodes(graph_data.graph, min(budget, graph_data.num_nodes))
    discount = select_degree_discount_nodes(graph_data.graph, min(budget, graph_data.num_nodes), probability)
    return [degree[:size] for size in range(len(degree) + 1)] + [discount[:size] for size in range(len(discount) + 1)]


def _make_seed_set(
    graph_data,
    nodes: list[int],
    partials: list[list[int]],
    size: int,
    index: int,
    rng: random.Random,
) -> list[int]:
    if size == 0:
        return []
    branch = index % 4
    if branch == 0:
        return rng.sample(nodes, size)
    if branch == 1 and partials:
        source = partials[index % len(partials)]
        if len(source) >= size:
            return list(source[:size])
    if branch == 2:
        ranked = select_high_degree_nodes(graph_data.graph, graph_data.num_nodes)
        return list(ranked[:size])
    source = partials[index % len(partials)] if partials else []
    if len(source) >= size:
        return list(source[:size])
    return rng.sample(nodes, size)


def _split_context_ids(
    context_ids: list[str],
    split: tuple[float, float, float],
) -> dict[str, set[str]]:
    if len(split) != 3 or not math.isclose(sum(split), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split must contain three fractions summing to 1")
    n = len(context_ids)
    train_end = int(n * split[0])
    valid_end = train_end + int(n * split[1])
    return {
        "train": set(context_ids[:train_end]),
        "validation": set(context_ids[train_end:valid_end]),
        "test": set(context_ids[valid_end:]),
    }


def build_marginal_dataset(
    graph_data,
    config: dict,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[MarginalGainSample]]:
    cfg = config.get("gnn", {}) | config.get("marginal_gain", {})
    candidates_per_context = max(1, int(cfg.get("candidates_per_context", 10)))
    total_samples = int(cfg.get("samples", 128))
    total_contexts = int(cfg.get("contexts", math.ceil(total_samples / candidates_per_context)))
    budget = min(int(config.get("seed", {}).get("budget", 10)), graph_data.num_nodes)
    mc_runs = int(cfg.get("mc_runs_train", config.get("diffusion", {}).get("mc_runs_train", 30)))
    base_seed = int(config.get("experiment", {}).get("random_seed", 42))
    probability = float(config.get("diffusion", {}).get("probability", 0.01))
    nodes = list(range(graph_data.num_nodes))
    rng = random.Random(base_seed)
    partials = _partial_seed_sequences(graph_data, budget, probability)
    contexts: list[tuple[str, list[MarginalGainSample]]] = []
    max_seed_size = min(
        max(budget - 1, 0),
        max(graph_data.num_nodes - candidates_per_context, 0),
    )
    for index in range(total_contexts):
        size = index % (max_seed_size + 1)
        seeds = _make_seed_set(graph_data, nodes, partials, size, index, rng)
        available = [node for node in nodes if node not in seeds]
        if not available:
            continue
        candidate_count = min(candidates_per_context, len(available))
        candidates = rng.sample(available, candidate_count)
        context_id = f"context_{index:05d}"
        estimates = estimate_marginal_gains(
            graph_data.graph,
            seeds,
            candidates,
            mc_runs,
            base_seed + index * mc_runs,
        )
        samples = []
        for candidate in candidates:
            estimate = estimates[candidate]
            samples.append(MarginalGainSample(
                context_id=context_id,
                seed_set=list(seeds),
                candidate=candidate,
                seed_set_size=len(seeds),
                base_spread=estimate["base_spread"],
                extended_spread=estimate["extended_spread"],
                marginal_gain=max(estimate["mean"], 0.0),
                label_std=estimate["std"],
            ))
        contexts.append((context_id, samples))

    result = {"train": [], "validation": [], "test": []}
    contexts_by_size: dict[int, list[tuple[str, list[MarginalGainSample]]]] = {}
    for context in contexts:
        contexts_by_size.setdefault(context[1][0].seed_set_size, []).append(context)
    for size_contexts in contexts_by_size.values():
        rng.shuffle(size_contexts)
        split_ids = _split_context_ids(
            [context_id for context_id, _ in size_contexts], split
        )
        for context_id, samples in size_contexts:
            destination = next(name for name, ids in split_ids.items() if context_id in ids)
            result[destination].extend(samples)
    return result


def save_marginal_dataset(
    splits: dict[str, list[MarginalGainSample]],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {name: [sample.to_dict() for sample in samples] for name, samples in splits.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def load_marginal_dataset(path: str | Path) -> dict[str, list[MarginalGainSample]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: [MarginalGainSample(**item) for item in items]
        for name, items in payload.items()
    }
