from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from grl.baselines import rank_degree_discount_candidates
from grl.diffusion import estimate_marginal_gains

FeatureRanker = Callable[[list[int]], list[int]]


class FeatureDQN(nn.Module):
    """Legacy FeatureDQN architecture used by the supplied checkpoint."""

    def __init__(self, state_dim: int, node_feat_dim: int):
        super().__init__()
        self.feature_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, node_feat_dim)
        )

    def forward(self, state, node_features):
        hidden = self.feature_layer(state)
        value = self.value_stream(hidden)
        weights = self.advantage_stream(hidden)
        advantage = torch.mm(weights, node_features.t())
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_feature_dqn_ranker(
    graph_data,
    benchmark_config: dict[str, Any],
    universe: list[int],
    project_root: Path,
) -> tuple[FeatureRanker | None, dict[str, str]]:
    model_path = _resolve_project_path(
        project_root,
        str(benchmark_config.get("feature_dqn_path", "param/dqn_model.pth")),
    )
    embedding_values = benchmark_config.get(
        "embedding_paths",
        [
            "param/node2vec_NetHEPT.txt.pth",
            "param/node2vec_nethept.pth",
            "param/node2vec_emb.pth",
        ],
    )
    embedding_paths = [
        _resolve_project_path(project_root, str(value)) for value in embedding_values
    ]
    embedding_path = next((path for path in embedding_paths if path.exists()), None)

    missing = []
    if not model_path.exists():
        missing.append(f"checkpoint not found: {model_path}")
    if embedding_path is None:
        checked = ", ".join(str(path) for path in embedding_paths)
        missing.append(f"embedding not found; checked: {checked}")
    if missing:
        return None, {"status": "skipped", "reason": "; ".join(missing)}

    try:
        embeddings = torch.load(
            embedding_path, map_location="cpu", weights_only=False
        ).float()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dim = int(state_dict["feature_layer.0.weight"].shape[1])
        node_feat_dim = int(state_dict["advantage_stream.2.weight"].shape[0])
        if embeddings.shape[0] < graph_data.num_nodes or embeddings.shape[1] != state_dim:
            raise ValueError(
                f"embedding shape={tuple(embeddings.shape)}, "
                f"expected rows>={graph_data.num_nodes}, dim={state_dim}"
            )
        degrees = dict(
            graph_data.graph.out_degree()
            if graph_data.graph.is_directed()
            else graph_data.graph.degree()
        )
        max_degree = max(degrees.values(), default=1) or 1
        norm_degree = torch.tensor(
            [[degrees.get(node, 0) / max_degree] for node in range(graph_data.num_nodes)],
            dtype=torch.float32,
        )
        node_features = torch.cat(
            [embeddings[: graph_data.num_nodes], norm_degree], dim=1
        )
        if node_features.shape[1] != node_feat_dim:
            raise ValueError(
                f"node feature dim={node_features.shape[1]}, "
                f"checkpoint expects {node_feat_dim}"
            )
        model = FeatureDQN(state_dim, node_feat_dim)
        model.load_state_dict(state_dict)
        model.eval()
    except (KeyError, RuntimeError, OSError, TypeError, ValueError) as exc:
        return None, {"status": "skipped", "reason": f"load failed: {exc}"}

    def rank(selected_seeds: list[int]) -> list[int]:
        state = (
            embeddings[selected_seeds].sum(dim=0)
            if selected_seeds
            else torch.zeros(state_dim)
        )
        with torch.no_grad():
            q_values = model(state.unsqueeze(0), node_features).squeeze(0)
        return sorted(universe, key=lambda node: (-float(q_values[node]), node))

    return rank, {
        "status": "loaded",
        "checkpoint": str(model_path),
        "embedding": str(embedding_path),
    }


def build_candidate_record(
    *,
    step: int,
    retriever: str,
    pool_size: int,
    pool: list[int],
    oracle_node: int,
    oracle_gain: float,
    gains: dict[int, float],
    runtime_seconds: float,
    oracle_gain_runtime_seconds: float,
    repeat: int | None = None,
) -> dict[str, Any]:
    candidate_gain = max((gains[node] for node in pool), default=0.0)
    recalled = oracle_node in pool
    candidate_loss = max(0.0, float(oracle_gain - candidate_gain))
    if recalled:
        candidate_loss = 0.0
    record: dict[str, Any] = {
        "step": step,
        "retriever": retriever,
        "M": pool_size,
        "restricted_oracle_node": oracle_node,
        "restricted_oracle_gain": float(oracle_gain),
        "recalled": float(recalled),
        "candidate_loss": candidate_loss,
        "runtime_seconds": float(runtime_seconds),
        "oracle_gain_runtime_seconds": float(oracle_gain_runtime_seconds),
    }
    if repeat is not None:
        record["repeat"] = repeat
    return record


def summarize_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault((item["retriever"], item["M"]), []).append(item)
    rows = []
    for (retriever, pool_size), values in sorted(grouped.items()):
        rows.append(
            {
                "retriever": retriever,
                "M": pool_size,
                "recall_at_m": sum(v["recalled"] for v in values) / len(values),
                "mean_candidate_loss": sum(v["candidate_loss"] for v in values) / len(values),
                "mean_runtime_seconds": sum(v["runtime_seconds"] for v in values) / len(values),
                "observations": len(values),
            }
        )
    return rows


def run_candidate_benchmark(
    graph_data,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    cfg = config.get("candidate_benchmark", {})
    pool_sizes = sorted({int(value) for value in cfg.get("pool_sizes", [10, 20, 50, 100])})
    if not pool_sizes or min(pool_sizes) <= 0:
        raise ValueError("candidate_benchmark.pool_sizes must contain positive values")
    max_nodes = min(int(cfg.get("max_nodes", 200)), graph_data.num_nodes)
    steps = min(int(cfg.get("steps", 3)), max_nodes)
    mc_runs = int(cfg.get("mc_runs", 3))
    random_repeats = int(cfg.get("random_repeats", 5))
    if max_nodes <= 0 or steps <= 0 or mc_runs <= 0 or random_repeats <= 0:
        raise ValueError("max_nodes, steps, mc_runs, and random_repeats must be positive")
    random_seed = int(config["experiment"]["random_seed"])
    probability = float(config.get("diffusion", {}).get("probability", 0.01))

    ranked_nodes = sorted(
        graph_data.graph.out_degree()
        if graph_data.graph.is_directed()
        else graph_data.graph.degree(),
        key=lambda item: (-item[1], item[0]),
    )
    universe = [node for node, _ in ranked_nodes[:max_nodes]]
    restricted_graph = graph_data.graph.subgraph(universe).copy()
    feature_ranker, feature_status = load_feature_dqn_ranker(
        graph_data, cfg, universe, project_root
    )
    oracle_scope = f"top-{max_nodes}-by-degree"

    selected: list[int] = []
    records: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for step_index in range(steps):
        available = [node for node in universe if node not in selected]
        gain_start = time.perf_counter()
        estimates = estimate_marginal_gains(
            graph_data.graph,
            selected,
            available,
            mc_runs,
            random_seed + step_index * 1000,
        )
        gain_seconds = time.perf_counter() - gain_start
        gains = {node: float(result["mean"]) for node, result in estimates.items()}
        oracle_node = max(available, key=lambda node: (gains[node], -node))
        oracle_gain = gains[oracle_node]

        start = time.perf_counter()
        degree_order = sorted(
            available,
            key=lambda node: (
                -(
                    graph_data.graph.out_degree(node)
                    if graph_data.graph.is_directed()
                    else graph_data.graph.degree(node)
                ),
                node,
            ),
        )
        retrieval_orders: dict[str, list[int]] = {"Degree": degree_order}
        runtimes = {"Degree": time.perf_counter() - start}

        start = time.perf_counter()
        retrieval_orders["DegreeDiscount"] = rank_degree_discount_candidates(
            restricted_graph, selected, available, probability
        )
        runtimes["DegreeDiscount"] = time.perf_counter() - start

        if feature_ranker is not None:
            start = time.perf_counter()
            retrieval_orders["FeatureDQN"] = [
                node for node in feature_ranker(selected) if node in available
            ]
            runtimes["FeatureDQN"] = time.perf_counter() - start

        for retriever, order in retrieval_orders.items():
            for pool_size in pool_sizes:
                pool = order[: min(pool_size, len(order))]
                records.append(
                    build_candidate_record(
                        step=step_index + 1,
                        retriever=retriever,
                        pool_size=pool_size,
                        pool=pool,
                        oracle_node=oracle_node,
                        oracle_gain=oracle_gain,
                        gains=gains,
                        runtime_seconds=runtimes[retriever],
                        oracle_gain_runtime_seconds=gain_seconds,
                    )
                )

        for repeat in range(random_repeats):
            start = time.perf_counter()
            order = available.copy()
            random.Random(random_seed + step_index * 1000 + repeat).shuffle(order)
            runtime = time.perf_counter() - start
            for pool_size in pool_sizes:
                pool = order[: min(pool_size, len(order))]
                records.append(
                    build_candidate_record(
                        step=step_index + 1,
                        retriever="Random",
                        pool_size=pool_size,
                        pool=pool,
                        oracle_node=oracle_node,
                        oracle_gain=oracle_gain,
                        gains=gains,
                        runtime_seconds=runtime,
                        oracle_gain_runtime_seconds=gain_seconds,
                        repeat=repeat,
                    )
                )

        selected.append(oracle_node)
        progress.append(
            {
                "step": step_index + 1,
                "restricted_oracle_node": oracle_node,
                "restricted_oracle_gain": oracle_gain,
                "oracle_gain_runtime_seconds": gain_seconds,
            }
        )

    return {
        "dataset": graph_data.name,
        "oracle_scope": oracle_scope,
        "oracle_note": "Restricted oracle over the configured top-N-by-degree universe; not a full-graph oracle.",
        "pool_sizes": pool_sizes,
        "steps": steps,
        "mc_runs": mc_runs,
        "random_repeats": random_repeats,
        "restricted_oracle_seed_sequence": selected,
        "feature_dqn": feature_status,
        "progress": progress,
        "summary": summarize_candidate_records(records),
        "records": records,
    }
