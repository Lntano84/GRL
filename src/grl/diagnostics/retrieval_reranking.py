from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

import torch

from grl.baselines import rank_degree_discount_candidates
from grl.diffusion import estimate_marginal_gains, estimate_spread
from grl.models import MarginalGainPredictor, build_node_features

from .candidate_benchmark import load_feature_dqn_ranker

MarginalScorer = Callable[[list[int], list[int]], dict[int, float]]


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_marginal_gain_ranker(
    graph_data,
    evaluation_config: dict[str, Any],
    project_root: Path,
) -> tuple[MarginalScorer | None, dict[str, str]]:
    model_path = _resolve_project_path(
        project_root,
        str(evaluation_config.get("marginal_model_path", "param/marginal_gain_nethept.pth")),
    )
    embedding_path = _resolve_project_path(
        project_root,
        str(
            evaluation_config.get(
                "marginal_embedding_path", "param/marginal_node2vec_nethept.pth"
            )
        ),
    )
    missing = []
    if not model_path.exists():
        missing.append(f"checkpoint not found: {model_path}")
    if not embedding_path.exists():
        missing.append(f"embedding not found: {embedding_path}")
    if missing:
        return None, {"status": "skipped", "reason": "; ".join(missing)}

    try:
        device = torch.device(str(evaluation_config.get("device", "cpu")))
        embeddings = torch.load(
            embedding_path, map_location=device, weights_only=False
        ).float()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        first_weight = state_dict["seed_encoder.node_mlp.0.weight"]
        hidden_dim = int(first_weight.shape[0])
        structural_dim = int(first_weight.shape[1] - embeddings.shape[1])
        if embeddings.shape[0] < graph_data.num_nodes:
            raise ValueError(
                f"embedding rows={embeddings.shape[0]}, expected >= {graph_data.num_nodes}"
            )
        if structural_dim != 1:
            raise ValueError(f"checkpoint structural_dim={structural_dim}, expected 1")
        embeddings = embeddings[: graph_data.num_nodes].to(device)
        norm_degrees, _ = build_node_features(graph_data.graph, device=device)
        model = MarginalGainPredictor(
            int(embeddings.shape[1]), hidden_dim, structural_dim=structural_dim
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
    except (KeyError, RuntimeError, OSError, TypeError, ValueError) as exc:
        return None, {"status": "skipped", "reason": f"load failed: {exc}"}

    def score(selected_seeds: list[int], candidates: list[int]) -> dict[int, float]:
        if not candidates:
            return {}
        seed_mask = torch.zeros(
            (graph_data.num_nodes, 1), dtype=torch.float32, device=device
        )
        if selected_seeds:
            seed_mask[selected_seeds, 0] = 1.0
        candidate_indices = torch.tensor(candidates, dtype=torch.long, device=device)
        with torch.no_grad():
            node_features = torch.cat([embeddings, norm_degrees], dim=-1)
            seed_repr = model.seed_encoder(node_features, seed_mask)
            candidate_repr = model.candidate_encoder(node_features[candidate_indices])
            expanded_seed = seed_repr.expand(candidate_repr.shape[0], -1)
            interaction = expanded_seed * candidate_repr
            difference = (expanded_seed - candidate_repr).abs()
            predictions = model.head(
                torch.cat(
                    [expanded_seed, candidate_repr, interaction, difference], dim=-1
                )
            ).reshape(-1)
        return {
            node: float(value)
            for node, value in zip(candidates, predictions.detach().cpu().tolist())
        }

    return score, {
        "status": "loaded",
        "checkpoint": str(model_path),
        "embedding": str(embedding_path),
        "model": MarginalGainPredictor.model_name,
        "model_version": MarginalGainPredictor.model_version,
    }


def build_reranking_record(
    *,
    step: int,
    retriever: str,
    ranker: str,
    pool_size: int,
    pool: list[int],
    oracle_node: int,
    oracle_gain: float,
    selected_node: int,
    gains: dict[int, float],
    retrieval_runtime_seconds: float,
    reranking_runtime_seconds: float,
    gain_runtime_seconds: float,
    repeat: int | None = None,
) -> dict[str, Any]:
    if not pool or selected_node not in pool:
        raise ValueError("pool must be non-empty and contain selected_node")
    pool_best_node = max(pool, key=lambda node: (gains[node], -node))
    pool_best_gain = float(gains[pool_best_node])
    selected_gain = float(gains[selected_node])
    candidate_loss = max(0.0, float(oracle_gain - pool_best_gain))
    ranking_loss = max(0.0, float(pool_best_gain - selected_gain))
    total_regret = max(0.0, float(oracle_gain - selected_gain))
    if oracle_node in pool:
        candidate_loss = 0.0
    if selected_node == pool_best_node:
        ranking_loss = 0.0
    record: dict[str, Any] = {
        "step": step,
        "retriever": retriever,
        "ranker": ranker,
        "M": pool_size,
        "pool": pool,
        "restricted_oracle_node": oracle_node,
        "restricted_oracle_gain": float(oracle_gain),
        "pool_best_node": pool_best_node,
        "pool_best_gain": pool_best_gain,
        "selected_node": selected_node,
        "selected_gain": selected_gain,
        "candidate_loss": candidate_loss,
        "ranking_loss": ranking_loss,
        "total_regret": total_regret,
        "decomposition_error": float(total_regret - candidate_loss - ranking_loss),
        "retrieval_runtime_seconds": float(retrieval_runtime_seconds),
        "reranking_runtime_seconds": float(reranking_runtime_seconds),
        "gain_evaluation_runtime_seconds": float(gain_runtime_seconds),
    }
    if repeat is not None:
        record["repeat"] = repeat
    return record


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_diagnostic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["retriever"], record["ranker"], record["M"])
        grouped.setdefault(key, []).append(record)
    summary = []
    for (retriever, ranker, pool_size), values in sorted(grouped.items()):
        summary.append(
            {
                "retriever": retriever,
                "ranker": ranker,
                "M": pool_size,
                "mean_candidate_loss": _mean([row["candidate_loss"] for row in values]),
                "mean_ranking_loss": _mean([row["ranking_loss"] for row in values]),
                "mean_total_regret": _mean([row["total_regret"] for row in values]),
                "mean_retrieval_runtime_seconds": _mean(
                    [row["retrieval_runtime_seconds"] for row in values]
                ),
                "mean_reranking_runtime_seconds": _mean(
                    [row["reranking_runtime_seconds"] for row in values]
                ),
                "observations": len(values),
            }
        )
    return summary


def _rank_retriever(
    retriever: str,
    graph_data,
    restricted_graph,
    selected: list[int],
    available: list[int],
    probability: float,
    random_seed: int,
    step_index: int,
    repeat: int | None,
    feature_ranker,
) -> tuple[list[int], float]:
    started = time.perf_counter()
    if retriever == "Degree":
        order = sorted(
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
    elif retriever == "DegreeDiscount":
        order = rank_degree_discount_candidates(
            restricted_graph, selected, available, probability
        )
    elif retriever == "FeatureDQN":
        if feature_ranker is None:
            raise ValueError("FeatureDQN ranker is unavailable")
        order = [node for node in feature_ranker(selected) if node in available]
    elif retriever == "Random":
        order = available.copy()
        random.Random(random_seed + step_index * 1000 + int(repeat or 0)).shuffle(order)
    else:
        raise ValueError(f"unsupported retriever: {retriever}")
    return order, time.perf_counter() - started


def _select_from_pool(
    ranker: str,
    selected: list[int],
    pool: list[int],
    marginal_scorer: MarginalScorer | None,
) -> tuple[int, float]:
    started = time.perf_counter()
    if ranker == "OriginalOrder":
        return pool[0], time.perf_counter() - started
    if ranker != "MarginalGainPredictor" or marginal_scorer is None:
        raise ValueError(f"ranker is unavailable: {ranker}")
    scores = marginal_scorer(selected, pool)
    chosen = max(pool, key=lambda node: (scores[node], -node))
    return chosen, time.perf_counter() - started


def _retriever_runs(
    feature_available: bool, random_repeats: int
) -> list[tuple[str, int | None]]:
    runs: list[tuple[str, int | None]] = [
        ("Degree", None),
        ("DegreeDiscount", None),
    ]
    if feature_available:
        runs.append(("FeatureDQN", None))
    runs.extend(("Random", repeat) for repeat in range(random_repeats))
    return runs


def run_oracle_trajectory_diagnostic(
    graph_data,
    cfg: dict[str, Any],
    universe: list[int],
    restricted_graph,
    feature_ranker,
    marginal_scorer: MarginalScorer | None,
    random_seed: int,
    probability: float,
) -> dict[str, Any]:
    rankers = ["OriginalOrder"]
    if marginal_scorer is not None:
        rankers.append("MarginalGainPredictor")
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    progress = []
    for step_index in range(cfg["steps"]):
        available = [node for node in universe if node not in selected]
        gain_started = time.perf_counter()
        estimates = estimate_marginal_gains(
            graph_data.graph,
            selected,
            available,
            cfg["mc_runs_diagnostic"],
            random_seed + step_index * 1000,
        )
        gain_runtime = time.perf_counter() - gain_started
        gains = {node: float(value["mean"]) for node, value in estimates.items()}
        oracle_node = max(available, key=lambda node: (gains[node], -node))
        oracle_gain = gains[oracle_node]
        for retriever, repeat in _retriever_runs(
            feature_ranker is not None, cfg["random_repeats"]
        ):
            order, retrieval_runtime = _rank_retriever(
                retriever,
                graph_data,
                restricted_graph,
                selected,
                available,
                probability,
                random_seed,
                step_index,
                repeat,
                feature_ranker,
            )
            for pool_size in cfg["pool_sizes"]:
                pool = order[: min(pool_size, len(order))]
                for ranker in rankers:
                    chosen, reranking_runtime = _select_from_pool(
                        ranker, selected, pool, marginal_scorer
                    )
                    records.append(
                        build_reranking_record(
                            step=step_index + 1,
                            retriever=retriever,
                            ranker=ranker,
                            pool_size=pool_size,
                            pool=pool,
                            oracle_node=oracle_node,
                            oracle_gain=oracle_gain,
                            selected_node=chosen,
                            gains=gains,
                            retrieval_runtime_seconds=retrieval_runtime,
                            reranking_runtime_seconds=reranking_runtime,
                            gain_runtime_seconds=gain_runtime,
                            repeat=repeat,
                        )
                    )
        selected.append(oracle_node)
        progress.append(
            {
                "step": step_index + 1,
                "seed_set_before": selected[:-1],
                "restricted_oracle_node": oracle_node,
                "restricted_oracle_gain": oracle_gain,
                "gain_evaluation_runtime_seconds": gain_runtime,
            }
        )
    return {
        "trajectory": "restricted_oracle",
        "restricted_oracle_seed_sequence": selected,
        "progress": progress,
        "summary": summarize_diagnostic_records(records),
        "records": records,
    }


def _method_specs(
    pool_sizes: list[int],
    feature_available: bool,
    marginal_available: bool,
    random_repeats: int,
) -> list[dict[str, Any]]:
    rankers = ["OriginalOrder"]
    if marginal_available:
        rankers.append("MarginalGainPredictor")
    specs = []
    for retriever, repeat in _retriever_runs(feature_available, random_repeats):
        for pool_size in pool_sizes:
            for ranker in rankers:
                specs.append(
                    {
                        "retriever": retriever,
                        "ranker": ranker,
                        "M": pool_size,
                        "repeat": repeat,
                        "selected": [],
                        "steps": [],
                        "selection_runtime_seconds": 0.0,
                        "gain_evaluation_runtime_seconds": 0.0,
                    }
                )
    return specs


def run_end_to_end_sequential_evaluation(
    graph_data,
    cfg: dict[str, Any],
    universe: list[int],
    restricted_graph,
    feature_ranker,
    marginal_scorer: MarginalScorer | None,
    random_seed: int,
    probability: float,
) -> dict[str, Any]:
    methods = _method_specs(
        cfg["pool_sizes"],
        feature_ranker is not None,
        marginal_scorer is not None,
        cfg["random_repeats"],
    )
    for step_index in range(cfg["steps"]):
        gain_cache: dict[tuple[int, ...], tuple[dict[int, float], float]] = {}
        retrieval_cache: dict[
            tuple[str, tuple[int, ...], int | None], tuple[list[int], float]
        ] = {}
        selection_cache: dict[
            tuple[str, tuple[int, ...], tuple[int, ...]], tuple[int, float]
        ] = {}
        for method in methods:
            selected = method["selected"]
            selected_key = tuple(selected)
            available = [node for node in universe if node not in selected]
            if selected_key not in gain_cache:
                gain_started = time.perf_counter()
                estimates = estimate_marginal_gains(
                    graph_data.graph,
                    selected,
                    available,
                    cfg["mc_runs_end_to_end"],
                    random_seed + step_index * 1000,
                )
                gain_cache[selected_key] = (
                    {node: float(value["mean"]) for node, value in estimates.items()},
                    time.perf_counter() - gain_started,
                )
            gains, gain_runtime = gain_cache[selected_key]
            oracle_node = max(available, key=lambda node: (gains[node], -node))
            oracle_gain = gains[oracle_node]
            retrieval_key = (method["retriever"], selected_key, method["repeat"])
            if retrieval_key not in retrieval_cache:
                retrieval_cache[retrieval_key] = _rank_retriever(
                    method["retriever"],
                    graph_data,
                    restricted_graph,
                    selected,
                    available,
                    probability,
                    random_seed,
                    step_index,
                    method["repeat"],
                    feature_ranker,
                )
            order, retrieval_runtime = retrieval_cache[retrieval_key]
            pool = order[: min(method["M"], len(order))]
            selection_key = (method["ranker"], selected_key, tuple(pool))
            if selection_key not in selection_cache:
                selection_cache[selection_key] = _select_from_pool(
                    method["ranker"], selected, pool, marginal_scorer
                )
            chosen, reranking_runtime = selection_cache[selection_key]
            record = build_reranking_record(
                step=step_index + 1,
                retriever=method["retriever"],
                ranker=method["ranker"],
                pool_size=method["M"],
                pool=pool,
                oracle_node=oracle_node,
                oracle_gain=oracle_gain,
                selected_node=chosen,
                gains=gains,
                retrieval_runtime_seconds=retrieval_runtime,
                reranking_runtime_seconds=reranking_runtime,
                gain_runtime_seconds=gain_runtime,
                repeat=method["repeat"],
            )
            record["seed_set_before"] = list(selected)
            method["steps"].append(record)
            method["selected"].append(chosen)
            method["selection_runtime_seconds"] += retrieval_runtime + reranking_runtime
            method["gain_evaluation_runtime_seconds"] += gain_runtime

    summary = []
    all_records = []
    for method_index, method in enumerate(methods):
        spread_started = time.perf_counter()
        spread = estimate_spread(
            graph_data.graph,
            method["selected"],
            cfg["mc_runs_spread"],
            random_seed + 900000,
        )
        spread_runtime = time.perf_counter() - spread_started
        method_id = f"method-{method_index + 1:03d}"
        for record in method["steps"]:
            record["method_id"] = method_id
            all_records.append(record)
        summary.append(
            {
                "method_id": method_id,
                "retriever": method["retriever"],
                "ranker": method["ranker"],
                "M": method["M"],
                "repeat": method["repeat"],
                "selected_seeds": method["selected"],
                "final_spread_mean": float(spread["mean"]),
                "final_spread_std": float(spread["std"]),
                "selection_runtime_seconds": float(
                    method["selection_runtime_seconds"]
                ),
                "gain_evaluation_runtime_seconds": float(
                    method["gain_evaluation_runtime_seconds"]
                ),
                "spread_evaluation_runtime_seconds": float(spread_runtime),
            }
        )
    return {
        "trajectory": "method_specific",
        "runtime_note": (
            "selection_runtime_seconds includes retrieval and reranking only; "
            "Monte Carlo gain and final-spread evaluation times are reported separately."
        ),
        "summary": summary,
        "records": all_records,
    }


def run_retrieval_reranking_evaluation(
    graph_data,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    raw_cfg = config.get("retrieval_reranking", {})
    pool_sizes = sorted(
        {int(value) for value in raw_cfg.get("pool_sizes", [10, 20, 50, 100])}
    )
    max_nodes = min(int(raw_cfg.get("max_nodes", 200)), graph_data.num_nodes)
    steps = min(int(raw_cfg.get("steps", 10)), max_nodes)
    cfg = {
        **raw_cfg,
        "pool_sizes": pool_sizes,
        "max_nodes": max_nodes,
        "steps": steps,
        "mc_runs_diagnostic": int(raw_cfg.get("mc_runs_diagnostic", 10)),
        "mc_runs_end_to_end": int(raw_cfg.get("mc_runs_end_to_end", 10)),
        "mc_runs_spread": int(raw_cfg.get("mc_runs_spread", 100)),
        "random_repeats": int(raw_cfg.get("random_repeats", 10)),
    }
    positive_values = [
        *pool_sizes,
        max_nodes,
        steps,
        cfg["mc_runs_diagnostic"],
        cfg["mc_runs_end_to_end"],
        cfg["mc_runs_spread"],
        cfg["random_repeats"],
    ]
    if not pool_sizes or min(positive_values) <= 0:
        raise ValueError("pool sizes, node/step limits, MC runs, and repeats must be positive")

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
    marginal_scorer, marginal_status = load_marginal_gain_ranker(
        graph_data, cfg, project_root
    )
    started = time.perf_counter()
    diagnostic = run_oracle_trajectory_diagnostic(
        graph_data,
        cfg,
        universe,
        restricted_graph,
        feature_ranker,
        marginal_scorer,
        random_seed,
        probability,
    )
    end_to_end = run_end_to_end_sequential_evaluation(
        graph_data,
        cfg,
        universe,
        restricted_graph,
        feature_ranker,
        marginal_scorer,
        random_seed,
        probability,
    )
    oracle_scope = f"top-{max_nodes}-by-degree"
    shared = {
        "dataset": graph_data.name,
        "oracle_scope": oracle_scope,
        "oracle_note": (
            "Restricted oracle over the configured top-N-by-degree universe; "
            "not a full-graph oracle."
        ),
        "pool_sizes": pool_sizes,
        "steps": steps,
        "mc_runs_diagnostic": cfg["mc_runs_diagnostic"],
        "mc_runs_end_to_end": cfg["mc_runs_end_to_end"],
        "mc_runs_spread": cfg["mc_runs_spread"],
        "random_repeats": cfg["random_repeats"],
        "feature_dqn": feature_status,
        "marginal_gain_predictor": marginal_status,
    }
    diagnostic = {**shared, **diagnostic}
    end_to_end = {**shared, **end_to_end}
    validation_records = diagnostic["records"] + end_to_end["records"]
    validation = {
        "loss_decomposition_holds": all(
            abs(record["decomposition_error"]) <= 1e-8
            for record in validation_records
        ),
        "maximum_decomposition_error": max(
            (abs(record["decomposition_error"]) for record in validation_records),
            default=0.0,
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {
        "diagnostic": diagnostic,
        "end_to_end": end_to_end,
        "validation": validation,
    }
