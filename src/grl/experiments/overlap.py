from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from grl.baselines import select_high_degree_nodes
from grl.diffusion import estimate_marginal_gains, estimate_spread
from grl.evaluation.ranking import regression_ranking_metrics
from grl.features import OVERLAP_FEATURE_NAMES, OverlapFeatureExtractor
from grl.models import MarginalGainPredictor, OverlapMarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from grl.training import MarginalGainSample, load_marginal_dataset


BASELINE = "Direct-MLP"
OVERLAP = "Direct-MLP+Overlap"


def _seed_mask(num_nodes: int, seed_set: list[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
    if seed_set:
        mask[seed_set, 0] = 1.0
    return mask


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grouped_metrics(samples: list[MarginalGainSample], predictions: list[float]) -> dict[str, float]:
    targets = [sample.marginal_gain for sample in samples]
    errors = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.context_id].append(index)
    rows = []
    for indices in groups.values():
        rows.append(regression_ranking_metrics(
            [predictions[index] for index in indices],
            [targets[index] for index in indices],
            top_ks=(1,),
        ))
    return {
        "contexts": len(groups),
        "samples": len(samples),
        "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "spearman": float(np.mean([row["spearman"] for row in rows])) if rows else 0.0,
        "top_1": float(np.mean([row["top_1_recall"] for row in rows])) if rows else 0.0,
    }


def _ranking_metrics(samples: list[MarginalGainSample], predictions: list[float]) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.context_id].append(index)
    losses = []
    for indices in groups.values():
        best = max(samples[index].marginal_gain for index in indices)
        selected_index = max(indices, key=lambda index: (predictions[index], -samples[index].candidate))
        losses.append(max(0.0, best - samples[selected_index].marginal_gain))
    return {
        "ranking_loss": float(np.mean(losses)) if losses else 0.0,
        "top_1": float(sum(loss == 0.0 for loss in losses) / len(losses)) if losses else 0.0,
        "contexts": len(losses),
    }


def _model_score(model, model_name: str, embeddings, norm_degrees, extractor, samples, device) -> list[float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for sample in samples:
            mask = _seed_mask(embeddings.shape[0], sample.seed_set, device)
            if model_name == OVERLAP:
                overlap = extractor.transform_one(sample.seed_set, sample.candidate)
                output = model(embeddings, norm_degrees, mask, sample.candidate, torch.tensor(overlap, device=device))
            else:
                output = model(embeddings, norm_degrees, mask, sample.candidate)
            predictions.append(float(output.item()))
    return predictions


def _train_model(model, model_name: str, samples: list[MarginalGainSample], embeddings, norm_degrees, extractor, graph_size: int, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    batch_size = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 8))
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        random.shuffle(samples)
        epoch_losses = []
        for begin in range(0, len(samples), batch_size):
            batch = samples[begin:begin + batch_size]
            masks = torch.stack([_seed_mask(graph_size, item.seed_set, device) for item in batch])
            candidates = torch.tensor([item.candidate for item in batch], dtype=torch.long, device=device)
            labels = torch.tensor([[item.marginal_gain] for item in batch], dtype=torch.float32, device=device)
            emb = embeddings.unsqueeze(0).expand(len(batch), -1, -1)
            deg = norm_degrees.unsqueeze(0).expand(len(batch), -1, -1)
            if model_name == OVERLAP:
                overlap = torch.tensor(extractor.transform(
                    [item.seed_set for item in batch], [item.candidate for item in batch]
                ), dtype=torch.float32, device=device)
                prediction = model(emb, deg, masks, candidates, overlap)
            else:
                prediction = model(emb, deg, masks, candidates)
            loss = F.mse_loss(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    return {"epochs": epochs, "training_examples": len(samples), "epoch_losses": losses, "seconds": time.perf_counter() - started}


def _sequential_spread(graph_data, model, model_name: str, embeddings, norm_degrees, extractor, cfg: dict[str, Any], seed: int, device: torch.device) -> list[dict[str, Any]]:
    max_nodes = min(int(cfg.get("sequential_max_nodes", 200)), graph_data.num_nodes)
    pool_size = min(int(cfg.get("sequential_pool_size", 20)), max_nodes)
    budgets = [int(value) for value in cfg.get("sequential_budgets", [1, 3, 5, 10])]
    max_budget = min(max(budgets, default=0), graph_data.num_nodes)
    universe = select_high_degree_nodes(graph_data.graph, max_nodes)
    selected = []
    rows = []
    ranking_losses = []
    top_1_values = []
    selection_started = time.perf_counter()
    for step in range(max_budget):
        available = [node for node in universe if node not in selected]
        pool = available[:min(pool_size, len(available))]
        gains = estimate_marginal_gains(
            graph_data.graph, selected, pool, int(cfg.get("mc_runs_sequential", 10)), seed + step * 1000
        )
        predictions = _model_score(model, model_name, embeddings, norm_degrees, extractor, [
            MarginalGainSample("sequential", list(selected), node, len(selected), 0.0, 0.0, 0.0, 0.0)
            for node in pool
        ], device)
        chosen = max(pool, key=lambda node: (predictions[pool.index(node)], -node))
        best_gain = max(float(gains[node]["mean"]) for node in pool)
        ranking_loss = max(0.0, best_gain - float(gains[chosen]["mean"]))
        ranking_losses.append(ranking_loss)
        top_1_values.append(float(ranking_loss == 0.0))
        selected.append(chosen)
        if step + 1 in budgets:
            spread = estimate_spread(graph_data.graph, selected, int(cfg.get("mc_runs_spread", 50)), seed + 900000 + step)
            rows.append({
                "seed_set_size": step + 1,
                "selected_seeds": list(selected),
                "final_spread": float(spread["mean"]),
                "spread_std": float(spread["std"]),
                "mean_ranking_loss": float(np.mean(ranking_losses)),
                "top_1": float(np.mean(top_1_values)),
                "candidate_pool": f"Degree Top-{pool_size} within restricted top-{max_nodes}-by-degree",
                "selection_runtime_seconds": time.perf_counter() - selection_started,
            })
    return rows


def run_overlap_experiment(graph_data, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    cfg = config.get("overlap_experiment", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("gnn", {}).get("device", "cpu"))
    dataset_path = Path(config["comparison"]["dataset_path"])
    splits = load_marginal_dataset(dataset_path)
    test_samples = splits["test"]
    train_samples = splits["train"]
    max_train = cfg.get("max_train_samples")
    if max_train is not None:
        train_samples = train_samples[:int(max_train)]
    model_dir = Path(config.get("gnn", {}).get("model_dir", output_dir / "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = model_dir / f"comparison_node2vec_{graph_data.name}.pth"
    embeddings = load_or_create_node2vec_embeddings(graph_data.graph, embedding_path, dimensions=int(config["gnn"].get("embedding_dim", 64)), walk_length=int(config["gnn"].get("walk_length", 10)), num_walks=int(config["gnn"].get("num_walks", 10)), window=int(config["gnn"].get("window", 10)), workers=int(config["gnn"].get("workers", 1))).to(device)
    norm_degrees, _ = build_node_features(graph_data.graph, device)
    extractor = OverlapFeatureExtractor(graph_data.graph, distance_cap=int(cfg.get("distance_cap", 6)))
    hidden_dim = int(config["gnn"].get("hidden_dim", 64))
    seeds = [int(value) for value in cfg.get("seeds", [20260821, 20260822, 20260823, 20260824, 20260825])]
    seed_results = []
    sequential_rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        baseline_model = MarginalGainPredictor(embeddings.shape[1], hidden_dim).to(device)
        baseline_training = _train_model(baseline_model, BASELINE, list(train_samples), embeddings, norm_degrees, extractor, graph_data.num_nodes, cfg, device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        overlap_model = OverlapMarginalGainPredictor(embeddings.shape[1], hidden_dim, overlap_dim=len(OVERLAP_FEATURE_NAMES)).to(device)
        overlap_training = _train_model(overlap_model, OVERLAP, list(train_samples), embeddings, norm_degrees, extractor, graph_data.num_nodes, cfg, device)
        baseline_predictions = _model_score(baseline_model, BASELINE, embeddings, norm_degrees, extractor, test_samples, device)
        overlap_predictions = _model_score(overlap_model, OVERLAP, embeddings, norm_degrees, extractor, test_samples, device)
        models = [(BASELINE, baseline_model, baseline_predictions, baseline_training), (OVERLAP, overlap_model, overlap_predictions, overlap_training)]
        rows = []
        for model_name, model, predictions, training in models:
            groups = [("overall", list(range(len(test_samples))))]
            groups.extend(
                (str(size), [i for i, item in enumerate(test_samples) if item.seed_set_size == size])
                for size in sorted({item.seed_set_size for item in test_samples})
            )
            for size, indices in groups:
                metrics = _grouped_metrics([test_samples[i] for i in indices], [predictions[i] for i in indices])
                metrics.update(_ranking_metrics([test_samples[i] for i in indices], [predictions[i] for i in indices]))
                rows.append({"seed": seed, "model": model_name, "seed_set_size": size, **metrics})
            checkpoint = seed_dir / ("direct_mlp.pth" if model_name == BASELINE else "direct_mlp_overlap.pth")
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "model_version": model.model_version, "dataset": graph_data.name, "dataset_sha256": _dataset_sha256(dataset_path), "seed": seed}, checkpoint)
            sequential = _sequential_spread(graph_data, model, model_name, embeddings, norm_degrees, extractor, cfg, seed, device)
            for item in sequential:
                item.update({"seed": seed, "model": model_name})
                sequential_rows.append(item)
            (seed_dir / f"{model_name.replace('+', '_plus_')}_sequential.json").write_text(json.dumps(sequential, ensure_ascii=False, indent=2), encoding="utf-8")
        (seed_dir / "metrics_by_seed_size.csv").open("w", encoding="utf-8-sig").close()
        seed_results.append({
            "seed": seed,
            "metrics": rows,
            "training": {BASELINE: baseline_training, OVERLAP: overlap_training},
            "checkpoints": {BASELINE: str(seed_dir / "direct_mlp.pth"), OVERLAP: str(seed_dir / "direct_mlp_overlap.pth")},
        })
        with (seed_dir / "metrics_by_seed_size.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fields = ["seed", "model", "seed_set_size", "mae", "spearman", "top_1", "ranking_loss", "contexts", "samples"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    all_rows = [row for result in seed_results for row in result["metrics"]]
    aggregate = []
    for key in sorted({(row["model"], row["seed_set_size"]) for row in all_rows}, key=lambda item: (item[0], item[1] != "overall", str(item[1]))):
        values = [row for row in all_rows if (row["model"], row["seed_set_size"]) == key]
        item = {"model": key[0], "seed_set_size": key[1], "seeds": len(values)}
        for metric in ("mae", "spearman", "top_1", "ranking_loss"):
            numbers = np.asarray([row[metric] for row in values], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(numbers))
            item[f"{metric}_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0
        aggregate.append(item)
    aggregate_path = output_dir / "aggregate_metrics.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(aggregate[0]) if aggregate else ["model", "seed_set_size"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    sequential_aggregate = []
    for key in sorted({(row["model"], row["seed_set_size"]) for row in sequential_rows}):
        matching = [row for row in sequential_rows if (row["model"], row["seed_set_size"]) == key]
        values = [row["final_spread"] for row in matching]
        ranking_values = [row["mean_ranking_loss"] for row in matching]
        top_1_values = [row["top_1"] for row in matching]
        sequential_aggregate.append({
            "model": key[0],
            "seed_set_size": key[1],
            "final_spread_mean": float(np.mean(values)),
            "final_spread_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ranking_loss_mean": float(np.mean(ranking_values)),
            "ranking_loss_std": float(np.std(ranking_values, ddof=1)) if len(ranking_values) > 1 else 0.0,
            "top_1_mean": float(np.mean(top_1_values)),
            "top_1_std": float(np.std(top_1_values, ddof=1)) if len(top_1_values) > 1 else 0.0,
            "seeds": len(values),
        })
    sequential_aggregate_path = output_dir / "sequential_aggregate.csv"
    with sequential_aggregate_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "seed_set_size", "final_spread_mean", "final_spread_std", "ranking_loss_mean", "ranking_loss_std", "top_1_mean", "top_1_std", "seeds"])
        writer.writeheader()
        writer.writerows(sequential_aggregate)
    result = {
        "dataset": graph_data.name,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _dataset_sha256(dataset_path),
        "split_samples": {name: len(items) for name, items in splits.items()},
        "split_contexts": {name: len({item.context_id for item in items}) for name, items in splits.items()},
        "feature_names": list(OVERLAP_FEATURE_NAMES),
        "feature_definition": "Topology-only shortest-path and k-hop neighborhood overlap; local coverage is a community-coverage proxy, not Louvain.",
        "same_labels_and_split": True,
        "models": [BASELINE, OVERLAP],
        "seeds": seeds,
        "aggregate_metrics": aggregate,
        "aggregate_metrics_csv": str(aggregate_path),
        "sequential_aggregate": sequential_aggregate,
        "sequential_aggregate_csv": str(sequential_aggregate_path),
        "seed_results": seed_results,
        "cross_graph_protocol": {
            "status": "protocol_only",
            "training_graph": "NetHEPT",
            "test_graph": "not run; requires teacher-specified graph file and matched preprocessing",
            "data_preparation": "Regenerate embeddings and marginal labels on the test graph with the same feature definitions and IC parameters; use a graph-specific disjoint test split.",
            "forbidden_leakage": ["test-graph marginal labels in training", "test-graph test split for tuning", "cross-graph node-id reuse", "embedding/cache files carrying labels"],
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
