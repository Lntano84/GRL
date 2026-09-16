from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from grl.evaluation.ranking import regression_ranking_metrics
from grl.models import MarginalGainPredictor, build_node_features, load_or_create_node2vec_embeddings
from grl.models.gnn import SpreadPredictorGNN

from .marginal_dataset import (
    MarginalGainSample,
    build_marginal_dataset,
    load_marginal_dataset,
    save_marginal_dataset,
)


MODEL_TOTAL = "TotalSpread-Diff"
MODEL_DIRECT = "MarginalGain-MLP"


def _seed_mask(num_nodes: int, seed_set: list[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
    if seed_set:
        mask[seed_set, 0] = 1.0
    return mask


def _mean_dict(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows])) if rows else 0.0
        for key in keys
    }


def grouped_metrics(
    samples: list[MarginalGainSample],
    predictions: list[float],
) -> dict[str, float]:
    targets = [sample.marginal_gain for sample in samples]
    errors = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.context_id].append(index)

    ranking_rows = []
    for indices in groups.values():
        group_predictions = [predictions[index] for index in indices]
        group_targets = [targets[index] for index in indices]
        ranking_rows.append(
            regression_ranking_metrics(group_predictions, group_targets, top_ks=(1, 5, 10))
        )
    ranking = _mean_dict(
        ranking_rows,
        (
            "spearman",
            "kendall",
            "pairwise_accuracy",
            "top_1_recall",
            "top_5_recall",
            "top_10_recall",
        ),
    )
    return {
        "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "rmse": float(math.sqrt(np.mean(errors ** 2))) if errors.size else 0.0,
        "contexts": len(groups),
        "samples": len(samples),
        **ranking,
    }


class MarginalComparisonExperiment:
    def __init__(self, graph_data, config: dict, output_dir: str | Path):
        self.graph_data = graph_data
        self.config = config
        self.cfg = config.get("comparison", {})
        self.device = torch.device(config.get("gnn", {}).get("device", "cpu"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(config.get("gnn", {}).get("model_dir", self.output_dir / "models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        embedding_dim = int(config.get("gnn", {}).get("embedding_dim", 64))
        self.embedding_path = model_dir / f"comparison_node2vec_{graph_data.name}.pth"
        self.embeddings = load_or_create_node2vec_embeddings(
            graph_data.graph,
            self.embedding_path,
            dimensions=embedding_dim,
            walk_length=int(config.get("gnn", {}).get("walk_length", 10)),
            num_walks=int(config.get("gnn", {}).get("num_walks", 10)),
            window=int(config.get("gnn", {}).get("window", 10)),
            workers=int(config.get("gnn", {}).get("workers", 1)),
        ).to(self.device)
        self.norm_degrees, _ = build_node_features(graph_data.graph, self.device)
        hidden_dim = int(config.get("gnn", {}).get("hidden_dim", 64))
        self.total_model = SpreadPredictorGNN(self.embeddings.shape[1], hidden_dim).to(self.device)
        self.direct_model = MarginalGainPredictor(self.embeddings.shape[1], hidden_dim).to(self.device)
        self.total_model_path = model_dir / f"total_spread_{graph_data.name}.pth"
        self.direct_model_path = model_dir / f"direct_marginal_{graph_data.name}.pth"

    def _load_or_build_splits(self) -> tuple[dict[str, list[MarginalGainSample]], Path]:
        configured_path = self.cfg.get("dataset_path")
        dataset_path = Path(configured_path) if configured_path else self.output_dir / "marginal_dataset.json"
        if dataset_path.exists() and bool(self.cfg.get("reuse_dataset", True)):
            return load_marginal_dataset(dataset_path), dataset_path
        splits = build_marginal_dataset(self.graph_data, self.config)
        save_marginal_dataset(splits, dataset_path)
        return splits, dataset_path

    def _direct_batch(self, samples: list[MarginalGainSample]):
        masks = torch.stack([
            _seed_mask(self.graph_data.num_nodes, sample.seed_set, self.device)
            for sample in samples
        ])
        candidates = torch.tensor(
            [sample.candidate for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.tensor(
            [[sample.marginal_gain] for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return masks, candidates, labels

    def _total_examples(
        self,
        samples: list[MarginalGainSample],
    ) -> list[tuple[list[int], float]]:
        examples: dict[tuple[int, ...], float] = {}
        for sample in samples:
            examples[tuple(sorted(sample.seed_set))] = sample.base_spread
            extended = tuple(sorted([*sample.seed_set, sample.candidate]))
            examples[extended] = sample.extended_spread
        return [(list(seed_set), target) for seed_set, target in examples.items()]

    def _train_direct(self, samples: list[MarginalGainSample]) -> dict:
        optimizer = optim.Adam(
            self.direct_model.parameters(),
            lr=float(self.cfg.get("learning_rate", 1e-3)),
        )
        batch_size = int(self.cfg.get("batch_size", 8))
        epochs = int(self.cfg.get("epochs", 5))
        epoch_losses = []
        started = time.perf_counter()
        self.direct_model.train()
        for _ in range(epochs):
            random.shuffle(samples)
            losses = []
            for begin in range(0, len(samples), batch_size):
                masks, candidates, labels = self._direct_batch(samples[begin:begin + batch_size])
                embeddings = self.embeddings.unsqueeze(0).expand(masks.shape[0], -1, -1)
                degrees = self.norm_degrees.unsqueeze(0).expand(masks.shape[0], -1, -1)
                prediction = self.direct_model(embeddings, degrees, masks, candidates)
                loss = F.mse_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_losses.append(float(np.mean(losses)) if losses else 0.0)
        torch.save(self.direct_model.state_dict(), self.direct_model_path)
        return {
            "epochs": epochs,
            "training_examples": len(samples),
            "epoch_losses": epoch_losses,
            "seconds": time.perf_counter() - started,
        }

    def _train_total(self, samples: list[MarginalGainSample]) -> dict:
        examples = self._total_examples(samples)
        optimizer = optim.Adam(
            self.total_model.parameters(),
            lr=float(self.cfg.get("learning_rate", 1e-3)),
        )
        batch_size = int(self.cfg.get("batch_size", 8))
        epochs = int(self.cfg.get("epochs", 5))
        epoch_losses = []
        started = time.perf_counter()
        self.total_model.train()
        for _ in range(epochs):
            random.shuffle(examples)
            losses = []
            for begin in range(0, len(examples), batch_size):
                batch = examples[begin:begin + batch_size]
                masks = torch.stack([
                    _seed_mask(self.graph_data.num_nodes, seed_set, self.device)
                    for seed_set, _ in batch
                ])
                labels = torch.tensor(
                    [[target] for _, target in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                embeddings = self.embeddings.unsqueeze(0).expand(masks.shape[0], -1, -1)
                degrees = self.norm_degrees.unsqueeze(0).expand(masks.shape[0], -1, -1)
                prediction = self.total_model(embeddings, degrees, masks)
                loss = F.mse_loss(prediction, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_losses.append(float(np.mean(losses)) if losses else 0.0)
        torch.save(self.total_model.state_dict(), self.total_model_path)
        return {
            "epochs": epochs,
            "training_examples": len(examples),
            "epoch_losses": epoch_losses,
            "seconds": time.perf_counter() - started,
        }

    def _predict(self, samples: list[MarginalGainSample]) -> dict[str, list[float]]:
        self.total_model.eval()
        self.direct_model.eval()
        total_predictions = []
        direct_predictions = []
        with torch.no_grad():
            for sample in samples:
                base_mask = _seed_mask(self.graph_data.num_nodes, sample.seed_set, self.device)
                extended_mask = base_mask.clone()
                extended_mask[sample.candidate, 0] = 1.0
                base_prediction = self.total_model(
                    self.embeddings,
                    self.norm_degrees,
                    base_mask,
                )
                extended_prediction = self.total_model(
                    self.embeddings,
                    self.norm_degrees,
                    extended_mask,
                )
                total_predictions.append(float((extended_prediction - base_prediction).item()))
                direct_predictions.append(float(self.direct_model(
                    self.embeddings,
                    self.norm_degrees,
                    base_mask,
                    sample.candidate,
                ).item()))
        return {MODEL_TOTAL: total_predictions, MODEL_DIRECT: direct_predictions}

    def _evaluate(
        self,
        samples: list[MarginalGainSample],
        predictions: dict[str, list[float]],
    ) -> tuple[dict, dict]:
        overall = {
            model_name: grouped_metrics(samples, model_predictions)
            for model_name, model_predictions in predictions.items()
        }
        by_size = {}
        for size in sorted({sample.seed_set_size for sample in samples}):
            indices = [index for index, sample in enumerate(samples) if sample.seed_set_size == size]
            size_samples = [samples[index] for index in indices]
            by_size[str(size)] = {
                model_name: grouped_metrics(
                    size_samples,
                    [model_predictions[index] for index in indices],
                )
                for model_name, model_predictions in predictions.items()
            }
        return overall, by_size

    def _write_csv(self, overall: dict, by_size: dict) -> None:
        fields = [
            "model", "seed_set_size", "contexts", "samples", "mae", "rmse",
            "spearman", "kendall", "pairwise_accuracy", "top_1_recall",
            "top_5_recall", "top_10_recall",
        ]
        with (self.output_dir / "comparison_metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for model_name, metrics in overall.items():
                writer.writerow({"model": model_name, "seed_set_size": "overall", **metrics})
            for size, model_rows in by_size.items():
                for model_name, metrics in model_rows.items():
                    writer.writerow({"model": model_name, "seed_set_size": size, **metrics})

    def _plot(self, by_size: dict) -> Path:
        sizes = [int(size) for size in by_size]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        panels = [
            ("mae", "MAE", False),
            ("spearman", "Spearman", True),
            ("top_1_recall", "Top-1 Accuracy", True),
        ]
        colors = {MODEL_TOTAL: "#D55E00", MODEL_DIRECT: "#0072B2"}
        for axis, (metric, label, bounded) in zip(axes, panels):
            for model_name in (MODEL_TOTAL, MODEL_DIRECT):
                values = [by_size[str(size)][model_name][metric] for size in sizes]
                axis.plot(sizes, values, marker="o", linewidth=2, label=model_name, color=colors[model_name])
            axis.set_xlabel("|S|")
            axis.set_ylabel(label)
            axis.set_xticks(sizes)
            axis.grid(alpha=0.25)
            if bounded:
                axis.set_ylim(-0.05, 1.05)
        axes[0].legend(frameon=False)
        fig.suptitle("NetHEPT: performance versus seed-set size")
        fig.tight_layout()
        output = self.output_dir / "performance_vs_seed_size.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return output

    def run(self) -> dict:
        splits, dataset_path = self._load_or_build_splits()
        training = {
            MODEL_TOTAL: self._train_total(list(splits["train"])),
            MODEL_DIRECT: self._train_direct(list(splits["train"])),
        }
        predictions = self._predict(splits["test"])
        overall, by_size = self._evaluate(splits["test"], predictions)
        self._write_csv(overall, by_size)
        plot_path = self._plot(by_size)
        result = {
            "dataset": self.graph_data.name,
            "dataset_path": str(dataset_path),
            "split_samples": {name: len(samples) for name, samples in splits.items()},
            "split_contexts": {
                name: len({sample.context_id for sample in samples})
                for name, samples in splits.items()
            },
            "training": training,
            "overall": overall,
            "by_seed_set_size": by_size,
            "artifacts": {
                "metrics_csv": str(self.output_dir / "comparison_metrics.csv"),
                "curve": str(plot_path),
                "total_model": str(self.total_model_path),
                "direct_model": str(self.direct_model_path),
                "embeddings": str(self.embedding_path),
            },
        }
        (self.output_dir / "comparison_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
