"""Gate-3b: can a learned state-conditioned scorer beat the analytic baseline?

The question
------------
Gate 2 established that ``exposure_scores_delta`` ("delta2") beats both out-degree and exact
Monte-Carlo greedy in the saturated regime.  That makes delta2 the real competitor: a paper
claiming a learned method has to beat *it*, not degree.

delta2 is a two-hop closed form that looks only at each out-neighbour's exposure ``delta``
and the change a candidate would cause.  It never looks at the **thresholds**.  Whether a
neighbour sits just inside ``[theta_kappa, theta_tau]`` or is about to be pushed past
``theta_tau`` is visible in the data but invisible to the analytic score, so there is a
concrete reason to expect a learned correction to help.

Protocol
--------
Leave-one-context-out.  The dataset is built from several independent (seed set, candidate
pool) draws; the model is trained on all draws but one and scored on the held-out draw.  This
tests generalization across *states*, which is the only interesting claim -- fitting the same
state that is being evaluated would prove nothing.

Models
------
``analytic``   delta2 itself, read straight from the feature vector (no fitting).
``linear``     ridge regression on the same features.  If this already matches the MLP, the
               relationship is close to linear and a GNN is not needed.
``mlp``        a small torch MLP.  CPU only.

Metrics
-------
``spearman``     rank correlation with the true marginal on held-out contexts.
``top5_recall``  overlap between the model's top-5 and the true top-5.
``regret``       (true gain of the true top-5 - true gain of the model's top-5) / |true top-5
                 gain|.  0 is perfect, 1 means the model captured nothing.

``regret`` is the metric that connects to spread: it says how much of the achievable gain
the ranking leaves on the table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_overexposure_pool_ranking import spearman  # noqa: E402


def top_k_indices(values: np.ndarray, k: int) -> list[int]:
    k = min(k, len(values))
    return list(np.argsort(-values, kind="stable")[:k])


def top_k_recall(truth: np.ndarray, scores: np.ndarray, k: int) -> float:
    a = set(top_k_indices(truth, k))
    b = set(top_k_indices(scores, k))
    return len(a & b) / max(1, k)


def regret(truth: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the achievable top-k gain that the scorer fails to capture."""
    best = sum(truth[i] for i in top_k_indices(truth, k))
    if abs(best) < 1e-9:
        return float("nan")
    got = sum(truth[i] for i in top_k_indices(scores, k))
    return 1.0 - got / best


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge with an intercept column."""
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ y)


def ridge_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return design @ weights


def standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (other - mean) / std


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    hidden: int,
    depth: int,
) -> MLP:
    torch.manual_seed(seed)
    model = MLP(x.shape[1], hidden=hidden, depth=depth)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    features = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)

    # Standardise the target so the learning rate is scale free across graphs and fractions
    # whose marginal gains differ by orders of magnitude.
    scale = float(target.std()) or 1.0
    normalised = (target - target.mean()) / scale
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(features), normalised)
        loss.backward()
        optimiser.step()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge regularisation")
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    names: list[str] = payload["feature_names"]
    records: list[dict] = payload["records"]
    analytic_index = names.index("analytic_score")

    # A "context" is one (seed fraction, draw) pair: the unit we hold out.
    contexts: dict[tuple[float, int], list[dict]] = {}
    for record in records:
        contexts.setdefault((record["seed_fraction"], record["context"]), []).append(record)

    keys = sorted(contexts)
    print(f"dataset={args.dataset.name}")
    print(f"graph={payload['graph']} n={payload['n']} records={len(records)} "
          f"contexts={len(keys)} frequencies={sorted({k[0] for k in keys})}")
    print(f"features={len(names)}  mc_trials_per_label={payload['trials']}")
    print()

    results: dict[str, dict[str, list[float]]] = {
        "analytic": {}, "linear": {}, "mlp": {},
    }
    rows: list[dict] = []

    for held_out in keys:
        train_records = [r for k, rows_ in contexts.items() if k != held_out for r in rows_]
        test_records = contexts[held_out]
        if not train_records or len(test_records) < 3:
            continue

        x_train = np.array([r["features"] for r in train_records], dtype=float)
        y_train = np.array([r["label"] for r in train_records], dtype=float)
        x_test = np.array([r["features"] for r in test_records], dtype=float)
        y_test = np.array([r["label"] for r in test_records], dtype=float)

        x_train_s, x_test_s = standardise(x_train, x_test)

        scores = {
            "analytic": x_test[:, analytic_index],
            "linear": ridge_predict(ridge_fit(x_train_s, y_train, args.alpha), x_test_s),
        }
        model = train_mlp(x_train_s, y_train, args.epochs, args.lr,
                          args.random_seed, args.hidden, args.depth)
        with torch.no_grad():
            scores["mlp"] = model(torch.tensor(x_test_s, dtype=torch.float32)).numpy()

        row = {
            "seed_fraction": held_out[0],
            "context": held_out[1],
            "n_test": len(test_records),
        }
        for name, values in scores.items():
            rho = spearman(list(values), list(y_test))
            recall = top_k_recall(y_test, values, args.top_k)
            reg = regret(y_test, values, args.top_k)
            results[name].setdefault("spearman", []).append(rho)
            results[name].setdefault("recall", []).append(recall)
            results[name].setdefault("regret", []).append(reg)
            row[f"{name}_spearman"] = rho
            row[f"{name}_recall"] = recall
            row[f"{name}_regret"] = reg
        rows.append(row)

        print(f"  |S|/n={held_out[0]*100:>5.1f}% ctx={held_out[1]} n={len(test_records):>3}  "
              f"rho: an={row['analytic_spearman']:>+6.3f} li={row['linear_spearman']:>+6.3f} "
              f"mlp={row['mlp_spearman']:>+6.3f}  |  "
              f"regret: an={row['analytic_regret']:>6.3f} li={row['linear_regret']:>6.3f} "
              f"mlp={row['mlp_regret']:>6.3f}", flush=True)

    print()
    print("=" * 92)
    print(f"LEAVE-ONE-CONTEXT-OUT SUMMARY  (top-{args.top_k})")
    print("=" * 92)
    print(f"  {'model':<12}{'spearman':>12}{'top-k recall':>15}{'regret':>10}"
          f"{'beats analytic':>17}")
    for name in ("analytic", "linear", "mlp"):
        block = results[name]
        if not block.get("spearman"):
            continue
        rho = statistics.fmean(block["spearman"])
        rec = statistics.fmean(block["recall"])
        reg = statistics.fmean([r for r in block["regret"] if r == r])
        if name == "analytic":
            wins = "-"
        else:
            wins = f"{sum(1 for a, b in zip(block['regret'], results['analytic']['regret']) if a < b)}/{len(block['regret'])}"
        print(f"  {name:<12}{rho:>+12.4f}{rec:>15.4f}{reg:>10.4f}{wins:>17}")

    print()
    print("VERDICT")
    print("-" * 92)
    an = results["analytic"]["regret"]
    for name in ("linear", "mlp"):
        block = results[name].get("regret", [])
        if not block:
            continue
        wins = sum(1 for a, b in zip(block, an) if a < b)
        delta = statistics.fmean(block) - statistics.fmean(an)
        print(f"  {name:<8} regret {statistics.fmean(block):.4f} vs analytic "
              f"{statistics.fmean(an):.4f}  ({delta:+.4f})  wins {wins}/{len(block)}")
    print()
    if results["mlp"].get("regret") and results["linear"].get("regret"):
        mlp_delta = statistics.fmean(results["mlp"]["regret"]) - statistics.fmean(an)
        lin_delta = statistics.fmean(results["linear"]["regret"]) - statistics.fmean(an)
        if mlp_delta < 0 or lin_delta < 0:
            print("  -> a learned scorer improves on the analytic baseline on held-out states.")
        else:
            print("  -> no learned model beats the analytic baseline on held-out states.")
            print("     delta2 is a strong baseline; the extra threshold features do not add")
            print("     enough signal at this sample size.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "dataset": str(args.dataset),
            "graph": payload["graph"],
            "top_k": args.top_k,
            "feature_names": names,
            "per_context": rows,
            "summary": {
                name: {
                    "spearman": statistics.fmean(block["spearman"]) if block.get("spearman") else None,
                    "recall": statistics.fmean(block["recall"]) if block.get("recall") else None,
                    "regret": (statistics.fmean([r for r in block["regret"] if r == r])
                               if block.get("regret") else None),
                }
                for name, block in results.items()
            },
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
