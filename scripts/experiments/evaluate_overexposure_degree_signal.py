"""Gate-1 diagnostic: does degree remain a good predictor of marginal gain?

This script answers the decisive question behind the DASFAA setting change:

    Under overexposure, is degree still correlated with the true conditional marginal
    gain Delta(v | S)?

Why it matters
--------------
The current GRL pipeline relies on a cheap degree-based retrieval stage achieving
Recall@M = 1.000 on NetHEPT under the standard IC model.  That is the reason a learned
proposal adds nothing there: the cheap scorer is already perfect.

The whole argument for moving to overexposure is that the objective becomes a peaked,
non-monotone function of cumulative exposure (``2 * delta * (1 - delta)``), so a
high-degree hub can saturate its own neighbourhood and stop being valuable.  If degree
survives overexposure unchanged, that argument collapses and Gate 1 fails.

What is measured
----------------
For each configuration (model x budget x context) we compute the exact conditional
marginal gain of every candidate under paired common random numbers, then report the
Spearman correlation between node degree and that gain.

  * ``ic``            -- standard independent cascade with the same edge weights
  * ``overexposure``  -- the threshold-window model (deterministic activation)

The gap between the two is the quantity of interest.  A secondary breakdown reports how
often the true top-k candidates are also the top-k by degree.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion import estimate_spread_over_configs  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
)
from grl.utils.config import load_yaml_config  # noqa: E402

MODEL_IC = "ic"
MODEL_OVEREXPOSURE = "overexposure"


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


@dataclass
class ContextResult:
    budget: int
    context: int
    model: str
    n_candidates: int
    n_with_positive_gain: int
    spearman_degree_gain: float
    top_k_overlap: float
    positive_gain_share_by_degree_rank: float


def _degrees(graph: nx.Graph | nx.DiGraph) -> dict[int, int]:
    source = graph.out_degree if graph.is_directed() else graph.degree
    return {node: int(degree) for node, degree in source}


def evaluate_context(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    candidates: list[int],
    degrees: dict[int, int],
    model: str,
    budget: int,
    context: int,
    mc_runs: int,
    random_seed: int,
) -> ContextResult:
    configurations = [list(seeds)] + [[*seeds, candidate] for candidate in candidates]

    if model == MODEL_OVEREXPOSURE:
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc_runs, random_seed
        )
    elif model == MODEL_IC:
        estimates = estimate_spread_over_configs(graph, configurations, mc_runs, random_seed)
    else:
        raise ValueError(f"unknown model: {model}")

    base = estimates[0]["mean"]
    gains = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]
    degree_values = [float(degrees[candidate]) for candidate in candidates]

    # How often is the best candidate by gain also the highest-degree one?
    best_gain_index = max(range(len(gains)), key=lambda i: gains[i])
    best_degree_index = max(range(len(degree_values)), key=lambda i: degree_values[i])
    top_k_overlap = 1.0 if best_gain_index == best_degree_index else 0.0

    positive = [g for g in gains if g > 0]
    # Share of total positive gain captured by the top half of candidates by degree.
    if positive:
        pairs = sorted(zip(degree_values, gains), key=lambda p: -p[0])
        half = max(1, len(pairs) // 2)
        top_half_gain = sum(g for _, g in pairs[:half] if g > 0)
        positive_gain_share = top_half_gain / sum(positive)
    else:
        positive_gain_share = float("nan")

    return ContextResult(
        budget=budget,
        context=context,
        model=model,
        n_candidates=len(candidates),
        n_with_positive_gain=len(positive),
        spearman_degree_gain=spearman(degree_values, gains),
        top_k_overlap=top_k_overlap,
        positive_gain_share_by_degree_rank=positive_gain_share,
    )


def run(
    config_path: Path,
    budgets: list[int],
    contexts_per_budget: int,
    mc_runs: int,
    random_seed: int,
    sample_size: int | None,
) -> dict:
    config = load_yaml_config(config_path)
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    degrees = _degrees(graph)
    n = graph_data.num_nodes
    rng = random.Random(random_seed)

    results: list[ContextResult] = []
    for budget in budgets:
        for context in range(contexts_per_budget):
            size = rng.randint(0, max(0, budget - 1))
            seeds = rng.sample(range(n), size) if size else []
            pool = [node for node in range(n) if node not in set(seeds)]
            candidates = rng.sample(pool, min(sample_size or n, len(pool)))
            for model in (MODEL_IC, MODEL_OVEREXPOSURE):
                results.append(
                    evaluate_context(
                        graph, seeds, candidates, degrees, model, budget, context,
                        mc_runs, random_seed + 1000 * context + budget,
                    )
                )
            print(
                f"  budget={budget} context={context} "
                f"|S|={len(seeds)} candidates={len(candidates)}",
                flush=True,
            )

    payload = {
        "config": str(config_path),
        "mc_runs": mc_runs,
        "random_seed": random_seed,
        "sample_size": sample_size,
        "budgets": budgets,
        "contexts_per_budget": contexts_per_budget,
        "results": [r.__dict__ for r in results],
        "summary": _summarise(results, budgets),
    }
    return payload


def _summarise(results: list[ContextResult], budgets: list[int]) -> dict:
    summary: dict = {"by_model": {}, "by_model_and_budget": {}}
    for model in (MODEL_IC, MODEL_OVEREXPOSURE):
        rows = [r for r in results if r.model == model]
        summary["by_model"][model] = {
            "spearman_mean": _mean([r.spearman_degree_gain for r in rows]),
            "spearman_std": _std([r.spearman_degree_gain for r in rows]),
            "top1_overlap": _mean([r.top_k_overlap for r in rows]),
            "gain_share_top_half_by_degree": _mean(
                [r.positive_gain_share_by_degree_rank for r in rows]
            ),
            "contexts": len(rows),
        }
        for budget in budgets:
            sub = [r for r in rows if r.budget == budget]
            summary["by_model_and_budget"][f"{model}_k{budget}"] = {
                "spearman_mean": _mean([r.spearman_degree_gain for r in sub]),
                "spearman_std": _std([r.spearman_degree_gain for r in sub]),
                "top1_overlap": _mean([r.top_k_overlap for r in sub]),
                "contexts": len(sub),
            }
    return summary


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return statistics.fmean(clean) if clean else float("nan")


def _std(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument("--mc-runs", type=int, default=60)
    parser.add_argument("--sample-size", type=int, default=800,
                        help="candidates sampled per context; 0 or negative means all nodes")
    parser.add_argument("--random-seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sample_size = None if args.sample_size <= 0 else args.sample_size
    print("Gate-1 diagnostic: degree vs true marginal gain")
    print(f"  config      : {args.config}")
    print(f"  budgets     : {args.budgets}")
    print(f"  contexts    : {args.contexts} per budget")
    print(f"  mc runs     : {args.mc_runs}")
    print(f"  sample size : {sample_size or 'all nodes'}")
    print()

    payload = run(
        args.config, args.budgets, args.contexts, args.mc_runs,
        args.random_seed, sample_size,
    )

    print()
    print("=" * 78)
    print("SUMMARY: Spearman(degree, true marginal gain)")
    print("=" * 78)
    print(f"{'model':<16}{'budget':>8}{'spearman':>12}{'std':>10}"
          f"{'top1 agree':>12}{'contexts':>10}")
    for model in (MODEL_IC, MODEL_OVEREXPOSURE):
        overall = payload["summary"]["by_model"][model]
        print(f"{model:<16}{'all':>8}{overall['spearman_mean']:>12.4f}"
              f"{overall['spearman_std']:>10.4f}{overall['top1_overlap']:>12.3f}"
              f"{overall['contexts']:>10}")
        for budget in args.budgets:
            row = payload["summary"]["by_model_and_budget"][f"{model}_k{budget}"]
            print(f"{'':<16}{budget:>8}{row['spearman_mean']:>12.4f}"
                  f"{row['spearman_std']:>10.4f}{row['top1_overlap']:>12.3f}"
                  f"{row['contexts']:>10}")
    print()

    ic = payload["summary"]["by_model"][MODEL_IC]["spearman_mean"]
    oe = payload["summary"]["by_model"][MODEL_OVEREXPOSURE]["spearman_mean"]
    print(f"GATE 1a: IC Spearman = {ic:.4f} | overexposure Spearman = {oe:.4f}")
    if oe == oe and ic == ic:
        if oe < 0.3 and oe < ic - 0.2:
            print("  -> PASS: degree signal collapses under overexposure.")
        elif oe < ic - 0.1:
            print("  -> WEAK PASS: signal degrades but stays substantial; review Gate 1.")
        else:
            print("  -> FAIL: degree remains a strong predictor; the setting-change "
                  "argument needs rethinking.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
