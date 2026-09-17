"""Weight-scale sweep for Gate 1a / 1a' on NetHEPT.

Motivation
----------
NetHEPT ships in-degree-normalised edge weights: every node's incoming weights sum to 1.
That keeps ``delta`` small, so ``2*delta*(1-delta)`` stays on its increasing branch and
overexposure never makes an extra seed harmful.  The earlier Gate 1 results were measured
in exactly that benign regime, which is why degree survived and no negative marginal gain
ever appeared.

Rescaling the weights changes the regime.  A first probe found that

    w = 0.25  ->  17-78 negative nodes,   min gain  +0.70   (no negative gains)
    w = 0.50  ->  71-190 negative nodes,  min gain  +0.60   (no negative gains)
    w = 0.80  ->  76-208 negative nodes,  min gain  -1.60   (negative gains appear)

so ``delta`` must be able to exceed 0.5 for the non-monotone branch to matter.  This script
re-runs both gate diagnostics across a weight ladder so the conclusions can be stated for
the regime where overexposure is actually active, rather than for a regime where it is
inert.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion import estimate_spread_over_configs as estimate_ic_spreads  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    positive_activation_probability,
    run_overexposure,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_overexposure_conditional_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores,
    spearman,
    _top_k_recall,
)


def rescale(graph: nx.DiGraph, weight: float) -> nx.DiGraph:
    scaled = nx.DiGraph()
    scaled.add_nodes_from(graph.nodes())
    for u, v, _ in graph.edges(data=True):
        scaled.add_edge(u, v, weight=weight)
    return scaled


def _mean(values):
    clean = [v for v in values if v == v]
    return statistics.fmean(clean) if clean else float("nan")


def evaluate_weight(
    graph: nx.DiGraph,
    nodes: list[int],
    weight: float,
    budget: int,
    contexts: int,
    mc_runs: int,
    sample_size: int,
    random_seed: int,
) -> dict:
    indeg = dict(graph.in_degree())
    outdeg = dict(graph.out_degree())
    rng = random.Random(random_seed)

    rows = []
    for context in range(contexts):
        # Seed from high in-degree nodes: they are the ones that saturate neighbours.
        size = round((budget - 1) * context / max(1, contexts - 1))
        seeds = sorted(nodes, key=lambda x: -indeg[x])[:size]
        pool = [v for v in nodes if v not in set(seeds)]
        candidates = rng.sample(pool, min(sample_size, len(pool)))

        configurations = [seeds] + [seeds + [c] for c in candidates]
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc_runs, random_seed + 131 * context
        )
        base = estimates[0]["mean"]
        truth = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]

        # Average exposure state under S.
        delta_accum = {node: 0.0 for node in nodes}
        negative_total = 0
        probe = min(mc_runs, 12)
        for offset in range(probe):
            probe_rng = random.Random(random_seed + 7 * context + offset)
            windows = sample_threshold_windows(nodes, probe_rng)
            run = run_overexposure(graph, seeds, windows, probe_rng)
            for node, value in run.delta.items():
                delta_accum[node] += value
            negative_total += len(run.negative)
        delta = {node: value / probe for node, value in delta_accum.items()}

        rows.append({
            "context": context,
            "seed_size": size,
            "spearman_degree": spearman(degree_scores(graph, candidates), truth),
            "spearman_exposure": spearman(
                exposure_scores(graph, candidates, delta, set(seeds)), truth
            ),
            "top1_degree": _top_k_recall(truth, degree_scores(graph, candidates), 1),
            "top1_exposure": _top_k_recall(
                truth, exposure_scores(graph, candidates, delta, set(seeds)), 1
            ),
            "negative_gain_share": (
                sum(1 for g in truth if g < 0) / len(truth) if truth else float("nan")
            ),
            "negative_nodes": negative_total / probe,
        })

    return {
        "weight": weight,
        "budget": budget,
        "contexts": contexts,
        "spearman_degree": _mean([r["spearman_degree"] for r in rows]),
        "spearman_exposure": _mean([r["spearman_exposure"] for r in rows]),
        "top1_degree": _mean([r["top1_degree"] for r in rows]),
        "top1_exposure": _mean([r["top1_exposure"] for r in rows]),
        "negative_gain_share": _mean([r["negative_gain_share"] for r in rows]),
        "negative_nodes": _mean([r["negative_nodes"] for r in rows]),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20])
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--mc-runs", type=int, default=15)
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    graph_data = load_graph_from_config(config)
    nodes = list(graph_data.graph.nodes())

    print("Weight-scale sweep: Gate 1a and 1a' across overexposure regimes")
    print(f"  weights={args.weights} budgets={args.budgets} contexts={args.contexts} "
          f"mc={args.mc_runs} candidates={args.sample_size}")
    print()

    results = []
    for weight in args.weights:
        graph = rescale(graph_data.graph, weight)
        for budget in args.budgets:
            row = evaluate_weight(
                graph, nodes, weight, budget, args.contexts,
                args.mc_runs, args.sample_size, args.random_seed,
            )
            results.append(row)
            print(
                f"  w={weight:.2f} k={budget:<3} neg_nodes={row['negative_nodes']:7.0f} "
                f"neg_gain={row['negative_gain_share']*100:5.1f}% | "
                f"spear deg={row['spearman_degree']:.4f} expo={row['spearman_exposure']:.4f} "
                f"(d={row['spearman_exposure']-row['spearman_degree']:+.4f}) | "
                f"top1 deg={row['top1_degree']:.2f} expo={row['top1_exposure']:.2f}",
                flush=True,
            )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'w':>6}{'k':>4}{'neg_nodes':>11}{'neg_gain%':>11}"
          f"{'spear_deg':>11}{'spear_expo':>12}{'delta':>9}{'top1_deg':>10}{'top1_expo':>11}")
    for row in results:
        print(f"{row['weight']:>6.2f}{row['budget']:>4}{row['negative_nodes']:>11.0f}"
              f"{row['negative_gain_share']*100:>11.1f}{row['spearman_degree']:>11.4f}"
              f"{row['spearman_exposure']:>12.4f}"
              f"{row['spearman_exposure']-row['spearman_degree']:>+9.4f}"
              f"{row['top1_degree']:>10.2f}{row['top1_exposure']:>11.2f}")
    print()

    active = [r for r in results if r["negative_gain_share"] > 0.01]
    print("Where overexposure is ACTIVE (negative marginal gains observed):")
    if not active:
        print("  none -- non-monotonicity never triggered at these settings.")
    for row in active:
        delta = row["spearman_exposure"] - row["spearman_degree"]
        verdict = "state helps" if delta > 0.02 else ("degree better" if delta < -0.02 else "tie")
        print(f"  w={row['weight']:.2f} k={row['budget']}: neg_gain="
              f"{row['negative_gain_share']*100:.1f}%, Spearman delta={delta:+.4f} -> {verdict}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
