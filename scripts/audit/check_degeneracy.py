"""Audit experiment: what does the overexposure model actually degenerate to?

The task brief requires a sanity/degeneracy case: when the threshold window collapses to the
standard setting (theta_tau = 1, i.e. ``overexposure_free=True``), the model should approach
standard IC/LT behaviour.

This script measures, on the same graph, seed sets and MC budget:

1. ``ic``            standard independent cascade at the graph's own edge weights
2. ``oe``            the overexposure threshold-window model (full windows)
3. ``oe_free``       windows with theta_tau clamped to 1 (the claimed LT-like limit)
4. ``oe_wide``       a milder window (theta_tau rescaled into [0.5, 1]) as an intermediate

Reported per configuration: mean spread, and the Spearman correlation between each model's
true marginal gain and (a) out-degree, (b) the other models' marginal gains.  The point is to
establish whether "no overexposure" really lands near IC, or somewhere else, before any
claim about degeneracy is written down.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from evaluate_overexposure_pool_ranking import degree_scores, spearman  # noqa: E402
from grl.diffusion import (  # noqa: E402
    run_independent_cascade,
    run_overexposure,
    sample_threshold_windows,
)

MODELS = ("ic", "oe", "oe_free", "oe_wide")


def paired_spread(
    model: str, graph: nx.DiGraph, candidates: list[int], trials: int, seed: int,
    wide_lo: float = 0.5,
) -> list[float]:
    """Mean spread of ``[candidate]`` alone under each model, paired over trials."""
    nodes = list(graph.nodes())
    totals = [0.0] * len(candidates)
    for t in range(trials):
        r = random.Random(seed + t)
        if model == "ic":
            for i, c in enumerate(candidates):
                totals[i] += run_independent_cascade(graph, [c], random.Random(seed + t))
            continue
        windows = sample_threshold_windows(nodes, r, overexposure_free=(model == "oe_free"))
        if model == "oe_wide":
            windows = {v: (k, wide_lo + (1.0 - wide_lo) * tau) for v, (k, tau) in windows.items()}
        for i, c in enumerate(candidates):
            totals[i] += run_overexposure(graph, [c], windows, r).spread
    return [v / trials for v in totals]


def paired_gains(
    model: str, graph: nx.DiGraph, seeds: list[int], candidates: list[int],
    trials: int, seed: int,
) -> list[float]:
    """Marginal gain of each candidate over ``seeds``, paired within a trial."""
    nodes = list(graph.nodes())
    gains = [0.0] * len(candidates)
    for t in range(trials):
        r = random.Random(seed + t)
        if model == "ic":
            base = run_independent_cascade(graph, list(seeds), random.Random(seed + t))
            for i, c in enumerate(candidates):
                v = run_independent_cascade(graph, list(seeds) + [c], random.Random(seed + t))
                gains[i] += v - base
            continue
        windows = sample_threshold_windows(nodes, r, overexposure_free=(model == "oe_free"))
        base = run_overexposure(graph, list(seeds), windows, r).spread
        for i, c in enumerate(candidates):
            v = run_overexposure(graph, list(seeds) + [c], windows, r).spread
            gains[i] += v - base
    return [v / trials for v in gains]


@dataclass
class Row:
    graph: str
    model: str
    seed_fraction: float
    mean_gain: float
    negative_share: float
    rho_degree: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--wide-lo", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} m={graph.number_of_edges()} "
          f"<k>={2*graph.number_of_edges()/n:.2f}")
    print()

    rows: list[Row] = []
    gain_matrix: dict[tuple[float, str], list[float]] = {}

    for fraction in args.fractions:
        size = int(round(fraction * n))
        base_seed = args.random_seed + 13 * size
        rng = random.Random(base_seed)
        seeds = rng.sample(nodes, size) if size else []
        pool = [v for v in nodes if v not in set(seeds)]
        candidates = rng.sample(pool, min(args.candidates, len(pool)))
        degree = degree_scores(graph, candidates)

        print(f"  |S|/n = {fraction*100:.1f}%  (|S| = {size}, candidates = {len(candidates)})")
        for model in MODELS:
            gains = paired_gains(model, graph, seeds, candidates, args.trials, base_seed + 7)
            gain_matrix[(fraction, model)] = gains
            row = Row(
                graph=args.graph, model=model, seed_fraction=size / n,
                mean_gain=statistics.fmean(gains),
                negative_share=sum(1 for g in gains if g < 0) / len(gains),
                rho_degree=spearman(degree, gains),
            )
            rows.append(row)
            print(f"    {model:<9} mean_gain={row.mean_gain:>+10.3f}  "
                  f"neg={row.negative_share*100:>5.1f}%  rho_degree={row.rho_degree:>+7.3f}")

        print(f"    --- pairwise Spearman of marginal gains ({fraction*100:.1f}%) ---")
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                ga = gain_matrix[(fraction, a)]
                gb = gain_matrix[(fraction, b)]
                print(f"      {a:<8} ~ {b:<8} {spearman(ga, gb):+.3f}")
        print()

    print("=" * 92)
    print("DEGENERACY CHECK")
    print("=" * 92)
    for fraction in args.fractions:
        oe_free = gain_matrix[(fraction, "oe_free")]
        ic = gain_matrix[(fraction, "ic")]
        oe = gain_matrix[(fraction, "oe")]
        print(f"  |S|/n = {fraction*100:.1f}%")
        print(f"    mean gain   ic={statistics.fmean(ic):+10.3f}  "
              f"oe_free={statistics.fmean(oe_free):+10.3f}  oe={statistics.fmean(oe):+10.3f}")
        print(f"    rho(ic, oe_free) = {spearman(ic, oe_free):+.3f}   "
              f"rho(ic, oe) = {spearman(ic, oe):+.3f}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "fractions": args.fractions,
            "candidates": args.candidates, "trials": args.trials,
            "wide_lo": args.wide_lo,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
