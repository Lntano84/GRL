"""Gate-2d: is the analytic ranking's advantage robust across selection draws?

Gate 2c picked seeds using a single fixed threshold-window draw and then evaluated with an
independent, paired draw.  That leaves an obvious failure mode: the result could be an
artefact of *that one* selection draw, which is only one sample from the threshold
distribution.

This script repeats the whole select-then-evaluate procedure over many independent selection
draws and reports the distribution of the paired marginal for each policy.  A policy is only
credible if it wins on most draws, not on average.

Policies
--------
``degree``      top-k by out-degree.
``analytic``    top-k by the analytic state-conditioned score, state observed from the
                selection draw itself.
``plateau@p``   analytic ordering with a patience-``p`` stop on the observed spread.

Everything is zero-noise at decision time except the state observation, which costs one
cascade per step.
"""

from __future__ import annotations

import argparse
import json
import math
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

from evaluate_overexposure_paper_graphs import LOADERS, normalise_in_weights  # noqa: E402
from evaluate_overexposure_plateau_stop import ranked_seeds  # noqa: E402
from evaluate_overexposure_pool_ranking import degree_scores  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    run_overexposure,
    sample_threshold_windows,
)


@dataclass
class Draw:
    graph: str
    seed_fraction: float
    selection_draw: int
    policy: str
    seeds_used: int
    marginal: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--selection-draws", type=int, default=8,
                        help="independent threshold draws used to PICK the seeds")
    parser.add_argument("--eval-trials", type=int, default=120,
                        help="paired trials used to SCORE the picked sets")
    parser.add_argument("--patience", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = LOADERS[args.graph]()
    if not args.no_normalise:
        graph = normalise_in_weights(graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} budget={args.budget} pool={args.pool_size} "
          f"selection_draws={args.selection_draws} eval_trials={args.eval_trials}")
    print()

    rows: list[Draw] = []
    for fraction in args.fractions:
        size = int(round(fraction * n))
        base_seed = args.random_seed + 13 * size
        rng = random.Random(base_seed)
        seeds = rng.sample(nodes, size) if size else []
        pool = [v for v in nodes if v not in set(seeds)]
        pool = rng.sample(pool, min(args.pool_size, len(pool)))
        degree = degree_scores(graph, pool)
        by_degree = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:args.budget]]

        # Shared evaluation draws: every selection is scored on the SAME windows, so the
        # comparison between policies is paired and only the selection varies.
        eval_windows = [
            sample_threshold_windows(nodes, random.Random(base_seed + 7001 + t))
            for t in range(args.eval_trials)
        ]

        def score(extra: list[int]) -> float:
            vals = []
            for t, windows in enumerate(eval_windows):
                r = random.Random(base_seed + 7001 + t)
                base = float(run_overexposure(graph, seeds, windows, r).spread)
                value = float(run_overexposure(graph, seeds + extra, windows, r).spread)
                vals.append(value - base)
            return statistics.fmean(vals)

        policies: dict[str, float] = {"degree": score(by_degree)}
        used: dict[str, int] = {"degree": len(by_degree)}

        for draw in range(args.selection_draws):
            selection_windows = sample_threshold_windows(
                nodes, random.Random(base_seed + 202 + 991 * draw)
            )
            pick, _ = ranked_seeds(
                graph, pool, args.budget, selection_windows,
                random.Random(base_seed + 202 + 991 * draw), 0,
            )
            policies[f"analytic#{draw}"] = score(pick)
            used[f"analytic#{draw}"] = len(pick)

            for patience in args.patience:
                pick, _ = ranked_seeds(
                    graph, pool, args.budget, selection_windows,
                    random.Random(base_seed + 202 + 991 * draw), patience,
                )
                policies[f"plateau{patience}#{draw}"] = score(pick)
                used[f"plateau{patience}#{draw}"] = len(pick)

        deg_value = policies["degree"]
        print(f"  |S|/n = {fraction*100:.1f}%   (|S| = {size}, pool = {len(pool)})")
        print(f"    degree           {deg_value:+9.2f}   (10 seeds)")
        for key in list(policies):
            if not key.startswith("analytic#") and not key.startswith("plateau"):
                continue
            print(f"    {key:<16}{policies[key]:+9.2f}   ({used[key]} seeds)"
                  f"   vs degree {policies[key]-deg_value:+8.2f}")
            rows.append(Draw(args.graph, fraction, int(key.split("#")[1]),
                             key.split("#")[0], used[key], policies[key]))
        rows.append(Draw(args.graph, fraction, -1, "degree", len(by_degree), deg_value))
        print()

    print("=" * 96)
    print("ROBUSTNESS ACROSS SELECTION DRAWS")
    print("=" * 96)
    for fraction in args.fractions:
        block = [r for r in rows if r.seed_fraction == fraction]
        deg = [r for r in block if r.policy == "degree"]
        deg_value = deg[0].marginal if deg else float("nan")
        print(f"  |S|/n = {fraction*100:.1f}%   degree = {deg_value:+.2f}")
        for name in sorted({r.policy for r in block if r.policy != "degree"}):
            values = [r.marginal for r in block if r.policy == name]
            wins = sum(1 for v in values if v > deg_value)
            print(f"    {name:<12} mean {statistics.fmean(values):+9.2f}"
                  f"   min {min(values):+9.2f}   max {max(values):+9.2f}"
                  f"   beats degree {wins}/{len(values)}")
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph,
            "n": n,
            "budget": args.budget,
            "pool_size": args.pool_size,
            "selection_draws": args.selection_draws,
            "eval_trials": args.eval_trials,
            "patience": args.patience,
            "in_weights_normalised": not args.no_normalise,
            "draws": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
