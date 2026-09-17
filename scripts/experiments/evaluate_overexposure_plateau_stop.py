"""Gate-2c: split the two jobs -- analytic ranking, observed-spread stopping.

What Gate 2b showed
-------------------
``exposure_scores_delta`` is a good *ranker* and a bad *calibrator* of the true marginal
gain.  On Congress-Twitter, using the realised exposure state (40 candidates, 60 paired
trials each):

    |S|/n   true mean marginal   analytic mean   ratio    Spearman
    0%          +193.586             +2.852       0.01      0.902
    5%            -0.661             -0.251       0.38      0.667
    10%           +0.167             -0.095      -0.57      0.263
    20%           +0.120             -0.192      -1.60      0.427

So the analytic score underestimates the marginal by roughly 100x in the unsaturated regime,
and its *sign* is wrong once saturated.  Rank correlation stays positive throughout.  A
policy that stops on the analytic score's zero crossing therefore stops almost immediately:
Gate 2b's rule took one seed and scored -0.28 where degree took ten and scored -3.60 -- safe,
but it leaves the remaining budget unused.

The fix: do not use the analytic score for the stopping decision at all.  Ranking and
stopping are different problems with different available information.

``analytic_ranked``  rank candidates by the analytic score, add seeds in that order.
``plateau``          same order, but stop when the *observed* spread has not improved for
                     ``patience`` consecutive seeds.  The observed spread is exact: with the
                     threshold windows fixed, a cascade is a deterministic function of the
                     seed set (Gate 1d), so there is no simulation noise in the signal --
                     unlike Monte-Carlo greedy, which estimates marginals with MC=25 draws
                     and is therefore noise-driven exactly where the marginal is ~0.

The observed spread is what an adaptive policy legitimately sees: the campaign is run, the
realised number of positively activated users is counted, and that count informs whether to
keep spending budget.  No estimator is involved.

Cost: ``patience`` cascades beyond the seeds actually taken, versus
``budget * pool * mc`` for Monte-Carlo greedy.
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
from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
)
from grl.diffusion.overexposure import (  # noqa: E402
    run_overexposure,
    sample_threshold_windows,
)


def ranked_seeds(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    windows: dict,
    rng: random.Random,
    patience: int,
    hops: int = 2,
) -> tuple[list[int], int]:
    """Add seeds in analytic-score order, optionally stopping on a spread plateau.

    ``patience = 0`` never stops early (pure analytic top-k).  Otherwise the policy stops
    once ``patience`` consecutive additions have failed to strictly increase the observed
    spread.  Every decision uses the *realised* state, so the ranking is state-conditioned.

    Returns ``(seeds, cascades_spent)``.
    """
    chosen: list[int] = []
    remaining = list(pool)
    cascades = 0
    best_spread = -1.0
    stale = 0

    for _ in range(budget):
        if not remaining:
            break
        state = dict(run_overexposure(graph, chosen, windows, rng).delta)
        cascades += 1
        scores = exposure_scores_delta(graph, remaining, state, set(chosen), hops=hops)
        pick = max(range(len(remaining)), key=lambda i: scores[i])
        candidate = remaining[pick]

        trial = chosen + [candidate]
        spread = float(run_overexposure(graph, trial, windows, rng).spread)
        cascades += 1

        if spread > best_spread:
            best_spread = spread
            stale = 0
        else:
            stale += 1
        chosen = trial
        remaining.pop(pick)

        if patience > 0 and stale >= patience:
            break

    return chosen, cascades


def paired_marginals(
    graph: nx.DiGraph,
    nodes: list[int],
    seeds: list[int],
    variants: dict[str, list[int]],
    trials: int,
    base_seed: int,
) -> dict[str, tuple[float, float]]:
    acc: dict[str, list[float]] = {name: [] for name in variants}
    for t in range(trials):
        r = random.Random(base_seed + 7 + t)
        windows = sample_threshold_windows(nodes, r)
        base = float(run_overexposure(graph, seeds, windows, r).spread)
        for name, extra in variants.items():
            value = float(run_overexposure(graph, seeds + extra, windows, r).spread)
            acc[name].append(value - base)
    return {
        name: (statistics.fmean(v), statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0)
        for name, v in acc.items()
    }


@dataclass
class Row:
    graph: str
    seed_size: int
    seed_fraction: float
    budget: int
    policy: str
    patience: int
    seeds_used: int
    cascades: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--patience", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="0 = never stop early (pure analytic top-k)")
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
          f"trials={args.trials} patience={args.patience}")
    print()

    all_rows: list[Row] = []
    summary: dict[str, list[float]] = {}

    for fraction in args.fractions:
        size = int(round(fraction * n))
        base_seed = args.random_seed + 13 * size
        rng = random.Random(base_seed)
        seeds = rng.sample(nodes, size) if size else []
        pool = [v for v in nodes if v not in set(seeds)]
        pool = rng.sample(pool, min(args.pool_size, len(pool)))

        # Selection uses one fixed threshold draw, as in the Gate 2 main table.
        select_windows = sample_threshold_windows(nodes, random.Random(base_seed + 202))

        variants: dict[str, list[int]] = {}
        meta: dict[str, tuple[int, int]] = {}

        degree = degree_scores(graph, pool)
        by_degree = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:args.budget]]
        variants["degree"] = by_degree
        meta["degree"] = (len(by_degree), 0)

        for patience in args.patience:
            pick, cascades = ranked_seeds(
                graph, pool, args.budget, select_windows,
                random.Random(base_seed + 202), patience,
            )
            name = f"plateau@{patience}"
            variants[name] = pick
            meta[name] = (len(pick), cascades)

        results = paired_marginals(graph, nodes, seeds, variants, args.trials, base_seed)

        print(f"  |S|/n = {fraction*100:.1f}%   (|S| = {size})")
        print(f"    {'policy':<14}{'marginal':>11}{'+-se':>8}{'seeds':>7}{'cascades':>10}")
        for name, (mean, se) in results.items():
            used, casc = meta[name]
            patience_value = 0 if name == "degree" else int(name.split("@")[1])
            print(f"    {name:<14}{mean:>+11.2f}{se:>8.2f}{used:>7}{casc:>10}")
            summary.setdefault(name, []).append(mean)
            all_rows.append(Row(args.graph, size, size / n, args.budget, name,
                                patience_value, used, casc))
        print()

    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  {'policy':<14}{'mean marginal':>15}{'worst':>10}{'mean seeds':>12}")
    for name, values in summary.items():
        used = statistics.fmean([r.seeds_used for r in all_rows if r.policy == name])
        print(f"  {name:<14}{statistics.fmean(values):>+15.2f}{min(values):>+10.2f}{used:>12.1f}")
    print()
    print("  Ranking and stopping are separate problems.  The analytic score ranks well")
    print("  (Spearman 0.26-0.90 across the sweep) but is not calibrated -- it underestimates")
    print("  the marginal ~100x when unsaturated and has the wrong sign when saturated -- so")
    print("  the stopping decision must come from the observed spread instead, which is exact")
    print("  given the fixed threshold windows.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph,
            "n": n,
            "budget": args.budget,
            "pool_size": args.pool_size,
            "trials": args.trials,
            "patience": args.patience,
            "in_weights_normalised": not args.no_normalise,
            "rows": [asdict(r) for r in all_rows],
            "mean_marginal_by_policy": {
                name: statistics.fmean(v) for name, v in summary.items()
            },
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
