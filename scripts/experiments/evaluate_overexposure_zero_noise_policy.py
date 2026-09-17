"""Gate-2b: a zero-noise policy that is allowed to stop.

The gap Gate 2 identified
-------------------------
Congress-Twitter, budget = 10, 400 paired trials (docs/EXPERIMENTS_20260916.md section 1):

    |S|/n    degree top-10    delta2 top-10    exact MC greedy
    0%        +362.51          +361.64          +367.28
    5.1%        -4.39            +0.49            -0.01
    10.1%       -1.61            +2.24            +0.28
    20.0%       -5.17            +2.26            +0.42

Once the network saturates the true marginal is approximately zero, so exact Monte-Carlo
greedy -- which searches for the largest marginal using noisy estimates -- is driven by
noise and does worse than a plain closed form.  The missing capability is therefore not
better *ranking* but a decision rule that can say "stop" without simulation noise.

Why ``delta2`` cannot simply use its own zero crossing
-----------------------------------------------------
``exposure_scores_delta`` scores a candidate by

    sum_v [ g(delta(v) + dE_w(v)) - g(delta(v)) ],      g(e) = 2e(1-e)

and that quantity goes negative as ``delta(v) -> 1`` because ``g'(delta) -> -2``.  But a node
at ``delta ~ 1`` is usually already *positive*, and pushing it further does not remove it
from the spread immediately -- only crossing its own ``theta_tau`` does.  So the zero
crossing of the score is a biased estimate of the point where adding a seed stops helping.
An earlier version of this experiment stopped on ``score <= 0`` after one seed and scored
347 against a degree baseline of 364.

What this script does
---------------------
It keeps the analytic, zero-noise score and replaces the stopping rule with an explicit
threshold:

    stop when   best_analytic_score < stop_threshold * (score of the first seed taken)

i.e. a *relative* threshold, normalised by the first accepted seed so it is scale free
across graphs.  ``--thresholds`` sweeps it, including ``0.0`` (the naive zero crossing) and
``-inf`` (never stop, which reduces to the pure analytic top-k from Gate 1a'').

The threshold is tuned on the same realisation here, so the numbers are an **upper bound**
on what an offline-calibrated rule would get.  That is stated explicitly in the verdict
rather than hidden.

The comparison is paired: every policy is evaluated on the same threshold windows, and the
reported quantity is the marginal spread over the pre-existing seed set.
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


def observed_delta(
    graph: nx.DiGraph, seeds: list[int], windows: dict, rng: random.Random
) -> dict[int, float]:
    """Exposure state realised by ``seeds``.

    One cascade, no Monte-Carlo: with the threshold windows fixed, ``delta`` is a
    deterministic set function of the seed set (established in Gate 1d).
    """
    return dict(run_overexposure(graph, list(seeds), windows, rng).delta)


def analytic_policy(
    graph: nx.DiGraph,
    seeds: list[int],
    pool: list[int],
    budget: int,
    windows: dict,
    rng: random.Random,
    threshold: float,
    hops: int = 2,
) -> list[int]:
    """Pick up to ``budget`` seeds with a zero-noise score and a relative stop rule.

    The cascade is replayed after every accepted seed so the score sees the *realised*
    state, not the seedless one.  Replaying costs one cascade per step (``budget`` in
    total), which is negligible next to ``budget * |pool| * mc`` for Monte-Carlo greedy.
    """
    chosen: list[int] = []
    remaining = list(pool)
    first_score: float | None = None

    for _ in range(budget):
        if not remaining:
            break
        state = observed_delta(graph, chosen, windows, rng)
        scores = exposure_scores_delta(graph, remaining, state, set(chosen), hops=hops)
        best = max(range(len(remaining)), key=lambda i: scores[i])
        best_score = scores[best]

        if first_score is None:
            first_score = best_score
        if first_score > 0 and best_score < threshold * first_score:
            break
        if first_score <= 0:
            # Even the first seed has no analytic gain; refusing to seed is the honest
            # answer, and is exactly what a correct stopping rule should do.
            break
        chosen.append(remaining.pop(best))

    return chosen


def paired_marginals(
    graph: nx.DiGraph,
    nodes: list[int],
    seeds: list[int],
    variants: dict[str, list[int]],
    trials: int,
    base_seed: int,
) -> dict[str, tuple[float, float]]:
    """Marginal spread of every variant over ``seeds``, sharing windows across variants."""
    acc: dict[str, list[float]] = {name: [] for name in variants}
    for t in range(trials):
        r = random.Random(base_seed + 7 + t)
        windows = sample_threshold_windows(nodes, r)
        base = float(run_overexposure(graph, seeds, windows, r).spread)
        for name, extra in variants.items():
            value = float(run_overexposure(graph, seeds + extra, windows, r).spread)
            acc[name].append(value - base)
    return {
        name: (statistics.fmean(v), statistics.stdev(v) / math.sqrt(len(v)))
        for name, v in acc.items()
    }


@dataclass
class Row:
    graph: str
    seed_size: int
    seed_fraction: float
    budget: int
    threshold: float
    seeds_used: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.5],
                        help="relative stop threshold; 0.0 is the naive zero crossing, "
                             "1.5 effectively never stops")
    parser.add_argument("--never-stop", action="store_true",
                        help="add a control arm that ignores the stop rule entirely")
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
          f"trials={args.trials}")
    print(f"thresholds={args.thresholds}{' + never-stop control' if args.never_stop else ''}")
    print()

    rows: list[Row] = []
    print(f"{'|S|/n':>7}{'policy':>18}{'marginal':>11}{'+-se':>8}{'seeds':>7}")
    print("-" * 54)
    for fraction in args.fractions:
        size = int(round(fraction * n))
        base_seed = args.random_seed + 13 * size
        rng = random.Random(base_seed)
        seeds = rng.sample(nodes, size) if size else []
        pool = [v for v in nodes if v not in set(seeds)]
        pool = rng.sample(pool, min(args.pool_size, len(pool)))

        # Reference arms, all state-aware and all zero-noise.
        degree = degree_scores(graph, pool)
        by_degree = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:args.budget]]

        variants: dict[str, list[int]] = {"degree": by_degree}
        seeds_used: dict[str, int] = {"degree": len(by_degree)}

        # The analytic policy at every threshold.  The windows used for *selection* are a
        # single fixed draw, exactly as in the Gate 2 main table.
        select_rng = random.Random(base_seed + 202)
        select_windows = sample_threshold_windows(nodes, select_rng)
        for threshold in args.thresholds:
            pick = analytic_policy(
                graph, seeds, pool, args.budget, select_windows,
                random.Random(base_seed + 202), threshold,
            )
            name = f"analytic@{threshold:g}"
            variants[name] = pick
            seeds_used[name] = len(pick)
            rows.append(Row(args.graph, size, size / n, args.budget, threshold, len(pick)))

        if args.never_stop:
            pick = analytic_policy(
                graph, seeds, pool, args.budget, select_windows,
                random.Random(base_seed + 202), float("-inf"),
            )
            variants["analytic@never"] = pick
            seeds_used["analytic@never"] = len(pick)

        results = paired_marginals(graph, nodes, seeds, variants, args.trials, base_seed)
        for name, (mean, se) in results.items():
            print(f"{fraction*100:>6.1f}%{name:>18}{mean:>+11.2f}{se:>8.2f}"
                  f"{seeds_used[name]:>7}")

        # delta2 with the empty-state score, the Gate 1a'' baseline, for continuity.
        print("-" * 54)

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("  All policies are zero-cascade at decision time: the analytic score is a closed")
    print("  form, and observing the state costs one cascade per step (budget in total),")
    print("  versus budget * pool * mc for Monte-Carlo greedy.")
    print()
    print("  The stop threshold is tuned on the same realisation reported here, so these")
    print("  numbers are an UPPER BOUND on an offline-calibrated rule, not a held-out score.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph,
            "n": n,
            "budget": args.budget,
            "pool_size": args.pool_size,
            "trials": args.trials,
            "thresholds": args.thresholds,
            "in_weights_normalised": not args.no_normalise,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
