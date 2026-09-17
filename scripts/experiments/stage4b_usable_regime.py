"""Stage 4b: find the regime where BOTH conditions hold.

The tension found in stage 4
----------------------------
On four graphs the two properties the learned pipeline needs did not coincide:

    graph              <k>    within/between (state dependence)   headroom at k=5..10
    congress_twitter  55.95        5.64  (strong)                 0.6%  (saturated)
    email_eu_core     50.89        1.94  (state dependent)        0.3-0.6% (saturated)
    ca_grqc           11.06        0.64  (moderate)               4-5% degree, 41-78% random
    nethept            4.23        0.075 (NOT state dependent)   0.7-40%

A state-conditioned predictor is only useful when (i) a candidate's marginal gain genuinely
depends on the current seed set, and (ii) a cheap heuristic still leaves room to be improved on.
Stage 4 suggests these pull in opposite directions on the four graphs tested: dense graphs are
state dependent but saturate almost immediately, sparse graphs have headroom but behave almost
like a static ranking problem.

This script looks for overlap by pushing the budget down to k = 1..3 and sweeping a denser set
of graphs, because saturation is a function of seed count and stage 4 only probed k >= 2.

Reported per (graph, k): state-dependence ratio and the degree-to-oracle gap.  A regime is
"usable" when the ratio is above 0.5 and the gap is above 2%.
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
from evaluate_overexposure_pool_ranking import degree_scores  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402


@dataclass
class Cell:
    graph: str
    mean_degree: float
    budget: int
    state_ratio: float
    between_sd: float
    within_sd: float
    degree_gap: float
    random_gap: float
    oracle_spread: float


def oracle_greedy(graph, pool, budget, oracle, seed):
    chosen: list[int] = []
    remaining = list(pool)
    for step in range(budget):
        if not remaining:
            break
        scores = oracle.score(chosen, remaining, step=seed + step)
        best = max(remaining, key=lambda v: (scores[v], -v))
        if scores[best] <= 0.0:
            break
        chosen.append(best)
        remaining.remove(best)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "facebook",
                                 "bitcoin_alpha", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--oracle-mc", type=int, default=40)
    parser.add_argument("--eval-mc", type=int, default=150)
    parser.add_argument("--state-contexts", type=int, default=4)
    parser.add_argument("--state-mc", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    params = resolve_overexposure_params(None)
    cells: list[Cell] = []

    for graph_name in args.graphs:
        graph = load_graph(graph_name)
        nodes = list(graph.nodes())
        n = len(nodes)
        md = 2 * graph.number_of_edges() / n
        print(f"=== {graph_name} n={n} <k>={md:.2f}")

        for budget in args.budgets:
            rng = random.Random(args.random_seed + 31 * budget)
            pool = rng.sample(nodes, min(args.pool_size, n))

            oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.oracle_mc,
                                                 random_seed=args.random_seed, params=params)
            evaluator = OverexposureMonteCarloOracle(graph, mc_runs=args.eval_mc,
                                                     random_seed=args.random_seed + 99,
                                                     params=params)

            greedy = oracle_greedy(graph, pool, budget, oracle, args.random_seed)
            oracle_spread = evaluator.spread(greedy)["mean"]

            degree = dict(graph.out_degree())
            by_degree = sorted(pool, key=lambda v: (-degree[v], v))[:budget]
            rand_order = list(pool)
            random.Random(args.random_seed + 7).shuffle(rand_order)

            deg_spread = evaluator.spread(by_degree)["mean"]
            rnd_spread = evaluator.spread(rand_order[:budget])["mean"]
            deg_gap = (oracle_spread - deg_spread) / oracle_spread if oracle_spread > 1e-9 else float("nan")
            rnd_gap = (oracle_spread - rnd_spread) / oracle_spread if oracle_spread > 1e-9 else float("nan")

            # state dependence at this budget: same candidates, several seed states
            states = []
            srng = random.Random(args.random_seed + 500 + budget)
            for i in range(args.state_contexts):
                size = min(budget * i, max(0, n - args.pool_size))
                states.append(srng.sample(nodes, size))
            cands = [v for v in pool if all(v not in set(s) for s in states)][:15]
            if len(cands) < 3:
                cands = pool[:15]

            st_oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.state_mc,
                                                     random_seed=args.random_seed, params=params)
            per_cand: dict[int, list[float]] = {c: [] for c in cands}
            for index, state in enumerate(states):
                sc = st_oracle.score(state, cands, step=args.random_seed + 100 * index)
                for c in cands:
                    per_cand[c].append(sc[c])
            means = [statistics.fmean(v) for v in per_cand.values()]
            between = statistics.pstdev(means) if len(means) > 1 else 0.0
            withins = [statistics.pstdev(v) for v in per_cand.values() if len(v) > 1]
            within = statistics.fmean(withins) if withins else 0.0
            ratio = within / between if between > 1e-12 else float("inf")

            cell = Cell(graph_name, md, budget, ratio, between, within, deg_gap, rnd_gap,
                        oracle_spread)
            cells.append(cell)
            print(f"  k={budget}  state_ratio={ratio:7.3f}  degree_gap={deg_gap*100:7.2f}%  "
                  f"random_gap={rnd_gap*100:7.2f}%  oracle={oracle_spread:9.2f}")
        print()

    print("=" * 100)
    print("USABLE REGIME SEARCH  (state_ratio > 0.5  AND  degree_gap > 2%)")
    print("=" * 100)
    print(f"  {'graph':<18}{'<k>':>7}{'k':>4}{'state_ratio':>13}{'degree_gap%':>13}"
          f"{'random_gap%':>13}{'usable':>8}")
    for c in cells:
        usable = c.state_ratio > 0.5 and c.degree_gap > 0.02
        print(f"  {c.graph:<18}{c.mean_degree:>7.2f}{c.budget:>4}{c.state_ratio:>13.3f}"
              f"{c.degree_gap*100:>13.2f}{c.random_gap*100:>13.2f}{'YES' if usable else '':>8}")
    print()
    good = [c for c in cells if c.state_ratio > 0.5 and c.degree_gap > 0.02]
    if good:
        print(f"  {len(good)} usable cell(s):")
        for c in good:
            print(f"    {c.graph} k={c.budget}: ratio {c.state_ratio:.3f}, "
                  f"degree gap {c.degree_gap*100:.2f}%, random gap {c.random_gap*100:.2f}%")
    else:
        print("  NO usable cell.  On these graphs state dependence and optimisation headroom do")
        print("  not coincide, which would mean a state-conditioned predictor has little to")
        print("  exploit exactly where a predictor is needed.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs, "budgets": args.budgets, "pool_size": args.pool_size,
            "oracle_mc": args.oracle_mc, "eval_mc": args.eval_mc, "state_mc": args.state_mc,
            "cells": [asdict(c) for c in cells],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
