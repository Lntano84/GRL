"""Baselines from the source model, implemented so the paper can compare against them.

Why this exists
---------------
The source model (Inf. Sci. 744 (2026) 123375) evaluates its own algorithms against
``Random``, ``Max_Degree``, ``IMRank``, ``PageRank`` and ``CELF``.  Our draft so far compares
only against out-degree, exact Monte-Carlo greedy and our own closed form, which invites the
obvious reviewer question: *why not compare against the method you are extending?*  This script
implements the two algorithms the source model actually proposes.

IGA -- incremental greedy (its Algorithm 1)
-------------------------------------------
Repeatedly add the candidate with the largest marginal gain in the overexposure objective,
stopping at the budget.  Because the objective is non-monotone the same rule can in principle
stop early when no candidate has a positive gain.  Its stated guarantee is only ``gamma / k``.
Cost: ``k * |pool| * MC`` cascades.

UB -- upper-bound greedy (its Algorithm 2, Sec. 5.2)
----------------------------------------------------
Optimise the surrogate ``lambda(S) = sigma^kappa(S) - sigma^tau(S)``, where ``sigma^kappa`` and
``sigma^tau`` are linear-threshold spreads at the lower and upper thresholds.  Two variants are
worth measuring, because the surrogate is a *difference* of two monotone submodular functions and
differences do not preserve submodularity:

``ub_greedy``   greedy on ``lambda`` directly.
``ub_kappa``    greedy on ``sigma^kappa`` alone.  This is the pragmatic reading of "optimise the
                upper bound": ``sigma^kappa`` is a genuine monotone submodular LT influence
                function, so greedy on it carries a ``(1 - 1/e)`` flavour and, unlike ``lambda``,
                it is not a difference.

Both are evaluated on the true objective ``sigma``, as any baseline must be.

IMPORTANT CAVEAT carried from docs/SURROGATE_GAP_FINDING.md: our reading of
``sigma^kappa / sigma^tau`` may not match the source model's.  Results from the ``ub_*`` arms
should be labelled as "our implementation of the stated definition" until that reading is
confirmed with the authors.
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

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
)
from grl.baselines.classic_im import (  # noqa: E402
    celf_seeds,
    imrank_seeds,
    max_degree_seeds,
    pagerank_seeds,
    random_seeds,
)
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    run_overexposure,
    sample_threshold_windows,
)


def lt_spread(
    graph: nx.DiGraph, seeds: list[int], thresholds: dict[int, float]
) -> int:
    """Fixed-threshold linear-threshold spread (strict inequality, as the source states)."""
    active: set[int] = set(seeds)
    weighted_in: dict[int, float] = {v: 0.0 for v in graph.nodes()}
    frontier = list(seeds)
    while frontier:
        newly: list[int] = []
        for node in frontier:
            for target in graph.successors(node):
                if target in active:
                    continue
                weighted_in[target] += float(graph[node][target].get("weight", 0.0))
                if weighted_in[target] > thresholds[target]:
                    active.add(target)
                    newly.append(target)
        frontier = newly
    return len(active)


def greedy_overexposure(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    windows: dict,
    rng: random.Random,
    mc: int,
) -> tuple[list[int], int]:
    """IGA: greedy on the true objective, with an early stop when no gain is positive."""
    seeds: list[int] = []
    remaining = list(pool)
    cascades = 0
    for _ in range(budget):
        if not remaining:
            break
        configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc, rng.randrange(1 << 30)
        )
        cascades += len(remaining) * mc
        base = estimates[0]["mean"]
        gains = [estimates[i + 1]["mean"] - base for i in range(len(remaining))]
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0.0:
            break
        seeds.append(remaining.pop(best))
    return seeds, cascades


def greedy_lt(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    windows: dict,
    which: str,
) -> list[int]:
    """Greedy on an LT surrogate: ``which`` in {"kappa", "tau", "lambda"}.

    The LT processes use fixed thresholds, so the spread is exact and no Monte Carlo is needed.
    """
    index = 0 if which == "kappa" else 1
    nodes = list(graph.nodes())
    seeds: list[int] = []
    remaining = list(pool)

    def score(candidate_set: list[int]) -> float:
        # ``lambda`` needs a fresh window draw to be an expectation over thresholds; we use the
        # supplied draw for kappa/tau and the same draw for lambda so the comparison is paired.
        kap = {v: windows[v][0] for v in nodes}
        tau = {v: windows[v][1] for v in nodes}
        if which == "kappa":
            return float(lt_spread(graph, candidate_set, kap))
        if which == "tau":
            return float(lt_spread(graph, candidate_set, tau))
        return float(lt_spread(graph, candidate_set, kap) - lt_spread(graph, candidate_set, tau))

    current = score(seeds)
    for _ in range(budget):
        if not remaining:
            break
        best, best_gain = -1, float("-inf")
        for candidate in remaining:
            value = score(seeds + [candidate])
            gain = value - current
            if gain > best_gain:
                best_gain, best = gain, candidate
        if best < 0 or best_gain <= 0:
            break
        seeds.append(best)
        remaining.remove(best)
        current = score(seeds)
    return seeds


@dataclass
class Row:
    graph: str
    seed_size: int
    seed_fraction: float
    policy: str
    marginal: float
    stderr: float
    seeds_used: int
    cascades: int


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
        name: (
            statistics.fmean(v),
            statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0,
        )
        for name, v in acc.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.10, 0.20])
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--mc-greedy", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[Row] = []
    for label in args.graphs:
        graph = load_graph(label)
        nodes = list(graph.nodes())
        n = len(nodes)
        print(f"=== {label}: n={n} <k>={2*graph.number_of_edges()/n:.2f}")
        for fraction in args.fractions:
            size = int(round(fraction * n))
            base_seed = args.random_seed + 13 * size
            rng = random.Random(base_seed)
            seeds = rng.sample(nodes, size) if size else []
            pool = [v for v in nodes if v not in set(seeds)]
            pool = rng.sample(pool, min(args.pool_size, len(pool)))
            select_windows = sample_threshold_windows(nodes, random.Random(base_seed + 202))

            degree = degree_scores(graph, pool)
            by_degree = [pool[i] for i in sorted(range(len(pool)),
                                                key=lambda i: -degree[i])[:args.budget]]

            # The Gate 2 closed form, so the source model's algorithms are compared against the
            # strongest training-free scorer we have and not only against degree.
            observed = dict(run_overexposure(
                graph, list(seeds), select_windows,
                random.Random(base_seed + 202),
            ).delta)
            delta2_scores = exposure_scores_delta(graph, pool, observed, set(seeds), hops=2)
            by_delta2 = [pool[i] for i in sorted(range(len(pool)),
                                                 key=lambda i: -delta2_scores[i])[:args.budget]]

            iga_seeds, iga_cascades = greedy_overexposure(
                graph, pool, args.budget, select_windows,
                random.Random(base_seed + 202), args.mc_greedy,
            )
            ub_lambda = greedy_lt(graph, pool, args.budget, select_windows, "lambda")
            ub_kappa = greedy_lt(graph, pool, args.budget, select_windows, "kappa")

            # The remaining baselines the source model reports against.
            celf_pick, celf_cascades = celf_seeds(
                graph, pool, args.budget, args.mc_greedy, base_seed + 909,
            )
            pr_pick = pagerank_seeds(graph, pool, args.budget)
            imr_pick = imrank_seeds(graph, pool, args.budget)
            rnd_pick = random_seeds(graph, pool, args.budget,
                                    random.Random(base_seed + 555))

            variants = {
                "degree": by_degree,
                "delta2 (ours)": by_delta2,
                "IGA": iga_seeds,
                "UB-lambda": ub_lambda,
                "UB-kappa": ub_kappa,
                "CELF": celf_pick,
                "IMRank": imr_pick,
                "PageRank": pr_pick,
                "Random": rnd_pick,
            }
            results = paired_marginals(graph, nodes, seeds, variants, args.trials, base_seed)

            used = {name: len(v) for name, v in variants.items()}
            cascades = {"IGA": iga_cascades, "CELF": celf_cascades}
            print(f"  |S|/n={fraction*100:>5.1f}%")
            for name, (mean, se) in results.items():
                rows.append(Row(label, size, size / n, name, mean, se, used[name],
                                cascades.get(name, 0)))
                print(f"     {name:<14} {mean:>+9.2f} +-{se:>6.2f}  seeds={used[name]:>3}"
                      f"  cascades={cascades.get(name, 0):>8}", flush=True)
        print()

    print("=" * 100)
    print("SOURCE-MODEL BASELINES vs the closed form from Gate 2")
    print("=" * 100)
    print("  NOTE: UB-* arms implement OUR READING of the source model's definition of")
    print("  sigma^kappa / sigma^tau.  See docs/SURROGATE_GAP_FINDING.md before quoting them.")
    print()
    by_policy: dict[str, list[float]] = {}
    for row in rows:
        by_policy.setdefault(row.policy, []).append(row.marginal)
    for policy, values in by_policy.items():
        print(f"  {policy:<10} mean marginal {statistics.fmean(values):>+10.2f}   "
              f"worst {min(values):>+10.2f}   n={len(values)}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs,
            "fractions": args.fractions,
            "budget": args.budget,
            "pool_size": args.pool_size,
            "trials": args.trials,
            "mc_greedy": args.mc_greedy,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
