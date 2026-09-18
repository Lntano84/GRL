"""Stage 4b: find the regime where BOTH conditions hold, without the two confounds of the first run.

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

Two confounds in the first run of this script (audit item P1-3.4 and P1-3.5)
--------------------------------------------------------------------------
**P1-3.4 --- mixed seed sizes in one statistic.**  The state-dependence contexts were built with
sizes ``0, k, 2k, 3k`` and the within/between ratio was computed over all of them at once.  The
``|S| = 0`` context has marginals orders of magnitude larger than the saturated ones, so it
dominated the *between*-candidate spread and deflated the ratio.  The verdict "not state
dependent" could therefore be produced by the pooling, not by the graph.  The ratio is now computed
**within one seed size at a time** and reported per size; nothing is pooled.

**P1-3.5 --- a single random candidate pool.**  One ``rng.sample`` of 30 nodes makes regime
selection an uncontrolled variable: the pool's degree profile decides how much headroom is
visible.  The pool is now drawn by **degree-stratified sampling** (the default) so it mirrors the
graph's degree distribution, and the whole measurement is repeated over several independent pools
so the reported gap comes with a spread across draws rather than a single number.

Both fixes change what a "usable" cell means, so the verdict is deliberately conservative: a cell
counts only when the state ratio clears the threshold at the seed size where the headroom was
measured, and the headroom clears it in the *median* pool draw.
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
from grl.data.weights import (  # noqa: E402
    AS_GIVEN,
    CLIP_TO_ONE,
    SUM_TO_ONE,
    describe_normalisation,
    normalise_in_weights,
)
from grl.diffusion.contract import (  # noqa: E402
    TARGET_MODE_ALL,
    TARGET_MODE_DEGREE_TAIL,
    ContractViolation,
    ModelContract,
    TargetedObjective,
    build_contract,
    resolve_target_contract,
)
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402

#: The state-dependence ratio must clear this to count as state dependent.
STATE_RATIO_THRESHOLD = 0.5
#: The degree-to-reference gap must clear this fraction to count as headroom.
GAP_THRESHOLD = 0.02


@dataclass
class GapCell:
    """Headroom measurement for one (graph, budget, pool draw)."""

    graph: str
    mean_degree: float
    budget: int
    pool_draw: int
    pool_strategy: str
    pool_mean_degree: float
    degree_gap: float
    random_gap: float
    reference_spread: float
    reference_policy: str


@dataclass
class StateCell:
    """State dependence at ONE seed size.  Never pooled across sizes (P1-3.4)."""

    graph: str
    mean_degree: float
    budget: int
    state_size: int
    state_fraction: float
    n_contexts: int
    n_candidates: int
    state_ratio: float
    between_sd: float
    within_sd: float


def degree_stratified_pool(
    graph: nx.DiGraph, nodes: list, size: int, rng: random.Random, bands: int = 5
) -> list:
    """Sample ``size`` nodes so the pool mirrors the graph's out-degree distribution.

    Confound P1-3.5: a single uniform draw can be unrepresentative in exactly the variable the
    study is about.  Banding by out-degree and sampling proportionally makes the pool's degree
    profile match the graph's by construction.
    """
    if size >= len(nodes):
        return list(nodes)
    degree = dict(graph.out_degree())
    ordered = sorted(nodes, key=lambda v: (degree[v], v))
    n = len(ordered)
    pool: list = []
    for band in range(bands):
        lo = band * n // bands
        hi = (band + 1) * n // bands
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        # proportional allocation, at least one node per non-empty band
        take = max(1, round(size * len(chunk) / n))
        take = min(take, len(chunk))
        pool.extend(rng.sample(chunk, take))
    # trim or top up to the exact size without disturbing the profile more than necessary
    if len(pool) > size:
        pool = rng.sample(pool, size)
    elif len(pool) < size:
        remaining = [v for v in nodes if v not in set(pool)]
        pool.extend(rng.sample(remaining, min(size - len(pool), len(remaining))))
    return pool


def greedy_reference(graph, pool, budget, oracle, seed, stopping_non_positive: bool) -> list:
    """Greedy on an ESTIMATOR.  Named for what it is: not exact, not globally optimal."""
    chosen: list[int] = []
    remaining = list(pool)
    for step in range(budget):
        if not remaining:
            break
        scores = oracle.score(chosen, remaining, step=seed + step)
        best = max(remaining, key=lambda v: (scores[v], -v))
        if stopping_non_positive and scores[best] <= 0.0:
            break
        chosen.append(best)
        remaining.remove(best)
    return chosen


def state_dependence(
    graph,
    nodes: list,
    pool: list,
    seed_size: int,
    contexts: int,
    candidates: int,
    mc_runs: int,
    seed: int,
    params,
) -> StateCell | None:
    """Within/between ratio of candidate marginals, computed at ONE seed size only."""
    if seed_size > len(pool):
        return None
    used: set = set()
    states: list[list[int]] = []
    srng = random.Random(seed)
    for _ in range(contexts):
        candidates_from = [v for v in pool if v not in used]
        if len(candidates_from) < seed_size + candidates:
            break
        state = srng.sample(candidates_from, seed_size)
        used.update(state)
        states.append(state)
    if len(states) < 2:
        return None

    available = [v for v in pool if v not in used]
    if len(available) < candidates:
        return None
    cands = srng.sample(available, candidates)

    oracle = OverexposureMonteCarloOracle(graph, mc_runs=mc_runs, random_seed=seed, params=params)
    per_cand: dict[int, list[float]] = {c: [] for c in cands}
    for index, state in enumerate(states):
        scores = oracle.score(state, cands, step=seed + 100 * index)
        for c in cands:
            per_cand[c].append(scores[c])

    means = [statistics.fmean(v) for v in per_cand.values()]
    between = statistics.pstdev(means) if len(means) > 1 else 0.0
    withins = [statistics.pstdev(v) for v in per_cand.values() if len(v) > 1]
    within = statistics.fmean(withins) if withins else 0.0
    ratio = within / between if between > 1e-12 else float("inf")
    return StateCell(
        graph="", mean_degree=0.0, budget=0, state_size=seed_size,
        state_fraction=seed_size / len(nodes) if nodes else 0.0,
        n_contexts=len(states), n_candidates=len(cands),
        state_ratio=ratio, between_sd=between, within_sd=within,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "facebook",
                                 "bitcoin_alpha", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--pool-draws", type=int, default=5,
                        help="independent pool draws per cell (confound P1-3.5)")
    parser.add_argument("--pool-strategy", default="degree_stratified",
                        choices=["degree_stratified", "random"])
    parser.add_argument("--oracle-mc", type=int, default=40)
    parser.add_argument("--eval-mc", type=int, default=150)
    parser.add_argument("--state-contexts", type=int, default=4,
                        help="contexts PER SEED SIZE; sizes are never pooled (confound P1-3.4)")
    parser.add_argument("--state-candidates", type=int, default=15)
    parser.add_argument("--state-mc", type=int, default=120)
    parser.add_argument("--normalisation", default=SUM_TO_ONE,
                        choices=[SUM_TO_ONE, CLIP_TO_ONE, AS_GIVEN],
                        help="confound P1-3.8: the model requires at most 1, not exactly 1")
    parser.add_argument("--target-mode", default=TARGET_MODE_ALL,
                        choices=[TARGET_MODE_ALL, TARGET_MODE_DEGREE_TAIL],
                        help=f"'{TARGET_MODE_ALL}' is D = V with seeds allowed inside it (what the "
                             f"earlier sweeps did, and a DIFFERENT problem from the source "
                             f"model's); '{TARGET_MODE_DEGREE_TAIL}' is the source model's "
                             f"formulation: D is the top --target-fraction by out-degree and seeds "
                             f"come from V \\ D")
    parser.add_argument("--target-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    params = resolve_overexposure_params(None)
    gap_cells: list[GapCell] = []
    state_cells: list[StateCell] = []

    for graph_name in args.graphs:
        graph = load_graph(graph_name)
        graph = normalise_in_weights(graph, args.normalisation)
        nodes = list(graph.nodes())
        n = len(nodes)
        md = 2 * graph.number_of_edges() / n
        scale = describe_normalisation(graph)
        contract = resolve_target_contract(graph, args.target_mode, args.target_fraction,
                                           max(args.budgets))
        describe = contract.describe()
        eligible = contract.objective.legal_candidates(graph)
        print(f"=== {graph_name} n={n} <k>={md:.2f} "
              f"normalisation={scale['strategy']} max_in_total={scale['max_in_weight_total']:.3f}")
        print(f"    contract: |D|={describe['objective']['target_set_size']} "
              f"seeds_in_D={describe['objective']['allow_seeds_in_target']} "
              f"budget<= {describe['objective']['budget']} "
              f"counting='{describe['counting']}'")
        print(f"    eligible seeds: {len(eligible)}/{n}")

        for budget in args.budgets:
            for draw in range(args.pool_draws):
                rng = random.Random(args.random_seed + 31 * budget + 101 * draw)
                pool_size = min(args.pool_size, len(eligible))
                if args.pool_strategy == "degree_stratified":
                    pool = degree_stratified_pool(graph, eligible, pool_size, rng)
                else:
                    pool = rng.sample(eligible, pool_size)
                # the contract forbids a seed inside D; a pool that slipped one through would be
                # caught here rather than silently measured
                contract.objective.check_seed_eligibility(pool)

                oracle = OverexposureMonteCarloOracle(
                    graph, mc_runs=args.oracle_mc, random_seed=args.random_seed, params=params)
                evaluator = OverexposureMonteCarloOracle(
                    graph, mc_runs=args.eval_mc, random_seed=args.random_seed + 99, params=params)

                greedy = greedy_reference(graph, pool, budget, oracle, args.random_seed, False)
                reference_spread = evaluator.spread(greedy)["mean"]

                degree = dict(graph.out_degree())
                by_degree = sorted(pool, key=lambda v: (-degree[v], v))[:budget]
                rand_order = list(pool)
                random.Random(args.random_seed + 7).shuffle(rand_order)
                deg_spread = evaluator.spread(by_degree)["mean"]
                rnd_spread = evaluator.spread(rand_order[:budget])["mean"]

                deg_gap = ((reference_spread - deg_spread) / reference_spread
                           if reference_spread > 1e-9 else float("nan"))
                rnd_gap = ((reference_spread - rnd_spread) / reference_spread
                           if reference_spread > 1e-9 else float("nan"))
                gap_cells.append(GapCell(
                    graph=graph_name, mean_degree=md, budget=budget, pool_draw=draw,
                    pool_strategy=args.pool_strategy,
                    pool_mean_degree=statistics.fmean(degree[v] for v in pool),
                    degree_gap=deg_gap, random_gap=rnd_gap,
                    reference_spread=reference_spread,
                    reference_policy=f"mc_greedy_mc{args.oracle_mc}",
                ))

            # state dependence, ONE seed size per cell
            for state_size in sorted({0, budget, 2 * budget}):
                cell = state_dependence(
                    graph, nodes, pool, state_size, args.state_contexts, args.state_candidates,
                    args.state_mc, args.random_seed + 500 + budget + 17 * state_size, params)
                if cell is None:
                    continue
                cell.graph = graph_name
                cell.mean_degree = md
                cell.budget = budget
                state_cells.append(cell)

            gaps = [c for c in gap_cells if c.graph == graph_name and c.budget == budget]
            deg_median = statistics.median(c.degree_gap for c in gaps)
            deg_lo = min(c.degree_gap for c in gaps)
            deg_hi = max(c.degree_gap for c in gaps)
            ratios = "  ".join(
                f"|S|={c.state_size}:{c.state_ratio:.3f}"
                for c in state_cells if c.graph == graph_name and c.budget == budget)
            print(f"  k={budget}  degree_gap median={deg_median*100:7.2f}% "
                  f"[{deg_lo*100:7.2f}, {deg_hi*100:7.2f}] over {len(gaps)} pools  "
                  f"state_ratio by size -> {ratios}")
        print()

    print("=" * 108)
    print("HEADROOM ACROSS POOL DRAWS  (confound P1-3.5: no single pool decides the verdict)")
    print("=" * 108)
    print(f"  {'graph':<18}{'<k>':>7}{'k':>4}{'pool_mean_k':>13}{'deg_gap_med%':>14}"
          f"{'deg_gap_min%':>14}{'rand_gap_med%':>15}")
    for graph_name in args.graphs:
        for budget in args.budgets:
            group = [c for c in gap_cells if c.graph == graph_name and c.budget == budget]
            if not group:
                continue
            print(f"  {graph_name:<18}{group[0].mean_degree:>7.2f}{budget:>4}"
                  f"{statistics.fmean(c.pool_mean_degree for c in group):>13.2f}"
                  f"{statistics.median(c.degree_gap for c in group)*100:>14.2f}"
                  f"{min(c.degree_gap for c in group)*100:>14.2f}"
                  f"{statistics.median(c.random_gap for c in group)*100:>15.2f}")

    print()
    print("=" * 108)
    print("STATE DEPENDENCE, PER SEED SIZE  (confound P1-3.4: never pooled)")
    print("=" * 108)
    print(f"  {'graph':<18}{'<k>':>7}{'k':>4}{'|S|':>5}{'|S|/n':>8}{'ctx':>5}"
          f"{'state_ratio':>13}{'between_sd':>12}{'within_sd':>11}{'dependent':>11}")
    for c in state_cells:
        print(f"  {c.graph:<18}{c.mean_degree:>7.2f}{c.budget:>4}{c.state_size:>5}"
              f"{c.state_fraction*100:>7.1f}%{c.n_contexts:>5}{c.state_ratio:>13.3f}"
              f"{c.between_sd:>12.4f}{c.within_sd:>11.4f}"
              f"{'YES' if c.state_ratio > STATE_RATIO_THRESHOLD else '':>11}")

    print()
    print("=" * 108)
    print(f"USABLE CELLS  (state_ratio > {STATE_RATIO_THRESHOLD} at the budget's OWN seed size "
          f"AND median degree gap > {GAP_THRESHOLD*100:.0f}%)")
    print("=" * 108)
    usable = []
    for graph_name in args.graphs:
        for budget in args.budgets:
            group = [c for c in gap_cells if c.graph == graph_name and c.budget == budget]
            if not group:
                continue
            median_gap = statistics.median(c.degree_gap for c in group)
            # the ratio must be at the seed size the headroom was measured at, i.e. |S| = k
            own = [c for c in state_cells
                   if c.graph == graph_name and c.budget == budget and c.state_size == budget]
            if not own:
                continue
            ratio = max(c.state_ratio for c in own)
            if ratio > STATE_RATIO_THRESHOLD and median_gap > GAP_THRESHOLD:
                usable.append((graph_name, budget, ratio, median_gap))
    if usable:
        for graph_name, budget, ratio, gap in usable:
            print(f"  {graph_name} k={budget}: ratio {ratio:.3f} at |S|={budget}, "
                  f"median degree gap {gap*100:.2f}%")
    else:
        print("  NO usable cell.  On these graphs state dependence at |S| = k and optimisation")
        print("  headroom do not coincide, which means a state-conditioned predictor has little")
        print("  to exploit exactly where a predictor is needed.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs, "budgets": args.budgets, "pool_size": args.pool_size,
            "pool_draws": args.pool_draws, "pool_strategy": args.pool_strategy,
            "oracle_mc": args.oracle_mc, "eval_mc": args.eval_mc, "state_mc": args.state_mc,
            "state_contexts_per_size": args.state_contexts,
            "normalisation": args.normalisation,
            "state_sizes_are_pooled": False,
            "reference_is_exact": False,
            "state_ratio_threshold": STATE_RATIO_THRESHOLD,
            "gap_threshold": GAP_THRESHOLD,
            "gap_cells": [asdict(c) for c in gap_cells],
            "state_cells": [asdict(c) for c in state_cells],
            "usable": [{"graph": g, "budget": b, "state_ratio": r, "median_degree_gap": d}
                       for g, b, r, d in usable],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
