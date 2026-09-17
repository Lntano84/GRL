"""Gate-2b: sequential vs adaptive seed selection in the *saturated* regime.

Why this exists alongside the original Gate 1b
----------------------------------------------
``evaluate_overexposure_sequential_value.py`` scores each candidate with its own
independent simulation (``spread_of`` per candidate), which is ``O(k * M * MC)`` fresh
cascades per query and does not share threshold windows between candidates.  On the paper's
small dense graphs at high seed fractions that is both slow and noisy.

This script answers the same question with the *paired* estimator
(:func:`estimate_overexposure_spread_over_configs`), so every candidate in a step is scored
against the same threshold draws.  That cuts the variance of the difference between
candidates -- which is the only thing the argmax depends on -- by a large factor at the
same cost.

The question matters because Gate 1c v3 showed the objective only becomes non-monotone once
the seed *fraction* ``|S| / n`` is large.  On Congress-Twitter (475 nodes) a budget of 50
already covers 10.5% of the network, so this is where a stopping rule can actually fire.

Policies
--------
``static_degree``    top-k out-degree, chosen up front.  No state, no order.
``nonadaptive``      greedy over the *empty-prefix* state only, chosen up front.  Knows the
                     objective but never observes the cascade.
``adaptive``         re-scores every remaining candidate against the realised state after
                     each step, and stops early when the best marginal gain is not positive.
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

from evaluate_overexposure_paper_graphs import LOADERS, normalise_in_weights  # noqa: E402
from evaluate_overexposure_pool_ranking import degree_scores  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    run_overexposure,
    sample_threshold_windows,
)


@dataclass
class Result:
    graph: str
    budget: int
    seed_fraction: float
    trial: int
    static_spread: float
    nonadaptive_spread: float
    adaptive_spread: float
    stateaware_spread: float
    adaptive_seeds_used: int
    stateaware_seeds_used: int


def observed_state(
    graph: nx.DiGraph, seeds: list[int], windows: dict, rng: random.Random
) -> dict[int, float]:
    """The realised exposure state after running the cascade for ``seeds``.

    This is what an adaptive policy *observes*.  It is a single cascade, because the
    threshold windows are already fixed -- given ``seeds`` and the draw, ``delta`` is a
    deterministic set function (Gate 1d).
    """
    return dict(run_overexposure(graph, list(seeds), windows, rng).delta)


def stateaware_step(
    graph: nx.DiGraph,
    seeds: list[int],
    remaining: list[int],
    delta_state: dict[int, float],
) -> tuple[int, float]:
    """Pick the next seed by the analytic score evaluated at the *observed* state.

    Zero cascades: ``exposure_scores_delta`` is a two-hop closed form in terms of the
    observed ``delta``.  This is the cheap alternative to Monte-Carlo greedy, and the
    thing a learned policy would have to beat.
    """
    from evaluate_overexposure_pool_ranking import exposure_scores_delta

    scores = exposure_scores_delta(graph, remaining, delta_state, set(seeds), hops=2)
    best = max(range(len(remaining)), key=lambda i: scores[i])
    return best, scores[best]


def _marginal_gains(
    graph: nx.DiGraph,
    seeds: list[int],
    remaining: list[int],
    windows: dict,
    rng: random.Random,
    mc: int,
) -> tuple[list[float], int]:
    """Paired marginal gain of every remaining candidate, sharing one set of windows.

    ``estimate_overexposure_spread_over_configs`` shares threshold windows across the
    configurations in one call, so the differences below are not contaminated by threshold
    noise.  It does not accept externally supplied windows, so the realised cascade is
    replayed through it -- the policy still observes state, it just also samples.
    """
    configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc, rng.randrange(1 << 30)
    )
    base = estimates[0]["mean"]
    gains = [estimates[i + 1]["mean"] - base for i in range(len(remaining))]
    return gains, len(remaining) * mc


def run_trial(
    graph: nx.DiGraph,
    label: str,
    budget: int,
    pool_size: int,
    mc: int,
    mc_eval: int,
    trial: int,
    base_seed: int,
) -> Result:
    nodes = list(graph.nodes())
    n = len(nodes)
    rng = random.Random(base_seed + 104729 * trial)
    pool = rng.sample(nodes, min(pool_size, n))

    # --- static degree ------------------------------------------------------------
    degree = degree_scores(graph, pool)
    static_seeds = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:budget]]

    # --- non-adaptive: greedily pick the whole set against the empty-prefix state ---
    nonadaptive_seeds: list[int] = []
    remaining = list(pool)
    for _ in range(budget):
        if not remaining:
            break
        gains, _ = _marginal_gains(graph, nonadaptive_seeds, remaining, {}, rng, mc)
        best = max(range(len(gains)), key=lambda i: gains[i])
        nonadaptive_seeds.append(remaining.pop(best))

    # --- adaptive: re-observe the realised state, then re-score -------------------
    # One threshold draw fixes the whole realisation; the policy sees the resulting delta
    # after each step.  "Adaptive" therefore means *state-conditioned*, which is the only
    # kind of adaptivity this model admits (Gate 1d: no per-edge randomness to re-sample).
    realisation = sample_threshold_windows(nodes, random.Random(base_seed + 53 * trial + 7))
    adaptive_seeds: list[int] = []
    remaining = list(pool)
    for _ in range(budget):
        if not remaining:
            break
        gains, _ = _marginal_gains(graph, adaptive_seeds, remaining, realisation, rng, mc)
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0.0:
            break  # non-monotone objective: a correct adaptive rule must be able to stop
        adaptive_seeds.append(remaining.pop(best))

    # --- state-aware: analytic re-scoring against the observed delta, zero cascades --
    # NOTE on the stopping rule: ``exposure_scores_delta`` is a *marginal* quantity
    # (the change in expected positive out-neighbours).  Once the neighbourhood is
    # saturated, ``delta(u) -> 1`` drives ``g'(delta) -> -2`` and the score goes negative
    # even though the node still adds itself to the spread.  Its zero crossing is
    # therefore NOT an unbiased estimate of the point where adding a seed stops helping,
    # so the rule is only allowed to stop after it has taken at least two seeds.
    state_seeds: list[int] = []
    remaining = list(pool)
    for _ in range(budget):
        if not remaining:
            break
        state = observed_state(graph, state_seeds, realisation,
                               random.Random(base_seed + 53 * trial + 7))
        best, score = stateaware_step(graph, state_seeds, remaining, state)
        if score <= 0.0 and len(state_seeds) >= 2:
            break
        state_seeds.append(remaining.pop(best))

    spreads = estimate_overexposure_spread_over_configs(
        graph, [static_seeds, nonadaptive_seeds, adaptive_seeds, state_seeds],
        mc_eval, base_seed + 31
    )
    return Result(
        graph=label,
        budget=budget,
        seed_fraction=budget / n,
        trial=trial,
        static_spread=spreads[0]["mean"],
        nonadaptive_spread=spreads[1]["mean"],
        adaptive_spread=spreads[2]["mean"],
        stateaware_spread=spreads[3]["mean"],
        adaptive_seeds_used=len(adaptive_seeds),
        stateaware_seeds_used=len(state_seeds),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=120)
    parser.add_argument("--mc", type=int, default=40)
    parser.add_argument("--mc-eval", type=int, default=200)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = LOADERS[args.graph]()
    if not args.no_normalise:
        graph = normalise_in_weights(graph)
    n = graph.number_of_nodes()
    print(f"graph={args.graph} n={n} pool={args.pool_size} trials={args.trials} "
          f"mc={args.mc} mc_eval={args.mc_eval}")
    print()

    results: list[Result] = []
    for budget in args.budgets:
        for trial in range(args.trials):
            result = run_trial(graph, args.graph, budget, args.pool_size, args.mc,
                               args.mc_eval, trial, args.random_seed)
            results.append(result)
            print(f"  k={budget:>3} ({budget/n*100:>4.1f}%) trial={trial}  "
                  f"static={result.static_spread:>8.2f}  "
                  f"nonadaptive={result.nonadaptive_spread:>8.2f}  "
                  f"adaptive={result.adaptive_spread:>8.2f} "
                  f"({result.adaptive_seeds_used:>2}s)  "
                  f"stateaware={result.stateaware_spread:>8.2f} "
                  f"({result.stateaware_seeds_used:>2}s)", flush=True)

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(f"  {'k':>5}{'|S|/n':>8}{'static':>10}{'nonadapt':>10}{'adaptive':>10}"
          f"{'stateaware':>12}{'sw-static':>11}{'ad-static':>11}{'sw-nonad':>10}")
    for budget in args.budgets:
        rows = [r for r in results if r.budget == budget]
        s = statistics.fmean([r.static_spread for r in rows])
        na = statistics.fmean([r.nonadaptive_spread for r in rows])
        ad = statistics.fmean([r.adaptive_spread for r in rows])
        sw = statistics.fmean([r.stateaware_spread for r in rows])
        print(f"  {budget:>5}{budget/n*100:>7.1f}%{s:>10.2f}{na:>10.2f}{ad:>10.2f}"
              f"{sw:>12.2f}{sw-s:>+11.2f}{ad-s:>+11.2f}{sw-na:>+10.2f}")

    print()
    print("VERDICT")
    print("-" * 104)
    all_rows = results
    d_sw = statistics.fmean([r.stateaware_spread - r.static_spread for r in all_rows])
    d_ad = statistics.fmean([r.adaptive_spread - r.static_spread for r in all_rows])
    sw_wins = sum(1 for r in all_rows if r.stateaware_spread > r.static_spread)
    stopped = sum(1 for r in all_rows if r.stateaware_seeds_used < r.budget)
    print(f"  stateaware - static : {d_sw:+.2f}   ({sw_wins}/{len(all_rows)} trials win)")
    print(f"  adaptive   - static : {d_ad:+.2f}")
    print(f"  stateaware stopped early : {stopped}/{len(all_rows)} trials")
    if stopped:
        used = [r.stateaware_seeds_used for r in all_rows
                if r.stateaware_seeds_used < r.budget]
        print(f"    when it stopped it used {min(used)}-{max(used)} seeds instead of the budget")
    if d_sw > 0:
        print("  -> the zero-cascade state-conditioned rule beats the state-blind degree")
        print("     baseline, so sequential state observation has value.")
    else:
        print("  -> the zero-cascade state-conditioned rule does NOT beat degree here.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph,
            "n": n,
            "budgets": args.budgets,
            "trials": args.trials,
            "pool_size": args.pool_size,
            "mc": args.mc,
            "mc_eval": args.mc_eval,
            "in_weights_normalised": not args.no_normalise,
            "results": [asdict(r) for r in results],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
