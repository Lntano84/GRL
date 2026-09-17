"""Gate-2 diagnostic: the cost/quality frontier of overexposure seed selection.

Why this experiment exists
--------------------------
Gate 1a'' showed that a *training-free* two-hop analytic scorer (``delta2``) recovers
about 90-91% of the true top-k marginal gain inside a degree-retrieved pool, while plain
out-degree recovers 80-89%.  That is a problem for a "learn a better ranker" story: the
analytic baseline already sits close to the oracle for a *single* decision, and a learned
model that merely edges it out on rank correlation will not carry a paper.

The defensible story is economic.  Under the threshold-window model there is no reverse
reachability to exploit (Gate 1d: RR sets are empty), so the only exact reference is
Monte-Carlo greedy -- and that costs

    k steps  x  M candidates  x  MC runs  =  k * M * MC  cascades per query

*every time a seed set is chosen*.  The analytic scorer costs zero cascades; a trained
policy costs one forward pass.  So the contribution is not "higher Spearman" but
"comparable spread at two to three orders of magnitude less simulation".

This script measures that frontier directly.  It reports, for each policy:

* ``spread``        -- mean Monte-Carlo positive spread of the chosen seed set
* ``gain_vs_degree``-- paired improvement over the degree incumbent
* ``cascades``      -- exact number of cascade simulations the policy spent per query
* ``cascades_per_seed``

Policies
--------
``degree``          out-degree top-k inside the retrieved pool.  Zero cascades.
``delta2``          the Gate-1a'' analytic scorer, top-k inside the pool.  Zero cascades.
``greedy_exact``    Monte-Carlo greedy: at each step score every remaining candidate by
                    its simulated marginal gain.  ``k * M * MC`` cascades.
``greedy_once``     Monte-Carlo greedy planned once against a *fixed* set of windows and
                    then replayed (state-blind planning).  ``k * M * MC_plan`` cascades,
                    but it never observes the realised state -- included because Gate 1b
                    found this family *loses* to plain degree.

The two greedy variants separate "sequential" from "adaptive", which is the distinction
Gate 1b already showed to matter.
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

from evaluate_overexposure_pool_ranking import (  # noqa: E402
    build_pools,
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
)
from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    run_overexposure,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402


class CascadeCounter:
    """Counts cascade simulations so the cost axis is measured, not assumed."""

    def __init__(self) -> None:
        self.n = 0

    def run(self, graph, seeds, windows, rng):
        self.n += 1
        return run_overexposure(graph, seeds, windows, rng)


# --------------------------------------------------------------------------------------
# policies.  Each returns (seed_list, cascades_spent)
# --------------------------------------------------------------------------------------
def policy_degree(graph, pool, k, windows, rng, counter) -> list[int]:
    scores = degree_scores(graph, pool)
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in order[:k]]


def policy_delta2(graph, pool, k, windows, rng, counter, hops: int = 2) -> list[int]:
    # The state is observed once, for free: it is the empirical exposure of the empty
    # prefix plus whatever the caller has already realised.  Here it is the seedless state,
    # matching the other policies' starting point.
    delta = mean_exposure_state(graph, [], min(20, 30), 12345)
    scores = exposure_scores_delta(graph, pool, delta, set(), hops=hops)
    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in order[:k]]


def policy_greedy_adaptive(graph, pool, k, windows, rng, counter) -> list[int]:
    """Re-score every remaining candidate after each realised transition.

    This is the expensive one: ``k`` rounds, each scoring ``M`` candidates with ``MC``
    simulations to reduce noise, plus one evaluation per accepted seed.
    """
    mc = MC_PER_CANDIDATE
    seeds: list[int] = []
    remaining = list(pool)
    for _ in range(k):
        if not remaining:
            break
        best_candidate, best_gain = -1, float("-inf")
        base = counter.run(graph, seeds, windows, rng)
        configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc, rng.randrange(1 << 30)
        )
        counter.n += len(remaining) * mc
        baseline = estimates[0]["mean"]
        for index in range(len(remaining)):
            gain = estimates[index + 1]["mean"] - baseline
            if gain > best_gain:
                best_gain, best_candidate = gain, remaining[index]
        if best_candidate < 0:
            break
        seeds.append(best_candidate)
        remaining.remove(best_candidate)
    return seeds


def policy_greedy_once(graph, pool, k, windows, rng, counter) -> list[int]:
    """State-blind planning: pick the whole set against a fixed window draw."""
    mc = MC_PER_CANDIDATE
    planning_windows = sample_threshold_windows(list(graph.nodes()), random.Random(999))
    seeds: list[int] = []
    remaining = list(pool)
    for _ in range(k):
        if not remaining:
            break
        configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc, 4242
        )
        counter.n += len(remaining) * mc
        baseline = estimates[0]["mean"]
        gains = [estimates[i + 1]["mean"] - baseline for i in range(len(remaining))]
        index = max(range(len(gains)), key=lambda i: gains[i])
        seeds.append(remaining[index])
        remaining.remove(remaining[index])
    return seeds


POLICIES = {
    "degree": policy_degree,
    "delta2": policy_delta2,
    "greedy_exact": policy_greedy_adaptive,
    "greedy_once": policy_greedy_once,
}


@dataclass
class Observation:
    policy: str
    replicate: int
    budget: int
    pool_size: int
    seed_size: int
    spread: float
    spread_std: float
    cascades: int
    cascades_per_seed: float


def evaluate(
    graph,
    config_path: Path,
    budgets: list[int],
    replicates: int,
    pool_size: int,
    mc_eval: int,
    random_seed: int,
    policies: list[str],
) -> tuple[list[Observation], dict]:
    nodes = list(graph.nodes())
    observations: list[Observation] = []

    for budget in budgets:
        for replicate in range(replicates):
            rng = random.Random(random_seed + 7919 * replicate + 13 * budget)
            pools = build_pools(graph, nodes, set(), budget, pool_size, rng)
            pool = pools["degree_pool"]
            # Evaluation windows are drawn once and shared by every policy, so the
            # comparison is paired on the realisation.
            eval_seed = random_seed + 104729 * replicate + 31 * budget
            for name in policies:
                counter = CascadeCounter()
                windows = sample_threshold_windows(nodes, random.Random(eval_seed))
                seeds = POLICIES[name](graph, pool, budget, windows, rng, counter)
                spread = estimate_overexposure_spread_over_configs(
                    graph, [seeds], mc_eval, eval_seed
                )[0]["mean"]
                observations.append(Observation(
                    policy=name,
                    replicate=replicate,
                    budget=budget,
                    pool_size=len(pool),
                    seed_size=len(seeds),
                    spread=spread,
                    spread_std=float("nan"),
                    cascades=counter.n,
                    cascades_per_seed=counter.n / max(1, len(seeds)),
                ))
                print(f"  k={budget} rep={replicate} {name:<14} |S|={len(seeds):>3}"
                      f" spread={spread:>9.2f} cascades={counter.n:>9}", flush=True)

    summary = _summarise(observations, budgets, policies)
    return observations, summary


def _summarise(observations: list[Observation], budgets: list[int],
               policies: list[str]) -> dict:
    summary: dict = {"by_policy": {}, "by_policy_and_budget": {}, "paired_vs_degree": {}}

    def block(rows: list[Observation]) -> dict:
        if not rows:
            return {}
        return {
            "replicates": len(rows),
            "spread": statistics.fmean([r.spread for r in rows]),
            "cascades": statistics.fmean([r.cascades for r in rows]),
            "cascades_per_seed": statistics.fmean([r.cascades_per_seed for r in rows]),
        }

    for policy in policies:
        rows = [r for r in observations if r.policy == policy]
        summary["by_policy"][policy] = block(rows)
        for budget in budgets:
            summary["by_policy_and_budget"][f"{policy}_k{budget}"] = block(
                [r for r in rows if r.budget == budget]
            )

    # Paired improvements over the degree incumbent, keyed by (budget, replicate).
    for policy in policies:
        if policy == "degree":
            continue
        diffs, ratios = [], []
        for budget in budgets:
            for replicate in range(max((r.replicate for r in observations), default=-1) + 1):
                a = [r for r in observations
                     if r.policy == policy and r.budget == budget and r.replicate == replicate]
                b = [r for r in observations
                     if r.policy == "degree" and r.budget == budget and r.replicate == replicate]
                if a and b:
                    diffs.append(a[0].spread - b[0].spread)
                    if b[0].spread:
                        ratios.append(a[0].spread / b[0].spread)
        if diffs:
            summary["paired_vs_degree"][policy] = {
                "mean_delta": statistics.fmean(diffs),
                "mean_ratio": statistics.fmean(ratios) if ratios else float("nan"),
                "n_pairs": len(diffs),
                "wins": sum(1 for d in diffs if d > 0),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--mc-per-candidate", type=int, default=20,
                        help="MC simulations per candidate inside the greedy policies")
    parser.add_argument("--mc-eval", type=int, default=60,
                        help="MC simulations used to score the final seed set")
    parser.add_argument("--policies", nargs="+", default=list(POLICIES),
                        choices=list(POLICIES))
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    global MC_PER_CANDIDATE
    MC_PER_CANDIDATE = args.mc_per_candidate

    config = load_yaml_config(args.config)
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    print(f"graph: {graph_data.name}  nodes={graph.number_of_nodes()} "
          f"edges={graph.number_of_edges()}")
    print(f"budgets={args.budgets} replicates={args.replicates} pool={args.pool_size} "
          f"mc_per_candidate={args.mc_per_candidate} mc_eval={args.mc_eval}")
    print()

    observations, summary = evaluate(
        graph, args.config, args.budgets, args.replicates, args.pool_size,
        args.mc_eval, args.random_seed, args.policies,
    )

    print()
    print("=" * 96)
    print("COST / QUALITY FRONTIER")
    print("=" * 96)
    print(f"  {'policy':<14}{'spread':>11}{'cascades':>12}{'per seed':>11}"
          f"{'gain vs degree':>17}")
    degree_spread = summary["by_policy"].get("degree", {}).get("spread")
    for policy in args.policies:
        row = summary["by_policy"].get(policy) or {}
        if not row:
            continue
        paired = summary["paired_vs_degree"].get(policy, {})
        delta = paired.get("mean_delta", 0.0) if policy != "degree" else 0.0
        pct = (f"{100*delta/degree_spread:+.1f}%" if degree_spread else "n/a")
        print(f"  {policy:<14}{row['spread']:>11.2f}{row['cascades']:>12.0f}"
              f"{row['cascades_per_seed']:>11.1f}{pct:>17}")

    print()
    print("VERDICT (amortised-cost positioning)")
    print("-" * 96)
    exact = summary["by_policy"].get("greedy_exact") or {}
    deg = summary["by_policy"].get("degree") or {}
    d2 = summary["by_policy"].get("delta2") or {}
    if exact and deg:
        speedup = (exact["cascades"] / d2["cascades"]) if d2.get("cascades") else float("inf")
        quality_gap = (exact["spread"] - d2["spread"]) / exact["spread"] if exact["spread"] else float("nan")
        print(f"  greedy_exact spends {exact['cascades']:.0f} cascades/query to reach "
              f"spread {exact['spread']:.2f}")
        print(f"  delta2       spends {d2.get('cascades', 0):.0f} cascades/query to reach "
              f"spread {d2.get('spread', float('nan')):.2f}")
        print(f"  => delta2 leaves {quality_gap*100:.1f}% of the exact greedy spread on "
              f"the table, at {(1-speedup)*100:.4f}% of the simulation cost")
        print()
        print(f"  The learned-policy target is therefore: capture part of that "
              f"{quality_gap*100:.1f}% gap while staying at ~0 cascades/query.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "config": str(args.config),
            "graph": graph_data.name,
            "budgets": args.budgets,
            "replicates": args.replicates,
            "pool_size": args.pool_size,
            "mc_per_candidate": args.mc_per_candidate,
            "mc_eval": args.mc_eval,
            "policies": args.policies,
            "observations": [asdict(o) for o in observations],
            "summary": summary,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
