"""Gate-1b diagnostic: does sequential/adaptive selection actually pay off?

Motivation
----------
The DASFAA plan rests on the claim that overexposure makes *sequential* decision
making valuable.  Under the standard IC/LT models the objective is a set function, so
"sequential" and "static" coincide and the claim would be vacuous.  Under overexposure
the state is a monotone accumulator, so a decision made at step t changes what is
available at step t+1 -- but that alone does not prove adaptivity helps.

This script separates three policies that are easy to conflate:

``static_degree``
    Take the top-``k`` nodes by out-degree, decided before anything runs.  No model, no
    observation, no order dependence.  This is the "just pick popular nodes" baseline.

``nonadaptive_greedy``
    Commit to a size-``k`` seed set chosen by greedy marginal gain, but decide the whole
    set *once*, before the cascade runs.  This is the best you can do if you must fix the
    plan in advance -- the realistic situation when a campaign is scheduled up front.

``adaptive_greedy``
    Re-select at every step using the observed state of the actual cascade.  This is
    what a state-aware learned policy is trying to approximate.

The quantity that decides the direction is the *gap*:

    adaptive_greedy  -  nonadaptive_greedy

If that gap is small, sequential observation is not worth a learned model and Gate 1b
fails.  If it is material, the setting genuinely rewards state conditioning, and a
learning contribution has somewhere to live.

Implementation note
-------------------
Thresholds are sampled once per trial and held fixed; under the deterministic activation
mode a realization is therefore fully determined by its threshold draw.  ``adaptive``
is evaluated exactly inside each realization (it observes the running state), while
``nonadaptive`` is decided on an independent threshold draw and then scored on the
evaluation draw -- i.e. it cannot see the realization it is being judged on.  That is
the honest comparison: the adaptive policy may look, the non-adaptive policy may not.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    DETERMINISTIC,
    STOCHASTIC,
    run_overexposure,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402

POLICY_STATIC = "static_degree"
POLICY_NONADAPTIVE = "nonadaptive_greedy"
POLICY_ADAPTIVE = "adaptive_greedy"
POLICIES = (POLICY_STATIC, POLICY_NONADAPTIVE, POLICY_ADAPTIVE)


def _out_degree(graph) -> dict[int, int]:
    return {node: int(degree) for node, degree in graph.out_degree()}


def spread_of(graph, seeds, windows, rng, activation_mode: str = DETERMINISTIC) -> int:
    return run_overexposure(
        graph, list(seeds), windows, rng, activation_mode=activation_mode
    ).spread


def greedy_step_gain(
    graph,
    seeds: list[int],
    pool: list[int],
    windows,
    base_spread: int,
    activation_mode: str = DETERMINISTIC,
) -> tuple[int, int]:
    """Exact best next seed for a *given* threshold realization.

    Returns ``(candidate, gain)``.  Exactness is affordable here because the deterministic
    activation mode makes a realization a pure function of the thresholds, so each
    candidate needs one cascade rather than a Monte-Carlo estimate.
    """
    best_candidate, best_gain = -1, float("-inf")
    for candidate in pool:
        spread = spread_of(graph, [*seeds, candidate], windows, random.Random(0))
        if spread > best_gain:
            best_candidate, best_gain = candidate, spread
    return best_candidate, best_gain - base_spread


def adaptive_seeds(
    graph, pool: list[int], k: int, windows, rng, activation_mode: str = DETERMINISTIC
) -> list[int]:
    seeds: list[int] = []
    remaining = list(pool)
    base = spread_of(graph, seeds, windows, rng, activation_mode)
    for _ in range(k):
        if not remaining:
            break
        candidate, gain = greedy_step_gain(graph, seeds, remaining, windows, base, activation_mode)
        if candidate < 0 or gain <= 0:
            # Overexposure makes further seeds worthless or harmful: stop early.
            # (The objective is non-monotone, so the greedy rule must be allowed to stop.)
            break
        seeds.append(candidate)
        remaining.remove(candidate)
        base = spread_of(graph, seeds, windows, rng, activation_mode)
    return seeds


def nonadaptive_seeds(
    graph, pool: list[int], k: int, planning_windows, planning_rng, rounds: int,
    activation_mode: str = DETERMINISTIC,
) -> list[int]:
    """Choose a size-k set once, scoring candidates by average exact gain.

    The planner may run its own simulations (``planning_windows`` / ``rounds``) but never
    sees the evaluation realization.
    """
    seeds: list[int] = []
    remaining = list(pool)
    for _ in range(k):
        if not remaining:
            break
        best_candidate, best_score = -1, float("-inf")
        for candidate in remaining:
            total = 0.0
            for _ in range(rounds):
                rng = random.Random(0)
                total += spread_of(graph, [*seeds, candidate], planning_windows, rng)
            score = total / rounds
            if score > best_score:
                best_candidate, best_score = candidate, score
        if best_candidate < 0:
            break
        seeds.append(best_candidate)
        remaining.remove(best_candidate)
    return seeds


def run(
    config_path: Path,
    budgets: list[int],
    trials: int,
    pool_size: int,
    planning_rounds: int,
    random_seed: int,
    mc_eval_runs: int,
    activation_mode: str = DETERMINISTIC,
) -> dict:
    config = load_yaml_config(config_path)
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    nodes = list(graph.nodes())
    degree = _out_degree(graph)

    rng = random.Random(random_seed)
    observations: list[dict] = []

    for budget in budgets:
        for trial in range(trials):
            # A fixed candidate pool shared by every policy in this trial.
            pool = rng.sample(nodes, min(pool_size, len(nodes)))
            pool_degree_sorted = sorted(pool, key=lambda x: (-degree[x], x))

            # Policy 1: static degree, no state at all.
            static_seeds = pool_degree_sorted[:budget]

            # Policy 2: non-adaptive greedy, planned on an independent threshold draw.
            planning_windows = sample_threshold_windows(nodes, random.Random(random_seed + 7 * trial + budget))
            nonadaptive = nonadaptive_seeds(
                graph, pool, budget, planning_windows, rng, planning_rounds,
                activation_mode,
            )

            spreads = {POLICY_STATIC: [], POLICY_NONADAPTIVE: [], POLICY_ADAPTIVE: []}
            adaptive_sizes: list[int] = []
            for run_index in range(mc_eval_runs):
                eval_rng = random.Random(random_seed + 100000 + 13 * trial + budget + run_index)
                windows = sample_threshold_windows(nodes, eval_rng)
                spreads[POLICY_STATIC].append(
                    spread_of(graph, static_seeds, windows, eval_rng, activation_mode)
                )
                spreads[POLICY_NONADAPTIVE].append(
                    spread_of(graph, nonadaptive, windows, eval_rng, activation_mode)
                )
                adaptive = adaptive_seeds(graph, pool, budget, windows, eval_rng, activation_mode)
                adaptive_sizes.append(len(adaptive))
                spreads[POLICY_ADAPTIVE].append(
                    spread_of(graph, adaptive, windows, eval_rng, activation_mode)
                )

            record = {
                "budget": budget,
                "trial": trial,
                "pool_size": len(pool),
                "means": {name: statistics.fmean(values) for name, values in spreads.items()},
                "stds": {
                    name: (statistics.pstdev(values) if len(values) > 1 else 0.0)
                    for name, values in spreads.items()
                },
                "mean_adaptive_seed_count": statistics.fmean(adaptive_sizes),
                "static_seeds": static_seeds,
                "nonadaptive_seeds": nonadaptive,
            }
            observations.append(record)
            m = record["means"]
            print(
                f"  k={budget} trial={trial} | static={m[POLICY_STATIC]:8.2f} "
                f"nonadaptive={m[POLICY_NONADAPTIVE]:8.2f} adaptive={m[POLICY_ADAPTIVE]:8.2f} "
                f"| adaptive-|\u003eS|={record['mean_adaptive_seed_count']:.1f}",
                flush=True,
            )

    payload = {
        "config": str(config_path),
        "budgets": budgets,
        "trials": trials,
        "pool_size": pool_size,
        "planning_rounds": planning_rounds,
        "mc_eval_runs": mc_eval_runs,
        "activation_mode": activation_mode,
        "random_seed": random_seed,
        "observations": observations,
        "summary": _summarise(observations, budgets),
    }
    return payload


def _summarise(observations: list[dict], budgets: list[int]) -> dict:
    summary: dict = {"overall": {}, "by_budget": {}}
    for name in POLICIES:
        values = [o["means"][name] for o in observations]
        summary["overall"][name] = {
            "mean_spread": statistics.fmean(values),
            "std_spread": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    gaps = [
        o["means"][POLICY_ADAPTIVE] - o["means"][POLICY_NONADAPTIVE] for o in observations
    ]
    summary["overall"]["adaptive_minus_nonadaptive"] = {
        "mean": statistics.fmean(gaps),
        "std": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
        "relative": statistics.fmean(
            (o["means"][POLICY_ADAPTIVE] - o["means"][POLICY_NONADAPTIVE])
            / o["means"][POLICY_NONADAPTIVE]
            for o in observations
        ),
    }
    summary["overall"]["static_minus_nonadaptive"] = {
        "mean": statistics.fmean(
            o["means"][POLICY_STATIC] - o["means"][POLICY_NONADAPTIVE] for o in observations
        )
    }
    for budget in budgets:
        rows = [o for o in observations if o["budget"] == budget]
        summary["by_budget"][f"k{budget}"] = {
            name: statistics.fmean([r["means"][name] for r in rows]) for name in POLICIES
        }
        summary["by_budget"][f"k{budget}"]["adaptive_minus_nonadaptive"] = statistics.fmean(
            r["means"][POLICY_ADAPTIVE] - r["means"][POLICY_NONADAPTIVE] for r in rows
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5])
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=150)
    parser.add_argument("--planning-rounds", type=int, default=3)
    parser.add_argument("--mc-eval-runs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument(
        "--activation-mode", choices=[DETERMINISTIC, STOCHASTIC], default=DETERMINISTIC
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print("Gate-1b diagnostic: is sequential/adaptive selection worth anything?")
    print(f"  budgets={args.budgets} trials={args.trials} pool={args.pool_size} "
          f"eval_runs={args.mc_eval_runs} mode={args.activation_mode}")
    print()

    payload = run(
        args.config, args.budgets, args.trials, args.pool_size,
        args.planning_rounds, args.random_seed, args.mc_eval_runs,
        args.activation_mode,
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'policy':<26}{'mean spread':>14}{'std':>10}")
    for name in POLICIES:
        row = payload["summary"]["overall"][name]
        print(f"{name:<26}{row['mean_spread']:>14.2f}{row['std_spread']:>10.2f}")
    print()
    gap = payload["summary"]["overall"]["adaptive_minus_nonadaptive"]
    static_gap = payload["summary"]["overall"]["static_minus_nonadaptive"]
    print(f"adaptive - nonadaptive   : {gap['mean']:+8.2f}  ({gap['relative']*100:+.2f}%)")
    print(f"nonadaptive - static     : {-static_gap['mean']:+8.2f}")
    print()
    print("GATE 1b judgement")
    rel = gap["relative"]
    if rel > 0.05:
        print(f"  -> PASS: adaptive beats non-adaptive by {rel*100:.1f}%; sequential state "
              "conditioning has real value.")
    elif rel > 0.01:
        print(f"  -> WEAK: gap is only {rel*100:.1f}%; sequential value is marginal.")
    else:
        print(f"  -> FAIL: gap is {rel*100:.2f}%; sequential decision making does not pay, "
              "so the setting-change argument needs rebuilding.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
