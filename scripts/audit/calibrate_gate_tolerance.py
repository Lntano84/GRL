"""Calibrate the absolute quality tolerance against what the experiment can actually resolve.

Why this is needed before the Go/No-Go sweep
--------------------------------------------
:mod:`grl.evaluation.gate` requires the paired confidence interval to be **narrower than the
tolerance**, and returns ``UNDECIDED`` otherwise, on the principle that a measurement which cannot
resolve the question has not answered it.  The smoke run showed almost every cell returning
``UNDECIDED`` at a tolerance of ``|D| / 1000``.

That is not a bug in the gate.  It is the gate working: the proposed relative criterion (``<= 1%``)
and the proposed absolute anchor ("one node per thousand") both sit below what this experiment can
measure at any feasible budget.  This script measures the actual paired standard error and reports,
for each candidate tolerance, the Monte-Carlo budget that would be required to resolve it --- so the
choice of anchor becomes an explicit, costed decision instead of an assumption.

It answers three questions:
  1. What is the paired standard error of a reference-vs-method comparison, per cell, at a given MC?
  2. How much MC would a given absolute tolerance require?  (SE scales as 1/sqrt(MC).)
  3. Which tolerances are resolvable *today*, and at what cost?

Run it, then set ``--tolerance-per-thousand`` (or switch to the resolution anchor) from the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from go_no_go import (  # noqa: E402
    SEQUENTIAL_ARM,
    STATIC_ARMS,
    degree_stratified_pool,
    run_delta2_sequential,
    run_mc_greedy,
    run_static_degree,
    run_static_delta2,
)
from grl.algorithms.sequential_im import FILL_BUDGET  # noqa: E402
from grl.data.weights import SUM_TO_ONE, normalise_in_weights  # noqa: E402
from grl.diffusion.contract import resolve_target_contract  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.evaluation.gate import tolerance_from_target_fraction  # noqa: E402


def paired_se_for_seed_sets(graph, nodes, a: list[int], b: list[int], trials: int, seed: int,
                            params) -> tuple[float, float, float]:
    """Paired mean and standard error of ``spread(a) - spread(b)`` on shared windows."""
    from grl.diffusion import overexposure as oe

    diffs: list[float] = []
    for offset in range(trials):
        rng = random.Random(seed + offset)
        windows = oe.sample_threshold_windows(nodes, rng)
        sa = float(oe.run_overexposure(graph, list(a), windows, rng,
                                       activation_mode=params.activation_mode).spread)
        sb = float(oe.run_overexposure(graph, list(b), windows, rng,
                                       activation_mode=params.activation_mode).spread)
        diffs.append(sa - sb)
    mean = statistics.fmean(diffs)
    se = (statistics.pstdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
    return mean, se, statistics.pstdev(diffs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--target-fraction", type=float, default=0.2)
    parser.add_argument("--mc", type=int, default=300)
    parser.add_argument("--reference-mc", type=int, default=200)
    parser.add_argument("--state-mc", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--per-thousand", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0],
                        help="candidate absolute anchors: target nodes per 1000 target nodes")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "gate_calibration.json")
    args = parser.parse_args()

    params = resolve_overexposure_params(None)
    rows: list[dict] = []
    started = time.time()

    for graph_name in args.graphs:
        graph = normalise_in_weights(load_graph(graph_name).copy(), SUM_TO_ONE)
        nodes = list(graph.nodes())
        n = len(nodes)
        contract = resolve_target_contract(graph, "degree-tail", args.target_fraction,
                                           max(args.budgets))
        eligible = contract.objective.legal_candidates(graph)
        target_size = len(contract.objective.target_set)
        tolerance = tolerance_from_target_fraction(target_size, per_thousand=1.0)
        print(f"=== {graph_name} n={n} |D|={target_size} "
              f"base tolerance (1 per 1000) = {tolerance:.4f} target nodes", flush=True)

        for budget in args.budgets:
            rng = random.Random(args.random_seed + 31 * budget)
            pool = degree_stratified_pool(graph, eligible, min(args.pool_size, len(eligible)), rng)
            seed = args.random_seed

            ref, _ = run_mc_greedy(graph, pool, budget, oracle_mc=args.reference_mc,
                                   params=params, random_seed=seed, stopping=FILL_BUDGET)
            seq, _ = run_delta2_sequential(graph, pool, budget, state_mc=args.state_mc,
                                           params=params, random_seed=seed)
            deg, _ = run_static_degree(graph, pool, budget)
            d2s, _ = run_static_delta2(graph, pool, budget, state_mc=args.state_mc,
                                       params=params, random_seed=seed)

            contrasts = {"ref_vs_seq": (ref, seq), "ref_vs_degree": (ref, deg),
                         "ref_vs_delta2_static": (ref, d2s), "seq_vs_degree": (seq, deg)}
            for label, (a, b) in contrasts.items():
                mean, se, sd = paired_se_for_seed_sets(graph, nodes, a, b, args.mc, seed + 5000,
                                                        params)
                # MC needed for the CI half-width to fall to a given tolerance
                needed = {}
                for per_thousand in args.per_thousand:
                    tol = tolerance_from_target_fraction(target_size, per_thousand=per_thousand)
                    if se > 0:
                        # half-width = 1.96 * se / sqrt(scale) <= tol  =>  scale >= (1.96 se / tol)^2
                        scale = (1.96 * se / tol) ** 2
                    else:
                        scale = 0.0
                    needed[f"{per_thousand:g}_per_1000"] = {
                        "tolerance": tol, "mc_scale_needed": scale,
                        "mc_needed": round(args.mc * scale),
                    }
                rows.append({
                    "graph": graph_name, "n": n, "budget": budget, "contrast": label,
                    "mc": args.mc, "paired_mean": mean, "paired_se": se, "paired_sd": sd,
                    "ci_width_95": 2 * 1.96 * se,
                    "target_size": target_size,
                    "mc_needed": needed,
                })
                print(f"  k={budget:>3} {label:<22} mean={mean:+8.3f} se={se:6.3f} "
                      f"CI95_width={2*1.96*se:7.3f}  "
                      f"MC for 1/1000: {needed['1_per_1000']['mc_needed']:>10,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "script": Path(__file__).name,
        "design": {"graphs": args.graphs, "budgets": args.budgets, "mc": args.mc,
                   "reference_mc": args.reference_mc, "state_mc": args.state_mc,
                   "pool_size": args.pool_size, "target_fraction": args.target_fraction},
        "candidates": args.per_thousand,
        "rows": rows,
    }, indent=2), encoding="utf-8")

    print()
    print("=" * 104)
    print("WHAT THE EXPERIMENT CAN RESOLVE")
    print("=" * 104)
    print(f"  {'graph':<18}{'k':>4}{'contrast':<24}{'se':>9}{'CI95 width':>12}"
          f"{'MC for 1/1000':>15}{'MC for 10/1000':>16}")
    for row in rows:
        print(f"  {row['graph']:<18}{row['budget']:>4}{row['contrast']:<24}"
              f"{row['paired_se']:>9.3f}{row['ci_width_95']:>12.3f}"
              f"{row['mc_needed']['1_per_1000']['mc_needed']:>15,}"
              f"{row['mc_needed']['10_per_1000']['mc_needed']:>16,}")
    print()
    worst = max((r["mc_needed"]["1_per_1000"]["mc_needed"] for r in rows), default=0)
    print(f"  Resolving a 1-node-per-1000 tolerance needs up to {worst:,} paired trials per")
    print(f"  contrast at the current design.  At MC = {args.mc} the CI is wider than the")
    print(f"  tolerance in every cell, which is why the gate returns UNDECIDED.")
    print()
    print("  Consequences, in order of preference:")
    print("    1. anchor the tolerance at the measurement resolution "
          "(grl.evaluation.gate.tolerance_from_reference_se)")
    print("       and say so in the caption --- it asks 'is the method indistinguishable from the")
    print("       reference', which is a weaker but answerable question;")
    print("    2. or raise --eval-mc by the scale printed above for the cells that matter;")
    print("    3. or report the quality difference with its CI and decline to gate on it at all.")
    print(f"\n  wall time {time.time()-started:.0f}s; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
