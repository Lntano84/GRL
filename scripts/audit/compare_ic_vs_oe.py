"""DEPRECATED -- kept for the record, do not use for the paper.

Why this file is deprecated
---------------------------
It reported "OE greedy 367.46 vs IC greedy 88.30" and presented that as a fair head-to-head.
It is not.  The script draws its candidate pool as 40 uniformly random nodes, and inside that
pool the overexposure process sits in a strongly percolating regime while IC stays subcritical.
The result therefore reports *process scale*, not optimisation quality.

A separate calibration run that used seeds drawn uniformly from all nodes reported near parity
at the same graph (|S| = 20: IC 115.81 vs OE 118.68).  The disagreement between the two is a
protocol difference, and stitching them together into one "comparison" was my error.

The protocol that replaced it is ``find_comparable_regime.py``, which holds the seed-source
protocol fixed while sweeping seed-set size so the comparable band can be identified, and
``docs/AUDIT_AND_PLAN.md`` section 6.3 records the outcome.  The paper does not do a cross-model
spread comparison at all; it compares each model against its own oracle.

Kept rather than deleted because the task brief forbids removing failed experiments and the
negative record is informative: it is the concrete instance of the "different process scales"
trap.

--- original docstring below ---

Fair head-to-head: IC versus overexposure, with a solver given to each.

Protocol (identical for both diffusion models)
----------------------------------------------
* same graph, same seed budget k, same candidate pool
* same evaluation budget: the final spread of every method is measured with the SAME number
  of Monte-Carlo trials under the SAME diffusion model
* every solver lives inside its own diffusion model, so no method is handicapped by having to
  optimise a process it was not designed for

Methods, per model:
  degree      -- out-degree top-k, no simulation
  greedy_mc   -- Monte-Carlo greedy inside that model (the model's own near-oracle)

Reported per model and budget:
  spread(degree), spread(greedy_mc), the gap closed, and the spread of greedy_mc relative to
  the other model's greedy_mc.  The last row is the actual claim: does an optimiser inside the
  overexposure model reach more influence than an optimiser inside IC, on the same graph and
  the same budget?

Why the comparison is now meaningful
------------------------------------
At the repository's native ``kappa_hi = 1.0`` the two processes percolate at comparable scale
on this graph (at |S| = 20: IC 115.8, OE 118.7, ratio 1.03).  An earlier audit reported a 10x gap
but that came from a single ``max_out_degree`` seed chosen deliberately for one diagnostic; it
does not generalise and is not used here.
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
from evaluate_overexposure_pool_ranking import degree_scores  # noqa: E402
from grl.diffusion import (  # noqa: E402
    estimate_spread_over_configs as ic_paired,
    estimate_overexposure_spread_over_configs as oe_paired,
    run_independent_cascade,
    run_overexposure,
    sample_threshold_windows,
)


def ic_gains(graph: nx.DiGraph, seeds: list[int], candidates: list[int],
             trials: int, seed: int) -> list[float]:
    """Paired marginal gains under IC, sharing one live-edge sample per trial."""
    configs = [list(seeds)] + [[*seeds, c] for c in candidates]
    return [e["mean"] for e in ic_paired(graph, configs, trials, seed)]


def oe_gains(graph: nx.DiGraph, seeds: list[int], candidates: list[int],
             trials: int, seed: int) -> list[float]:
    configs = [list(seeds)] + [[*seeds, c] for c in candidates]
    return [e["mean"] for e in oe_paired(graph, configs, trials, seed)]


def greedy(graph: nx.DiGraph, pool: list[int], budget: int, model: str,
           mc: int, seed: int) -> tuple[list[int], int]:
    """MC greedy inside ``model``; returns (seeds, candidate_evaluations)."""
    chosen: list[int] = []
    remaining = list(pool)
    evals = 0
    for step in range(budget):
        if not remaining:
            break
        fn = ic_gains if model == "ic" else oe_gains
        vals = fn(graph, chosen, remaining, mc, seed + 1009 * step)
        base, gains = vals[0], vals[1:]
        evals += len(remaining)
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0:
            break
        chosen.append(remaining.pop(best))
    return chosen, evals


@dataclass
class Row:
    model: str
    budget: int
    method: str
    spread: float
    stderr: float
    candidate_evals: int


def measure_ic(graph: nx.DiGraph, seeds: list[int], trials: int, seed: int) -> tuple[float, float]:
    vals = [float(run_independent_cascade(graph, seeds, random.Random(seed + t)))
            for t in range(trials)]
    return statistics.fmean(vals), statistics.stdev(vals) / math.sqrt(trials)


def measure_oe(graph: nx.DiGraph, seeds: list[int], trials: int, seed: int) -> tuple[float, float]:
    nodes = list(graph.nodes())
    vals = []
    for t in range(trials):
        r = random.Random(seed + t)
        w = sample_threshold_windows(nodes, r)
        vals.append(float(run_overexposure(graph, seeds, w, r).spread))
    return statistics.fmean(vals), statistics.stdev(vals) / math.sqrt(trials)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--greedy-mc", type=int, default=20)
    parser.add_argument("--eval-trials", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} m={graph.number_of_edges()} "
          f"<k>={2*graph.number_of_edges()/n:.2f}")
    print(f"pool={args.pool_size} greedy_mc={args.greedy_mc} eval_trials={args.eval_trials}")
    print()

    rows: list[Row] = []
    for budget in args.budgets:
        rng = random.Random(args.random_seed + 31 * budget)
        pool = rng.sample(nodes, min(args.pool_size, n))
        degree = degree_scores(graph, pool)
        by_degree = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:budget]]

        print(f"k = {budget}")
        for model in ("ic", "oe"):
            g_seeds, evals = greedy(graph, pool, budget, model, args.greedy_mc,
                                    args.random_seed + 77 * budget)
            measure = measure_ic if model == "ic" else measure_oe
            s_deg, se_deg = measure(graph, by_degree, args.eval_trials, args.random_seed + 5)
            s_gre, se_gre = measure(graph, g_seeds, args.eval_trials, args.random_seed + 5)
            rows.append(Row(model, budget, "degree", s_deg, se_deg, 0))
            rows.append(Row(model, budget, "greedy_mc", s_gre, se_gre, evals))
            print(f"  {model:<4} degree    {s_deg:9.2f} +-{se_deg:5.2f}")
            print(f"  {model:<4} greedy_mc {s_gre:9.2f} +-{se_gre:5.2f}   "
                  f"({evals} candidate evals, {len(g_seeds)} seeds)")
        ic_g = [r for r in rows if r.budget == budget and r.model == "ic"
                and r.method == "greedy_mc"][0]
        oe_g = [r for r in rows if r.budget == budget and r.model == "oe"
                and r.method == "greedy_mc"][0]
        delta = oe_g.spread - ic_g.spread
        pooled_se = math.sqrt(ic_g.stderr ** 2 + oe_g.stderr ** 2)
        print(f"  --> greedy_mc: OE {oe_g.spread:.2f} vs IC {ic_g.spread:.2f}   "
              f"delta={delta:+.2f} (pooled se {pooled_se:.2f}, {abs(delta)/pooled_se:.1f} sigma)")
        print()

    print("=" * 88)
    print("HEAD-TO-HEAD SUMMARY (same graph, budget, pool and evaluation budget)")
    print("=" * 88)
    print(f"  {'k':>4}{'IC greedy':>12}{'OE greedy':>12}{'delta':>10}{'sigma':>8}")
    for budget in args.budgets:
        ic_g = [r for r in rows if r.budget == budget and r.model == "ic"
                and r.method == "greedy_mc"][0]
        oe_g = [r for r in rows if r.budget == budget and r.model == "oe"
                and r.method == "greedy_mc"][0]
        pooled_se = math.sqrt(ic_g.stderr ** 2 + oe_g.stderr ** 2)
        delta = oe_g.spread - ic_g.spread
        print(f"  {budget:>4}{ic_g.spread:>12.2f}{oe_g.spread:>12.2f}{delta:>+10.2f}"
              f"{(abs(delta)/pooled_se if pooled_se else float('nan')):>8.1f}")
    print()
    print("  NOTE: this compares each model's own near-oracle, which is the fair test of")
    print("  whether the overexposure setting yields more influence at equal budget.  It is")
    print("  NOT a claim that our GRL pipeline beats IC -- that claim needs the pipeline")
    print("  running inside the overexposure model, which is stage 5.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "budgets": args.budgets,
            "pool_size": args.pool_size, "greedy_mc": args.greedy_mc,
            "eval_trials": args.eval_trials,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
