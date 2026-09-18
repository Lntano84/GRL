"""Quantify how much the P0 state-machine fix changes the earlier measurements.

The fix changed two things that every previous overexposure number depended on:

1. A positive node could previously never become negative, because the implementation marked it
   ``settled`` on its first transition.  Exposure therefore never removed anyone from the spread
   count, which inflated spreads and made the objective look more nearly monotone than it is.
2. The ``delta < 1`` guard turned ``delta = 1`` nodes negative even when their drawn window was
   ``[0, 1]``.

Rather than re-run every experiment, this script measures the effect on the two quantities the
retracted claims rested on:

* mean spread of a fixed seed set, before/after;
* the share of candidates with a negative marginal gain, before/after.

The "before" behaviour is reconstructed here by a local reimplementation of the old state rule
(settle on first transition, delta < 1 guard), so both variants run against the same graph, seed
sets, windows and trial indices.  Keeping the old rule in this file rather than in the library is
deliberate: the library should have one meaning.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from grl.diffusion import overexposure as oe  # noqa: E402


def old_rule_spread(
    graph: nx.DiGraph, seeds: list[int], windows: dict, rng: random.Random
) -> int:
    """Reimplementation of the pre-fix state machine, for comparison only.

    Differences from the library: a node that turns positive is settled forever, and
    ``delta < 1`` is required for a positive transition.
    """
    state = {v: oe.INACTIVE for v in graph.nodes()}
    delta = {v: 0.0 for v in graph.nodes()}
    settled: set[int] = set()
    positive: set[int] = set()
    negative: set[int] = set()
    directed = graph.is_directed()

    frontier = []
    for s in seeds:
        if state[s] != oe.POSITIVE:
            state[s] = oe.POSITIVE
            positive.add(s)
            frontier.append(s)

    rounds = 0
    limit = graph.number_of_nodes() + 1
    while frontier and rounds < limit:
        rounds += 1
        for node in frontier:
            edges = graph.out_edges(node, data=True) if directed else graph.edges(node, data=True)
            for u, v, data in edges:
                target = v if directed else (v if u == node else u)
                if target == node:
                    continue
                delta[target] += float(data.get("weight", 0.0))

        newly: list[int] = []
        for node in graph.nodes():
            if state[node] != oe.INACTIVE or node in settled:
                continue
            current = delta[node]
            if current <= 0.0:
                continue
            kappa, tau = windows[node]
            if current > tau:
                state[node] = oe.NEGATIVE
                negative.add(node)
                settled.add(node)
            elif current >= kappa and current < 1.0:
                state[node] = oe.POSITIVE
                positive.add(node)
                newly.append(node)
                settled.add(node)
            elif current >= 1.0:
                state[node] = oe.NEGATIVE
                negative.add(node)
                settled.add(node)
        frontier = newly

    return len(positive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} <k>={2*graph.number_of_edges()/n:.2f}")
    print(f"trials={args.trials}  (both rules share seed sets, windows and trial indices)")
    print()
    print(f"  {'k':>3}{'old spread':>12}{'new spread':>12}{'delta%':>9}"
          f"{'old neg%':>10}{'new neg%':>10}")

    rows = []
    for budget in args.budgets:
        rng = random.Random(args.random_seed + 13 * budget)
        seeds = rng.sample(nodes, budget)
        pool = [v for v in nodes if v not in set(seeds)]
        candidates = rng.sample(pool, min(args.candidates, len(pool)))

        old_spreads, new_spreads = [], []
        for t in range(args.trials):
            r = random.Random(args.random_seed + 7 + t)
            windows = oe.sample_threshold_windows(nodes, r)
            old_spreads.append(float(old_rule_spread(graph, seeds, windows, r)))
            new_spreads.append(float(oe.run_overexposure(graph, seeds, windows, r).spread))

        old_gains, new_gains = [], []
        for t in range(args.trials):
            r = random.Random(args.random_seed + 101 + t)
            windows = oe.sample_threshold_windows(nodes, r)
            base_old = old_rule_spread(graph, seeds, windows, r)
            base_new = oe.run_overexposure(graph, seeds, windows, r).spread
            for c in candidates:
                old_gains.append(
                    old_rule_spread(graph, seeds + [c], windows, r) - base_old
                )
                new_gains.append(
                    oe.run_overexposure(graph, seeds + [c], windows, r).spread - base_new
                )

        old_mean = statistics.fmean(old_spreads)
        new_mean = statistics.fmean(new_spreads)
        pct = (new_mean - old_mean) / old_mean * 100 if old_mean else float("nan")
        old_neg = sum(1 for g in old_gains if g < 0) / len(old_gains)
        new_neg = sum(1 for g in new_gains if g < 0) / len(new_gains)
        rows.append({"k": budget, "old_spread": old_mean, "new_spread": new_mean,
                     "pct_change": pct, "old_neg": old_neg, "new_neg": new_neg})
        print(f"  {budget:>3}{old_mean:>12.2f}{new_mean:>12.2f}{pct:>9.2f}"
              f"{old_neg*100:>10.1f}{new_neg*100:>10.1f}")

    print()
    print("=" * 92)
    print("IMPACT ON PRIOR CLAIMS")
    print("=" * 92)
    spread_drop = [r for r in rows if r["pct_change"] < -1.0]
    neg_increase = [r for r in rows if r["new_neg"] > r["old_neg"] + 0.02]
    print(f"  spread changed by more than 1% at k = {[r['k'] for r in spread_drop] or 'nowhere'}")
    print(f"  negative-marginal share increased at k = {[r['k'] for r in neg_increase] or 'nowhere'}")
    print()
    print("  Any earlier statement about spread levels, saturation points or the monotonicity of")
    print("  the objective was measured with the old rule and must be re-derived, not quoted.")
    print()

    if args.output:
        import json
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"graph": args.graph, "rows": rows}, indent=2),
                               encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
