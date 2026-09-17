"""Audit: why is overexposure spread ~30x larger than IC spread on the same graph?

Measured on Congress-Twitter at |S|/n = 0 (single candidate seed):
    ic       mean gain  +9.18
    oe       mean gain  +206.7
    oe_free  mean gain  +277.0     <- clamping tau to 1 INCREASES spread

That ordering is the opposite of "removing overexposure reduces spread", so either the model
has a property we have not understood, or something in the implementation is off.  This script
isolates the mechanism by measuring, for a single seed:

* the seed's out-degree and the sum of its out-edge weights,
* the distribution of delta reached by its out-neighbours after one propagation round,
* the activation rate under three rules:
    (a) the window rule as implemented, i.e. activate when kappa <= delta <= tau
    (b) the pure probability rule g(delta) = 2*delta*(1-delta) with an independent coin,
* and the same quantities under IC for comparison.

The point is to find which quantity differs by an order of magnitude from the intuition
"one seed reaches about one neighbour".
"""

from __future__ import annotations

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
from grl.diffusion import (  # noqa: E402
    positive_activation_probability,
    run_independent_cascade,
    run_overexposure,
    sample_threshold_windows,
)


def main() -> int:
    graph_name = sys.argv[1] if len(sys.argv) > 1 else "congress_twitter"
    graph = load_graph(graph_name)
    nodes = list(graph.nodes())
    n = len(nodes)

    # Pick a high-out-degree seed: the most favourable case for both models.
    seed = max(nodes, key=lambda v: graph.out_degree(v))
    out_edges = [(v, float(d.get("weight", 0.0)))
                 for _, v, d in graph.out_edges(seed, data=True)]
    out_deg = len(out_edges)
    w_sum = sum(w for _, w in out_edges)
    in_deg_seed = graph.in_degree(seed)

    print(f"graph={graph_name} n={n}")
    print(f"seed={seed}  out_degree={out_deg}  sum_out_weights={w_sum:.4f}  "
          f"in_degree={in_deg_seed}")
    print(f"  edges are in-weight-normalised: each out-edge weight ~ 1/in_degree(target)")
    print()

    trials = 200
    ic_vals, oe_vals, oefree_vals = [], [], []
    delta_means, within_window, prob_rule = [], [], []

    for t in range(trials):
        ic_vals.append(run_independent_cascade(graph, [seed], random.Random(1000 + t)))

        r = random.Random(1000 + t)
        windows = sample_threshold_windows(nodes, r)
        oe_vals.append(run_overexposure(graph, [seed], windows, r).spread)

        r2 = random.Random(1000 + t)
        w2 = sample_threshold_windows(nodes, r2, overexposure_free=True)
        oefree_vals.append(run_overexposure(graph, [seed], w2, r2).spread)

        # One deterministic propagation round from the seed, by hand.
        r3 = random.Random(1000 + t)
        w3 = sample_threshold_windows(nodes, r3)
        deltas = {}
        for target, weight in out_edges:
            deltas[target] = deltas.get(target, 0.0) + weight
        if deltas:
            delta_means.append(statistics.fmean(deltas.values()))
            within_window.append(sum(
                1 for v, d in deltas.items() if w3[v][0] <= d <= w3[v][1]
            ) / len(deltas))
            # expected activations under the window rule, for a single round
            prob_rule.append(sum(
                positive_activation_probability(d) for d in deltas.values()
            ))

    print(f"after {trials} trials, single seed:")
    print(f"  ic spread        mean={statistics.fmean(ic_vals):8.3f}")
    print(f"  oe spread        mean={statistics.fmean(oe_vals):8.3f}")
    print(f"  oe_free spread   mean={statistics.fmean(oefree_vals):8.3f}")
    print()
    print(f"  one-round neighbour delta: mean={statistics.fmean(delta_means):.6f} "
          f"(out_degree={out_deg})")
    print(f"  fraction of out-neighbours whose delta lands in their window: "
          f"{statistics.fmean(within_window)*100:.2f}%")
    print(f"  expected activations from one round via g(delta): "
          f"{statistics.fmean(prob_rule):.3f}")
    print()
    print("  interpretation:")
    print(f"    a single seed has {out_deg} out-neighbours, each receiving about "
          f"{w_sum/out_deg if out_deg else 0:.6f} of influence")
    print(f"    if the window rule fires on ~{statistics.fmean(within_window)*100:.2f}% of them, "
          f"one round yields ~{statistics.fmean(within_window)*out_deg:.2f} activations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
