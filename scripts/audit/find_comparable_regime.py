"""Locate the regime where IC and overexposure percolate comparably.

Why this is needed
------------------
Two measurements of the SAME graph disagreed by 6x:

  * calibration sweep, seeds drawn uniformly from all nodes:
        |S| = 20  ->  IC 115.81,  OE 118.68   (ratio 1.03)
  * head-to-head inside a 40-node random pool:
        k = 10    ->  IC  88.30,  OE 367.46   (ratio 4.2)

The difference is seed composition and process scale, not a contradiction in the code: the
overexposure process crosses a percolation threshold somewhere between "a couple of arbitrary
seeds" and "twenty uniform seeds", and inside the strongly percolating regime every method
saturates near the component size, so method differences compress to nothing and the IC/OE
contrast reflects process scale rather than optimisation quality.

Any claim of the form "setting X yields more influence" is only meaningful when both processes
are in a comparable regime.  This script sweeps seed-set size with a FIXED seed-source protocol
(uniform over all nodes) and reports, at each size:

  * mean spread under IC and under overexposure,
  * the ratio, and
  * the head-room each model leaves, i.e. how far degree is from MC greedy inside that model,

so the comparable band can be identified by the ratio being near 1 AND neither model being
saturated.

It deliberately does NOT mix protocols: every row uses uniform seed sets of the stated size and
the same trial index for both models.
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
from grl.diffusion import (  # noqa: E402
    run_independent_cascade,
    run_overexposure,
    sample_threshold_windows,
)


@dataclass
class Row:
    graph: str
    seed_size: int
    coverage: float
    ic_spread: float
    ic_stderr: float
    oe_spread: float
    oe_stderr: float
    ratio: float


def measure(graph: nx.DiGraph, seeds: list[int], trials: int, seed: int) -> tuple[float, float, float, float]:
    nodes = list(graph.nodes())
    ic_vals: list[float] = []
    oe_vals: list[float] = []
    for t in range(trials):
        ic_vals.append(float(run_independent_cascade(graph, seeds, random.Random(seed + t))))
        r = random.Random(seed + t)
        w = sample_threshold_windows(nodes, r)
        oe_vals.append(float(run_overexposure(graph, seeds, w, r).spread))
    return (
        statistics.fmean(ic_vals), statistics.stdev(ic_vals) / math.sqrt(trials),
        statistics.fmean(oe_vals), statistics.stdev(oe_vals) / math.sqrt(trials),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--seed-sizes", type=int, nargs="+",
                        default=[1, 2, 3, 5, 8, 12, 20, 40, 80])
    parser.add_argument("--repeats", type=int, default=5,
                        help="independent seed sets per size, to average out seed composition")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} m={graph.number_of_edges()} "
          f"<k>={2*graph.number_of_edges()/n:.2f}")
    print(f"uniform seed sets over ALL nodes, {args.repeats} independent sets per size, "
          f"{args.trials} trials, IC and OE share the trial index")
    print()
    print(f"  {'|S|':>5}{'cov':>7}{'IC':>10}{'OE':>10}{'OE/IC':>8}{'IC max':>9}{'OE max':>9}")
    print("  " + "-" * 62)

    rows: list[Row] = []
    for size in args.seed_sizes:
        ic_means, oe_means, ic_ses, oe_ses = [], [], [], []
        for rep in range(args.repeats):
            rng = random.Random(args.random_seed + 1013 * rep + 13 * size)
            seeds = rng.sample(nodes, min(size, n))
            a, ae, b, be = measure(graph, seeds, args.trials,
                                   args.random_seed + 1013 * rep)
            ic_means.append(a)
            ic_ses.append(ae)
            oe_means.append(b)
            oe_ses.append(be)
        ic_m = statistics.fmean(ic_means)
        oe_m = statistics.fmean(oe_means)
        ratio = oe_m / ic_m if ic_m > 1e-9 else float("inf")
        row = Row(args.graph, size, size / n, ic_m, statistics.fmean(ic_ses),
                  oe_m, statistics.fmean(oe_ses), ratio)
        rows.append(row)
        print(f"  {size:>5}{size/n*100:>6.1f}%{ic_m:>10.2f}{oe_m:>10.2f}{ratio:>8.2f}"
              f"{ic_m + 3*row.ic_stderr:>9.1f}{oe_m + 3*row.oe_stderr:>9.1f}")

    print()
    print("=" * 88)
    print("COMPARABLE BAND")
    print("=" * 88)
    band = [r for r in rows if 0.5 <= r.ratio <= 2.0]
    if band:
        print("  seed sizes where the two processes are within 2x of each other:")
        for r in band:
            print(f"    |S| = {r.seed_size:>3} ({r.coverage*100:.1f}% coverage): "
                  f"IC {r.ic_spread:.2f}, OE {r.oe_spread:.2f}, ratio {r.ratio:.2f}")
        print()
        print("  Use a seed size inside this band for any 'more influence' claim; outside it")
        print("  the comparison reports process scale rather than optimisation quality.")
    else:
        print("  no seed size in the swept range puts the two processes within 2x.")
        print("  The two models percolate at fundamentally different scales on this graph,")
        print("  and a head-to-head spread comparison is not meaningful without rescaling")
        print("  the diffusion parameters (see calibrate_oe_to_ic.py).")
    saturated = [r for r in rows if r.oe_spread > 0.9 * n]
    if saturated:
        print()
        print(f"  WARNING: overexposure already covers >90% of the graph at |S| = "
              f"{[r.seed_size for r in saturated]} -- method differences there are")
        print("  compressed by saturation and should not be used to rank methods.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "repeats": args.repeats, "trials": args.trials,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
