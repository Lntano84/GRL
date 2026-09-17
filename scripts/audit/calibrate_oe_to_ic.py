"""Find the overexposure regime that is comparable to standard IC.

Why this exists
---------------
Auditing showed a 10x spread-scale gap on the same graph and seed:

    ic       43.3
    oe      366.7
    oe_free 454.9

so a naive "OE beats IC" table compares a barely-spreading process against a percolating one,
which is not a comparison.  The instruction is to make the comparison *fair* and then win it,
not to avoid it.

The calibration lever
---------------------
Activation is ``kappa <= delta <= tau`` with ``kappa ~ U[0,1]``.  Scaling all weights by ``s``
maps ``delta -> s*delta``, so ``P(kappa <= s*delta) = P(kappa/s <= delta)``: rescaling weights
is distributionally the same as rescaling the threshold support.  We therefore parameterise the
threshold support directly as ``kappa ~ U[0, kappa_hi]`` (keeping ``tau >= kappa``), which leaves
the graph untouched and gives one clean knob for the process's percolation scale.

This script sweeps ``kappa_hi`` and reports, for identical seed sets and identical evaluation
budget:

  * mean spread under overexposure,
  * mean spread under standard IC on the same graph,
  * the ratio, so the ``kappa_hi`` at which the two processes are comparable is identifiable.

It also reports the same quantities for a *modest* seed budget rather than a single seed, because
single-seed spread is dominated by one node's out-degree.
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
from grl.diffusion import (  # noqa: E402
    run_independent_cascade,
    run_overexposure,
    sample_threshold_windows,
)


def windows_with_support(
    nodes: list[int], rng: random.Random, kappa_hi: float, tau_hi: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """Windows with kappa ~ U[0, kappa_hi], tau ~ U[kappa, tau_hi].

    ``kappa_hi = 1`` reduces to the repository's default sampler.  Smaller values make
    activation harder, which is how the process is brought down to IC's scale.
    """
    out: dict[int, tuple[float, float]] = {}
    for v in nodes:
        kappa = rng.random() * kappa_hi
        tau = kappa + rng.random() * max(1e-12, tau_hi - kappa)
        out[v] = (kappa, min(tau, tau_hi))
    return out


@dataclass
class Row:
    graph: str
    kappa_hi: float
    seed_size: int
    ic_spread: float
    oe_spread: float
    ratio: float


def paired_spread(
    graph: nx.DiGraph,
    seeds: list[int],
    kappa_hi: float,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """IC and OE mean spread for the SAME seed set, paired over trials."""
    nodes = list(graph.nodes())
    ic_vals: list[float] = []
    oe_vals: list[float] = []
    for t in range(trials):
        ic_vals.append(float(run_independent_cascade(graph, seeds, random.Random(seed + t))))
        r = random.Random(seed + t)
        w = windows_with_support(nodes, r, kappa_hi)
        oe_vals.append(float(run_overexposure(graph, seeds, w, r).spread))
    return statistics.fmean(ic_vals), statistics.fmean(oe_vals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--kappa-hi", type=float, nargs="+",
                        default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01])
    parser.add_argument("--seed-sizes", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    print(f"graph={args.graph} n={n} m={graph.number_of_edges()} "
          f"<k>={2*graph.number_of_edges()/n:.2f}")
    print(f"trials={args.trials}  (IC and OE share the same seed set and trial index)")
    print()

    rows: list[Row] = []
    for size in args.seed_sizes:
        rng = random.Random(args.random_seed + 13 * size)
        seeds = rng.sample(nodes, size)
        print(f"|S| = {size} ({size/n*100:.2f}% coverage)")
        print(f"  {'kappa_hi':>9}{'IC':>12}{'OE':>12}{'OE/IC':>9}")
        for kappa_hi in args.kappa_hi:
            ic, oe = paired_spread(graph, seeds, kappa_hi, args.trials, args.random_seed)
            ratio = oe / ic if ic > 1e-9 else float("inf")
            rows.append(Row(args.graph, kappa_hi, size, ic, oe, ratio))
            print(f"  {kappa_hi:>9.3f}{ic:>12.2f}{oe:>12.2f}{ratio:>9.3f}")
        print()

    print("=" * 88)
    print("CALIBRATION READ-OUT")
    print("=" * 88)
    for size in args.seed_sizes:
        block = [r for r in rows if r.seed_size == size]
        # the kappa_hi whose ratio is closest to 1 -- the comparable operating point
        best = min(block, key=lambda r: abs(r.ratio - 1.0))
        print(f"  |S| = {size}: closest to parity at kappa_hi = {best.kappa_hi:.3f} "
              f"(IC {best.ic_spread:.2f}, OE {best.oe_spread:.2f}, ratio {best.ratio:.3f})")
    print()
    print("  Use the parity kappa_hi as the comparable setting for any IC-vs-OE table.")
    print("  Report the native kappa_hi = 1.0 numbers separately as the modelling contrast,")
    print("  never as a head-to-head win.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "trials": args.trials,
            "kappa_hi": args.kappa_hi, "seed_sizes": args.seed_sizes,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
