"""Measure the surrogate gap lambda(S) - sigma(S) that the source model leaves unquantified.

What the surrogate is
---------------------
The source model (Inf. Sci. 744 (2026) 123375) cannot optimise the overexposure objective
directly, so it optimises a surrogate.  Its Section 5 defines:

    sigma^kappa(.)   the objective value under the LINEAR THRESHOLD model instantiated with the
                     LOWER threshold theta^kappa, i.e. a node activates when
                     sum_{u in active in-neighbours} omega_uv > theta^kappa_v
    sigma^tau(.)     the same with the UPPER threshold theta^tau
    lambda(.)      = sigma^kappa(.) - sigma^tau(.)                    (its eq. 5-6)
    Theorem 6        lambda(S) >= sigma(S) for all S

Because theta^kappa <= theta^tau, the lower threshold activates at least as many nodes, so
sigma^kappa >= sigma^tau and lambda is the "excess" activation that overexposure removes.

Two things about this are worth measuring rather than assuming:

1. **How loose is the bound?**  lambda(S) - sigma(S) is the price of optimising the surrogate
   instead of the objective.  The source model never reports it.  If the gap is large, solving
   the surrogate says little about the objective; if it is small, the surrogate is the right
   thing to attack with sampling methods (RR *can* be applied to the two LT processes, since
   each is a genuine LT influence function).

2. **Is lambda well behaved?**  sigma^kappa and sigma^tau are each monotone submodular as LT
   influence functions, but their DIFFERENCE need not be.  The source model proves submodularity
   of the objective only for outward-tree networks, so the difference is the object of interest.

Implementation note
-------------------
The LT processes are run with a *fixed* threshold vector, so no live-edge sampling is needed:
a node activates when its active in-neighbours' weight exceeds the threshold.  To keep this
affordable on graphs with mean degree up to 56 we track the active weighted in-degree
incrementally, updating only the out-neighbours of newly activated nodes, rather than
recomputing sums for every node each round.
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
from grl.diffusion.overexposure import run_overexposure, sample_threshold_windows  # noqa: E402


def linear_threshold_spread(
    graph: nx.DiGraph,
    seeds: list[int],
    thresholds: dict[int, float],
    target: set[int] | None = None,
) -> int:
    """Spread of a fixed-threshold linear-threshold cascade.

    A node activates when the total weight of its active in-neighbours strictly exceeds its
    threshold and it is not a seed.  ``target`` restricts the count to a target set; ``None``
    counts every positively activated node.
    """
    active: set[int] = set(seeds)
    weighted_in: dict[int, float] = {v: 0.0 for v in graph.nodes()}
    frontier = list(seeds)

    while frontier:
        newly: list[int] = []
        for node in frontier:
            for target_node in graph.successors(node):
                if target_node in active:
                    continue
                weight = float(graph[node][target_node].get("weight", 0.0))
                weighted_in[target_node] += weight
                if weighted_in[target_node] > thresholds[target_node]:
                    active.add(target_node)
                    newly.append(target_node)
        frontier = newly

    if target is None:
        return len(active)
    return len(active & target)


@dataclass
class Row:
    graph: str
    seed_size: int
    seed_fraction: float
    sigma_true: float
    sigma_kappa: float
    sigma_tau: float
    lam: float
    gap: float
    gap_relative: float
    lambda_negative: bool


def measure(
    graph: nx.DiGraph,
    label: str,
    size: int,
    trials: int,
    base_seed: int,
    target: set[int] | None,
) -> Row:
    nodes = list(graph.nodes())
    rng = random.Random(base_seed)
    seeds = rng.sample(nodes, size) if size else []

    true_vals: list[float] = []
    kappa_vals: list[float] = []
    tau_vals: list[float] = []
    for t in range(trials):
        r = random.Random(base_seed + 11 + t)
        windows = sample_threshold_windows(nodes, r)
        # The overexposure objective itself.
        run = run_overexposure(graph, list(seeds), windows, r)
        true_vals.append(float(len(run.positive & target)) if target is not None
                         else float(run.spread))
        # The two LT surrogates, same fixed threshold vectors.
        kappa = {v: windows[v][0] for v in nodes}
        tau = {v: windows[v][1] for v in nodes}
        kappa_vals.append(float(linear_threshold_spread(graph, list(seeds), kappa, target)))
        tau_vals.append(float(linear_threshold_spread(graph, list(seeds), tau, target)))

    sigma_true = statistics.fmean(true_vals)
    sigma_kappa = statistics.fmean(kappa_vals)
    sigma_tau = statistics.fmean(tau_vals)
    lam = sigma_kappa - sigma_tau
    gap = lam - sigma_true
    return Row(
        graph=label,
        seed_size=len(seeds),
        seed_fraction=len(seeds) / len(nodes) if nodes else 0.0,
        sigma_true=sigma_true,
        sigma_kappa=sigma_kappa,
        sigma_tau=sigma_tau,
        lam=lam,
        gap=gap,
        gap_relative=gap / lam if abs(lam) > 1e-9 else float("nan"),
        lambda_negative=lam < 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[Row] = []
    for label in args.graphs:
        graph = load_graph(label)
        if args.no_normalise:
            # ``load_graph`` already normalises in-weights; there is no un-normalised path
            # here, so say so rather than silently ignoring the flag.
            raise SystemExit("--no-normalise is not supported: load_graph always normalises")
        n = graph.number_of_nodes()
        print(f"=== {label}: n={n} m={graph.number_of_edges()} "
              f"<k>={2*graph.number_of_edges()/n:.2f}")
        for fraction in args.fractions:
            size = int(round(fraction * n))
            row = measure(graph, label, size, args.trials,
                          args.random_seed + 13 * size, None)
            rows.append(row)
            flag = "  <-- lambda NEGATIVE" if row.lambda_negative else ""
            print(f"    |S|/n={row.seed_fraction*100:>5.1f}%  "
                  f"sigma={row.sigma_true:>9.2f}  sigma_kappa={row.sigma_kappa:>9.2f}  "
                  f"sigma_tau={row.sigma_tau:>9.2f}  lambda={row.lam:>9.2f}  "
                  f"gap={row.gap:>+10.2f}{flag}", flush=True)
        print()

    print("=" * 100)
    print("SURROGATE GAP  lambda(S) - sigma(S)      (Theorem 6 claims lambda >= sigma)")
    print("=" * 100)
    print(f"  {'graph':<18}{'|S|/n':>7}{'lambda':>11}{'sigma':>11}{'gap':>11}{'gap/lambda':>12}")
    for row in rows:
        print(f"  {row.graph:<18}{row.seed_fraction*100:>6.1f}%{row.lam:>11.2f}"
              f"{row.sigma_true:>11.2f}{row.gap:>+11.2f}{row.gap_relative:>12.3f}")

    negatives = [r for r in rows if r.lambda_negative]
    print()
    if negatives:
        print(f"  !! lambda < 0 in {len(negatives)}/{len(rows)} settings, which would contradict")
        print("     Theorem 6. Inspect those rows before quoting the bound.")
    else:
        print("  lambda >= 0 in every setting, consistent with Theorem 6.")
    worst = max(rows, key=lambda r: r.gap_relative if r.gap_relative == r.gap_relative else -9)
    print(f"  largest relative gap: {worst.gap_relative:.3f} at {worst.graph}, "
          f"|S|/n={worst.seed_fraction*100:.1f}%")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs,
            "fractions": args.fractions,
            "trials": args.trials,
            "in_weights_normalised": not args.no_normalise,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
