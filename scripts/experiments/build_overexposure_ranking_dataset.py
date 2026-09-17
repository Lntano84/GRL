"""Gate-3a: build a state-conditioned ranking dataset with trustworthy labels.

Why the labels need care
------------------------
The saturated regime is where the interesting behaviour lives, and it is also where the true
marginal gain collapses toward zero.  Measured on Congress-Twitter (30 candidates, paired
labels, shared windows per trial):

    |S|/n   MC    mean_gain   sd(gain)   mean_se   SNR = sd/se   rho_degree   rho_delta2
    10%     60      +0.084      0.514      0.525       0.98        -0.143       +0.010
    10%     250     -0.245      0.338      0.249       1.36        -0.558       +0.588
    10%     800     -0.179      0.349      0.132       2.65        -0.527       +0.613
    20%     60      -0.018      0.521      0.383       1.36        -0.299       +0.469
    20%     800     -0.106      0.268      0.096       2.79        -0.580       +0.828

At MC=60 the label noise is the same size as the spread of the labels themselves, so the
measured rank correlation is dominated by noise -- an earlier run reported ``rho_delta2`` of
about 0.01 at ``|S|/n = 10%`` purely because of this.  Labels here therefore default to
MC=400, and the script reports the SNR of the produced labels so the user can see whether a
given configuration is measurable at all.

Features
--------
The analytic baseline ``exposure_scores_delta`` only looks at each out-neighbour's exposure
``delta`` and the change a candidate would cause.  It never looks at the *thresholds*.  The
overexposure band is ``[theta_kappa, theta_tau]``, and whether a neighbour is close to
falling out of it is exactly the information a learned ranker could exploit and the analytic
score cannot.  Features therefore include, per candidate:

* ``delta`` at the candidate itself;
* over its out-neighbours: mean/std/min/max ``delta``, and the fraction already past
  ``theta_tau`` (overexposed), still below ``theta_kappa`` (untouched), or inside the band;
* the mean margin to ``theta_tau`` and to ``theta_kappa``, in ``delta`` units;
* the analytic score itself, so a model can learn a *correction* to it rather than
  rediscover it;
* structural features: out-degree, in-degree, out-weight sum.

Label
-----
Paired marginal gain ``sigma(S + {v}) - sigma(S)`` under the realised threshold windows.
"""

from __future__ import annotations

import argparse
import json
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

from evaluate_overexposure_paper_graphs import LOADERS, normalise_in_weights  # noqa: E402
from evaluate_overexposure_pool_ranking import exposure_scores_delta  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    run_overexposure,
    sample_threshold_windows,
)

FEATURE_NAMES = [
    "delta_self",
    "out_degree",
    "in_degree",
    "out_weight_sum",
    "nb_delta_mean",
    "nb_delta_std",
    "nb_delta_min",
    "nb_delta_max",
    "frac_overexposed",
    "frac_untouched",
    "frac_in_band",
    "mean_margin_to_tau",
    "mean_margin_to_kappa",
    "analytic_score",
]


def neighbour_lists(graph: nx.DiGraph) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {v: [] for v in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        out[u].append((v, float(data.get("weight", 0.0))))
    return out


def candidate_features(
    graph: nx.DiGraph,
    out_edges: dict[int, list[tuple[int, float]]],
    candidate: int,
    delta: dict[int, float],
    windows: dict[int, tuple[float, float]],
    analytic: float,
) -> list[float]:
    weights_in = graph.in_degree(candidate)
    edges = out_edges.get(candidate, [])
    neighbour_deltas: list[float] = []
    overexposed = untouched = in_band = 0
    margins_tau: list[float] = []
    margins_kappa: list[float] = []

    for target, weight in edges:
        if target == candidate:
            continue
        value = delta.get(target, 0.0)
        kappa, tau = windows[target]
        neighbour_deltas.append(value)
        margins_tau.append(tau - value)
        margins_kappa.append(value - kappa)
        if value > tau:
            overexposed += 1
        elif value < kappa:
            untouched += 1
        else:
            in_band += 1

    count = max(1, len(neighbour_deltas))
    if neighbour_deltas:
        mean_delta = statistics.fmean(neighbour_deltas)
        std_delta = statistics.pstdev(neighbour_deltas)
        min_delta, max_delta = min(neighbour_deltas), max(neighbour_deltas)
    else:
        mean_delta = std_delta = min_delta = max_delta = 0.0

    return [
        delta.get(candidate, 0.0),
        float(graph.out_degree(candidate)),
        float(weights_in),
        float(sum(w for _, w in edges)),
        mean_delta,
        std_delta,
        min_delta,
        max_delta,
        overexposed / count,
        untouched / count,
        in_band / count,
        statistics.fmean(margins_tau) if margins_tau else 0.0,
        statistics.fmean(margins_kappa) if margins_kappa else 0.0,
        analytic,
    ]


def paired_labels(
    graph: nx.DiGraph,
    nodes: list[int],
    seeds: list[int],
    candidates: list[int],
    trials: int,
    base_seed: int,
) -> tuple[list[float], list[float]]:
    """Mean and standard error of ``sigma(S + {v}) - sigma(S)`` for every candidate.

    All candidates share the threshold windows within a trial, so the differences are not
    contaminated by window-to-window variation.
    """
    acc: dict[int, list[float]] = {c: [] for c in candidates}
    for t in range(trials):
        r = random.Random(base_seed + 7 + t)
        windows = sample_threshold_windows(nodes, r)
        base = float(run_overexposure(graph, seeds, windows, r).spread)
        for c in candidates:
            value = float(run_overexposure(graph, seeds + [c], windows, r).spread)
            acc[c].append(value - base)
    means = [statistics.fmean(acc[c]) for c in candidates]
    errors = [
        statistics.stdev(acc[c]) / len(acc[c]) ** 0.5 if len(acc[c]) > 1 else 0.0
        for c in candidates
    ]
    return means, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--contexts-per-fraction", type=int, default=3,
                        help="independent (seed set, candidate pool) draws per fraction")
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--trials", type=int, default=400,
                        help="paired MC trials per label; see the module docstring on SNR")
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = LOADERS[args.graph]()
    if not args.no_normalise:
        graph = normalise_in_weights(graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    out_edges = neighbour_lists(graph)

    print(f"graph={args.graph} n={n} candidates={args.candidates} trials={args.trials} "
          f"contexts/fraction={args.contexts_per_fraction}")
    print()

    records: list[dict] = []
    for fraction in args.fractions:
        size = int(round(fraction * n))
        for context in range(args.contexts_per_fraction):
            base_seed = args.random_seed + 13 * size + 7919 * context
            rng = random.Random(base_seed)
            seeds = rng.sample(nodes, size) if size else []
            pool = [v for v in nodes if v not in set(seeds)]
            candidates = rng.sample(pool, min(args.candidates, len(pool)))

            # The state a policy observes: one realised draw.
            observe_rng = random.Random(base_seed + 202)
            windows = sample_threshold_windows(nodes, observe_rng)
            delta = dict(run_overexposure(graph, seeds, windows, observe_rng).delta)
            analytic = exposure_scores_delta(graph, candidates, delta, set(seeds), hops=2)

            labels, errors = paired_labels(graph, nodes, seeds, candidates,
                                           args.trials, base_seed)
            snr = (statistics.stdev(labels) / statistics.fmean(errors)
                   if statistics.fmean(errors) > 0 else float("nan"))

            for index, candidate in enumerate(candidates):
                records.append({
                    "seed_fraction": size / n,
                    "seed_size": size,
                    "context": context,
                    "candidate": int(candidate),
                    "features": candidate_features(graph, out_edges, candidate, delta,
                                                   windows, analytic[index]),
                    "label": labels[index],
                    "label_se": errors[index],
                })

            print(f"  |S|/n={size/n*100:>5.1f}% ctx={context} "
                  f"mean_label={statistics.fmean(labels):>+9.3f} "
                  f"sd_label={statistics.stdev(labels):>7.3f} "
                  f"mean_se={statistics.fmean(errors):>7.3f} "
                  f"SNR={snr:>6.2f}", flush=True)
        print()

    payload = {
        "graph": args.graph,
        "n": n,
        "fractions": args.fractions,
        "contexts_per_fraction": args.contexts_per_fraction,
        "candidates": args.candidates,
        "trials": args.trials,
        "in_weights_normalised": not args.no_normalise,
        "feature_names": FEATURE_NAMES,
        "records": records,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}  ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
