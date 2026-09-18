"""Re-derive the regime (sign-flip) table under the FIXED model and an adequately powered MC.

Why this script exists
----------------------
``paper/dasfaa2027`` Table~\\ref{tab:signflip} was produced by
``scripts/experiments/evaluate_density_degree_signflip.py`` with ``--candidates 50 --mc-runs 12``.
``docs/GATE3A_REPORT.md`` records that exact command *and* warns in the same breath that ``mc-runs=12``
is "only good for trends" and that the correlation numbers must not be cited from it.  The paper cites
them anyway, and the honesty constraints carried from the earlier analysis say a rank correlation in
this regime must never be reported below MC = 300.  So the table has two independent defects:

1. it was measured with the pre-fix state machine (a positive node could never turn negative);
2. it was measured at MC = 12, where the negative-share and the rank correlations are dominated by
   Monte-Carlo noise.

This script re-measures it under the fixed model at high MC.

Estimator
---------
``estimate_overexposure_spread_over_configs`` returns only ``mean`` and the per-trial ``std`` of the
spread, so it cannot express the uncertainty of a *paired marginal*.  This script therefore runs the
pairing itself: within each trial the base configuration and every candidate configuration share one
window draw, and the per-trial marginal ``spread(S + c) - spread(S)`` is formed before any averaging.
That gives an honest paired standard error, which is what makes the "negative marginal" claim
testable at all: the raw negative share is exactly the quantity that MC = 12 inflated.

It does not overwrite the old JSON.  The old numbers stay on disk as the record of what was measured;
this writes a new file so the two can be compared.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from grl.diffusion.overexposure import (  # noqa: E402
    run_overexposure,
    sample_threshold_windows,
)

# Import from the SAME modules the original table used, so the re-derivation differs only in the
# model fix and the Monte-Carlo budget, not in the feature definitions.
from evaluate_density_degree_signflip import (  # noqa: E402
    GRAPHS,
    load,
    structural_stats,
)
from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
)


@dataclass
class Cell:
    graph: str
    graph_mean_degree: float
    n: int
    seed_size: int
    seed_fraction: float
    mc_runs: int
    candidates: int
    mean_gain: float
    paired_se: float
    gain_sd: float
    negative_share: float
    negative_beyond_noise: float
    rho_degree: float
    rho_delta2: float
    seconds: float


def spearman(xs: list[float], ys: list[float]) -> float:
    """Average-rank Spearman, ties included.  Local so the audit does not depend on scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 1e-12 else float("nan")


def paired_marginals(
    graph,
    base: list[int],
    candidates: list[int],
    mc_runs: int,
    random_seed: int,
) -> list[list[float]]:
    """Per-trial ``spread(base + c) - spread(base)`` for every candidate, sharing windows.

    Returns one row per trial, one column per candidate, so the caller can form both the paired mean
    and the paired standard error without any independence assumption between configurations.
    """
    nodes = list(graph.nodes())
    configs = [list(base)] + [[*base, c] for c in candidates]
    per_trial: list[list[float]] = []
    for offset in range(mc_runs):
        rng = random.Random(random_seed + offset)
        windows = sample_threshold_windows(nodes, rng)
        spreads = [
            float(run_overexposure(graph, seeds, windows, rng).spread) for seeds in configs
        ]
        base_spread = spreads[0]
        per_trial.append([s - base_spread for s in spreads[1:]])
    return per_trial


def probe_cell(
    graph,
    name: str,
    stats: dict,
    fraction: float,
    n_candidates: int,
    mc_runs: int,
    base_seed: int,
) -> Cell:
    started = time.time()
    nodes = list(graph.nodes())
    n = len(nodes)
    size = int(round(fraction * n))
    rng = random.Random(base_seed + 13 * size)
    seeds = rng.sample(nodes, size) if size else []
    pool = [v for v in nodes if v not in set(seeds)]
    candidates = rng.sample(pool, min(n_candidates, len(pool)))

    trials = paired_marginals(graph, seeds, candidates, mc_runs, base_seed + 5)
    columns = list(zip(*trials))
    truth = [statistics.fmean(col) for col in columns]
    se = [
        statistics.pstdev(col) / math.sqrt(len(col)) if len(col) > 1 else 0.0
        for col in columns
    ]

    state = mean_exposure_state(graph, seeds, min(mc_runs, 15), base_seed + 5)
    degree = degree_scores(graph, candidates)
    delta2 = exposure_scores_delta(graph, candidates, state, set(seeds), hops=2)

    return Cell(
        graph=name,
        graph_mean_degree=stats["mean_degree"],
        n=n,
        seed_size=len(seeds),
        seed_fraction=size / n if n else 0.0,
        mc_runs=mc_runs,
        candidates=len(candidates),
        mean_gain=statistics.fmean(truth),
        paired_se=statistics.fmean(se),
        gain_sd=statistics.pstdev(truth),
        negative_share=sum(1 for g in truth if g < 0) / len(truth),
        # A candidate counts as negative only if its paired mean is more than two paired standard
        # errors below zero.  This is the test the raw share cannot make.
        negative_beyond_noise=sum(
            1 for g, s in zip(truth, se) if g + 2.0 * s < 0.0
        ) / len(truth),
        rho_degree=spearman(degree, truth),
        rho_delta2=spearman(delta2, truth),
        seconds=time.time() - started,
    )


def summarise(cells: list[Cell], graphs: list[str]) -> None:
    print("=" * 108)
    print("REGIME TABLE, FIXED MODEL, PAIRED ESTIMATOR")
    print("=" * 108)
    print(f"  {'graph':<20}{'<k>':>7}{'frac':>7}{'neg%':>7}{'neg>noise%':>12}"
          f"{'rho_deg':>10}{'rho_d2':>9}{'mean':>9}{'+-se':>8}")
    for name in graphs:
        for cell in [c for c in cells if c.graph == name]:
            print(f"  {name:<20}{cell.graph_mean_degree:>7.2f}"
                  f"{cell.seed_fraction*100:>6.0f}%{cell.negative_share*100:>7.1f}"
                  f"{cell.negative_beyond_noise*100:>12.1f}{cell.rho_degree:>+10.3f}"
                  f"{cell.rho_delta2:>+9.3f}{cell.mean_gain:>+9.3f}"
                  f"{cell.paired_se:>8.3f}")

    saturated = [c for c in cells if c.seed_fraction >= 0.20]
    if not saturated:
        return
    print()
    print("  Hypothesis (paper): out-degree is an anti-signal once saturated, and a large share of")
    print("  candidates has a negative marginal.")
    per_graph: dict[str, list[Cell]] = {}
    for cell in saturated:
        per_graph.setdefault(cell.graph, []).append(cell)

    print(f"    mean raw negative share          : "
          f"{statistics.fmean([c.negative_share for c in saturated])*100:5.1f}%")
    print(f"    mean negative share beyond noise : "
          f"{statistics.fmean([c.negative_beyond_noise for c in saturated])*100:5.1f}%")
    neg_cells = [c for c in saturated if c.rho_degree < 0.0]
    print(f"    rho_degree < 0 in {len(neg_cells)}/{len(saturated)} saturated cells; "
          f"mean {statistics.fmean([c.rho_degree for c in saturated]):+.3f}")
    print(f"    mean rho_delta2 over saturated cells: "
          f"{statistics.fmean([c.rho_delta2 for c in saturated]):+.3f}")

    graphs_negative = 0
    wins = 0
    for name, group in per_graph.items():
        d = statistics.fmean([c.rho_degree for c in group])
        d2 = statistics.fmean([c.rho_delta2 for c in group])
        if d < 0.0:
            graphs_negative += 1
        if d2 > d:
            wins += 1
    print(f"    graphs whose saturated mean rho_degree is negative: "
          f"{graphs_negative}/{len(per_graph)}")
    print(f"    graphs where rho_delta2 > rho_degree             : {wins}/{len(per_graph)}")
    if len(per_graph) >= 3:
        xs = [statistics.fmean([c.graph_mean_degree for c in g]) for g in per_graph.values()]
        ys = [statistics.fmean([c.rho_degree for c in g]) for g in per_graph.values()]
        print(f"    Spearman(<k>, saturated rho_degree) = {spearman(xs, ys):+.3f} "
              f"over {len(xs)} graphs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.20, 0.40])
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--mc-runs", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "signflip_fixedmodel_mc300.json")
    parser.add_argument("--resume", action="store_true",
                        help="keep cells already present in --output")
    args = parser.parse_args()

    cells: list[Cell] = []
    done: set[tuple[str, float]] = set()
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        cells = [Cell(**row) for row in payload.get("cells", [])]
        done = {(c.graph, round(c.seed_fraction, 6)) for c in cells}
        print(f"resuming: {len(cells)} cells already measured", flush=True)

    stats_by_graph: dict[str, dict] = {}
    for name in args.graphs:
        graph = load(name)
        stats = structural_stats(graph)
        stats_by_graph[name] = stats
        print(f"=== {name}: n={stats['n']} m={stats['m']} <k>={stats['mean_degree']:.2f}",
              flush=True)
        for fraction in args.fractions:
            if (name, round(fraction, 6)) in done:
                print(f"    |S|/n={fraction*100:>5.1f}%  (cached)", flush=True)
                continue
            cell = probe_cell(graph, name, stats, fraction, args.candidates,
                              args.mc_runs, args.random_seed)
            cells.append(cell)
            print(f"    |S|/n={cell.seed_fraction*100:>5.1f}% "
                  f"mean={cell.mean_gain:>+8.3f}+-{cell.paired_se:.3f} "
                  f"sd={cell.gain_sd:>6.3f} "
                  f"neg={cell.negative_share*100:>5.1f}% "
                  f"neg>noise={cell.negative_beyond_noise*100:>5.1f}% "
                  f"rho_deg={cell.rho_degree:>+7.3f} rho_d2={cell.rho_delta2:>+7.3f} "
                  f"[{cell.seconds:.0f}s]", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "script": Path(__file__).name,
                "model": "fixed state machine (positive nodes may turn negative)",
                "estimator": "paired per-trial marginal, common window draw within a trial",
                "mc_runs": args.mc_runs,
                "candidates": args.candidates,
                "fractions": args.fractions,
                "random_seed": args.random_seed,
                "structural_stats": stats_by_graph,
                "cells": [asdict(c) for c in cells],
            }, indent=2), encoding="utf-8")
        print(flush=True)

    summarise(cells, args.graphs)
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
