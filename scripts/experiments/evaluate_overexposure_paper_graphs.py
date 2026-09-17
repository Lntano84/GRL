"""Gate-2 benchmark on the Inf. Sci. 2026 paper's own graphs.

Why NetHEPT misled us
---------------------
Every earlier Gate in this project ran on NetHEPT (15,233 nodes).  Over eight separate
diagnostics the objective looked essentially monotone there: per-candidate marginal gains
stayed non-negative even with weight inflation, and ``sigma(S)`` rose monotonically up to
``|S| = 4000``.

That was a scale artefact.  Saturation is governed by the *seed fraction* ``|S| / n``, not
by ``|S|``.  NetHEPT at ``|S| = 200`` is 1.3% seeded, nowhere near saturation.  On the
paper's own graphs -- Congress-Twitter is 475 nodes, Wiki-Vote 7,115 -- the same absolute
budget covers 42% and 2.8% of the network, and there the objective becomes genuinely
non-monotone: Congress-Twitter at ``|S| = 200`` has a *negative* mean marginal gain
(-0.094) with 55/120 candidates negative.

Weight regime
-------------
The model needs in-weights normalised to sum to 1 per node.  This is not an artefact: the
positive-activation window requires ``delta`` to reach ``[theta_kappa, theta_tau]``, so
weights must be large enough that a realistic number of active in-neighbours lands inside
the window.  Uniform ``p = 0.01`` on NetHEPT caps ``delta`` at ``0.01 * 60 = 0.6`` in the
best case and in practice at ~0.15, so the cascade barely fires and the objective is
trivially monotone (mean marginal gain 1.046, i.e. the seed itself plus almost nothing).
The paper's own datasets already carry such normalised weights.

What this script measures
-------------------------
For each configured graph, with in-weights normalised:

1. the **saturation curve** ``sigma(S)`` against the seed *fraction* ``|S| / n``;
2. the **non-monotonicity rate** (fraction of candidates with negative marginal gain);
3. the **headroom above the analytic baseline**, i.e. how much of the gap between
   ``degree``, the training-free ``delta2`` scorer, and Monte-Carlo greedy remains.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
    gain_capture,
    mean_exposure_state,
    spearman,
)
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    run_overexposure,
    sample_threshold_windows,
)

PAPER_DATA = ROOT / "data" / "paper"


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def load_nethept() -> nx.DiGraph:
    graph = nx.DiGraph()
    for line in (ROOT / "data" / "NetHEPT.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        graph.add_edge(int(parts[0]), int(parts[1]), weight=float(parts[2]))
    return graph


def load_congress() -> nx.DiGraph:
    """Congress-Twitter ships as a NetworkX edge list: ``u v {'weight': w}``.

    ``grl.data.graph_loader._parse_graph_file`` splits on whitespace and casts field 3 to
    float, which raises ``ValueError`` on this format, so it is parsed here.
    """
    graph = nx.DiGraph()
    path = PAPER_DATA / "congress" / "congress_network" / "congress.edgelist"
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        match = re.search(r"'weight':\s*([0-9.eE+-]+)", line)
        weight = float(match.group(1)) if match else 0.01
        graph.add_edge(int(parts[0]), int(parts[1]), weight=weight)
    return graph


def load_wikivote() -> nx.DiGraph:
    graph = nx.DiGraph()
    for line in (PAPER_DATA / "wiki-Vote.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        graph.add_edge(parts[0], parts[1], weight=0.01)
    return graph


LOADERS = {
    "nethept": load_nethept,
    "congress_twitter": load_congress,
    "wiki_vote": load_wikivote,
}


def normalise_in_weights(graph: nx.DiGraph) -> nx.DiGraph:
    """Scale each node's in-edge weights to sum to 1 (the regime the model requires)."""
    totals: dict = {v: 0.0 for v in graph.nodes()}
    for _, v, data in graph.edges(data=True):
        totals[v] += float(data.get("weight", 0.0))
    for _, v, data in graph.edges(data=True):
        total = totals[v]
        data["weight"] = float(data.get("weight", 0.0)) / total if total > 0 else 0.0
    return graph


# --------------------------------------------------------------------------------------
@dataclass
class CurvePoint:
    graph: str
    seed_size: int
    seed_fraction: float
    spread: float
    marginal_per_seed: float
    min_gain: float
    mean_gain: float
    negative_share: float
    spearman_degree: float
    spearman_delta2: float
    capture_degree: float
    capture_delta2: float


def probe_point(
    graph: nx.DiGraph,
    label: str,
    seed_size: int,
    n_candidates: int,
    mc_runs: int,
    top_k: int,
    base_seed: int,
) -> CurvePoint:
    nodes = list(graph.nodes())
    n = len(nodes)
    rng = random.Random(base_seed)
    seeds = rng.sample(nodes, min(seed_size, n)) if seed_size else []
    pool = [v for v in nodes if v not in set(seeds)]
    candidates = rng.sample(pool, min(n_candidates, len(pool)))

    configurations = [list(seeds)] + [[*seeds, c] for c in candidates]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc_runs, base_seed + 5
    )
    base = estimates[0]["mean"]
    truth = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]
    negative = sum(1 for g in truth if g < 0)

    state = mean_exposure_state(graph, seeds, min(mc_runs, 20), base_seed + 5)
    degree = degree_scores(graph, candidates)
    delta2 = exposure_scores_delta(graph, candidates, state, set(seeds), hops=2)
    k = min(top_k, len(candidates))

    return CurvePoint(
        graph=label,
        seed_size=len(seeds),
        seed_fraction=len(seeds) / n,
        spread=base,
        marginal_per_seed=0.0,  # filled by the caller from consecutive points
        min_gain=min(truth),
        mean_gain=statistics.fmean(truth),
        negative_share=negative / len(truth),
        spearman_degree=spearman(degree, truth),
        spearman_delta2=spearman(delta2, truth),
        capture_degree=gain_capture(truth, degree, k),
        capture_delta2=gain_capture(truth, delta2, k),
    )


def run_graph(
    label: str,
    graph: nx.DiGraph,
    fractions: list[float],
    n_candidates: int,
    mc_runs: int,
    top_k: int,
    base_seed: int,
) -> list[CurvePoint]:
    n = graph.number_of_nodes()
    sizes = sorted({int(round(f * n)) for f in fractions})
    points: list[CurvePoint] = []
    previous_spread = None
    previous_size = None
    for size in sizes:
        point = probe_point(graph, label, size, n_candidates, mc_runs, top_k, base_seed)
        if previous_spread is not None and size > previous_size:
            point.marginal_per_seed = (point.spread - previous_spread) / (size - previous_size)
        previous_spread, previous_size = point.spread, size
        points.append(point)
        print(f"  |S|={point.seed_size:>5} ({point.seed_fraction*100:>5.1f}%)  "
              f"spread={point.spread:>9.2f}  marg/seed={point.marginal_per_seed:>8.3f}  "
              f"min_gain={point.min_gain:>8.3f}  neg={point.negative_share*100:>5.1f}%  "
              f"rho_deg={point.spearman_degree:>7.3f}  rho_d2={point.spearman_delta2:>7.3f}",
              flush=True)
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["congress_twitter", "wiki_vote",
                                                       "nethept"],
                        choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--mc-runs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-normalise", action="store_true",
                        help="keep the graph's own weights instead of normalising in-weights")
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    all_points: list[CurvePoint] = []
    for label in args.graphs:
        graph = LOADERS[label]()
        if not args.no_normalise:
            graph = normalise_in_weights(graph)
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        print(f"=== {label}: n={n} m={m} <k>={2*m/n:.2f}"
              f"{' (in-weights normalised)' if not args.no_normalise else ''}")
        all_points.extend(run_graph(
            label, graph, args.fractions, args.candidates, args.mc_runs,
            args.top_k, args.random_seed,
        ))
        print()

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    for label in args.graphs:
        rows = [p for p in all_points if p.graph == label]
        if not rows:
            continue
        worst = max(rows, key=lambda p: p.negative_share)
        rising = all(rows[i].spread >= rows[i - 1].spread for i in range(1, len(rows)))
        d_rho = statistics.fmean([p.spearman_delta2 - p.spearman_degree for p in rows])
        d_cap = statistics.fmean([p.capture_delta2 - p.capture_degree for p in rows])
        print(f"  {label}")
        print(f"    sigma(S) monotone over the sampled fractions : {rising}")
        print(f"    worst non-monotonicity: {worst.negative_share*100:.1f}% of candidates "
              f"negative at |S|={worst.seed_size} ({worst.seed_fraction*100:.1f}%), "
              f"min_gain={worst.min_gain:+.3f}")
        print(f"    delta2 - degree: spearman {d_rho:+.4f}   gain_capture {d_cap:+.4f}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs,
            "fractions": args.fractions,
            "candidates": args.candidates,
            "mc_runs": args.mc_runs,
            "in_weights_normalised": not args.no_normalise,
            "points": [asdict(p) for p in all_points],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
