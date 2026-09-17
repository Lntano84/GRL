"""Gate 3a: which graphs turn out-degree into an anti-signal?

The claim under test
--------------------
Gate 2 established, on Congress-Twitter, that adding the highest-degree nodes *reduces*
spread once the network saturates, and that a state-conditioned analytic score avoids that
damage.  On NetHEPT the same manoeuvre stays beneficial and degree even wins at
``|S|/n = 20%``.  That is a two-point observation and cannot support a paper claim.

The mechanism proposed for the difference is density.  After in-weight normalisation
``delta(v) <= 1`` is driven by a node's **in**-degree, while a seed's value is driven by its
**out**-degree.  In a dense graph a high-degree node has so many in-neighbours that its own
neighbourhood is pushed into the overexposure band, so seeding it drives more neighbours past
``theta_tau``; in a sparse graph saturation comes from the *number* of seeds rather than from
any single node's degree, so degree should not flip sign.

This script tests that by sweeping the seed fraction across graphs spanning
``<k>`` from about 4 to about 56 and correlating the outcome with density.

Reported per (graph, fraction)
------------------------------
``rho_degree``   Spearman(out-degree, true marginal gain).  The sign of this is the claim:
                 it should start positive and go negative only in dense graphs.
``rho_delta2``   the same for the state-conditioned analytic score.
``negative_share`` fraction of candidates with a negative marginal gain.
``mean_gain``    average marginal gain, i.e. how saturated the regime is.

Every graph is loaded through :mod:`grl.data.graph_loader`, so this also exercises the
comment-prefix and NetworkX-dict-format handling added in this session.
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

from evaluate_overexposure_paper_graphs import normalise_in_weights  # noqa: E402
from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
    spearman,
)
from grl.data import graph_loader  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
)

DATA = ROOT / "data"
PAPER = DATA / "paper"

# name -> (path, directed).  Chosen to span a wide range of mean degree.
GRAPHS: dict[str, tuple[Path, bool]] = {
    "nethept": (DATA / "NetHEPT.txt", True),
    "p2p_gnutella08": (PAPER / "p2p-Gnutella08.txt", True),
    "ca_grqc": (PAPER / "ca-GrQc.txt", True),
    "wiki_vote": (PAPER / "wiki-Vote.txt", True),
    "ca_hepph": (PAPER / "ca-HepPh.txt", True),
    "facebook": (PAPER / "facebook_combined.txt", True),
    "email_eu_core": (PAPER / "email-Eu-core.txt", True),
    "congress_twitter": (PAPER / "congress" / "congress_network" / "congress.edgelist", True),
    # One of the source model's own four datasets.  Converted from the signed trust ratings by
    # scripts/data/convert_bitcoin_alpha.py.  Note the mean-degree convention difference: the
    # source reports <k> = 6.39, which is m/n for this edge set (i.e. the undirected reading);
    # as a directed graph 2m/n = 12.79.  We keep it directed because the exposure sum is
    # defined over in-edges.
    "bitcoin_alpha": (PAPER / "bitcoin-alpha.txt", True),
}


def load(name: str) -> nx.DiGraph:
    path, directed = GRAPHS[name]
    graph, _ = graph_loader._parse_graph_file(path, directed, 0.01)
    return normalise_in_weights(graph)


def structural_stats(graph: nx.DiGraph) -> dict:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    indeg = dict(graph.in_degree())
    outdeg = dict(graph.out_degree())
    nodes = list(graph.nodes())
    return {
        "n": n,
        "m": m,
        "mean_degree": 2 * m / n if n else 0.0,
        "max_out_degree": max(outdeg.values(), default=0),
        "max_in_degree": max(indeg.values(), default=0),
        "rho_in_out": spearman([indeg[v] for v in nodes], [outdeg[v] for v in nodes]),
    }


@dataclass
class Point:
    graph: str
    # Graph-level mean degree (2m/n).  Named distinctly from ``mean_gain`` so the two are
    # never confused: one describes the graph, the other describes the marginals.
    graph_mean_degree: float
    n: int
    seed_size: int
    seed_fraction: float
    mean_gain: float
    negative_share: float
    rho_degree: float
    rho_delta2: float


def probe(
    graph: nx.DiGraph,
    name: str,
    stats: dict,
    fraction: float,
    n_candidates: int,
    mc_runs: int,
    base_seed: int,
) -> Point:
    nodes = list(graph.nodes())
    n = len(nodes)
    size = int(round(fraction * n))
    rng = random.Random(base_seed + 13 * size)
    seeds = rng.sample(nodes, size) if size else []
    pool = [v for v in nodes if v not in set(seeds)]
    candidates = rng.sample(pool, min(n_candidates, len(pool)))

    configurations = [list(seeds)] + [[*seeds, c] for c in candidates]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc_runs, base_seed + 5
    )
    base = estimates[0]["mean"]
    truth = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]

    state = mean_exposure_state(graph, seeds, min(mc_runs, 15), base_seed + 5)
    degree = degree_scores(graph, candidates)
    delta2 = exposure_scores_delta(graph, candidates, state, set(seeds), hops=2)

    return Point(
        graph=name,
        graph_mean_degree=stats["mean_degree"],
        n=n,
        seed_size=len(seeds),
        seed_fraction=size / n if n else 0.0,
        mean_gain=statistics.fmean(truth),
        negative_share=sum(1 for g in truth if g < 0) / len(truth),
        rho_degree=spearman(degree, truth),
        rho_delta2=spearman(delta2, truth),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--mc-runs", type=int, default=15)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    points: list[Point] = []
    stats_by_graph: dict[str, dict] = {}

    for name in args.graphs:
        graph = load(name)
        stats = structural_stats(graph)
        stats_by_graph[name] = stats
        print(f"=== {name}: n={stats['n']} m={stats['m']} "
              f"<k>={stats['mean_degree']:.2f} "
              f"max_out={stats['max_out_degree']} max_in={stats['max_in_degree']} "
              f"rho(in,out)={stats['rho_in_out']:+.3f}", flush=True)
        for fraction in args.fractions:
            point = probe(graph, name, stats, fraction, args.candidates,
                          args.mc_runs, args.random_seed)
            points.append(point)
            print(f"    |S|/n={point.seed_fraction*100:>5.1f}% "
                  f"mean_gain={point.mean_gain:>9.3f} neg={point.negative_share*100:>5.1f}% "
                  f"rho_deg={point.rho_degree:>+7.3f} rho_d2={point.rho_delta2:>+7.3f}",
                  flush=True)
        print()

    print("=" * 96)
    print("HYPOTHESIS TEST: does density predict where out-degree flips sign?")
    print("=" * 96)
    saturated = [p for p in points if p.seed_fraction >= 0.20]
    print(f"  {'graph':<18}{'<k>':>8}{'rho_deg @ >=20%':>18}{'mean_gain':>12}{'neg%':>8}")
    for name in args.graphs:
        block = [p for p in saturated if p.graph == name]
        if not block:
            continue
        stats = stats_by_graph[name]
        print(f"  {name:<18}{stats['mean_degree']:>8.2f}"
              f"{statistics.fmean([p.rho_degree for p in block]):>+18.3f}"
              f"{statistics.fmean([p.mean_gain for p in block]):>12.3f}"
              f"{statistics.fmean([p.negative_share for p in block])*100:>8.1f}")

    if len(saturated) >= 6:
        dense = [p for p in saturated if p.graph_mean_degree >= 20]
        sparse = [p for p in saturated if p.graph_mean_degree < 20]
        print()
        if dense and sparse:
            print(f"  dense  (n={len(dense)} points, <k> >= 20): "
                  f"mean rho_degree {statistics.fmean([p.rho_degree for p in dense]):+.3f}")
            print(f"  sparse (n={len(sparse)} points, <k> <  20): "
                  f"mean rho_degree {statistics.fmean([p.rho_degree for p in sparse]):+.3f}")
            print()
            if (statistics.fmean([p.rho_degree for p in dense])
                    < statistics.fmean([p.rho_degree for p in sparse])):
                print("  -> density DOES predict the sign flip: degree degrades faster in")
                print("     dense graphs.  This supports the graph-level criterion.")
            else:
                print("  -> density does NOT separate the two groups in the expected")
                print("     direction; the criterion needs to be revised.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs,
            "fractions": args.fractions,
            "candidates": args.candidates,
            "mc_runs": args.mc_runs,
            "structural_stats": stats_by_graph,
            "points": [asdict(p) for p in points],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
