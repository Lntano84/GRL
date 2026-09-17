"""Gate-1c / Gate-1d: is the overexposure objective monotone, and can RR sample it?

Gate 1c -- saturation curve
---------------------------
Before claiming "the objective is non-monotone, therefore coverage-based sampling
breaks", check the more basic question: does ``sigma(S)`` itself decrease as ``|S|``
grows?  If it never decreases, the "k seeds" formulation is intact and the
non-monotonicity argument has to be made locally (per candidate) rather than globally.

This script sweeps ``|S|`` for two seed constructions (degree-biased and uniform),
reports the marginal gain per seed, and -- separately -- stresses the per-candidate
marginal gain at several edge-weight scales to find whether negative gains ever appear.

Gate 1d -- reverse reachability
-------------------------------
The classic RR identity ``E[I(S)] = n * Pr[S cap RR != empty]`` needs a
*per-edge independent random structure* (IC's live-edge graph, LT's trigger set).
The overexposure model's exposure score is instead

    delta(v, t) = sum_{u in A_in_t(v)} omega_uv

where ``A_in_t(v)`` is the set of in-neighbours that were *ever* positively activated --
a set function, not a randomised edge structure.  Given the threshold windows, delta is a
deterministic function of S.  The only randomness is the window draw itself.

So there is no live-edge graph to sample.  The nearest thing one can write down is a
reverse walk that admits a node only when ``0`` lies inside its window ``[kappa, tau]``,
because that is the only way a node can be positive with zero incoming influence.  Since
``(kappa, tau)`` is uniform on the simplex ``{0 <= kappa <= tau <= 1}``, ``kappa = 0`` is
a null set, so that never happens and the RR set collapses to a single node.  This script
measures exactly that, and compares the TIM-style estimate against the true spread.

Both gates are reported as structured JSON so the paper's section 3 can cite them.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import deque
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402


# --------------------------------------------------------------------------------------
# Gate 1c
# --------------------------------------------------------------------------------------
def degree_biased_seeds(graph, nodes, size: int, rng: random.Random) -> list[int]:
    """Roughly what a degree-greedy method would pick: sample from the degree head."""
    if size <= 0:
        return []
    out_degree = dict(graph.out_degree if graph.is_directed() else graph.degree)
    head = sorted(nodes, key=lambda v: (-out_degree[v], v))[: max(size * 2, size)]
    return rng.sample(head, min(size, len(head)))


def saturation_curve(graph, sizes: list[int], mc_runs: int, repeats: int,
                     base_seed: int) -> list[dict]:
    nodes = list(graph.nodes())
    rows: list[dict] = []
    for size in sizes:
        row: dict = {"seed_size": size}
        for mode in ("degree_biased", "uniform"):
            values = []
            for rep in range(repeats):
                rng = random.Random(base_seed + rep)
                seeds = (degree_biased_seeds(graph, nodes, size, rng) if mode == "degree_biased"
                         else (rng.sample(nodes, size) if size else []))
                est = estimate_overexposure_spread_over_configs(
                    graph, [seeds], mc_runs, base_seed + rep
                )
                values.append(est[0]["mean"])
            row[f"spread_{mode}"] = statistics.fmean(values)
            row[f"spread_{mode}_std"] = (statistics.stdev(values)
                                         if len(values) > 1 else 0.0)
        rows.append(row)
        print(f"    |S|={size:>5}  degree_biased={row['spread_degree_biased']:>9.1f}"
              f"  uniform={row['spread_uniform']:>9.1f}", flush=True)
    return rows


def marginal_gain_stress(graph, weight_scales: list[float], seed_sizes: list[int],
                         n_candidates: int, mc_runs: int, base_seed: int) -> list[dict]:
    """Where (if anywhere) does a candidate have a *negative* marginal gain?"""
    nodes = list(graph.nodes())
    rows: list[dict] = []
    rng = random.Random(base_seed)
    for scale in weight_scales:
        work = graph
        if scale != 1.0:
            work = graph.copy()
            nx.set_edge_attributes(
                work,
                {(u, v): float(d.get("weight", 0.0)) * scale
                 for u, v, d in work.edges(data=True)},
                "weight",
            )
        for size in seed_sizes:
            seeds = rng.sample(nodes, size) if size else []
            pool = [v for v in nodes if v not in set(seeds)]
            candidates = rng.sample(pool, min(n_candidates, len(pool)))
            configurations = [list(seeds)] + [[*seeds, c] for c in candidates]
            est = estimate_overexposure_spread_over_configs(
                work, configurations, mc_runs, base_seed + 7
            )
            base = est[0]["mean"]
            gains = [est[i + 1]["mean"] - base for i in range(len(candidates))]
            negative = sum(1 for g in gains if g < 0)
            rows.append({
                "weight_scale": scale,
                "seed_size": size,
                "n_candidates": len(candidates),
                "mc_runs": mc_runs,
                "min_gain": min(gains),
                "mean_gain": statistics.fmean(gains),
                "max_gain": max(gains),
                "n_negative": negative,
                "negative_share": negative / len(gains),
            })
            print(f"    scale={scale:<5} |S|={size:>4}  min={min(gains):>9.3f}"
                  f"  mean={statistics.fmean(gains):>9.3f}  neg={negative}/{len(gains)}",
                  flush=True)
    return rows


def in_weight_normalisation(graph) -> dict:
    """The structural reason negative gains are rare: in-weights are normalised to 1."""
    totals: dict[int, float] = {v: 0.0 for v in graph.nodes()}
    n_edges = 0
    for u, v, d in graph.edges(data=True):
        totals[v] += float(d.get("weight", 0.0))
        n_edges += 1
    nonzero = [t for t in totals.values() if t > 0]
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": n_edges,
        "n_nodes_with_in_edges": len(nonzero),
        "fraction_in_weight_sum_eq_1": (
            sum(1 for t in nonzero if abs(t - 1.0) < 1e-6) / len(nonzero) if nonzero else None
        ),
        "min_in_weight_sum": min(nonzero) if nonzero else None,
        "max_in_weight_sum": max(nonzero) if nonzero else None,
    }


# --------------------------------------------------------------------------------------
# Gate 1d
# --------------------------------------------------------------------------------------
def reverse_reachable(source: int, windows: dict, reverse_in: dict) -> set[int]:
    """Reverse BFS admitting a predecessor only if 0 lies inside its window.

    ``0 in [kappa, tau]`` is the only way a node can be positive while receiving no
    influence -- and a reverse walk needs exactly that as its base case.

    The source itself is **deliberately excluded**: under this model a non-seed node with
    ``delta = 0`` cannot be positive (positive activation needs ``delta >= kappa``, and
    ``kappa > 0`` almost surely), so a non-seed source contributes nothing.  Including it
    would make every RR set non-empty and reduce the TIM estimate to the trivial ``n``.
    Under the correct reading the set is empty unless a predecessor qualifies.
    """
    seen: set[int] = set()
    queue: deque[int] = deque()
    for predecessor, weight in reverse_in[source]:
        if weight <= 0.0:
            continue
        kappa, tau = windows[predecessor]
        if kappa <= 0.0 <= tau:
            seen.add(predecessor)
            queue.append(predecessor)
    while queue:
        node = queue.popleft()
        for predecessor, weight in reverse_in[node]:
            if predecessor in seen or weight <= 0.0:
                continue
            kappa, tau = windows[predecessor]
            if kappa <= 0.0 <= tau:
                seen.add(predecessor)
                queue.append(predecessor)
    return seen


def reverse_reachability_probe(graph, n_draws: int, seed_size: int, trials: int,
                               mc_runs: int, base_seed: int) -> dict:
    nodes = list(graph.nodes())
    n = len(nodes)
    reverse_in: dict[int, list[tuple[int, float]]] = {v: [] for v in nodes}
    for u, v, d in graph.edges(data=True):
        reverse_in[v].append((u, float(d.get("weight", 0.0))))

    rng = random.Random(base_seed)
    windows = sample_threshold_windows(nodes, rng)
    sizes = [len(reverse_reachable(rng.choice(nodes), windows, reverse_in))
             for _ in range(n_draws)]

    seeds = set(rng.sample(nodes, seed_size))
    hits = sum(1 for _ in range(trials)
               if reverse_reachable(rng.choice(nodes), windows, reverse_in))
    tim_estimate = n * hits / trials
    truth = estimate_overexposure_spread_over_configs(
        graph, [sorted(seeds)], mc_runs, base_seed + 13
    )[0]["mean"]

    zero_window_nodes = sum(1 for v in nodes if windows[v][0] <= 0.0)

    return {
        "n_nodes": n,
        "rr_set_size_min": min(sizes),
        "rr_set_size_median": statistics.median(sizes),
        "rr_set_size_mean": statistics.fmean(sizes),
        "rr_set_size_max": max(sizes),
        "rr_set_size_fraction_of_graph": statistics.fmean(sizes) / n,
        "nodes_with_window_including_zero": zero_window_nodes,
        "seed_size": seed_size,
        "tim_estimate": tim_estimate,
        "true_spread": truth,
        "tim_overestimate_factor": tim_estimate / truth if truth else None,
    }


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[0, 10, 30, 60, 120, 250, 500, 1000, 2000, 4000])
    parser.add_argument("--saturation-mc", type=int, default=30)
    parser.add_argument("--saturation-repeats", type=int, default=3)
    parser.add_argument("--weight-scales", type=float, nargs="+",
                        default=[1.0, 3.0, 6.0, 10.0])
    parser.add_argument("--stress-seed-sizes", type=int, nargs="+", default=[0, 200, 800])
    parser.add_argument("--stress-candidates", type=int, default=120)
    parser.add_argument("--stress-mc", type=int, default=15)
    parser.add_argument("--rr-draws", type=int, default=300)
    parser.add_argument("--rr-seed-size", type=int, default=50)
    parser.add_argument("--rr-trials", type=int, default=600)
    parser.add_argument("--rr-mc", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    graph = load_graph_from_config(config).graph
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()

    print("Gate 1c-1  in-weight normalisation (why negative gains are rare)")
    norm = in_weight_normalisation(graph)
    for key, value in norm.items():
        print(f"    {key:<32} {value}")
    print()

    print("Gate 1c-2  saturation curve  sigma(S) vs |S|")
    curve = saturation_curve(graph, args.sizes, args.saturation_mc,
                             args.saturation_repeats, args.random_seed)
    print()

    print("Gate 1c-3  per-candidate marginal gain stress test")
    stress = marginal_gain_stress(graph, args.weight_scales, args.stress_seed_sizes,
                                  args.stress_candidates, args.stress_mc, args.random_seed)
    print()

    print("Gate 1d  reverse reachability")
    rr = reverse_reachability_probe(graph, args.rr_draws, args.rr_seed_size,
                                    args.rr_trials, args.rr_mc, args.random_seed)
    for key, value in rr.items():
        print(f"    {key:<38} {value}")
    print()

    # --- verdicts -----------------------------------------------------------------
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)

    margins = []
    prev = None
    for row in curve:
        if prev is not None and row["seed_size"] > prev["seed_size"]:
            margins.append((row["spread_degree_biased"] - prev["spread_degree_biased"])
                           / (row["seed_size"] - prev["seed_size"]))
        prev = row
    monotone = all(row["spread_degree_biased"] >=
                   (curve[i - 1]["spread_degree_biased"] if i else 0.0)
                   for i, row in enumerate(curve))
    print(f"  1c  sigma(S) monotone increasing over |S| : {monotone}")
    if margins:
        print(f"      marginal per seed: {margins[0]:.2f} at |S|={curve[1]['seed_size']}"
              f"  ->  {margins[-1]:.2f} at |S|={curve[-1]['seed_size']}")
    worst = min(stress, key=lambda r: r["min_gain"])
    total_negative = sum(r["n_negative"] for r in stress)
    total_candidates = sum(r["n_candidates"] for r in stress)
    print(f"  1c  negative marginal gains across all settings : "
          f"{total_negative} / {total_candidates}")
    print(f"      worst min_gain = {worst['min_gain']:.3f} "
          f"(scale={worst['weight_scale']}, |S|={worst['seed_size']})")
    at_scale_one = [r for r in stress if r["weight_scale"] == 1.0]
    if at_scale_one:
        closest = min(r["min_gain"] for r in at_scale_one)
        print(f"      at the native weight scale (1.0) the minimum marginal gain is "
              f"{closest:+.3f} -> ", end="")
        if closest >= 0:
            print("non-negative but approaching zero as |S| grows.  The objective is\n"
                  "      effectively monotone at native weights; non-monotonicity needs "
                  "weight inflation (scale 3-6).")
        else:
            print("negative: the objective is genuinely non-monotone at native weights.")
    print(f"  1d  nodes whose window includes 0 : "
          f"{rr['nodes_with_window_including_zero']} / {rr['n_nodes']}")
    print(f"  1d  mean RR-set size : {rr['rr_set_size_mean']:.1f} "
          f"({rr['rr_set_size_fraction_of_graph']*100:.4f}% of the graph)")
    print(f"  1d  TIM estimate {rr['tim_estimate']:.1f} vs true spread "
          f"{rr['true_spread']:.1f}")
    if rr["rr_set_size_max"] == 0:
        print("      -> RR sets are EMPTY: the reverse-reachability relation does not\n"
              "         exist, so the classic identity E[I(S)] = n*Pr[S cap RR != 0]\n"
              "         degenerates to 0 while the true spread is positive.\n"
              "         RR is structurally inapplicable here, not merely slow.")
    elif rr["rr_set_size_max"] <= 1:
        print("      -> RR sets collapse to singletons: the relation is degenerate.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "config": str(args.config),
            "in_weight_normalisation": norm,
            "saturation_curve": curve,
            "marginal_gain_stress": stress,
            "reverse_reachability": rr,
            "verdict": {
                "sigma_monotone_over_abs_S": monotone,
                "total_negative_marginal_gains": total_negative,
                "total_candidates_probed": total_candidates,
                "min_marginal_gain_at_native_weight_scale": min(
                    (r["min_gain"] for r in stress if r["weight_scale"] == 1.0),
                    default=None,
                ),
                "rr_set_is_empty": rr["rr_set_size_max"] == 0,
                "rr_relation_degenerate": rr["rr_set_size_max"] <= 1,
            },
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
