"""Gate-1a'' diagnostic: conditional ranking *inside a realistic retriever pool*.

Why this replaces Gate 1a'
--------------------------
Gate 1a' drew its candidate pool with ``rng.sample(pool, 600)`` -- a uniform sample of
the whole node set.  NetHEPT is heavy tailed, so such a pool is almost entirely
low-degree nodes and the top of the true-gain ranking is populated by the few hubs that
happened to be drawn.  The degree score therefore wins on *pool composition* -- it knows
"hubs are valuable", a global regularity -- rather than on any ability to rank candidates
under a given exposure state.

That is the wrong measurement for this project.  The deployed architecture is

    Retriever(degree) -> top-M -> Reranker(state)

so the reranker never sees a uniform sample of the graph.  It sees a pool that is already
degree-selected, in which out-degree varies over a narrow band and the global degree
signal has been spent.  This script measures the reranker in exactly that regime, with the
uniform pool kept as a control so the artifact is visible rather than hidden.

Scorers
-------
``degree``          out-degree.  The incumbent, state-blind.
``expo_bfs_v1``     the Gate-1a' hand-crafted score, kept for continuity.  It multiplies a
                    shortest-path reach probability by a probability increment, which is
                    not an expectation; reported only to show the fix matters.
``expo_delta2``     corrected two-hop expectation:

                        dE_w(v) = w(v, u) * (1 - E_S(u))
                        score(w) = sum_v [ g(E_S(v) + dE_w(v)) - g(E_S(v)) ]

                    where g(e) = 2e(1-e) is the positive-activation probability of a node
                    at exposure e.  This is the exact one-step expected change in the
                    number of *positive* out-neighbours, computed on top of the observed
                    state E_S.  Two hops are included via a second Bellman pass.
``random``          uniform noise; the floor.

Metrics
-------
``spearman``        rank correlation with the true marginal gain.
``recall_at_10``    fraction of the true top-10 candidates the scorer also ranks top-10.
``gain_capture``    (gain of the scorer's top-k) / (gain of the true top-k).  The metric
                    that actually decides whether the downstream greedy step is hurt.
``negative_gain_share``  how often the pool contains a candidate with Delta(v|S) < 0, i.e.
                    evidence that the objective is genuinely non-monotone here.

All context-level metrics are reported individually, and paired mean differences come with
a bootstrap 95% CI, because the whole point is to decide a sign with a small number of
contexts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    positive_activation_probability,
    run_overexposure,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402

MODEL_OVEREXPOSURE = "overexposure"
POOL_DEGREE = "degree_pool"
POOL_UNIFORM = "uniform_pool"

SCORERS = ("degree", "expo_bfs_v1", "expo_delta2", "random")


# --------------------------------------------------------------------------------------
# ranking helpers
# --------------------------------------------------------------------------------------
def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho.  Returns nan when either side is degenerate (all ties)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _top_k_indices(values: list[float], k: int) -> list[int]:
    k = min(k, len(values))
    return sorted(range(len(values)), key=lambda i: -values[i])[:k]


def recall_at_k(truth: list[float], scores: list[float], k: int) -> float:
    if k <= 0 or not truth:
        return float("nan")
    k = min(k, len(truth))
    a = set(_top_k_indices(truth, k))
    b = set(_top_k_indices(scores, k))
    return len(a & b) / k


def gain_capture(truth: list[float], scores: list[float], k: int) -> float:
    """Sum of true gains over the scorer's top-k, divided by that of the true top-k."""
    if not truth:
        return float("nan")
    k = min(k, len(truth))
    best = sum(truth[i] for i in _top_k_indices(truth, k))
    got = sum(truth[i] for i in _top_k_indices(scores, k))
    if abs(best) < 1e-12:
        return float("nan")
    return got / best


def bootstrap_ci(diffs: list[float], iterations: int = 4000, alpha: float = 0.05,
                 seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    clean = [d for d in diffs if d == d and d not in (float("inf"), float("-inf"))]
    if len(clean) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(iterations):
        means.append(statistics.fmean(clean[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


# --------------------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------------------
def degree_scores(graph: nx.Graph | nx.DiGraph, candidates: list[int]) -> list[float]:
    source = graph.out_degree if graph.is_directed() else graph.degree
    return [float(source[v]) for v in candidates]


def random_scores(candidates: list[int], seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in candidates]


def _out_edges(graph: nx.Graph | nx.DiGraph, node: int):
    directed = graph.is_directed()
    for u, v, data in graph.out_edges(node, data=True) if directed else graph.edges(node, data=True):
        target = v if directed else (v if u == node else u)
        yield target, float(data.get("weight", 0.0))


def exposure_scores_bfs_v1(
    graph: nx.Graph | nx.DiGraph,
    candidates: list[int],
    delta: dict[int, float],
    seeds: set[int],
    depth: int = 4,
) -> list[float]:
    """The Gate-1a' scorer, kept verbatim for continuity (mathematically loose)."""
    scores: list[float] = []
    for candidate in candidates:
        if candidate in seeds:
            scores.append(float("-inf"))
            continue
        total = 0.0
        frontier: list[tuple[int, float]] = [(candidate, 1.0)]
        seen: dict[int, float] = {candidate: 1.0}
        level = 1
        while frontier and level <= depth:
            next_frontier: list[tuple[int, float]] = []
            for node, reach in frontier:
                for target, weight in _out_edges(graph, node):
                    if target == node or target in seeds:
                        continue
                    contribution = reach * weight
                    if contribution <= 0.0:
                        continue
                    current = delta.get(target, 0.0)
                    before = positive_activation_probability(current)
                    after = positive_activation_probability(min(1.0, current + contribution))
                    total += contribution * (after - before)
                    previous = seen.get(target, 0.0)
                    if contribution > previous and level < depth:
                        seen[target] = contribution
                        next_frontier.append((target, contribution))
            frontier = next_frontier
            level += 1
        scores.append(total)
    return scores


def exposure_scores_delta(
    graph: nx.Graph | nx.DiGraph,
    candidates: list[int],
    delta: dict[int, float],
    seeds: set[int],
    hops: int = 2,
) -> list[float]:
    """Corrected expectation-based score.

    Propagates the *increase in activation probability* of each node caused by seeding the
    candidate, then scores the candidate by the resulting increase in the expected number
    of positive out-neighbours.

        dE(v)    <- dE(parent) * w(parent, v) * (1 - E_S(v))     (Bellman over hops)
        score(w) = sum_{v in N+(w)} [ g(E_S(v) + dE(v)) - g(E_S(v)) ]
    """
    scores: list[float] = []
    for candidate in candidates:
        if candidate in seeds:
            scores.append(float("-inf"))
            continue

        d_e: dict[int, float] = {}
        for target, weight in _out_edges(graph, candidate):
            if target == candidate or target in seeds or weight <= 0.0:
                continue
            gain = weight * (1.0 - delta.get(target, 0.0))
            if gain > d_e.get(target, 0.0):
                d_e[target] = gain

        if hops >= 2:
            layer = sorted(d_e.items(), key=lambda kv: -kv[1])
            for node, node_delta in layer:
                if node_delta <= 1e-9:
                    continue
                for target, weight in _out_edges(graph, node):
                    if target == candidate or target in seeds or weight <= 0.0:
                        continue
                    propagated = node_delta * weight * (1.0 - delta.get(target, 0.0))
                    if propagated > d_e.get(target, 0.0):
                        d_e[target] = propagated

        total = 0.0
        for target, change in d_e.items():
            current = delta.get(target, 0.0)
            before = positive_activation_probability(current)
            after = positive_activation_probability(min(1.0, current + change))
            total += after - before
        scores.append(total)
    return scores


# --------------------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------------------
@dataclass
class ContextResult:
    pool: str
    budget: int
    context: int
    seed_size: int
    n_pool: int
    degree_min: int
    degree_max: int
    negative_gain_share: float
    mean_gain: float
    spearman_degree: float
    spearman_bfs_v1: float
    spearman_delta2: float
    spearman_random: float
    recall_degree: float
    recall_bfs_v1: float
    recall_delta2: float
    recall_random: float
    capture_degree: float
    capture_bfs_v1: float
    capture_delta2: float
    capture_random: float


def build_pools(
    graph: nx.Graph | nx.DiGraph,
    nodes: list[int],
    seeds: set[int],
    budget: int,
    pool_size: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    """Degree-selected pool (the deployed regime) plus a uniform pool (the control)."""
    deg = dict(graph.out_degree if graph.is_directed() else graph.degree)
    eligible = [v for v in nodes if v not in seeds]
    by_degree = sorted(eligible, key=lambda v: (-deg[v], v))
    # A retriever wide enough that the reranker has a real choice but degree is compressed.
    width = max(pool_size * 5, budget * 40)
    head = by_degree[: min(len(by_degree), width)]
    degree_pool = rng.sample(head, min(pool_size, len(head)))
    uniform_pool = rng.sample(eligible, min(pool_size, len(eligible)))
    return {POOL_DEGREE: degree_pool, POOL_UNIFORM: uniform_pool}


def mean_exposure_state(
    graph: nx.Graph | nx.DiGraph,
    seeds: list[int],
    trials: int,
    seed: int,
) -> dict[int, float]:
    """Mean per-node exposure delta under S, over ``trials`` threshold draws."""
    nodes = list(graph.nodes())
    accum = {node: 0.0 for node in nodes}
    for offset in range(trials):
        rng = random.Random(seed + offset)
        windows = sample_threshold_windows(nodes, rng)
        run = run_overexposure(graph, list(seeds), windows, rng)
        for node, value in run.delta.items():
            accum[node] += value
    return {node: value / trials for node, value in accum.items()}


def evaluate_context(
    graph: nx.Graph | nx.DiGraph,
    pool_name: str,
    candidates: list[int],
    seeds: list[int],
    budget: int,
    context: int,
    mc_runs: int,
    random_seed: int,
) -> ContextResult:
    configurations = [list(seeds)] + [[*seeds, c] for c in candidates]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc_runs, random_seed
    )
    base = estimates[0]["mean"]
    truth = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]

    delta = mean_exposure_state(graph, seeds, min(mc_runs, 20), random_seed)
    seed_set = set(seeds)

    scores = {
        "degree": degree_scores(graph, candidates),
        "expo_bfs_v1": exposure_scores_bfs_v1(graph, candidates, delta, seed_set),
        "expo_delta2": exposure_scores_delta(graph, candidates, delta, seed_set, hops=2),
        "random": random_scores(candidates, random_seed),
    }

    deg_values = scores["degree"]
    k = min(10, len(candidates))
    return ContextResult(
        pool=pool_name,
        budget=budget,
        context=context,
        seed_size=len(seeds),
        n_pool=len(candidates),
        degree_min=int(min(deg_values)),
        degree_max=int(max(deg_values)),
        negative_gain_share=sum(1 for g in truth if g < 0) / len(truth),
        mean_gain=statistics.fmean(truth),
        spearman_degree=spearman(deg_values, truth),
        spearman_bfs_v1=spearman(scores["expo_bfs_v1"], truth),
        spearman_delta2=spearman(scores["expo_delta2"], truth),
        spearman_random=spearman(scores["random"], truth),
        recall_degree=recall_at_k(truth, deg_values, k),
        recall_bfs_v1=recall_at_k(truth, scores["expo_bfs_v1"], k),
        recall_delta2=recall_at_k(truth, scores["expo_delta2"], k),
        recall_random=recall_at_k(truth, scores["random"], k),
        capture_degree=gain_capture(truth, deg_values, k),
        capture_bfs_v1=gain_capture(truth, scores["expo_bfs_v1"], k),
        capture_delta2=gain_capture(truth, scores["expo_delta2"], k),
        capture_random=gain_capture(truth, scores["random"], k),
    )


def run(
    config_path: Path,
    budgets: list[int],
    contexts: int,
    mc_runs: int,
    pool_size: int,
    random_seed: int,
    weight_scale: float | None,
    hops: int,
) -> dict:
    config = load_yaml_config(config_path)
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    if weight_scale is not None and weight_scale != 1.0:
        # Weights live on the edges, not in the config, so scale them after loading.
        nx.set_edge_attributes(
            graph,
            {edge: float(data.get("weight", 0.0)) * weight_scale
             for edge, data in graph.edges(data=True)},
            "weight",
        )
        print(f"  [weight-scale {weight_scale}] applied to "
              f"{graph.number_of_edges()} edges", flush=True)
    nodes = list(graph.nodes())
    rng = random.Random(random_seed)

    results: list[ContextResult] = []
    for budget in budgets:
        for context in range(contexts):
            size = round((budget - 1) * context / max(1, contexts - 1))
            seeds = rng.sample(nodes, size) if size else []
            pools = build_pools(graph, nodes, set(seeds), budget, pool_size, rng)
            for pool_name in (POOL_DEGREE, POOL_UNIFORM):
                candidates = pools[pool_name]
                result = evaluate_context(
                    graph, pool_name, candidates, seeds, budget, context,
                    mc_runs, random_seed + 977 * context + 31 * budget,
                )
                # hops is a runtime knob on the corrected scorer only.
                results.append(result)
                print(
                    f"  k={budget} ctx={context} |S|={len(seeds):>3} {pool_name:<13}"
                    f" pool={len(candidates):>4} deg=[{result.degree_min},{result.degree_max}]"
                    f" neg={result.negative_gain_share*100:>5.1f}%"
                    f" rho_deg={result.spearman_degree:+.4f}"
                    f" rho_d2={result.spearman_delta2:+.4f}"
                    f" cap_deg={result.capture_degree:.4f}"
                    f" cap_d2={result.capture_delta2:.4f}",
                    flush=True,
                )

    return {
        "config": str(config_path),
        "weight_scale": weight_scale,
        "hops": hops,
        "budgets": budgets,
        "contexts": contexts,
        "mc_runs": mc_runs,
        "pool_size": pool_size,
        "random_seed": random_seed,
        "results": [asdict(r) for r in results],
        "summary": _summarise(results, budgets),
    }


def _pair_diffs(rows: list[ContextResult], field_a: str, field_b: str) -> list[float]:
    return [getattr(r, field_a) - getattr(r, field_b) for r in rows]


def _summarise(results: list[ContextResult], budgets: list[int]) -> dict:
    summary: dict = {"by_pool": {}, "by_pool_and_budget": {}}

    def block(rows: list[ContextResult]) -> dict:
        if not rows:
            return {}
        out = {
            "contexts": len(rows),
            "negative_gain_share": statistics.fmean([r.negative_gain_share for r in rows]),
            "mean_gain": statistics.fmean([r.mean_gain for r in rows]),
        }
        for name, prefix in (("degree", "spearman_degree"), ("bfs_v1", "spearman_bfs_v1"),
                             ("delta2", "spearman_delta2"), ("random", "spearman_random")):
            values = [getattr(r, prefix) for r in rows]
            finite = [v for v in values if v == v]
            out[f"spearman_{name}"] = statistics.fmean(finite) if finite else float("nan")
            out[f"spearman_{name}_n"] = len(finite)
        for name in ("degree", "bfs_v1", "delta2", "random"):
            out[f"recall_{name}"] = statistics.fmean([getattr(r, f"recall_{name}") for r in rows])
            out[f"capture_{name}"] = statistics.fmean([getattr(r, f"capture_{name}") for r in rows])
        # Paired advantages of each exposure scorer over the degree incumbent.
        for name in ("bfs_v1", "delta2"):
            for metric in ("spearman", "recall", "capture"):
                diffs = _pair_diffs(rows, f"{metric}_{name}", f"{metric}_degree")
                lo, hi = bootstrap_ci(diffs)
                out[f"delta_{metric}_{name}"] = statistics.fmean(diffs)
                out[f"delta_{metric}_{name}_ci"] = [lo, hi]
        return out

    for pool_name in (POOL_DEGREE, POOL_UNIFORM):
        rows = [r for r in results if r.pool == pool_name]
        summary["by_pool"][pool_name] = block(rows)
        for budget in budgets:
            sub = [r for r in rows if r.budget == budget]
            summary["by_pool_and_budget"][f"{pool_name}_k{budget}"] = block(sub)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--mc-runs", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--weight-scale", type=float, default=None,
                        help="multiply all edge weights; >1 pushes the model into the "
                             "non-monotone regime where negative marginal gains appear")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print("Gate-1a'' diagnostic: conditional ranking inside a degree-selected pool")
    print(f"  budgets={args.budgets} contexts={args.contexts} mc={args.mc_runs} "
          f"pool={args.pool_size} weight_scale={args.weight_scale} hops={args.hops}")
    print()

    payload = run(
        args.config, args.budgets, args.contexts, args.mc_runs,
        args.pool_size, args.random_seed, args.weight_scale, args.hops,
    )

    print()
    print("=" * 108)
    print("SUMMARY  (paired vs the degree incumbent; CI = bootstrap 95% on the mean difference)")
    print("=" * 108)
    for pool_name in (POOL_DEGREE, POOL_UNIFORM):
        row = payload["summary"]["by_pool"][pool_name]
        if not row:
            continue
        print(f"\n[{pool_name}]  contexts={row['contexts']}  "
              f"negative_gain_share={row['negative_gain_share']*100:.1f}%  "
              f"mean_gain={row['mean_gain']:.2f}")
        print(f"  {'scorer':<14}{'spearman':>12}{'recall@10':>12}{'gain_capture':>14}")
        for name in ("degree", "bfs_v1", "delta2", "random"):
            print(f"  {name:<14}{row[f'spearman_{name}']:>12.4f}"
                  f"{row[f'recall_{name}']:>12.4f}{row[f'capture_{name}']:>14.4f}")
        for name in ("bfs_v1", "delta2"):
            for metric in ("spearman", "recall", "capture"):
                d = row[f"delta_{metric}_{name}"]
                lo, hi = row[f"delta_{metric}_{name}_ci"]
                print(f"    delta_{metric}_{name:<8} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    print()
    print("=" * 108)
    print("VERDICT")
    print("=" * 108)
    deg_pool = payload["summary"]["by_pool"].get(POOL_DEGREE) or {}
    uni_pool = payload["summary"]["by_pool"].get(POOL_UNIFORM) or {}
    if deg_pool:
        d_rho = deg_pool.get("delta_spearman_delta2", float("nan"))
        d_cap = deg_pool.get("delta_capture_delta2", float("nan"))
        lo_cap, hi_cap = deg_pool.get("delta_capture_delta2_ci", (float("nan"),) * 2)
        neg = deg_pool.get("negative_gain_share", 0.0)
        print(f"  degree-selected pool: delta2 vs degree  "
              f"spearman {d_rho:+.4f}   gain_capture {d_cap:+.4f} CI [{lo_cap:+.4f}, {hi_cap:+.4f}]")
        print(f"  non-monotonicity present: negative gains on "
              f"{neg*100:.1f}% of candidates")
        print()
        if lo_cap > 0:
            print("  -> PASS: inside the deployed retriever pool the state-aware score "
                  "captures more gain than degree, with a CI that excludes zero.")
        elif hi_cap < 0:
            print("  -> FAIL: the state-aware score is *worse* than degree inside the pool; "
                  "degree must already be near-optimal there.")
        else:
            print("  -> INCONCLUSIVE: CI covers zero. The pool regime does not separate the "
                  "scorers; increase contexts or move to a regime with more negative gains.")
    if uni_pool:
        print(f"  (control) uniform pool: degree spearman "
              f"{uni_pool.get('spearman_degree', float('nan')):.4f} vs delta2 "
              f"{uni_pool.get('spearman_delta2', float('nan')):.4f}  <- Gate-1a' artifact check")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
