"""Gate-1a' diagnostic: does *state* information improve conditional ranking?

Why this experiment exists
--------------------------
Gate 1a measured the aggregate Spearman correlation between node degree and the true
marginal gain, sampling candidates at random.  It found almost no difference between IC
and overexposure (0.885 vs 0.888), which killed the naive claim "degree stops working
under overexposure".

But that is the wrong comparison for a *state-conditioned* method.  The claim under test
is not "degree is a bad node-level score"; it is:

    given a concrete seed state S, does using the observed exposure state rank
    candidates better than using degree alone?

So this script fixes a concrete S, computes the true Delta(v | S) by paired Monte Carlo,
and compares two *learning-free* scorers:

``degree``
    Out-degree, i.e. a purely node-level score that ignores S entirely.

``exposure_aware``
    A hand-crafted state-aware score: the expected number of *newly* positively
    activated out-neighbours when v is added,

        q(v | S) = sum_{u in N+(v)} (1 - delta(u)) * p(1 - delta(u))

    where ``delta(u)`` is u's cumulative influence under S and ``p(d) = 2d(1-d)`` is the
    positive-activation probability.  This needs the state and nothing else.

Both scorers are cheap and require no training.  The gap between them is therefore a
*lower bound* on what state conditioning can buy, and it isolates the question from any
learning effect.  If ``exposure_aware`` beats ``degree`` on the conditional ranking under
overexposure but not under IC, the setting genuinely rewards state conditioning and a
learned policy has somewhere to live.  If it does not, Gate 1a' fails and the direction
needs rebuilding.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from grl.data.graph_loader import load_graph_from_config  # noqa: E402
from grl.diffusion import estimate_spread_over_configs as estimate_ic_spreads  # noqa: E402
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    positive_activation_probability,
    run_overexposure,
    sample_threshold_windows,
)
from grl.utils.config import load_yaml_config  # noqa: E402

SCORER_DEGREE = "degree"
SCORER_EXPOSURE = "exposure_aware"
SCORERS = (SCORER_DEGREE, SCORER_EXPOSURE)

MODEL_IC = "ic"
MODEL_OVEREXPOSURE = "overexposure"


def spearman(xs: list[float], ys: list[float]) -> float:
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


def degree_scores(graph: nx.Graph | nx.DiGraph, candidates: list[int]) -> list[float]:
    source = graph.out_degree if graph.is_directed() else graph.degree
    return [float(source[v]) for v in candidates]


def exposure_scores(
    graph: nx.Graph | nx.DiGraph,
    candidates: list[int],
    delta: dict[int, float],
    seeds: set[int],
    depth: int = 4,
) -> list[float]:
    """State-aware, learning-free score: expected new positive activations.

    A single-hop score cannot rank anything when the state is empty, and a seed's value
    comes mostly from the multi-hop cascade it triggers.  So we propagate the *change* in
    activation probability outwards from the candidate:

        q(v | S) = sum over u reachable within ``depth`` hops of
                     (reach_prob(u | v) * [P(u | S, v) - P(u | S)])

    where the bracket is the change in u's positive-activation probability caused by
    activating v, and ``reach_prob`` is the product of edge weights along a shortest
    weighted path (a standard influence-propagation proxy).  The score uses the observed
    exposure state ``delta`` and the graph only -- no learning.
    """
    directed = graph.is_directed()
    scores: list[float] = []

    for candidate in candidates:
        if candidate in seeds:
            scores.append(float("-inf"))
            continue

        total = 0.0
        # BFS carrying (node, reach_probability)
        frontier: list[tuple[int, float]] = [(candidate, 1.0)]
        seen: dict[int, float] = {candidate: 1.0}
        level = 1
        while frontier and level <= depth:
            next_frontier: list[tuple[int, float]] = []
            for node, reach in frontier:
                edges = graph.out_edges(node, data=True) if directed else graph.edges(node, data=True)
                for u, w, data in edges:
                    target = w if directed else (w if u == node else u)
                    if target == node or target in seeds:
                        continue
                    weight = float(data.get("weight", 0.0))
                    contribution = reach * weight
                    if contribution <= 0.0:
                        continue
                    # Change in target's activation probability when it receives this
                    # extra influence on top of the current state.
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


@dataclass
class ContextResult:
    model: str
    budget: int
    context: int
    seed_size: int
    n_candidates: int
    spearman_degree: float
    spearman_exposure: float
    top1_degree: float
    top1_exposure: float
    recall10_degree: float
    recall10_exposure: float
    negative_gain_share: float


def _top_k_recall(truth: list[float], scores: list[float], k: int) -> float:
    """Fraction of the true top-k candidates that the scorer also puts in its top-k."""
    if k <= 0 or not truth:
        return float("nan")
    k = min(k, len(truth))
    truth_top = {i for i in sorted(range(len(truth)), key=lambda i: -truth[i])[:k]}
    score_top = {i for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:k]}
    return len(truth_top & score_top) / k


def evaluate_context(
    graph: nx.Graph | nx.DiGraph,
    model: str,
    seeds: list[int],
    candidates: list[int],
    budget: int,
    context: int,
    mc_runs: int,
    random_seed: int,
) -> ContextResult:
    configurations = [list(seeds)] + [[*seeds, c] for c in candidates]

    if model == MODEL_OVEREXPOSURE:
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc_runs, random_seed
        )
    elif model == MODEL_IC:
        estimates = estimate_ic_spreads(graph, configurations, mc_runs, random_seed)
    else:
        raise ValueError(model)

    base = estimates[0]["mean"]
    truth = [estimates[i + 1]["mean"] - base for i in range(len(candidates))]

    deg = degree_scores(graph, candidates)

    # Exposure state under S: mean delta over the same trials.  Computed from the
    # threshold model regardless of which model supplies the ground truth, because the
    # scorers are only meaningful where an exposure state exists.
    nodes = list(graph.nodes())
    delta_accum = {node: 0.0 for node in nodes}
    for offset in range(min(mc_runs, 20)):
        rng = random.Random(random_seed + offset)
        windows = sample_threshold_windows(nodes, rng)
        run = run_overexposure(graph, list(seeds), windows, rng)
        for node, value in run.delta.items():
            delta_accum[node] += value
    divisor = float(min(mc_runs, 20))
    delta = {node: value / divisor for node, value in delta_accum.items()}

    expo = exposure_scores(graph, candidates, delta, set(seeds))

    negative_share = (
        sum(1 for g in truth if g < 0) / len(truth) if truth else float("nan")
    )

    return ContextResult(
        model=model,
        budget=budget,
        context=context,
        seed_size=len(seeds),
        n_candidates=len(candidates),
        spearman_degree=spearman(deg, truth),
        spearman_exposure=spearman(expo, truth),
        top1_degree=_top_k_recall(truth, deg, 1),
        top1_exposure=_top_k_recall(truth, expo, 1),
        recall10_degree=_top_k_recall(truth, deg, 10),
        recall10_exposure=_top_k_recall(truth, expo, 10),
        negative_gain_share=negative_share,
    )


def run(
    config_path: Path,
    budgets: list[int],
    contexts: int,
    mc_runs: int,
    sample_size: int | None,
    random_seed: int,
) -> dict:
    config = load_yaml_config(config_path)
    graph_data = load_graph_from_config(config)
    graph = graph_data.graph
    nodes = list(graph.nodes())
    n = len(nodes)
    rng = random.Random(random_seed)

    results: list[ContextResult] = []
    for budget in budgets:
        for context in range(contexts):
            # Vary seed size so we see both unsaturated and saturated states.
            size = 0 if contexts == 1 else round((budget - 1) * context / (contexts - 1))
            seeds = rng.sample(nodes, size) if size else []
            pool = [node for node in nodes if node not in set(seeds)]
            candidates = rng.sample(pool, min(sample_size or n, len(pool)))
            for model in (MODEL_IC, MODEL_OVEREXPOSURE):
                results.append(
                    evaluate_context(
                        graph, model, seeds, candidates, budget, context,
                        mc_runs, random_seed + 977 * context + 31 * budget,
                    )
                )
            print(
                f"  k={budget} context={context} |S|={len(seeds)} candidates={len(candidates)}",
                flush=True,
            )

    return {
        "config": str(config_path),
        "budgets": budgets,
        "contexts": contexts,
        "mc_runs": mc_runs,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "results": [r.__dict__ for r in results],
        "summary": _summarise(results, budgets),
    }


def _summarise(results: list[ContextResult], budgets: list[int]) -> dict:
    def block(rows: list[ContextResult]) -> dict:
        return {
            "spearman_degree": statistics.fmean([r.spearman_degree for r in rows]),
            "spearman_exposure": statistics.fmean([r.spearman_exposure for r in rows]),
            "top1_degree": statistics.fmean([r.top1_degree for r in rows]),
            "top1_exposure": statistics.fmean([r.top1_exposure for r in rows]),
            "recall10_degree": statistics.fmean([r.recall10_degree for r in rows]),
            "recall10_exposure": statistics.fmean([r.recall10_exposure for r in rows]),
            "negative_gain_share": statistics.fmean([r.negative_gain_share for r in rows]),
            "contexts": len(rows),
        }

    summary: dict = {"by_model": {}, "by_model_and_budget": {}}
    for model in (MODEL_IC, MODEL_OVEREXPOSURE):
        summary["by_model"][model] = block([r for r in results if r.model == model])
        for budget in budgets:
            summary["by_model_and_budget"][f"{model}_k{budget}"] = block(
                [r for r in results if r.model == model and r.budget == budget]
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "nethept.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[10])
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--mc-runs", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=600,
                        help="candidates per context; <=0 means all non-seed nodes")
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sample_size = None if args.sample_size <= 0 else args.sample_size
    print("Gate-1a' diagnostic: conditional ranking, degree vs exposure-aware score")
    print(f"  budgets={args.budgets} contexts={args.contexts} mc={args.mc_runs} "
          f"candidates={sample_size or 'all'}")
    print()

    payload = run(
        args.config, args.budgets, args.contexts, args.mc_runs,
        sample_size, args.random_seed,
    )

    print()
    print("=" * 90)
    print("SUMMARY  (conditional ranking under a fixed concrete state S)")
    print("=" * 90)
    header = (f"{'model':<14}{'budget':>8}{'spear_deg':>11}{'spear_expo':>12}"
              f"{'top1_deg':>10}{'top1_expo':>11}{'R@10_deg':>10}{'R@10_expo':>11}"
              f"{'neg%':>8}")
    print(header)
    for model in (MODEL_IC, MODEL_OVEREXPOSURE):
        row = payload["summary"]["by_model"][model]
        print(f"{model:<14}{'all':>8}{row['spearman_degree']:>11.4f}"
              f"{row['spearman_exposure']:>12.4f}{row['top1_degree']:>10.3f}"
              f"{row['top1_exposure']:>11.3f}{row['recall10_degree']:>10.3f}"
              f"{row['recall10_exposure']:>11.3f}{row['negative_gain_share']*100:>8.1f}")
        for budget in args.budgets:
            sub = payload["summary"]["by_model_and_budget"][f"{model}_k{budget}"]
            print(f"{'':<14}{budget:>8}{sub['spearman_degree']:>11.4f}"
                  f"{sub['spearman_exposure']:>12.4f}{sub['top1_degree']:>10.3f}"
                  f"{sub['top1_exposure']:>11.3f}{sub['recall10_degree']:>10.3f}"
                  f"{sub['recall10_exposure']:>11.3f}{sub['negative_gain_share']*100:>8.1f}")
    print()

    ic = payload["summary"]["by_model"][MODEL_IC]
    oe = payload["summary"]["by_model"][MODEL_OVEREXPOSURE]
    ic_gain = ic["spearman_exposure"] - ic["spearman_degree"]
    oe_gain = oe["spearman_exposure"] - oe["spearman_degree"]
    print("GATE 1a' judgement (Spearman gain of exposure-aware over degree)")
    print(f"  IC          : degree {ic['spearman_degree']:.4f} -> exposure "
          f"{ic['spearman_exposure']:.4f}  ({ic_gain:+.4f})")
    print(f"  overexposure: degree {oe['spearman_degree']:.4f} -> exposure "
          f"{oe['spearman_exposure']:.4f}  ({oe_gain:+.4f})")
    print()
    if oe_gain > 0.05 and oe_gain > ic_gain + 0.03:
        print("  -> PASS: exposure state carries ranking information that degree lacks, "
              "and predominantly under overexposure. State conditioning has real value.")
    elif oe_gain > 0.02:
        print("  -> WEAK PASS: some advantage, but small. Review before committing.")
    else:
        print("  -> FAIL: the exposure-aware score does not beat degree on conditional "
              "ranking; the setting does not reward state conditioning beyond degree.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
