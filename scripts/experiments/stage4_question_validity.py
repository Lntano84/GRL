"""Stage 4: does the overexposure setting actually pose a research question?

This runs before any learning.  It establishes four things on identical graphs, budgets,
diffusion parameters and evaluation budget:

Q1. Is a candidate's marginal gain **state dependent**?  If ``Delta(v | S)`` barely moves with S,
    a static node score would suffice and a state-conditioned predictor would have nothing to
    learn.  Measured by evaluating the same candidate set under several seed states and
    decomposing its variation into between-candidate and within-candidate parts.

Q2. Do cheap heuristics **rank** candidates well?  ``degree`` and ``degree discount`` are scored
    against the oracle's marginal gains by Spearman correlation, per budget.

Q3. Does overexposure **change the ordering** relative to a no-overexposure limit?  If the two
    produce the same ranking, the overexposure machinery adds nothing.

Q4. How much room is there between the cheap heuristics and the oracle?  ``gap to oracle`` is the
    quantity the learned pipeline is supposed to close at lower oracle cost.

Baseline fairness
-----------------
Every method is evaluated with the same graph, the same budget ``k``, the same diffusion
parameters and the same evaluation Monte-Carlo budget.  The final spread of a seed set is always
measured by an independent evaluation oracle, never by the scores a method used to choose it.

``degree discount`` is used in its state-conditioned form (``rank_degree_discount_candidates``),
which discounts candidates by the already-selected seeds.  Note its discount formula is derived
for independent cascade and takes a ``probability`` argument; here edges carry the graph's own
weights, whose mean is passed in and reported, so the comparison is transparent rather than
silently tuned.
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
from evaluate_overexposure_pool_ranking import degree_scores, spearman  # noqa: E402
from grl.baselines import rank_degree_discount_candidates  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402


def mean_edge_weight(graph: nx.DiGraph) -> float:
    weights = [float(d.get("weight", 0.0)) for _, _, d in graph.edges(data=True)]
    return statistics.fmean(weights) if weights else 0.0


def spearman_xy(xs: list[float], ys: list[float]) -> float:
    return spearman(xs, ys)


@dataclass
class RankRow:
    graph: str
    budget: int
    method: str
    spread: float
    spread_stderr: float
    gap_to_oracle: float
    oracle_cascades: int


@dataclass
class StateRow:
    graph: str
    budget: int
    between_sd: float
    within_sd: float
    state_dependence_ratio: float
    rho_no_overexposure: float
    n_candidates: int


def rank_methods(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    oracle: OverexposureMonteCarloOracle,
    evaluator: OverexposureMonteCarloOracle,
    mean_weight: float,
    seed: int,
) -> list[RankRow]:
    """Run the cheap heuristics and the oracle greedy, all evaluated identically."""
    degree = dict(graph.out_degree())
    ranked_degree = sorted(pool, key=lambda v: (-degree[v], v))
    dd_order = rank_degree_discount_candidates(graph, [], pool, mean_weight)
    random_order = list(pool)
    random.Random(seed).shuffle(random_order)

    greedy_seeds, greedy_cost = oracle_greedy(graph, pool, budget, oracle, seed)

    rows: list[RankRow] = []
    for name, order in (
        ("degree", ranked_degree),
        ("degree_discount", dd_order),
        ("random", random_order),
    ):
        seeds = order[:budget]
        spread = evaluator.spread(seeds)
        rows.append(RankRow(graph=graph.name if hasattr(graph, "name") else "",
                            budget=budget, method=name, spread=spread["mean"],
                            spread_stderr=spread["stderr"], gap_to_oracle=float("nan"),
                            oracle_cascades=0))

    oracle_spread = evaluator.spread(greedy_seeds)
    rows.append(RankRow(graph="", budget=budget, method="mc_greedy_oracle",
                        spread=oracle_spread["mean"], spread_stderr=oracle_spread["stderr"],
                        gap_to_oracle=0.0, oracle_cascades=greedy_cost))
    return rows


def oracle_greedy(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    oracle: OverexposureMonteCarloOracle,
    seed: int,
) -> tuple[list[int], int]:
    """MC greedy inside the overexposure model: the ground-truth reference."""
    chosen: list[int] = []
    remaining = list(pool)
    for step in range(budget):
        if not remaining:
            break
        scores = oracle.score(chosen, remaining, step=seed + step)
        best = max(remaining, key=lambda v: (scores[v], -v))
        if scores[best] <= 0.0:
            break
        chosen.append(best)
        remaining.remove(best)
    return chosen, oracle.stats.mc_cascades


def state_dependence(
    graph: nx.DiGraph,
    nodes: list[int],
    candidates: list[int],
    states: list[list[int]],
    oracle: OverexposureMonteCarloOracle,
    oe_oracle_free: OverexposureMonteCarloOracle,
    seed: int,
) -> StateRow:
    """Variance decomposition of Delta(v | S) across seed states, plus OE vs no-OE ranking."""
    gains_by_candidate: dict[int, list[float]] = {c: [] for c in candidates}
    for index, state in enumerate(states):
        scores = oracle.score(state, candidates, step=seed + 100 * index)
        for c in candidates:
            gains_by_candidate[c].append(scores[c])

    candidate_means = {c: statistics.fmean(v) for c, v in gains_by_candidate.items()}
    between = statistics.pstdev(list(candidate_means.values())) if len(candidate_means) > 1 else 0.0
    within_terms = []
    for c, values in gains_by_candidate.items():
        if len(values) > 1:
            within_terms.append(statistics.pstdev(values))
    within = statistics.fmean(within_terms) if within_terms else 0.0
    ratio = within / between if between > 1e-12 else float("inf")

    # Q3: does removing overexposure change the ranking of the same candidates?
    reference = states[0]
    oe_scores = [oracle.score(reference, candidates, step=seed + 7)[c] for c in candidates]
    no_oe_scores = [
        oe_oracle_free.score(reference, candidates, step=seed + 7)[c] for c in candidates
    ]
    rho = spearman_xy(oe_scores, no_oe_scores)

    return StateRow(graph="", budget=0, between_sd=between, within_sd=within,
                    state_dependence_ratio=ratio, rho_no_overexposure=rho,
                    n_candidates=len(candidates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core"], choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--oracle-mc", type=int, default=60,
                        help="MC cascades per candidate inside the oracle greedy")
    parser.add_argument("--eval-mc", type=int, default=300,
                        help="MC cascades used to evaluate a final seed set")
    parser.add_argument("--state-contexts", type=int, default=4,
                        help="number of distinct seed states for the state-dependence test")
    parser.add_argument("--state-mc", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    params = resolve_overexposure_params(None)
    rank_rows: list[RankRow] = []
    state_rows: list[StateRow] = []

    for graph_name in args.graphs:
        graph = load_graph(graph_name)
        graph.name = graph_name  # type: ignore[attr-defined]
        nodes = list(graph.nodes())
        n = len(nodes)
        mw = mean_edge_weight(graph)
        print(f"=== {graph_name}: n={n} m={graph.number_of_edges()} "
              f"<k>={2*graph.number_of_edges()/n:.2f} mean_edge_weight={mw:.5f}")

        for budget in args.budgets:
            rng = random.Random(args.random_seed + 31 * budget)
            pool = rng.sample(nodes, min(args.pool_size, n))

            oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.oracle_mc,
                                                  random_seed=args.random_seed, params=params)
            evaluator = OverexposureMonteCarloOracle(graph, mc_runs=args.eval_mc,
                                                     random_seed=args.random_seed + 99,
                                                     params=params)

            rows = rank_methods(graph, pool, budget, oracle, evaluator, mw,
                                args.random_seed + 5 * budget)
            oracle_spread = [r for r in rows if r.method == "mc_greedy_oracle"][0].spread
            for r in rows:
                r.graph = graph_name
                if r.method != "mc_greedy_oracle":
                    r.gap_to_oracle = (oracle_spread - r.spread) / oracle_spread \
                        if oracle_spread > 1e-9 else float("nan")
                rank_rows.append(r)
                print(f"  k={budget:<3} {r.method:<18} spread={r.spread:9.2f} "
                      f"+-{r.spread_stderr:5.2f}  gap={r.gap_to_oracle*100:6.2f}%")
            print()

        # Q1/Q3 at the largest budget, using distinct seed states
        budget = max(args.budgets)
        rng = random.Random(args.random_seed + 999)
        states = [rng.sample(nodes, min(budget * i, n)) for i in range(args.state_contexts)]
        pool = rng.sample([v for v in nodes if v not in set(states[-1])],
                          min(args.pool_size, n))
        candidates = pool[:min(20, len(pool))]

        oe_oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.state_mc,
                                                 random_seed=args.random_seed, params=params)
        no_oe_params = resolve_overexposure_params({"overexposure": {"overexposure_free": True}})
        no_oe_oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.state_mc,
                                                    random_seed=args.random_seed,
                                                    params=no_oe_params)
        srow = state_dependence(graph, nodes, candidates, states, oe_oracle, no_oe_oracle,
                                args.random_seed)
        srow.graph = graph_name
        srow.budget = budget
        state_rows.append(srow)
        print(f"  state dependence (candidates={srow.n_candidates}, "
              f"states={args.state_contexts}):")
        print(f"    between-candidate sd = {srow.between_sd:.4f}")
        print(f"    within-candidate  sd = {srow.within_sd:.4f}")
        print(f"    ratio within/between = {srow.state_dependence_ratio:.3f}")
        print(f"    rho(overexposure, no-overexposure) marginal gains = "
              f"{srow.rho_no_overexposure:+.3f}")
        print()

    print("=" * 92)
    print("STAGE 4 VERDICT")
    print("=" * 92)
    print(f"  {'graph':<18}{'k':>4}{'method':<20}{'spread':>10}{'gap%':>9}")
    for r in rank_rows:
        print(f"  {r.graph:<18}{r.budget:>4}{r.method:<20}{r.spread:>10.2f}"
              f"{r.gap_to_oracle*100:>9.2f}")
    print()
    for s in state_rows:
        verdict = ("STATE-DEPENDENT" if s.state_dependence_ratio > 0.5 else
                   "weakly state-dependent" if s.state_dependence_ratio > 0.2 else
                   "NOT state-dependent")
        print(f"  {s.graph}: within/between = {s.state_dependence_ratio:.3f} -> {verdict}")
        print(f"    rho(OE, no-OE) = {s.rho_no_overexposure:+.3f} "
              f"({'ranking changes' if s.rho_no_overexposure < 0.9 else 'ranking preserved'})")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graphs": args.graphs, "budgets": args.budgets,
            "pool_size": args.pool_size, "oracle_mc": args.oracle_mc,
            "eval_mc": args.eval_mc, "state_mc": args.state_mc,
            "state_contexts": args.state_contexts,
            "rank_rows": [asdict(r) for r in rank_rows],
            "state_rows": [asdict(s) for s in state_rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
