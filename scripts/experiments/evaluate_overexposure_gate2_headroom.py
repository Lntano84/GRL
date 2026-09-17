"""Gate-2: does a *state-conditioned* scorer beat the static analytic baseline?

What Gate 1c v3 established
---------------------------
On the paper's own graphs, swept by seed *fraction* ``|S|/n`` rather than absolute ``|S|``:

* ``rho_degree`` (out-degree vs true marginal gain) flips sign as the network saturates --
  ``+0.912 -> -0.141`` on NetHEPT, ``+0.792 -> -0.229`` on Wiki-Vote,
  ``+0.472 -> -0.397`` on Congress-Twitter.  So the original hypothesis ("hubs overexpose
  their own neighbourhood, degree becomes an anti-signal") is correct in the saturated
  regime.
* ``delta2`` stays positive everywhere and is the only scorer that does.

But ``delta2`` as implemented in Gate 1a'' has a specific weakness: it evaluates
``g(delta(u) + dE) - g(delta(u))`` using the exposure state of the **seedless** graph
(``delta ~ 0``).  That is the right thing at ``|S| = 0`` -- where ``g' = 2`` and the score
degenerates to the weighted out-sum -- but it is the wrong thing once the state is
non-trivial, because the whole point of overexposure is that ``g'`` collapses near
``delta = 1``.

This script measures the three-way frontier on Congress-Twitter, the small dense graph
where the saturated regime is reachable and where an *exact* Monte-Carlo greedy is
affordable:

``degree``         out-degree top-k.  Zero cascades.
``delta2_static``  the Gate-1a'' scorer at the seedless state.  Zero cascades.
``greedy_exact``   Monte-Carlo greedy that re-simulates every remaining candidate at every
                   step.  ``k * M * MC`` cascades -- the exact reference, and the thing a
                   learned policy is supposed to replace.

and, per seed size, reports the **capture ratio** of each zero-cost scorer against the
exact greedy, together with the cascade count that greedy needed.  The capture ratio is
the quantity a learned policy has to improve; the cascade count is the quantity it saves.

Everything is swept over seed *fraction*, because that is the variable that controls
whether the objective is non-monotone at all.
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

from evaluate_overexposure_paper_graphs import LOADERS, normalise_in_weights  # noqa: E402
from evaluate_overexposure_pool_ranking import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
    spearman,
)
from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
    run_overexposure,
    sample_threshold_windows,
)


@dataclass
class Row:
    graph: str
    seed_size: int
    seed_fraction: float
    pool_size: int
    spread_degree: float
    spread_delta2: float
    spread_greedy: float
    capture_degree: float
    capture_delta2: float
    spearman_degree: float
    spearman_delta2: float
    negative_share: float
    greedy_cascades: int
    greedy_steps_used: int


def exact_greedy(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    mc: int,
    base_seed: int,
) -> tuple[list[int], int, int]:
    """Monte-Carlo greedy, re-simulating every remaining candidate at every step.

    ``budget`` is the number of *additional* seeds to select.  It must not be conflated
    with the size of the pre-existing seed set: at ``|S| = 0`` the budget is still the full
    ``k``, and passing the seed-set size here instead makes the loop run zero times, so the
    greedy selects nothing and reports a spread of 0.

    Returns ``(seeds, cascades, steps_used)``.  ``cascades`` counts only the simulations
    the policy itself spends; the final scoring of the produced set is not charged to it.
    """
    seeds: list[int] = []
    remaining = list(pool)
    cascades = 0
    steps_used = 0
    for step in range(budget):
        if not remaining:
            break
        configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
        estimates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc, base_seed + 17 * step
        )
        cascades += len(remaining) * mc + 1
        baseline = estimates[0]["mean"]
        gains = [estimates[i + 1]["mean"] - baseline for i in range(len(remaining))]
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0.0:
            # No candidate improves the spread.  Because the objective is non-monotone
            # this is a legitimate stop, not a numerical artefact.
            break
        seeds.append(remaining.pop(best))
        steps_used = step + 1
    return seeds, cascades, steps_used


def evaluate_size(
    graph: nx.DiGraph,
    label: str,
    seed_size: int,
    budget: int,
    pool_size: int,
    mc_greedy: int,
    mc_eval: int,
    base_seed: int,
) -> Row:
    """Score every policy at one pre-existing seed size.

    ``seed_size`` is how many seeds the network already has (this is what drives
    saturation).  ``budget`` is how many *further* seeds each policy may add, and it is
    held fixed across the sweep so the policies are compared at equal cost.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    rng = random.Random(base_seed)
    seeds = rng.sample(nodes, min(seed_size, n)) if seed_size else []
    pool = [v for v in nodes if v not in set(seeds)]
    pool = rng.sample(pool, min(pool_size, len(pool)))

    # --- zero-cost scorers, evaluated against the *current* state ------------------
    state = mean_exposure_state(graph, seeds, min(mc_eval, 20), base_seed + 3)
    degree = degree_scores(graph, pool)
    delta2 = exposure_scores_delta(graph, pool, state, set(seeds), hops=2)

    # --- exact greedy -------------------------------------------------------------
    greedy_seeds, cascades, steps_used = exact_greedy(
        graph, pool, budget, mc_greedy, base_seed + 101
    )

    # --- paired evaluation of every candidate set on the same realisation ----------
    by_degree = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -degree[i])[:budget]]
    by_delta2 = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -delta2[i])[:budget]]
    configurations = [seeds, by_degree, by_delta2, greedy_seeds]
    estimates = estimate_overexposure_spread_over_configs(
        graph, configurations, mc_eval, base_seed + 7
    )
    base = estimates[0]["mean"]
    spread_degree = estimates[1]["mean"] - base
    spread_delta2 = estimates[2]["mean"] - base
    spread_greedy = estimates[3]["mean"] - base

    # --- true marginal gains, for the *ranking* metrics ---------------------------
    truth_configs = [list(seeds)] + [[*seeds, c] for c in pool]
    truth_est = estimate_overexposure_spread_over_configs(
        graph, truth_configs, mc_eval, base_seed + 7
    )
    truth_base = truth_est[0]["mean"]
    true_gains = [truth_est[i + 1]["mean"] - truth_base for i in range(len(pool))]

    # --- capture, measured in realised spread -------------------------------------
    # NOTE 1: capture must be the ratio of *evaluated spreads*, not of summed one-step
    # marginal gains.  Summing ``Delta(v | S)`` over a scorer's top-k double counts, because
    # those gains are single additions to the same state and ignore the diminishing returns
    # of adding them together.  That version produced captures above 1.0, impossible for a
    # greedy reference.
    #
    # NOTE 2: the ratio is still numerically fragile in the saturated regime, because the
    # greedy reference gain tends to zero there (at |S|/n = 20% on Congress-Twitter the
    # exact greedy adds only +0.5 to a base of 366).  Dividing by that produces wild values
    # for every policy.  ``spread_*`` is therefore the primary quantity and ``capture_*``
    # is reported only for reference.
    def capture(spread: float) -> float:
        return spread / spread_greedy if abs(spread_greedy) > 1e-9 else float("nan")

    return Row(
        graph=label,
        seed_size=len(seeds),
        seed_fraction=len(seeds) / n,
        pool_size=len(pool),
        spread_degree=spread_degree,
        spread_delta2=spread_delta2,
        spread_greedy=spread_greedy,
        capture_degree=capture(spread_degree),
        capture_delta2=capture(spread_delta2),
        spearman_degree=spearman(degree, true_gains),
        spearman_delta2=spearman(delta2, true_gains),
        negative_share=sum(1 for g in true_gains if g < 0) / len(true_gains),
        greedy_cascades=cascades,
        greedy_steps_used=steps_used,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(LOADERS))
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--pool-size", type=int, default=80)
    parser.add_argument("--budget", type=int, default=10,
                        help="seeds each policy may ADD at every sweep point; held fixed so "
                             "the zero-cost scorers and the exact greedy are compared at "
                             "equal cost.  Distinct from |S|, which is what drives saturation.")
    parser.add_argument("--mc-greedy", type=int, default=60,
                        help="MC runs per candidate inside the exact greedy")
    parser.add_argument("--mc-eval", type=int, default=200,
                        help="MC runs used for the paired evaluation and the true gains")
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = LOADERS[args.graph]()
    if not args.no_normalise:
        graph = normalise_in_weights(graph)
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    sizes = sorted({int(round(f * n)) for f in args.fractions})
    print(f"graph={args.graph} n={n} m={m} <k>={2*m/n:.2f} "
          f"pool={args.pool_size} budget={args.budget} "
          f"mc_greedy={args.mc_greedy} mc_eval={args.mc_eval}")
    print()

    rows: list[Row] = []
    for size in sizes:
        row = evaluate_size(graph, args.graph, size, args.budget, args.pool_size,
                            args.mc_greedy, args.mc_eval, args.random_seed + 13 * size)
        rows.append(row)
        print(f"  |S|={row.seed_size:>3} ({row.seed_fraction*100:>5.1f}%)  "
              f"neg={row.negative_share*100:>5.1f}%  "
              f"rho_deg={row.spearman_degree:>7.3f}  rho_d2={row.spearman_delta2:>7.3f}  "
              f"capture deg={row.capture_degree:>6.3f} d2={row.capture_delta2:>6.3f}  "
              f"greedy_spread={row.spread_greedy:>8.2f} cascades={row.greedy_cascades:>7} "
              f"(steps={row.greedy_steps_used})", flush=True)

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    sat = [r for r in rows if r.seed_fraction >= 0.10]
    unsat = [r for r in rows if r.seed_fraction < 0.10]
    for name, block in (("unsaturated (|S|/n < 10%)", unsat),
                        ("saturated   (|S|/n >= 10%)", sat)):
        if not block:
            continue
        print(f"  {name}")
        print(f"    marginal spread added by the budget:"
              f"  degree {statistics.fmean([r.spread_degree for r in block]):+8.2f}"
              f"   delta2 {statistics.fmean([r.spread_delta2 for r in block]):+8.2f}"
              f"   greedy {statistics.fmean([r.spread_greedy for r in block]):+8.2f}")
        print(f"    rank correlation with the true marginal gain:"
              f"  rho_degree {statistics.fmean([r.spearman_degree for r in block]):+.3f}"
              f"   rho_delta2 {statistics.fmean([r.spearman_delta2 for r in block]):+.3f}"
              f"   negative-gain share {statistics.fmean([r.negative_share for r in block])*100:.1f}%")
    print()
    total_cascades = sum(r.greedy_cascades for r in rows)
    print(f"  exact greedy spent {total_cascades} cascades across {len(rows)} settings;"
          f" both zero-cost scorers spent 0")
    print()
    print("  PRIMARY QUANTITY is the marginal spread, not the capture ratio: the greedy")
    print("  reference gain collapses toward zero once the network saturates, so the ratio")
    print("  is numerically unstable there and is reported only for reference.")
    print()
    worst_deg = min(rows, key=lambda r: r.spread_degree)
    best_d2 = max(rows, key=lambda r: r.spread_delta2)
    print(f"  worst degree marginal   : {worst_deg.spread_degree:+8.2f} "
          f"at |S|/n = {worst_deg.seed_fraction*100:.1f}%")
    print(f"  best  delta2 marginal   : {best_d2.spread_delta2:+8.2f} "
          f"at |S|/n = {best_d2.seed_fraction*100:.1f}%")
    if sat and statistics.fmean([r.spread_degree for r in sat]) < 0:
        print("  -> in the saturated regime the out-degree scorer is actively HARMFUL")
        print("     (it reduces spread below the pre-existing seed set), while delta2 stays")
        print("     non-negative.  That is the gap a learned policy has to close, and it is")
        print("     a stopping/selection problem, not a ranking problem.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph,
            "n": n,
            "m": m,
            "fractions": args.fractions,
            "pool_size": args.pool_size,
            "mc_greedy": args.mc_greedy,
            "mc_eval": args.mc_eval,
            "in_weights_normalised": not args.no_normalise,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
