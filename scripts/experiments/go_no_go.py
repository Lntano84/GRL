"""Go/No-Go: sequential vs static, and the quality--cost curves, under the frozen contract.

What this script answers
------------------------
**Gate 1(b)**, the only experiment that gates the DASFAA submission: does a *sequential* policy ---
one that observes the cascade and re-ranks after each seed --- beat the best *static* seed set by
more than 5%?  Gate 1(a) already passes (`docs/results/signflip_fixedmodel_mc300.json`); this is the
other half, and its deadline is 2026-10-13.

And the supporting evidence the plan asks for: quality--cost curves for a high-accuracy Monte-Carlo
reference, degree, the analytic `delta2`, `random-pruning`, and adaptive sampling, over several
graphs, budgets, candidate pools and random seeds, under both weight normalisations.

Design decisions that are load-bearing
--------------------------------------
**Budgets are derived from seed FRACTIONS, per graph.**  This is the paper's own thesis: the regime
is controlled by ``|S|/n``, not by the number of seeds.  A fixed ``k`` across graphs is therefore a
design error, and the first calibration run made it --- ``k = 5`` is ``1.05%`` of Congress-Twitter
(475 nodes, unsaturated, where degree is still a fine ranker) and ``0.03%`` of NetHEPT.  The runner
takes ``--fractions`` and computes ``k = round(f * n)`` per graph, so every graph is measured in the
same regime.  The realised ``k`` and ``|S|/n`` are recorded per cell.

**Every arm declares its stopping rule.**  The budget is *at most* ``k``, so an arm may stop early;
comparing arms that stop differently measures the stopping rule, not the selector (audit P1-3.1).

**State acquisition is priced.**  A state-conditioned arm must read ``delta``, which costs cascades.
``mc_cascades`` is the primary cost unit and *includes* the state spend (audit P1-3.2), so an arm
cannot be made to look free by forgetting a second counter.

**The reference is not an oracle.**  ``mc_greedy`` is greedy on an estimator.  At ``--reference-mc``
runs it is neither exact nor globally optimal, so its name carries the budget and its spread is a
reference point, not an upper bound (audit P1-3.3).

**All arms are evaluated on shared windows.**  Pairing is what makes a sub-percent difference
measurable at all; an unpaired comparison in this regime resolves nothing.

**The gate is absolute, and anchored at the measurement resolution by default.**  See
:mod:`grl.evaluation.gate`.  The plan's ``<= 1%`` cannot be evaluated here, and neither can an
arbitrary "one node per thousand": `scripts/audit/calibrate_gate_tolerance.py` measures that
resolving ``|D|/1000`` on Congress-Twitter at ``k = 5`` would need **~790,000 paired trials per
contrast** against the 300 used.  The default anchor is therefore the paired standard error, with
``--tolerance-per-thousand`` available for a caller who wants the stricter question and can pay for
it; ``--tolerance-anchor`` records which was asked.

**The normalisation is an argument.**  ``sum_to_one`` and ``clip_to_one`` differ by up to 50x in
exposure scale on a sparse graph, which decides whether any node is over-exposed at all (audit
P1-3.8), so both are run and both are recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from grl.data import graph_loader  # noqa: E402
from grl.algorithms.sequential_im import (  # noqa: E402
    FILL_BUDGET,
    PATIENCE_2,
    STOP_ON_NON_POSITIVE,
    adaptive_selective_greedy,
    full_oracle_greedy,
    reference_policy_name,
    selective_greedy,
)
from grl.baselines import random_pruning_greedy  # noqa: E402
from grl.data.weights import (  # noqa: E402
    CLIP_TO_ONE,
    SUM_TO_ONE,
    describe_normalisation,
    normalise_in_weights,
)
from grl.diffusion.contract import resolve_target_contract  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.evaluation.gate import (  # noqa: E402
    DEFAULT_CASCADE_SAVING,
    evaluate_gate,
    explain,
    tolerance_from_reference_se,
    tolerance_from_target_fraction,
)
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402
from grl.scoring import (  # noqa: E402
    degree_scores,
    exposure_scores_delta,
    mean_exposure_state,
    rank_by_score,
)

STOPPING_RULES = {
    "fill_budget": FILL_BUDGET,
    "stop_on_non_positive": STOP_ON_NON_POSITIVE,
    "patience_2": PATIENCE_2,
}

#: The static arm that Gate 1(b) is actually about: rank once at S = empty, take the top k, never
#: look at the cascade again.  ``degree_static`` is the same idea with a cost-free score.
SEQUENTIAL_ARM = "delta2_sequential"
#: The same sequential ranking but with a stopping rule fed by the observed spread.  Added because
#: ``delta2_sequential`` uses observation only to re-rank, and re-ranking is the one lever a static
#: arm may already have; deciding *how many* seeds to take is a lever only a sequential arm has.
#: This is the mechanism the paper's own framing proposes, so it is reported alongside rather than
#: substituted silently.
PATIENCE_ARM = "delta2_patience"
STATIC_ARMS = ("degree_static", "delta2_static")

#: The candidate pool must be at least this many times the budget.  Below it, the arms converge on
#: the same set and the cell measures nothing --- the first smoke run had pool=15 against k=95 and
#: produced identical spreads for every arm with a CI of exactly [0, 0].
#:
#: Set to 2 rather than 4 deliberately.  On a graph with a 20% target set the eligible seeds are
#: 0.8n, so a 4x guard caps the reachable seed fraction at |S|/n <= 0.20 and makes the 40% regime
#: --- where this paper's sign flip is strongest --- unreachable by construction.  At 2x the 40%
#: cell is exactly reachable (k = 0.4n needs pool = 0.8n = all eligible nodes).
POOL_TO_BUDGET = 2


@dataclass
class ArmResult:
    """One arm's outcome in one cell."""

    arm: str
    seeds: list[int]
    seed_count: int
    seed_fraction: float
    spread: float
    mc_cascades: int
    state_cascades: int
    state_reads: int
    stopping_rule: str
    stopped_early: bool
    is_reference: bool


@dataclass
class Cell:
    """One (graph, budget, pool, seed, normalisation) cell: every arm, paired."""

    graph: str
    n: int
    mean_degree: float
    normalisation: str
    max_in_weight_total: float
    target_size: int
    budget: int
    pool_size: int
    pool_draw: int
    pool_strategy: str
    #: True when the budget forced the pool to be the whole eligible set, so the pool draws are not
    #: independent replicates of anything.
    pool_covers_all_eligible: bool
    #: pool_size / budget.  The arms need room to differ; 1.0 means no choice at all.
    pool_to_budget: float
    random_seed: int
    reference_mc: int
    eval_mc: int
    state_mc: int
    stopping_rule: str
    arms: dict = field(default_factory=dict)
    #: arm -> per-trial spread, all arms on the SAME evaluation windows
    paired_spread: dict = field(default_factory=dict)
    gate: list = field(default_factory=list)
    seconds: float = 0.0


# ----------------------------------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------------------------------
def run_static_degree(graph, pool, budget, **_) -> tuple[list[int], dict]:
    """Top-k by out-degree.  No cascade, no state, no observation."""
    degree = dict(graph.out_degree())
    seeds = sorted(pool, key=lambda v: (-degree[v], v))[:budget]
    return seeds, {"mc_cascades": 0, "state_cascades": 0, "state_reads": 0,
                   "stopping_rule": "single_shot", "stopped_early": False}


def run_static_delta2(graph, pool, budget, *, state_mc, params, random_seed, **_):
    """Top-k by the analytic score computed **once** at S = empty, then frozen.

    One state read, then no observation.  This is the static counterpart of the sequential
    state-conditioned arm, and the pair of them is what Gate 1(b) compares.

    ``mc_cascades`` must include the state read.  The first version of this arm called
    ``mean_exposure_state`` directly and reported the oracle's counter, which was zero, so the arm
    looked free --- the exact confound (P1-3.2) this project already fixed once.  The read is
    charged explicitly here.
    """
    delta = mean_exposure_state(graph, [], state_mc, random_seed)   # the paid observation
    scores = exposure_scores_delta(graph, pool, delta, set())
    seeds = rank_by_score(pool, scores)[:budget]
    return seeds, {"mc_cascades": state_mc, "state_cascades": state_mc,
                   "state_reads": 1, "stopping_rule": "single_shot", "stopped_early": False}


def run_delta2_sequential(graph, pool, budget, *, state_mc, params, random_seed, **_):
    """Observe the state after every seed, re-rank by the closed form, take the argmax.

    This is the sequential arm Gate 1(b) is about.  Cost is one state read per step; the score
    itself is free.
    """
    selected: list[int] = []
    state_cascades = 0
    for step in range(budget):
        available = [v for v in pool if v not in set(selected)]
        if not available:
            break
        delta = mean_exposure_state(graph, selected, state_mc, random_seed + 1009 * step)
        state_cascades += state_mc
        scores = exposure_scores_delta(graph, available, delta, set(selected))
        selected.append(rank_by_score(available, scores)[0])
    return selected, {"mc_cascades": state_cascades, "state_cascades": state_cascades,
                      "state_reads": len(selected), "stopping_rule": "fill_budget",
                      "stopped_early": False}


def run_mc_greedy(graph, pool, budget, *, oracle_mc, params, random_seed, stopping, **_):
    """Greedy on a Monte-Carlo estimate of the marginal.  The expensive reference.

    Not exact, not globally optimal: at finite ``oracle_mc`` it is an estimator, and the arm's
    reported name carries that budget.
    """
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=oracle_mc,
                                          random_seed=random_seed, params=params)
    result = full_oracle_greedy(pool, budget, oracle, stopping=stopping)
    return result.selected_seeds, {
        "mc_cascades": oracle.stats.mc_cascades,
        "state_cascades": oracle.stats.state_cascades,
        "state_reads": oracle.stats.state_reads,
        "stopping_rule": result.steps[-1]["stopping_rule"] if result.steps else "n/a",
        "stopped_early": any(s.get("stopped_early") for s in result.steps),
    }


def run_random_pruning(graph, pool, budget, *, oracle_mc, params, random_seed, stopping,
                       shortlist, **_):
    """The matched control: same shortlist budget, random shortlist (Gate 2)."""
    oracle = OverexposureMonteCarloOracle(graph, mc_runs=oracle_mc,
                                          random_seed=random_seed, params=params)
    result = random_pruning_greedy(pool, budget, oracle, shortlist_size=shortlist,
                                   random_seed=random_seed, stopping=stopping)
    return result.selected_seeds, {
        "mc_cascades": oracle.stats.mc_cascades,
        "state_cascades": oracle.stats.state_cascades,
        "state_reads": oracle.stats.state_reads,
        "stopping_rule": result.steps[-1]["stopping_rule"] if result.steps else "n/a",
        "stopped_early": any(s.get("stopped_early") for s in result.steps),
    }


def run_selective(graph, pool, budget, *, oracle_mc, params, random_seed, stopping,
                  shortlist, state_mc, **_):
    """Learned/analytic shortlist + exact refinement, at the same shortlist budget as the control."""
    exact = OverexposureMonteCarloOracle(graph, mc_runs=oracle_mc,
                                         random_seed=random_seed, params=params)

    class AnalyticScorer:
        """Scores by the closed form against the state observed so far."""

        def __init__(self):
            self.state_cascades = 0
            self.state_reads = 0

        def score(self, seeds, candidates, step=0):
            delta = mean_exposure_state(graph, list(seeds), state_mc,
                                        random_seed + 7717 * step)
            self.state_cascades += state_mc
            self.state_reads += 1
            values = exposure_scores_delta(graph, list(candidates), delta, set(seeds))
            return {c: v for c, v in zip(candidates, values)}

    scorer = AnalyticScorer()
    result = selective_greedy(pool, budget, scorer, exact, top_m=shortlist, stopping=stopping)
    return result.selected_seeds, {
        "mc_cascades": exact.stats.mc_cascades + scorer.state_cascades,
        "state_cascades": scorer.state_cascades,
        "state_reads": scorer.state_reads,
        "stopping_rule": result.steps[-1]["stopping_rule"] if result.steps else "n/a",
        "stopped_early": any(s.get("stopped_early") for s in result.steps),
    }


def run_adaptive(graph, pool, budget, *, oracle_mc, params, random_seed, stopping,
                 shortlist, state_mc, **_):
    """Adaptive refinement: expand the exact-evaluation frontier until the envelope closes."""
    exact = OverexposureMonteCarloOracle(graph, mc_runs=oracle_mc,
                                         random_seed=random_seed, params=params)

    class AnalyticScorer:
        def __init__(self):
            self.state_cascades = 0
            self.state_reads = 0

        def score(self, seeds, candidates, step=0):
            delta = mean_exposure_state(graph, list(seeds), state_mc,
                                        random_seed + 7717 * step)
            self.state_cascades += state_mc
            self.state_reads += 1
            values = exposure_scores_delta(graph, list(candidates), delta, set(seeds))
            return {c: v for c, v in zip(candidates, values)}

    scorer = AnalyticScorer()
    result = adaptive_selective_greedy(
        pool, budget, scorer, exact,
        initial_m=max(1, shortlist // 2), batch_m=max(1, shortlist // 2),
        max_m=shortlist, stopping=stopping,
    )
    return result.selected_seeds, {
        "mc_cascades": exact.stats.mc_cascades + scorer.state_cascades,
        "state_cascades": scorer.state_cascades,
        "state_reads": scorer.state_reads,
        "stopping_rule": result.steps[-1]["stopping_rule"] if result.steps else "n/a",
        "stopped_early": any(s.get("stopped_early") for s in result.steps),
    }


def run_delta2_patience(graph, pool, budget, *, state_mc, params, random_seed, stopping,
                        eval_mc=None, **_):
    """Sequential ranking **plus a stopping rule driven by the observed spread**.

    Why this arm exists, and how it differs from :func:`run_delta2_sequential`
    ---------------------------------------------------------------------------
    ``delta2_sequential`` uses observation for one thing only: re-ranking.  A static arm already
    ranks well in the unsaturated regime, so if re-ranking is the only lever, sequential selection
    has little to add --- and the first cells of the Gate 1(b) sweep show it adding nothing.

    But observation licenses a second decision that a static arm structurally cannot make: *how many*
    seeds to take.  A static arm commits to ``k`` seeds before seeing anything; a sequential arm can
    stop when the realised spread stops improving.  That is the mechanism the paper's own framing
    proposes ("analytic ranking + a patience stop on the observed spread"), so testing it is testing
    the method as described, not a post-hoc variant.

    Both arms are reported side by side and the distinction is stated in the results, because a
    reader is entitled to know that the stronger sequential arm was also the more complex one.

    Cost: one state read per step (for ranking) **plus one cascade per step** to measure the realised
    spread that the patience rule reads.  Both are charged.
    """
    from grl.diffusion import overexposure as oe

    selected: list[int] = []
    state_cascades = 0
    spread_cascades = 0
    observed: list[float] = []

    for step in range(int(budget)):
        available = [v for v in pool if v not in set(selected)]
        if not available:
            break
        delta = mean_exposure_state(graph, selected, state_mc, random_seed + 1009 * step)
        state_cascades += state_mc
        scores = exposure_scores_delta(graph, available, delta, set(selected))
        chosen = rank_by_score(available, scores)[0]
        selected.append(chosen)

        # the observation the patience rule reads: one cascade on the realised seed set
        rng = random.Random(random_seed + 4241 * step)
        windows = oe.sample_threshold_windows(list(graph.nodes()), rng)
        observed.append(float(oe.run_overexposure(
            graph, list(selected), windows, rng,
            activation_mode=params.activation_mode).spread))
        spread_cascades += 1

        if stopping.should_stop([], observed):
            break

    return selected, {"mc_cascades": state_cascades + spread_cascades,
                      "state_cascades": state_cascades,
                      "state_reads": len(selected),
                      "stopping_rule": f"patience_on_observed_spread+{len(observed)}_spread_reads",
                      "stopped_early": len(selected) < int(budget)}


ARMS = {
    "degree_static": run_static_degree,
    "delta2_static": run_static_delta2,
    SEQUENTIAL_ARM: run_delta2_sequential,
    PATIENCE_ARM: run_delta2_patience,
    "random_pruning": run_random_pruning,
    "selective_analytic": run_selective,
    "adaptive_selective": run_adaptive,
}

#: Cheap arms: no full-pool Monte-Carlo greedy.  Gate 1(b) needs only these, because
#: ``sequential vs static`` is a contrast between ``delta2_sequential`` and
#: ``degree_static`` / ``delta2_static`` --- three arms that cost no per-candidate MC at all.  That
#: separation matters: the full-pool reference costs ``O(k * |pool| * MC)`` cascades, which at
#: ``k = 95``, ``|pool| = 380``, ``MC = 25`` is roughly 900,000 cascades for ONE cell, so bundling
#: it with the gate experiment would make the gate unaffordable for no reason.
CHEAP_ARMS = ("degree_static", "delta2_static", SEQUENTIAL_ARM, PATIENCE_ARM)
#: Arms that need the expensive full-pool Monte-Carlo reference to be meaningful.
REFERENCE_ARMS = ("random_pruning", "selective_analytic", "adaptive_selective")


def reference_arm_name(reference_mc: int) -> str:
    return f"mc_greedy_mc{reference_mc}"


# ----------------------------------------------------------------------------------------------
# pool construction
# ----------------------------------------------------------------------------------------------
def degree_stratified_pool(graph, eligible, size, rng, bands=5):
    """Sample so the pool's degree profile matches the graph's (audit P1-3.5)."""
    if size >= len(eligible):
        return list(eligible)
    degree = dict(graph.out_degree())
    ordered = sorted(eligible, key=lambda v: (degree[v], v))
    n = len(ordered)
    pool: list = []
    for band in range(bands):
        chunk = ordered[band * n // bands:(band + 1) * n // bands]
        if not chunk:
            continue
        take = min(max(1, round(size * len(chunk) / n)), len(chunk))
        pool.extend(rng.sample(chunk, take))
    if len(pool) > size:
        pool = rng.sample(pool, size)
    elif len(pool) < size:
        rest = [v for v in eligible if v not in set(pool)]
        pool.extend(rng.sample(rest, min(size - len(pool), len(rest))))
    return pool


def paired_spreads(graph, nodes, seed_sets: dict[str, list[int]], trials: int, seed: int,
                   params) -> dict[str, list[float]]:
    """Evaluate every arm's seed set on the SAME threshold windows.

    Pairing is not an optimisation here, it is the experiment: in the saturated regime the
    window-draw variance is larger than the between-arm differences, so an unpaired comparison
    cannot see the effect at all.
    """
    from grl.diffusion import overexposure as oe

    names = list(seed_sets)
    per_trial = {name: [] for name in names}
    for offset in range(trials):
        rng = random.Random(seed + offset)
        windows = oe.sample_threshold_windows(nodes, rng)
        for name in names:
            per_trial[name].append(float(
                oe.run_overexposure(graph, list(seed_sets[name]), windows, rng,
                                    activation_mode=params.activation_mode).spread))
    return per_trial


def load_raw(name: str) -> nx.DiGraph:
    """Load a graph **without** normalising its in-weights.

    The obvious loader, ``evaluate_density_degree_signflip.load``, normalises internally.  Feeding
    its output to ``normalise_in_weights`` therefore applies the strategy to an
    already-normalised graph: every strategy becomes a no-op and ``sum_to_one`` and ``clip_to_one``
    produce byte-identical numbers.  That is exactly what the first version of this runner did, and
    the identical columns were the tell.

    Comparing weight strategies requires the file's own weights, so this bypasses the wrapper.
    """
    path, directed = GRAPHS[name]
    graph, _ = graph_loader._parse_graph_file(path, directed, 0.01)
    return graph


def raw_weight_profile(graph: nx.DiGraph) -> dict:
    """What the file actually carries, so a no-op strategy is visible before it is run."""
    from grl.data.weights import in_weight_totals

    totals = sorted(in_weight_totals(graph).values())
    nonzero = [t for t in totals if t > 0.0]
    return {
        "max_in_total": totals[-1] if totals else 0.0,
        "min_nonzero_in_total": nonzero[0] if nonzero else 0.0,
        "nodes_over_one": sum(1 for t in totals if t > 1.0 + 1e-9),
        "nodes_at_one": sum(1 for t in totals if abs(t - 1.0) <= 1e-9),
        "n": len(totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+",
                        default=["congress_twitter", "email_eu_core", "ca_grqc", "nethept"],
                        choices=list(GRAPHS))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.20, 0.40],
                        help="seed FRACTIONS; k is derived per graph as round(f * n) so every "
                             "graph is measured in the same regime. A fixed k across graphs of "
                             "different size is the design error the first calibration run made.")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="explicit budgets, overriding --fractions. Recorded as such.")
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--pool-draws", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260917, 20260918, 20260919])
    parser.add_argument("--normalisations", nargs="+", default=[SUM_TO_ONE, CLIP_TO_ONE],
                        choices=[SUM_TO_ONE, CLIP_TO_ONE])
    parser.add_argument("--target-mode", default="degree-tail",
                        choices=["all", "degree-tail"],
                        help="degree-tail is the source model's formulation: D is the top "
                             "--target-fraction by out-degree and seeds come from V \\ D")
    parser.add_argument("--target-fraction", type=float, default=0.2)
    parser.add_argument("--reference-mc", type=int, default=200,
                        help="Monte-Carlo budget of the greedy REFERENCE. Its name carries this "
                             "number because at finite MC it is neither exact nor optimal.")
    parser.add_argument("--eval-mc", type=int, default=300)
    parser.add_argument("--state-mc", type=int, default=25,
                        help="cascades to read the exposure state; charged to every arm that reads it")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--stopping", default="fill_budget", choices=list(STOPPING_RULES))
    parser.add_argument("--tolerance-anchor", default="resolution",
                        choices=["resolution", "target_fraction"],
                        help="'resolution' = one paired standard error of the reference, the "
                             "smallest difference this experiment can resolve; 'target_fraction' = "
                             "--tolerance-per-thousand target nodes, which is stricter and may be "
                             "unaffordable (see scripts/audit/calibrate_gate_tolerance.py)")
    parser.add_argument("--tolerance-per-thousand", type=float, default=1.0,
                        help="absolute quality tolerance in target nodes per 1000 target nodes; "
                             "only used when --tolerance-anchor target_fraction")
    parser.add_argument("--required-saving", type=float, default=DEFAULT_CASCADE_SAVING)
    parser.add_argument("--arms", nargs="+", default=None,
                        choices=list(ARMS) + ["reference"],
                        help="which arms to run. Default: every cheap arm plus 'reference', which "
                             "expands to the full-pool MC-greedy reference and the three arms that "
                             "need it. Pass --arms delta2_sequential degree_static delta2_static for "
                             "Gate 1(b) alone: the full-pool reference costs O(k*|pool|*MC) and "
                             "would otherwise dominate the run.")
    parser.add_argument("--max-seeds", type=int, default=1200,
                        help="skip cells whose realised k exceeds this. The sequential arm reads the "
                             "state once per seed, so its cost is O(k * state_mc) cascades; past a "
                             "few thousand seeds that is hours per cell on a 15k-node graph. "
                             "Skipped cells are recorded with the reason, not silently dropped.")
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "go_no_go.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    stopping = STOPPING_RULES[args.stopping]
    params = resolve_overexposure_params(None)
    reference = reference_arm_name(args.reference_mc)
    if args.arms is None:
        selected_arms = dict(ARMS)
        want_reference = True
    else:
        want_reference = "reference" in args.arms
        selected_arms = {k: v for k, v in ARMS.items()
                         if k in args.arms or (want_reference and k in REFERENCE_ARMS)}
    if not selected_arms:
        raise SystemExit("no arms selected")
    print(f"arms: {', '.join(selected_arms)}"
          f"{'  + full-pool reference' if want_reference else '  (no full-pool reference)'}")
    cells: list[Cell] = []
    skipped: list[dict] = []
    failures: list[dict] = []

    def flush() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "script": Path(__file__).name,
            "contract": {
                "target_mode": args.target_mode,
                "target_fraction": args.target_fraction,
                "budget_is_at_most": True,
                "stopping_rule": args.stopping,
                "state_acquisition_is_priced": True,
                "reference_is_exact": False,
                "reference_mc": args.reference_mc,
            },
            "design": {
                "graphs": args.graphs,
                "fractions": None if args.budgets else args.fractions,
                "explicit_budgets": args.budgets,
                "pool_size": args.pool_size, "pool_draws": args.pool_draws,
                "random_seeds": args.seeds, "normalisations": args.normalisations,
                "eval_mc": args.eval_mc, "state_mc": args.state_mc,
                "shortlist": args.shortlist,
                "sequential_arm": SEQUENTIAL_ARM, "static_arms": list(STATIC_ARMS),
                "reference_arm": reference,
            },
            "gate": {
                "criterion": "absolute pair tolerance in target nodes + required cascade saving",
                "relative_form_is_report_only": True,
                "tolerance_anchor": args.tolerance_anchor,
                "tolerance_per_thousand": (args.tolerance_per_thousand
                                           if args.tolerance_anchor == "target_fraction" else None),
                "required_saving": args.required_saving,
                "reason_absolute": (
                    "1% of a near-zero marginal is below the measurement noise, and even "
                    "|D|/1000 needs ~790k paired trials on Congress-Twitter k=5; see "
                    "docs/results/gate_calibration.json and grl/evaluation/gate.py"
                ),
            },
            "cells": [asdict(c) for c in cells],
            "skipped_cells": skipped,
            "arm_failures": failures,
        }, indent=2), encoding="utf-8")

    started_all = time.time()
    for graph_name in args.graphs:
        raw_graph = load_raw(graph_name)
        profile = raw_weight_profile(raw_graph)
        print(f"  [{graph_name} raw in-weights] max_total={profile['max_in_total']:.4f} "
              f"min_nonzero={profile['min_nonzero_in_total']:.4f} "
              f"nodes_over_1={profile['nodes_over_one']} nodes_at_1={profile['nodes_at_one']}"
              f"/{profile['n']}", flush=True)
        if profile["nodes_over_one"] == 0 and profile["nodes_at_one"] == profile["n"]:
            print(f"  NOTE: {graph_name} already has every in-weight total at exactly 1, so "
                  f"sum_to_one and clip_to_one are identical here and the strategy comparison is "
                  f"degenerate on this graph.  Reported rather than silently duplicated.")
        for normalisation in args.normalisations:
            graph = normalise_in_weights(raw_graph.copy(), normalisation)
            scale = describe_normalisation(graph)
            nodes = list(graph.nodes())
            n = len(nodes)
            mean_degree = 2 * graph.number_of_edges() / n

            # Budgets come from fractions so that every graph is measured in the same regime.
            if args.budgets:
                budget_spec = [(None, int(k)) for k in args.budgets]
            else:
                budget_spec = [(f, max(1, int(round(f * n)))) for f in args.fractions]

            contract = resolve_target_contract(graph, args.target_mode, args.target_fraction,
                                               max(k for _, k in budget_spec))
            eligible = contract.objective.legal_candidates(graph)
            target_size = len(contract.objective.target_set)
            print(f"=== {graph_name} n={n} <k>={mean_degree:.2f} normalisation={normalisation} "
                  f"|D|={target_size} eligible={len(eligible)}", flush=True)

            for fraction, budget in budget_spec:
                realised_fraction = budget / n
                if fraction is not None and abs(realised_fraction - fraction) > 0.01:
                    print(f"  NOTE: requested |S|/n={fraction:.2f} realised "
                          f"{realised_fraction:.4f} after rounding k={budget} on n={n}")
                for draw in range(args.pool_draws):
                    for seed in args.seeds:
                        rng = random.Random(seed + 31 * budget + 101 * draw)
                        # The pool is the domain the arms choose from, and it shrinks as seeds are
                        # taken.  If the pool is not comfortably larger than the budget every arm
                        # converges on the same set and the cell cannot discriminate anything ---
                        # the first smoke run had pool=15 against k=95 and produced identical
                        # spreads with a CI of exactly [0, 0] for every arm.  So the pool is
                        # widened to at least POOL_TO_BUDGET x k, and a cell that cannot reach a
                        # usable ratio is recorded as skipped rather than silently measured.
                        wanted = max(args.pool_size, POOL_TO_BUDGET * budget)
                        pool_size = min(wanted, len(eligible))
                        if pool_size < POOL_TO_BUDGET * budget:
                            print(f"  SKIP k={budget} draw={draw} seed={seed}: only {pool_size} "
                                  f"eligible nodes, need {POOL_TO_BUDGET * budget} "
                                  f"({POOL_TO_BUDGET}x the budget) for the arms to differ",
                                  flush=True)
                            skipped.append({
                                "graph": graph_name, "normalisation": normalisation,
                                "budget": budget, "fraction": realised_fraction,
                                "pool_draw": draw, "random_seed": seed,
                                "eligible": len(eligible), "needed_pool": POOL_TO_BUDGET * budget,
                                "reason": "pool too small relative to the budget; every arm would "
                                          "select the same set",
                            })
                            continue
                        if budget > args.max_seeds:
                            # The sequential arm reads the state once per seed, so its cost is
                            # O(k * state_mc) cascades.  Past a few thousand seeds that is hours per
                            # cell on a 15k-node graph, and the cell is not worth it: |S|/n is what
                            # matters, and a smaller graph reaches the same fraction for less.
                            print(f"  SKIP k={budget} (|S|/n={realised_fraction:.3f}) on "
                                  f"{graph_name}: exceeds --max-seeds {args.max_seeds}; the "
                                  f"sequential arm's cost is O(k * state_mc)", flush=True)
                            skipped.append({
                                "graph": graph_name, "normalisation": normalisation,
                                "budget": budget, "fraction": realised_fraction,
                                "pool_draw": draw, "random_seed": seed,
                                "eligible": len(eligible),
                                "reason": f"k > --max-seeds ({args.max_seeds}); sequential state "
                                          f"reads cost O(k * state_mc)",
                            })
                            continue
                        pool = degree_stratified_pool(graph, eligible, pool_size, rng)
                        # When the budget forces the pool to be the entire eligible set, the pool
                        # draws are not independent: there is nothing left to vary.  Observed on
                        # congress_twitter at |S|/n = 0.20, where 4 x 95 = 380 equals the number of
                        # eligible nodes and all three draws gave identical spreads.  Recorded
                        # rather than reported as three independent replicates.
                        pool_covers_all = pool_size >= len(eligible)
                        contract.objective.check_seed_eligibility(pool)

                        t0 = time.time()
                        arms: dict[str, ArmResult] = {}
                        seed_sets: dict[str, list[int]] = {}
                        meta: dict[str, dict] = {}

                        # the reference first: it defines the cost baseline
                        if want_reference:
                            try:
                                seeds_ref, info_ref = run_mc_greedy(
                                    graph, pool, budget, oracle_mc=args.reference_mc,
                                    params=params, random_seed=seed, stopping=stopping)
                                seed_sets[reference] = seeds_ref
                                meta[reference] = info_ref
                            except Exception as exc:      # a failing arm must not lose the cell
                                print(f"    !! reference failed: {type(exc).__name__}: {exc}",
                                      flush=True)
                                want_reference = False

                        for arm_name, fn in selected_arms.items():
                            try:
                                seeds_arm, info = fn(
                                    graph, pool, budget, oracle_mc=args.reference_mc,
                                    params=params, random_seed=seed, stopping=stopping,
                                    shortlist=args.shortlist, state_mc=args.state_mc,
                                    eval_mc=args.eval_mc)
                            except Exception as exc:
                                # A failing arm must be loud, not silently absent: an arm that
                                # disappears from the table looks like an arm that was never run,
                                # and the first version of delta2_patience did exactly that when its
                                # signature did not match the dispatcher.
                                print(f"    !! {arm_name} FAILED and is absent from this cell: "
                                      f"{type(exc).__name__}: {exc}", flush=True)
                                failures.append({"graph": graph_name, "arm": arm_name,
                                                 "budget": budget, "error": repr(exc)})
                                continue
                            seed_sets[arm_name] = seeds_arm
                            meta[arm_name] = info

                        paired = paired_spreads(graph, nodes, seed_sets, args.eval_mc,
                                                seed + 5000, params)

                        for arm_name, seeds_arm in seed_sets.items():
                            info = meta[arm_name]
                            arms[arm_name] = ArmResult(
                                arm=arm_name, seeds=list(seeds_arm),
                                seed_count=len(seeds_arm),
                                seed_fraction=len(seeds_arm) / n if n else 0.0,
                                spread=statistics.fmean(paired[arm_name]),
                                mc_cascades=int(info["mc_cascades"]),
                                state_cascades=int(info["state_cascades"]),
                                state_reads=int(info["state_reads"]),
                                stopping_rule=str(info["stopping_rule"]),
                                stopped_early=bool(info["stopped_early"]),
                                is_reference=(arm_name == reference),
                            )

                        gate_rows = []
                        if reference in paired:
                            ref_trials = paired[reference]
                            ref_spread = statistics.fmean(ref_trials)
                            ref_cost = arms[reference].mc_cascades
                            # Anchor the tolerance.  'resolution' is one paired standard error of
                            # the reference against the cheapest arm, i.e. the smallest difference
                            # this experiment can actually see; 'target_fraction' is the stricter
                            # |D|/1000 question, which the calibration shows is unaffordable here.
                            if args.tolerance_anchor == "resolution":
                                ref_vals = list(ref_trials)
                                ref_se = (statistics.pstdev(ref_vals) / math.sqrt(len(ref_vals))
                                          if len(ref_vals) > 1 else float("inf"))
                                tolerance = tolerance_from_reference_se(ref_se, multiple=1.0)
                                anchor = "reference_se"
                            else:
                                tolerance = tolerance_from_target_fraction(
                                    target_size, per_thousand=args.tolerance_per_thousand)
                                anchor = f"target_fraction:{args.tolerance_per_thousand}"
                            for arm_name in paired:
                                if arm_name == reference:
                                    continue
                                gains = [r - m for r, m in zip(ref_trials, paired[arm_name])]
                                row = evaluate_gate(
                                    graph=graph_name, budget=budget, reference=reference,
                                    method=arm_name, paired_gains=gains,
                                    reference_spread=ref_spread,
                                    method_spread=statistics.fmean(paired[arm_name]),
                                    reference_cascades=ref_cost,
                                    method_cascades=arms[arm_name].mc_cascades,
                                    abs_tolerance=tolerance,
                                    tolerance_anchor=anchor,
                                    required_saving=args.required_saving,
                                )
                                gate_rows.append(row.as_dict())

                        cell = Cell(
                            graph=graph_name, n=n, mean_degree=mean_degree,
                            normalisation=normalisation,
                            max_in_weight_total=scale["max_in_weight_total"],
                            target_size=target_size, budget=budget, pool_size=len(pool),
                            pool_draw=draw, pool_strategy="degree_stratified",
                            pool_covers_all_eligible=pool_covers_all,
                            pool_to_budget=(len(pool) / budget if budget else float("nan")),
                            random_seed=seed, reference_mc=args.reference_mc,
                            eval_mc=args.eval_mc, state_mc=args.state_mc,
                            stopping_rule=args.stopping,
                            arms={k: asdict(v) for k, v in arms.items()},
                            paired_spread={k: [round(x, 4) for x in v] for k, v in paired.items()},
                            gate=gate_rows,
                            seconds=time.time() - t0,
                        )
                        cells.append(cell)
                        best_static = max(
                            (arms[a].spread for a in STATIC_ARMS if a in arms), default=float("nan"))
                        seq = arms[SEQUENTIAL_ARM].spread if SEQUENTIAL_ARM in arms else float("nan")
                        lift = ((seq - best_static) / abs(best_static) * 100.0
                                if best_static and math.isfinite(best_static) else float("nan"))
                        ref_txt = (f"ref={arms[reference].spread:8.2f}"
                                   f"({arms[reference].mc_cascades:>7})"
                                   if reference in arms else "ref=  (none)")
                        seq_txt = (f"seq={seq:8.2f}({arms[SEQUENTIAL_ARM].mc_cascades:>6})"
                                   if SEQUENTIAL_ARM in arms else "seq=  (n/a)")
                        print(f"  k={budget:>3} |S|/n={realised_fraction:.3f} draw={draw} "
                              f"seed={seed} {ref_txt} {seq_txt} "
                              f"static={best_static:8.2f} lift={lift:+6.2f}% "
                              f"[{cell.seconds:.0f}s]", flush=True)
                        flush()

    # ---------------- summary ----------------
    print()
    print("=" * 112)
    print("GATE 1(b): sequential vs best static")
    print("=" * 112)
    print("  The pre-registered test is the arm `delta2_sequential`: it uses observation only to")
    print("  re-rank, which is the weakest possible use of it and therefore the most conservative")
    print("  reading of the gate.  `delta2_patience` additionally uses observation to decide how")
    print("  many seeds to take --- a lever a static arm structurally lacks --- and is reported")
    print("  alongside because it is the mechanism the paper's own framing proposes.  A reader is")
    print("  entitled to see both; a PASS on the patience arm alone is not a PASS on the gate.")
    for arm_label, arm_key in (("re-rank only", SEQUENTIAL_ARM),
                               ("re-rank + observed stop", PATIENCE_ARM)):
        lifts: list[float] = []
        per_cell: list[tuple] = []
        for cell in cells:
            if arm_key not in cell.arms:
                continue
            statics = [cell.arms[a]["spread"] for a in STATIC_ARMS if a in cell.arms]
            if not statics:
                continue
            best = max(statics)
            if abs(best) < 1e-9:
                continue
            lift = (cell.arms[arm_key]["spread"] - best) / abs(best)
            lifts.append(lift)
            per_cell.append((cell.graph, cell.budget, cell.normalisation, lift,
                             cell.arms[arm_key]["mc_cascades"]))
        if not lifts:
            continue
        positive = sum(1 for x in lifts if x > 0)
        median_lift = statistics.median(lifts)
        print()
        print(f"  --- {arm_label}  ({arm_key}) ---")
        print(f"  cells compared          : {len(lifts)}")
        print(f"  mean relative lift      : {statistics.fmean(lifts)*100:+.2f}%")
        print(f"  median relative lift    : {median_lift*100:+.2f}%")
        print(f"  cells with positive lift: {positive}/{len(lifts)}")
        print(f"  Gate 1(b) threshold     : +5.00%")
        print(f"  VERDICT (median)        : "
              f"{'PASSES' if median_lift > 0.05 else 'DOES NOT PASS'}")
        # per-fraction breakdown, because the regime is the fraction and a pooled median hides it
        by_fraction: dict[float, list[float]] = {}
        for cell in cells:
            if arm_key not in cell.arms:
                continue
            statics = [cell.arms[a]["spread"] for a in STATIC_ARMS if a in cell.arms]
            if not statics:
                continue
            best = max(statics)
            if abs(best) < 1e-9:
                continue
            frac = round(cell.arms[arm_key]["seed_fraction"], 3)
            by_fraction.setdefault(frac, []).append(
                (cell.arms[arm_key]["spread"] - best) / abs(best))
        for frac in sorted(by_fraction):
            vals = by_fraction[frac]
            print(f"    |S|/n={frac:<6.3f} n={len(vals):<4} "
                  f"median lift {statistics.median(vals)*100:+7.2f}%  "
                  f"positive {sum(1 for x in vals if x > 0)}/{len(vals)}")
    print("  NOTE: Gate 1(b) is decided on the median across cells; the per-cell values are in the")
    print("        artifact.  A single favourable cell is not evidence.")

    print()
    print("=" * 112)
    print("QUALITY--COST, absolute gate")
    print("=" * 112)
    tally: dict[str, dict[str, int]] = {}
    for cell in cells:
        for row in cell.gate:
            slot = tally.setdefault(row["method"], {"PASS": 0, "FAIL": 0, "UNDECIDED": 0})
            slot[row["verdict"]] += 1
    print(f"  {'method':<22}{'PASS':>7}{'FAIL':>7}{'UNDECIDED':>11}")
    for method, counts in sorted(tally.items()):
        print(f"  {method:<22}{counts['PASS']:>7}{counts['FAIL']:>7}{counts['UNDECIDED']:>11}")
    print()
    for cell in cells[:3]:
        for row in cell.gate[:1]:
            print("  sample:", explain(type("V", (), row)()) if False else
                  f"{cell.graph} k={cell.budget} {row['method']}: {row['verdict']} "
                  f"CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] tol={row['abs_tolerance']:.3f} "
                  f"saving={row['cascade_saving']*100:.0f}%")
    print()
    print(f"  total wall time: {time.time() - started_all:.0f}s over {len(cells)} cells")
    flush()
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
