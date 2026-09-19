"""Candidate-screening headroom diagnosis on three fixed configurations.

The question
------------
Is there a worthwhile candidate difference to learn, and does the target-aware ``delta2`` score
throw away good candidates?  Four screening methods face the **identical** existing seed set and the
**identical** candidate list, and the diagnosis asks whether the score's top-8 contains the
candidate an independent Monte-Carlo pass would have picked.

Three configurations only, no cross-graph pooling:

===============  ===================  ==================================================
graph            existing seeds       why this configuration
===============  ===================  ==================================================
Congress-Twitter 1% of nodes          where the current score performs poorly
email-Eu-core    5% of nodes          re-check the existing positive ranking signal
ca-GrQc          1% of nodes          re-check the target-aware score's improvement
===============  ===================  ==================================================

Construction (graph, weight normalisation, target set, seeds, pool) is reused verbatim from
``scripts/audit/diagnose_scorer.py`` so the two diagnoses are comparable, with the same base seed.

Two independent Monte-Carlo batches
-----------------------------------
**Batch 1** (``BATCH1_NS``, MC = 1000).  Within a trial the threshold windows are drawn once and
shared, and each candidate's marginal is ``|P_final(S ∪ {c}) ∩ D| - |P_final(S) ∩ D|`` --- a paired
difference in the contracted objective, the same quantity the arms are scored on.  Batch 1 is used
for **every selection decision**: the full-pool MC best candidate, and each method's shortlist best.

**Batch 2** (``BATCH2_NS``, MC = 1000).  A disjoint random stream, used only to *evaluate* the
candidates batch 1 selected.  **Nothing is re-selected on batch 2.**  Without this split the
selection and the scorecard share their noise, and every method looks better than it is.

The full-pool winner is called the **MC reference candidate**, never the true optimum: it is the best
of 50 candidates under one finite-sample pass, and on an independent pass it can be overtaken.  The
differences are therefore reported signed, and **not truncated at zero**.

Cost accounting
---------------
The state read is charged.  Batch 1, batch 2 and the state acquisition draw from disjoint streams,
and the disjointness is checked rather than assumed --- see ``--check-only``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from diagnose_scorer import true_marginals  # noqa: E402  (construction reuse)
from evaluate_density_degree_signflip import GRAPHS  # noqa: E402
from go_no_go import degree_stratified_pool, load_raw  # noqa: E402
from grl.data.weights import SUM_TO_ONE, normalise_in_weights  # noqa: E402
from grl.diffusion.contract import resolve_target_contract  # noqa: E402
from grl.oracle import TargetedMonteCarloOracle, trial_seeds  # noqa: E402
from grl.scoring import exposure_scores_delta, rank_by_score  # noqa: E402

#: Stream namespaces.  Disjoint by construction because trial_seeds multiplies the base by a stride
#: far larger than any trial count; the check below verifies it rather than trusting it.
STATE_NS = 900_000
BATCH1_NS = 1_100_000
BATCH2_NS = 1_300_000

CONFIGS = [
    ("congress_twitter", 0.01, "where the current score performs poorly"),
    ("email_eu_core", 0.05, "re-check the existing positive ranking signal"),
    ("ca_grqc", 0.01, "re-check the target-aware score's improvement"),
]

SHORTLIST = 8
RANDOM_REPEATS = 100


# --------------------------------------------------------------------------------------------
# paired Monte-Carlo marginals in the contracted objective
# --------------------------------------------------------------------------------------------
def paired_marginals_in_target(graph, contract, seeds, candidates, trials, base_seed):
    """Per-trial ``|P_final(S ∪ {c}) ∩ D| - |P_final(S) ∩ D|`` for every candidate.

    Returns ``(per_trial, base_counts)`` where ``per_trial[c]`` is a list of ``trials`` paired
    differences and ``base_counts`` the per-trial ``|P_final(S) ∩ D|``.  Windows are drawn once per
    trial and shared across all candidates, so the differences carry no window-draw noise.

    Every level is checked against ``[0, |D|]``.  Marginals are **not** range-checked to be
    non-negative: a negative marginal is the phenomenon, not an error --- adding a seed can push a
    target past its upper threshold and reduce the number of positive targets.
    """
    from grl.diffusion import overexposure as oe

    target = set(contract.objective.target_set)
    limit = len(target)
    nodes = list(graph.nodes())
    per_trial: dict[int, list[float]] = {int(c): [] for c in candidates}
    base_counts: list[float] = []

    for trial_seed in trial_seeds(base_seed, trials):
        rng = random.Random(trial_seed)
        windows = oe.sample_threshold_windows(
            nodes, rng, threshold_law=contract.diffusion.threshold_law,
            window_lo=contract.diffusion.window_lo)
        run = oe.run_overexposure(graph, list(seeds), windows, rng,
                                  activation_mode=contract.diffusion.activation_mode)
        base = float(len(run.positive & target))
        if not (-1e-9 <= base <= limit + 1e-9):
            raise ValueError(f"base target count {base} outside [0, |D|] = [0, {limit}]")
        base_counts.append(base)
        for candidate in candidates:
            extended = oe.run_overexposure(graph, [*seeds, candidate], windows, rng,
                                           activation_mode=contract.diffusion.activation_mode)
            count = float(len(extended.positive & target))
            if not (-1e-9 <= count <= limit + 1e-9):
                raise ValueError(f"target count {count} outside [0, |D|] = [0, {limit}]")
            per_trial[int(candidate)].append(count - base)
    return per_trial, base_counts


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    ordered = sorted(values)
    return {
        "mean": mean, "sd": sd, "se": se, "n": n,
        "ci95_low": mean - 1.96 * se if math.isfinite(se) else float("-inf"),
        "ci95_high": mean + 1.96 * se if math.isfinite(se) else float("inf"),
        "min": ordered[0], "p05": ordered[max(0, int(0.05 * n) - 1)],
        "median": statistics.median(ordered),
        "p95": ordered[min(n - 1, int(0.95 * n))], "max": ordered[-1],
        "negative_share": sum(1 for v in values if v < 0) / n if n else float("nan"),
    }


def paired_difference(a: list[float], b: list[float]) -> dict:
    """Paired difference ``a - b`` on shared trials, with its 95% CI."""
    if len(a) != len(b) or not a:
        return {}
    diffs = [x - y for x, y in zip(a, b)]
    return summarise(diffs)


def random_shortlist_statistics(per_trial_batch1, per_trial_batch2, candidates,
                                size: int, repeats: int, seed: int) -> dict:
    """Draw ``repeats`` random shortlists of ``size`` and record what each one would have chosen.

    The two sources of variation are reported separately and must not be conflated:

    * **shortlist-draw variation** --- which 8 candidates the draw happened to contain, captured by
      the quantiles over repeats of the chosen candidate's batch-2 marginal;
    * **Monte-Carlo estimation uncertainty** --- the paired CI of that marginal within a repeat,
      carried through from batch 2.

    Batch-2 results are reused across repeats: the draw selects a candidate, it does not re-simulate
    one, so a single batch-2 pass covers every repeat.
    """
    rng = random.Random(seed)
    choices: list[dict] = []
    for _ in range(repeats):
        shortlist = rng.sample(candidates, min(size, len(candidates)))
        best = max(shortlist, key=lambda c: (statistics.fmean(per_trial_batch1[int(c)]), -c))
        chosen_b2 = per_trial_batch2[int(best)]
        choices.append({
            "chosen": int(best),
            "shortlist": sorted(int(c) for c in shortlist),
            "batch1_mean": statistics.fmean(per_trial_batch1[int(best)]),
            "batch2_summary": summarise(chosen_b2),
        })
    means = [c["batch2_summary"]["mean"] for c in choices]
    return {
        "repeats": repeats,
        "shortlist_size": size,
        "batch2_mean_over_repeats": statistics.fmean(means),
        "batch2_mean_quantiles": {
            "min": min(means), "p05": sorted(means)[max(0, int(0.05 * repeats) - 1)],
            "median": statistics.median(means),
            "p95": sorted(means)[min(repeats - 1, int(0.95 * repeats))], "max": max(means),
        },
        # median width of the per-repeat MC CI, so the two noise sources are readable side by side
        "median_mc_ci_width": statistics.median(
            c["batch2_summary"]["ci95_high"] - c["batch2_summary"]["ci95_low"] for c in choices),
        "mean_chosen_batch1": statistics.fmean(c["batch1_mean"] for c in choices),
        "choices": choices,
    }


def check_streams(trials: int, seed: int) -> dict:
    """The pre-run check: the three streams must not overlap."""
    streams = {
        "state": set(trial_seeds(seed + STATE_NS, max(trials, 1))),
        "batch1": set(trial_seeds(seed + BATCH1_NS, trials)),
        "batch2": set(trial_seeds(seed + BATCH2_NS, trials)),
    }
    overlaps = {
        "state_batch1": len(streams["state"] & streams["batch1"]),
        "state_batch2": len(streams["state"] & streams["batch2"]),
        "batch1_batch2": len(streams["batch1"] & streams["batch2"]),
    }
    return {"trials": trials, "overlaps": overlaps,
            "disjoint": all(v == 0 for v in overlaps.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+",
                        default=[f"{g}:{f}" for g, f, _ in CONFIGS],
                        help="graph:fraction pairs; defaults to the three specified configurations")
    parser.add_argument("--pool-size", type=int, default=120)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--shortlist", type=int, default=SHORTLIST)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--mc", type=int, default=1000,
                        help="trials per batch; the same budget is used for both")
    parser.add_argument("--target-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "shortlist_headroom.json")
    parser.add_argument("--check-only", action="store_true",
                        help="run the two pre-run checks and exit")
    args = parser.parse_args()

    checks = check_streams(args.mc, args.random_seed)
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return 0 if checks["disjoint"] else 1
    if not checks["disjoint"]:
        raise SystemExit(f"stream overlap: {checks['overlaps']}")

    try:
        code_version = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()
    except Exception:
        code_version = "unknown"

    started = time.time()
    blocks: list[dict] = []
    for spec in args.configs:
        name, fraction_text = spec.split(":")
        fraction = float(fraction_text)
        why = next((w for g, f, w in CONFIGS if g == name and abs(f - fraction) < 1e-9), "")

        graph = normalise_in_weights(load_raw(name).copy(), SUM_TO_ONE)
        n = len(graph.nodes())
        k = max(1, int(round(fraction * n)))
        contract = resolve_target_contract(graph, "degree-tail", args.target_fraction, k)
        target = set(contract.objective.target_set)
        eligible = contract.objective.legal_candidates(graph)

        rng = random.Random(args.random_seed + 31 * k)
        pool = degree_stratified_pool(graph, eligible, min(args.pool_size, len(eligible)), rng)
        seeds = sorted(pool, key=lambda v: -graph.out_degree(v))[:k]
        candidates = [v for v in pool if v not in set(seeds)][:args.candidates]
        if len(candidates) < args.shortlist:
            print(f"=== {name} |S|/n={fraction}: only {len(candidates)} candidates, skipped")
            blocks.append({"graph": name, "fraction": fraction, "skipped": True,
                           "reason": f"only {len(candidates)} candidates"})
            continue

        print(f"=== {name} n={n} |S|={k} (|S|/n={k/n:.3f}) |D|={len(target)} "
              f"eligible={len(eligible)} candidates={len(candidates)}", flush=True)

        # ---- state reads (charged, own stream) ----
        oracle = TargetedMonteCarloOracle(graph, contract, mc_runs=args.mc,
                                          random_seed=args.random_seed + STATE_NS)
        state_current = oracle.state(seeds, step=0)
        state_empty = oracle.state([], step=1)
        state_cascades = oracle.stats.state_cascades

        degree_score = [float(graph.out_degree(v)) for v in candidates]
        static_score = exposure_scores_delta(graph, candidates, state_empty, set(),
                                            target_set=target)
        state_score = exposure_scores_delta(graph, candidates, state_current, set(seeds),
                                           target_set=target)

        # ---- batch 1: selection only ----
        t0 = time.time()
        b1, base1 = paired_marginals_in_target(graph, contract, seeds, candidates, args.mc,
                                              args.random_seed + BATCH1_NS)
        # ---- batch 2: evaluation only, disjoint stream ----
        b2, base2 = paired_marginals_in_target(graph, contract, seeds, candidates, args.mc,
                                              args.random_seed + BATCH2_NS)
        batch_seconds = time.time() - t0

        b1_mean = {c: statistics.fmean(b1[int(c)]) for c in candidates}
        reference = max(candidates, key=lambda c: (b1_mean[c], -c))

        by_id = {int(c): i for i, c in enumerate(candidates)}
        methods: dict[str, dict] = {}
        shortlists = {
            "degree": rank_by_score(candidates, degree_score),
            "static_delta2": rank_by_score(candidates, static_score),
            "state_delta2": rank_by_score(candidates, state_score),
        }
        for label, order in shortlists.items():
            short = order[:args.shortlist]
            chosen = max(short, key=lambda c: (b1_mean[c], -c))
            methods[label] = {
                "top8": [int(c) for c in short],
                "contains_reference": reference in short,
                "chosen": int(chosen),
                "chosen_batch1_summary": summarise(b1[int(chosen)]),
                "chosen_batch2_summary": summarise(b2[int(chosen)]),
                "gain_vs_reference_batch2": paired_difference(b2[int(chosen)], b2[int(reference)]),
                "gain_vs_reference_batch1": paired_difference(b1[int(chosen)], b1[int(reference)]),
            }

        methods["random"] = random_shortlist_statistics(
            b1, b2, candidates, args.shortlist, args.random_repeats, args.random_seed + 17)

        # score-level contrast: does the target-aware score point at the reference?
        def rank_of(score_list, target_candidate):
            order = rank_by_score(candidates, score_list)
            return order.index(target_candidate) + 1

        block = {
            "graph": name, "fraction": fraction, "why": why,
            "n": n, "seed_size": k, "seed_fraction": k / n,
            "target_size": len(target), "target_set": sorted(int(t) for t in target),
            "existing_seeds": [int(s) for s in seeds],
            "candidates": [int(c) for c in candidates],
            "candidate_count": len(candidates),
            "shortlist_size": args.shortlist,
            "pool_size": len(pool),
            "mc_per_batch": args.mc,
            "state_cascades": state_cascades,
            "base_counts": {
                "batch1": summarise(base1), "batch2": summarise(base2),
                "batch1_vs_batch2": paired_difference(base1, base2),
            },
            "reference_candidate": {
                "candidate": int(reference),
                "note": "MC reference candidate: best of the pool under batch 1. NOT the true "
                        "optimum; it can be overtaken on an independent pass.",
                "batch1_summary": summarise(b1[int(reference)]),
                "batch2_summary": summarise(b2[int(reference)]),
                "degree_score": degree_score[by_id[int(reference)]],
                "static_delta2_score": static_score[by_id[int(reference)]],
                "state_delta2_score": state_score[by_id[int(reference)]],
                "degree_rank": rank_of(degree_score, reference),
                "static_delta2_rank": rank_of(static_score, reference),
                "state_delta2_rank": rank_of(state_score, reference),
            },
            "methods": methods,
            "batch_seconds": batch_seconds,
            # candidate-level table, including the per-trial paired marginals for later recomputation
            "candidate_rows": [
                {
                    "candidate": int(c),
                    "degree": degree_score[i],
                    "static_delta2": static_score[i],
                    "state_delta2": state_score[i],
                    "batch1": summarise(b1[int(c)]),
                    "batch2": summarise(b2[int(c)]),
                    "batch1_paired": b1[int(c)],
                    "batch2_paired": b2[int(c)],
                    "is_reference": int(c) == int(reference),
                    "b1_rank": None,
                }
                for i, c in enumerate(candidates)
            ],
        }
        order_b1 = sorted(candidates, key=lambda c: (-b1_mean[c], c))
        for row in block["candidate_rows"]:
            row["b1_rank"] = order_b1.index(row["candidate"]) + 1
        blocks.append(block)

        print(f"  reference candidate {reference}: batch1 {b1_mean[reference]:+.3f} "
              f"batch2 {statistics.fmean(b2[int(reference)]):+.3f} "
              f"(degree rank {block['reference_candidate']['degree_rank']}/{len(candidates)}, "
              f"state-delta2 rank {block['reference_candidate']['state_delta2_rank']})")
        for label in ("degree", "static_delta2", "state_delta2"):
            m = methods[label]
            gain = m["gain_vs_reference_batch2"]
            print(f"    {label:<15} chosen {m['chosen']:<6} "
                  f"contains_ref={str(m['contains_reference']):<6} "
                  f"batch2 {m['chosen_batch2_summary']['mean']:+.3f} "
                  f"gain vs ref {gain['mean']:+.3f} "
                  f"[{gain['ci95_low']:+.3f}, {gain['ci95_high']:+.3f}]")
        rnd = methods["random"]
        print(f"    {'random':<15} mean over {args.random_repeats} draws "
              f"{rnd['batch2_mean_over_repeats']:+.3f} "
              f"[{rnd['batch2_mean_quantiles']['min']:+.3f}, "
              f"{rnd['batch2_mean_quantiles']['max']:+.3f}]")
        print(flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "script": Path(__file__).name,
        "code_version": code_version,
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "objective": "F_D(S) = E[|P_final(S) ∩ D|]  (contracted objective)",
        "stream_namespaces": {"state": STATE_NS, "batch1": BATCH1_NS, "batch2": BATCH2_NS},
        "stream_check": checks,
        "range_checks": {
            "levels_asserted_in_range": True,
            "marginals_may_be_negative": True,
            "note": "every |P_final ∩ D| must lie in [0, |D|]; a marginal is a difference of two "
                    "levels and is allowed to be negative",
        },
        "not_done": [
            "no three-hop comparison",
            "no 'activation frequency below 50%' test for whether a target is active",
            "no negative-marginal share quoted without a confidence interval",
        ],
        "configs": [b.get("graph") for b in blocks],
        "blocks": blocks,
        "wall_seconds": time.time() - started,
        "complete": all(not b.get("skipped") for b in blocks),
    }, indent=2), encoding="utf-8")

    print(f"\n  wall time {time.time() - started:.0f}s   wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
