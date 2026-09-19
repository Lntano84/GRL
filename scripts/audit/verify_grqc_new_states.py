"""Four new states on the same ca-GrQc configuration: is there stable screening headroom?

The question
------------
``docs/GRQC_RANK_REVERSAL.md`` showed that the two ``delta2`` scores disagree about which candidate
to pick, and that at that one configuration the disagreement costs nothing measurable
(``+0.104 [-0.194, +0.403]``).  One configuration cannot answer the question that decides whether a
learned screener is worth building: **does expensive Monte-Carlo screening still beat simple
screening when the existing seeds and the candidate pool change?**

This script changes exactly one thing --- the random draws that produce the existing seed set ``S``
and the candidate pool --- and holds everything else frozen.

Frozen
------
graph ``ca_grqc``; weight normalisation ``sum_to_one``; the *same* target set ``D``; ``|S| = 52``;
50 candidates; shortlist 8; score formula and hop depth; diffusion parameters; the seed and candidate
generation rule.  Nothing is tuned, and no threshold is applied to any comparison.

Changed
-------
Four pre-fixed new random seeds (``NEW_SEEDS``) regenerate ``S`` and the candidate pool.  The actual
node lists are saved and checked: no two configurations are duplicates, and none reproduces the
original one.

Per configuration, the existing two-batch framework
---------------------------------------------------
1. **state**: 1000 cascades, used only to compute the state-conditioned score;
2. **selection**: 1000 cascades, paired marginals for all 50 candidates;
3. **choices frozen here**: ``degree`` top-8, ``static_delta2`` top-8, ``state_delta2`` top-8 and the
   full-50 argmax each pick their best candidate by selection-batch marginal; ``random`` top-8 is
   drawn 100 times and picks its best by the same rule each time;
4. **evaluation**: 1000 cascades on an independent stream, for all 50 candidates, with the per-trial
   arrays saved.

Every choice is frozen before the evaluation batch is generated, and the frozen choices are written
to disk and hashed first.  The full-50 pick is called the **MC reference candidate**, never the
optimum: it is the best of 50 under one finite pass.

Reporting
---------
Per configuration: the evaluation marginals of each method's chosen candidate, plus three paired
differences with 95% intervals --- MC reference minus degree, MC reference minus static, and state
minus static.  Then a four-row table and a sign count across configurations.  **Nothing is averaged
across configurations**: the four are separate states, not replicates of one population, and the
random method's 100 repetitions are a baseline *within* a configuration, never 100 study scenarios.

Cost
----
Per configuration ``1000 + 51 * 1000 + 51 * 1000 = 103,000`` cascades.  ``--config-index`` runs one
configuration so the four can be executed in parallel; ``--merge`` combines them.
"""

from __future__ import annotations

import argparse
import hashlib
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

from go_no_go import degree_stratified_pool, load_raw  # noqa: E402
from grl.data.weights import SUM_TO_ONE, normalise_in_weights  # noqa: E402
from grl.diffusion import overexposure as oe  # noqa: E402
from grl.diffusion.contract import resolve_target_contract  # noqa: E402
from grl.oracle import TargetedMonteCarloOracle, trial_seeds  # noqa: E402
from grl.scoring import exposure_scores_delta, rank_by_score  # noqa: E402

GRAPH = "ca_grqc"
FRACTION = 0.01
TARGET_FRACTION = 0.2
SHORTLIST = 8
POOL_SIZE = 120
TRIALS = 1000
RANDOM_REPEATS = 100

#: Pre-fixed, chosen before any of these configurations was generated.  No seed was selected,
#: re-drawn or dropped on the basis of any result.
NEW_SEEDS = [20261001, 20261002, 20261003, 20261004]

#: Stream namespaces, the same three purposes the original two-batch framework used.
STATE_NS = 900_000
SELECT_NS = 1_100_000
EVAL_NS = 1_300_000
#: The original configuration's four streams, checked for overlap alongside the twelve new ones.
ORIGINAL_SEED = 20260917
ORIGINAL_NS = {"state": 900_000, "batch1": 1_100_000, "batch2": 1_300_000, "batch3": 1_500_000}

SOURCE = ROOT / "docs" / "results" / "shortlist_headroom.json"
PER_CONFIG = ROOT / "docs" / "results" / "grqc_new_states_cfg{index}.json"
#: One frozen-choices file PER configuration.  The four configurations run as separate processes, so
#: a single shared file would be a read-modify-write race that could silently drop an entry.
FROZEN = ROOT / "docs" / "results" / "grqc_new_states_choices_cfg{index}.json"
CHOICES_GLOB = "grqc_new_states_choices_cfg*.json"
OUTPUT = ROOT / "docs" / "results" / "grqc_new_states.json"


def two_sided_p(mean: float, se: float) -> float:
    if not math.isfinite(se) or se <= 0:
        return 1.0
    return 2.0 * (1.0 - 0.5 * math.erfc(-abs(mean) / se / math.sqrt(2.0)))


def describe(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else float("inf")
    ordered = sorted(values)
    return {
        "mean": mean, "sd": math.sqrt(var), "se": se, "n": n,
        "ci95_low": mean - 1.96 * se if math.isfinite(se) else float("-inf"),
        "ci95_high": mean + 1.96 * se if math.isfinite(se) else float("inf"),
        "min": ordered[0], "median": ordered[n // 2], "max": ordered[-1],
        "negative_share": sum(1 for v in values if v < 0) / n,
    }


def paired(a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    out = describe(diffs)
    out["p_two_sided"] = two_sided_p(out["mean"], out["se"])
    return out


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def frozen_digest(entry: dict) -> str:
    """SHA-256 over everything in a frozen-choices entry except the digest field itself.

    The writer and the merge step both call THIS function, so the two cannot drift apart.  They did:
    the first version added two documentation fields after computing the digest, and the merge would
    then have rejected its own files.
    """
    body = {k: v for k, v in entry.items() if k != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def stream_overlap_check() -> dict:
    """Every stream any of the sixteen (four old, twelve new) will draw from, checked pairwise."""
    streams: dict[str, set[int]] = {}
    for name, ns in ORIGINAL_NS.items():
        streams[f"orig_{name}"] = set(trial_seeds(ORIGINAL_SEED + ns, TRIALS))
    for i, seed in enumerate(NEW_SEEDS):
        streams[f"cfg{i}_state"] = set(trial_seeds(seed + STATE_NS, TRIALS))
        streams[f"cfg{i}_select"] = set(trial_seeds(seed + SELECT_NS, TRIALS))
        streams[f"cfg{i}_eval"] = set(trial_seeds(seed + EVAL_NS, TRIALS))
    names = sorted(streams)
    overlaps = {f"{a}|{b}": len(streams[a] & streams[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    bad = {k: v for k, v in overlaps.items() if v}
    return {"streams_checked": len(names), "pairs_checked": len(overlaps),
            "overlapping_pairs": bad, "disjoint": not bad}


def paired_marginals(graph, contract, seeds, candidates, trials, base_seed):
    """Per-trial paired marginals, newly-positive and lost-positive counts, in the contracted D."""
    target = set(contract.objective.target_set)
    nodes = list(graph.nodes())
    mode = contract.diffusion.activation_mode
    law = contract.diffusion.threshold_law
    lo = contract.diffusion.window_lo
    per = {int(c): {"marginal": [], "newly_positive": [], "lost_positive": []}
           for c in candidates}
    base_counts: list[float] = []
    for trial_seed in trial_seeds(base_seed, trials):
        rng = random.Random(trial_seed)
        windows = oe.sample_threshold_windows(nodes, rng, threshold_law=law, window_lo=lo)
        base_run = oe.run_overexposure(graph, list(seeds), windows, rng, activation_mode=mode)
        base_pos = {int(v) for v in base_run.positive} & target
        base_counts.append(float(len(base_pos)))
        for candidate in candidates:
            run = oe.run_overexposure(graph, [*seeds, candidate], windows, rng,
                                      activation_mode=mode)
            pos = {int(v) for v in run.positive} & target
            gained = len(pos - base_pos)
            lost = len(base_pos - pos)
            delta = float(len(pos) - len(base_pos))
            if delta != float(gained - lost):
                raise SystemExit(f"decomposition broken: {delta} != {gained} - {lost}")
            per[int(candidate)]["marginal"].append(delta)
            per[int(candidate)]["newly_positive"].append(float(gained))
            per[int(candidate)]["lost_positive"].append(float(lost))
    return per, base_counts


def build_configuration(index: int, graph, contract, k: int):
    """Regenerate S and the candidate pool with the frozen rule, changing only the random seed."""
    eligible = contract.objective.legal_candidates(graph)
    rng = random.Random(NEW_SEEDS[index] + 31 * k)
    pool = degree_stratified_pool(graph, eligible, min(POOL_SIZE, len(eligible)), rng)
    seeds = sorted(pool, key=lambda v: -graph.out_degree(v))[:k]
    candidates = [v for v in pool if v not in set(seeds)][:50]
    return [int(s) for s in seeds], [int(c) for c in candidates], len(pool)


def run_one(index: int, args) -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    block = next(b for b in source["blocks"] if b["graph"] == GRAPH)
    graph = normalise_in_weights(load_raw(GRAPH).copy(), SUM_TO_ONE)
    n = len(graph.nodes())
    k = max(1, int(round(FRACTION * n)))
    contract = resolve_target_contract(graph, "degree-tail", TARGET_FRACTION, k)
    target = set(contract.objective.target_set)
    if sorted(int(t) for t in target) != sorted(block["target_set"]):
        raise SystemExit("target set D differs from the original; it must be held fixed")

    seeds, candidates, pool_size = build_configuration(index, graph, contract, k)
    base = NEW_SEEDS[index]

    # ---- 1. state, own stream ----
    oracle = TargetedMonteCarloOracle(graph, contract, mc_runs=args.trials,
                                      random_seed=base + STATE_NS)
    state_current = oracle.state(seeds, step=0)
    state_cascades = oracle.stats.state_cascades

    # ---- 2. selection batch, own stream ----
    t0 = time.time()
    sel, sel_base = paired_marginals(graph, contract, seeds, candidates, args.trials,
                                     base + SELECT_NS)

    # ---- 3. freeze choices BEFORE the evaluation batch exists ----
    sel_mean = {c: statistics.fmean(sel[c]["marginal"]) for c in candidates}
    target_set = target
    static_scores = exposure_scores_delta(graph, candidates, {v: 0.0 for v in graph.nodes()},
                                         set(), target_set=target_set)
    state_scores = exposure_scores_delta(graph, candidates, state_current, set(seeds),
                                        target_set=target_set)
    degree_scores = [float(graph.out_degree(v)) for v in candidates]
    score_lists = {"degree": degree_scores, "static_delta2": static_scores,
                   "state_delta2": state_scores}

    def pick(short: list[int]) -> int:
        return max(short, key=lambda c: (sel_mean[c], -c))

    choices: dict[str, dict] = {}
    for label, scores in score_lists.items():
        short = rank_by_score(candidates, scores)[:SHORTLIST]
        choices[label] = {"top8": [int(c) for c in short], "chosen": int(pick(short))}
    reference = int(pick(candidates))
    choices["mc_reference"] = {"top8": None, "chosen": reference,
                               "note": "best of all 50 by selection-batch marginal; a finite-sample "
                                       "reference, not the optimum"}

    rng = random.Random(base + 17)
    draws = []
    for _ in range(RANDOM_REPEATS):
        short = rng.sample(candidates, min(SHORTLIST, len(candidates)))
        draws.append({"shortlist": sorted(int(c) for c in short), "chosen": int(pick(short))})

    # everything the file will contain except the digest itself, so the digest covers it all
    frozen = {
        "index": index, "seed": base, "graph": GRAPH, "fraction": FRACTION,
        "n": n, "seed_size": k, "target_size": len(target), "pool_size": pool_size,
        "existing_seeds": seeds, "candidates": candidates,
        "choices": choices, "random_draws": draws, "select_trials": args.trials,
        "code_version": code_version(), "frozen_before_evaluation": True,
        "new_seeds": NEW_SEEDS,
        "note": ("this configuration's choices, written BEFORE its evaluation batch is generated; "
                 "--merge refuses to proceed without all four files and re-hashes each one"),
    }
    digest = frozen_digest(frozen)
    frozen["payload_sha256"] = digest

    # this configuration's choices, written BEFORE its evaluation batch exists.  One file per
    # configuration because the four run concurrently.
    frozen_path = FROZEN.with_name(FROZEN.name.format(index=index))
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"  cfg{index} frozen: degree={choices['degree']['chosen']} "
          f"static={choices['static_delta2']['chosen']} state={choices['state_delta2']['chosen']} "
          f"mc_ref={reference}  sha={digest[:12]}", flush=True)

    # ---- 4. evaluation batch, own stream, all 50 candidates ----
    ev, ev_base = paired_marginals(graph, contract, seeds, candidates, args.trials, base + EVAL_NS)
    eval_seconds = time.time() - t0

    ev_mean = {c: statistics.fmean(ev[c]["marginal"]) for c in candidates}
    method_eval = {label: {"chosen": ch["chosen"],
                           "evaluation": describe(ev[ch["chosen"]]["marginal"]),
                           "selection": describe(sel[ch["chosen"]]["marginal"])}
                   for label, ch in choices.items()}
    rand_eval_means = [ev_mean[d["chosen"]] for d in draws]
    method_eval["random"] = {
        "repeats": RANDOM_REPEATS,
        "mean_over_repeats": statistics.fmean(rand_eval_means),
        "quantiles": {"min": min(rand_eval_means),
                      "p05": sorted(rand_eval_means)[max(0, int(0.05 * RANDOM_REPEATS) - 1)],
                      "median": statistics.median(rand_eval_means),
                      "p95": sorted(rand_eval_means)[min(RANDOM_REPEATS - 1,
                                                         int(0.95 * RANDOM_REPEATS))],
                      "max": max(rand_eval_means)},
        "note": "100 repetitions of the SAME configuration; a within-configuration baseline, not "
                "100 study scenarios and not pooled with anything",
    }

    contrasts = {
        "mc_reference_minus_degree": paired(ev[reference]["marginal"],
                                            ev[choices["degree"]["chosen"]]["marginal"]),
        "mc_reference_minus_static": paired(ev[reference]["marginal"],
                                            ev[choices["static_delta2"]["chosen"]]["marginal"]),
        "state_minus_static": paired(ev[choices["state_delta2"]["chosen"]]["marginal"],
                                     ev[choices["static_delta2"]["chosen"]]["marginal"]),
    }

    headroom = {"reference_selection": describe(sel[reference]["marginal"]),
                "reference_evaluation": describe(ev[reference]["marginal"]),
                "pool_evaluation_mean": statistics.fmean(ev_mean.values()),
                "pool_evaluation_best": max(ev_mean.values()),
                "pool_best_flagged": "argmax over the evaluation batch; upward-biased",
                "pool_evaluation_sd": statistics.pstdev(list(ev_mean.values()))}

    payload = {
        "script": Path(__file__).name, "code_version": code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "objective": "Delta(v|S) = |P_final(S u {v}) cap D| - |P_final(S) cap D|",
        "index": index, "random_seed": base,
        "configuration": {"graph": GRAPH, "fraction": FRACTION, "n": n, "seed_size": k,
                          "target_size": len(target), "candidate_count": len(candidates),
                          "pool_size": pool_size, "existing_seeds": seeds,
                          "candidates": candidates,
                          "seeds_in_target": len(set(seeds) & target)},
        "streams": {"state": base + STATE_NS, "selection": base + SELECT_NS,
                    "evaluation": base + EVAL_NS, "trials": args.trials},
        "cost": {"state_cascades": state_cascades,
                 "selection_cascades": (1 + len(candidates)) * args.trials,
                 "evaluation_cascades": (1 + len(candidates)) * args.trials,
                 "seconds": eval_seconds},
        "frozen": {"payload_sha256": digest, "file": frozen_path.name,
                   "written_before_evaluation": True},
        "methods": method_eval, "contrasts": contrasts, "headroom": headroom,
        "selection_base_counts": describe(sel_base),
        "evaluation_base_counts": describe(ev_base),
        "per_trial_evaluation": {str(c): ev[c] for c in candidates},
        "per_trial_selection_marginals": {str(c): sel[c]["marginal"] for c in candidates},
        "complete": True,
    }
    out = PER_CONFIG.with_name(PER_CONFIG.name.format(index=index))
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  cfg{index} done in {eval_seconds:.0f}s -> {out.name}", flush=True)
    return 0


def merge(args) -> int:
    """Combine the four configuration files into one artifact plus the four-row summary table."""
    frozen: dict[str, dict] = {}
    for i in range(len(NEW_SEEDS)):
        path = FROZEN.with_name(FROZEN.name.format(index=i))
        if not path.exists():
            raise SystemExit(f"configuration {i} has no frozen choices at {path.name}; refusing to "
                             f"merge, because selection must be frozen before evaluation")
        entry = json.loads(path.read_text(encoding="utf-8"))
        got = entry.get("payload_sha256")
        if got != frozen_digest(entry):
            raise SystemExit(f"configuration {i}: frozen choices changed after being written")
        frozen[str(i)] = entry

    blocks = []
    for i in range(len(NEW_SEEDS)):
        path = PER_CONFIG.with_name(PER_CONFIG.name.format(index=i))
        if not path.exists():
            raise SystemExit(f"missing {path.name}; run --config-index {i} first")
        blocks.append(json.loads(path.read_text(encoding="utf-8")))

    # configurations must be genuinely different, and none may repeat the original
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    orig_block = next(b for b in original["blocks"] if b["graph"] == GRAPH)
    orig_key = (frozenset(int(s) for s in orig_block["existing_seeds"]),
                tuple(int(c) for c in orig_block["candidates"]))
    keys, distinct = {}, {}
    for b in blocks:
        key = (frozenset(b["configuration"]["existing_seeds"]),
               tuple(b["configuration"]["candidates"]))
        keys[b["index"]] = key
    for i, key in keys.items():
        dup = [j for j, other in keys.items() if j != i and other == key]
        distinct[str(i)] = {"duplicate_of": dup, "repeats_original": key == orig_key,
                            "seeds_shared_with_original":
                                len(key[0] & orig_key[0]),
                            "candidates_shared_with_original":
                                len(set(key[1]) & set(orig_key[1]))}
    if any(v["duplicate_of"] or v["repeats_original"] for v in distinct.values()):
        raise SystemExit(f"configurations are not distinct: {distinct}")

    names = ["degree", "static_delta2", "state_delta2", "random", "mc_reference"]
    table = []
    for b in blocks:
        row = {"configuration": b["index"], "random_seed": b["random_seed"],
               "degree": b["methods"]["degree"]["evaluation"]["mean"],
               "static": b["methods"]["static_delta2"]["evaluation"]["mean"],
               "state": b["methods"]["state_delta2"]["evaluation"]["mean"],
               "random_mean": b["methods"]["random"]["mean_over_repeats"],
               "mc_reference": b["methods"]["mc_reference"]["evaluation"]["mean"]}
        row["chosen"] = {n: b["methods"][n]["chosen"] for n in names if n != "random"}
        row["chosen"]["random"] = f"{RANDOM_REPEATS} draws of {SHORTLIST} from 50, best each time"
        row["random_quantiles"] = b["methods"]["random"]["quantiles"]
        row["headroom"] = {"pool_evaluation_mean": b["headroom"]["pool_evaluation_mean"],
                           "pool_evaluation_best": b["headroom"]["pool_evaluation_best"],
                           "pool_evaluation_sd": b["headroom"]["pool_evaluation_sd"],
                           "reference_evaluation": b["headroom"]["reference_evaluation"]["mean"],
                           "reference_selection": b["headroom"]["reference_selection"]["mean"]}
        table.append(row)

    sign_counts = {}
    for cname, c in (("mc_reference_minus_degree", "mc_reference_minus_degree"),
                     ("mc_reference_minus_static", "mc_reference_minus_static"),
                     ("state_minus_static", "state_minus_static")):
        pos = sum(1 for b in blocks if b["contrasts"][c]["mean"] > 0)
        sig = sum(1 for b in blocks
                  if b["contrasts"][c]["ci95_low"] > 0 or b["contrasts"][c]["ci95_high"] < 0)
        sign_counts[cname] = {"positive_point_estimates": pos, "configurations": len(blocks),
                              "intervals_excluding_zero": sig}

    artifact = {
        "script": Path(__file__).name, "code_version": code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "question": "with new existing seeds and a new candidate pool, does expensive Monte-Carlo "
                    "screening still beat simple screening?",
        "frozen_across_configurations": [
            "graph ca_grqc", "weight normalisation sum_to_one", "the same target set D",
            "|S| = 52", "50 candidates", "shortlist 8", "score formula and two-hop depth",
            "diffusion parameters", "seed and candidate generation rule"],
        "changed_only": "the four pre-fixed random seeds that generate S and the candidate pool",
        "new_seeds": NEW_SEEDS,
        "trials_per_batch": args.trials,
        "framework": "state 1000 -> selection 1000 (freeze) -> independent evaluation 1000, "
                     "per configuration, on disjoint streams",
        "stream_check": stream_overlap_check(),
        "distinctness": distinct,
        "frozen_choices_files": [FROZEN.name.format(index=i) for i in range(len(NEW_SEEDS))],
        "frozen_choices_sha256": {k: v["payload_sha256"] for k, v in frozen.items()},
        "summary_table": table,
        "sign_counts": sign_counts,
        "pooling": "none; the four configurations are separate states, not replicates of one "
                   "population, and the random method's 100 repetitions are a baseline within a "
                   "configuration rather than 100 study scenarios",
        "configurations": blocks,
        "complete": all(b.get("complete") for b in blocks),
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("=" * 108)
    print("  FOUR NEW STATES, SAME CONFIGURATION -- independent-evaluation marginals")
    print("=" * 108)
    print(f"  streams checked: {artifact['stream_check']['streams_checked']} "
          f"({artifact['stream_check']['pairs_checked']} pairs), "
          f"disjoint: {artifact['stream_check']['disjoint']}")
    print()
    print(f"  {'cfg':>4}{'seed':>10}{'degree':>10}{'static':>10}{'state':>10}"
          f"{'random':>10}{'MC ref':>10}")
    for row in table:
        print(f"  {row['configuration']:>4}{row['random_seed']:>10}{row['degree']:>10.3f}"
              f"{row['static']:>10.3f}{row['state']:>10.3f}{row['random_mean']:>10.3f}"
              f"{row['mc_reference']:>10.3f}")
    print()
    print("  PAIRED DIFFERENCES (evaluation batch, same windows within a configuration)")
    for name in ("mc_reference_minus_degree", "mc_reference_minus_static", "state_minus_static"):
        print(f"    {name}")
        for b in blocks:
            c = b["contrasts"][name]
            print(f"      cfg{b['index']}  {c['mean']:+.3f} +-{c['se']:.3f}  "
                  f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  p={c['p_two_sided']:.3f}")
        sc = sign_counts[name]
        print(f"      positive in {sc['positive_point_estimates']}/{sc['configurations']} "
              f"configurations; intervals excluding zero in "
              f"{sc['intervals_excluding_zero']}/{sc['configurations']}")
    print()
    print(f"  wrote {OUTPUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-index", type=int, default=None, choices=range(len(NEW_SEEDS)))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--check-streams-only", action="store_true")
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    if args.check_streams_only:
        check = stream_overlap_check()
        print(json.dumps(check, indent=2))
        return 0 if check["disjoint"] else 1
    if args.merge:
        return merge(args)
    if args.config_index is None:
        parser.error("give --config-index N, --merge, or --check-streams-only")
    check = stream_overlap_check()
    if not check["disjoint"]:
        raise SystemExit(f"stream overlap: {check['overlapping_pairs']}")
    return run_one(args.config_index, args)


if __name__ == "__main__":
    raise SystemExit(main())
