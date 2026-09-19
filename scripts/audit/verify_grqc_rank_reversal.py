"""Independent replication and input ablation of the ca-GrQc rank reversal.

The claim under test
--------------------
``docs/SHORTLIST_HEADROOM.md`` reported that on ca-GrQc 1% the two ``delta2`` scores disagree
completely about which candidate is best: with an all-zero exposure baseline and no seed mask the
closed form ranks the Monte-Carlo reference candidate (8150) **1st of 50**, and with the realised
current-state exposure it ranks the same candidate **50th of 50**.  The accompanying text explained
that as the state-conditioning term ``g(E_S + dE) - g(E_S)`` going negative, and the phrase "the
state-conditioned score anti-ranks the best candidate" invited the stronger reading that **the state
information is harmful**.

That stronger reading is not what was measured.  What was measured is that the *score* reversed
rank.  Whether the reversal costs real objective value is a separate question, and it is the question
this script answers.  Nothing here is allowed to infer a negative true gain from a negative score.

What is held fixed
------------------
Graph, weight normalisation, target set ``D``, existing seed set ``S``, the 50 candidates and the
batch-1 Monte-Carlo marginals are all read from ``docs/results/shortlist_headroom.json`` and the
construction is re-derived only to *check* them, never to replace them.  No candidate is re-drawn, no
target is added, no existing seed is changed.

The ablation separates the two inputs
-------------------------------------
``exposure_scores_delta`` takes two things that the static and state versions both vary at once:

===========  ==========================  ==========================
version      exposure input              seed mask passed to score
===========  ==========================  ==========================
``A``        all-zero (empty state)      empty set
``B``        all-zero (empty state)      the current ``S``
``C``        current mean exposure       empty set
``D``        current mean exposure       the current ``S``
===========  ==========================  ==========================

``A`` and ``D`` reproduce the two scores the original run used; ``B`` and ``C`` isolate one factor
each.  The hop depth stays at the paper's two, and the score function itself is untouched.  Every
candidate is still evaluated from the **same actual seed set ``S``** in the real simulation.

The evaluation batch is new and independent
-------------------------------------------
**Batch 3** (``BATCH3_NS``) is a fourth random stream, disjoint from the state stream and from
batches 1 and 2, and the disjointness is checked before the run rather than asserted afterwards.  It
runs a **fixed 6000 trials to completion**, with no early stop on significance in either direction.
Old batches select; batch 3 only confirms.  The two are never pooled.

Within a trial the threshold windows are drawn once and shared by every candidate, which is what makes
the differences paired.  The activation coin flips inside ``run_overexposure`` are drawn sequentially
from the same stream, so the result depends on the frozen candidate order; the order is written to the
candidate file before any evaluation happens and is recorded in the artifact.  This is the same
convention batches 1 and 2 used, kept deliberately so the batches stay comparable.

Frozen before evaluation
------------------------
Each version's chosen candidate is decided by the *existing* batch-1 marginals, the candidate list is
written to ``docs/results/grqc_rank_reversal_candidates.json`` with a SHA-256 of its own payload, and
the main artifact records that hash.  No candidate can be swapped after the new numbers are in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from grl.scoring import exposure_scores_delta, out_edges, rank_by_score  # noqa: E402

GRAPH = "ca_grqc"
FRACTION = 0.01
SOURCE = ROOT / "docs" / "results" / "shortlist_headroom.json"
CANDIDATE_FILE = ROOT / "docs" / "results" / "grqc_rank_reversal_candidates.json"
OUTPUT = ROOT / "docs" / "results" / "grqc_rank_reversal.json"

#: Stream namespaces, identical to the original diagnostic for the first three.
BASE_SEED = 20260917
STATE_NS = 900_000
BATCH1_NS = 1_100_000
BATCH2_NS = 1_300_000
#: A fourth stream, far outside the three originals; verified disjoint, not assumed.
BATCH3_NS = 1_500_000

STATE_TRIALS = 1000
SELECT_TRIALS = 1000
CONFIRM_TRIALS = 6000
SHORTLIST = 8
TARGET_FRACTION = 0.2

#: The nodes the original report singled out, kept as an explicit cross-check.
NODES_OF_INTEREST = {8150: "original MC reference and static-delta2 choice",
                     15960: "original state-delta2 choice",
                     3058: "original degree choice"}

VERSIONS = ("A", "B", "C", "D")
PRIMARY = ("A", "D")
SECONDARY = (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"))


# -------------------------------------------------------------------------------------- helpers
def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def two_sided_p(mean: float, se: float) -> float:
    if not math.isfinite(se) or se <= 0:
        return 1.0
    return 2.0 * (1.0 - normal_cdf(abs(mean) / se))


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


def holm(pairs: dict[str, dict], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over the secondary comparisons, reported alongside the raw p-values."""
    ordered = sorted(pairs.items(), key=lambda kv: kv[1]["p_two_sided"])
    m = len(ordered)
    running = 0.0
    adjusted = {}
    for i, (name, res) in enumerate(ordered):
        factor = m - i
        value = min(1.0, factor * res["p_two_sided"])
        running = max(running, value)
        adjusted[name] = {"factor": factor, "p_adjusted": running,
                          "significant_at_0_05": bool(running < alpha),
                          "p_raw": res["p_two_sided"]}
    return {"method": "Holm-Bonferroni over the four secondary comparisons",
            "alpha": alpha, "comparisons": m, "per_comparison": adjusted}


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def stream_overlaps() -> dict:
    """Disjointness of the four streams, checked before anything is run."""
    streams = {
        "state": set(trial_seeds(BASE_SEED + STATE_NS, STATE_TRIALS)),
        "batch1": set(trial_seeds(BASE_SEED + BATCH1_NS, SELECT_TRIALS)),
        "batch2": set(trial_seeds(BASE_SEED + BATCH2_NS, SELECT_TRIALS)),
        "batch3": set(trial_seeds(BASE_SEED + BATCH3_NS, CONFIRM_TRIALS)),
    }
    names = sorted(streams)
    overlaps = {f"{a}_{b}": len(streams[a] & streams[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    return {"trials": {"state": STATE_TRIALS, "batch1": SELECT_TRIALS,
                       "batch2": SELECT_TRIALS, "batch3": CONFIRM_TRIALS},
            "namespaces": {"state": STATE_NS, "batch1": BATCH1_NS,
                           "batch2": BATCH2_NS, "batch3": BATCH3_NS},
            "base_seed": BASE_SEED,
            "overlaps": overlaps,
            "disjoint": all(v == 0 for v in overlaps.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--candidate-file", type=Path, default=CANDIDATE_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--trials", type=int, default=CONFIRM_TRIALS,
                        help="batch-3 trials; fixed in advance and never stopped early")
    parser.add_argument("--check-only", action="store_true",
                        help="reproduce the old scores, freeze the candidate list, then exit")
    args = parser.parse_args()

    t_start = time.time()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    block = next(b for b in source["blocks"] if b["graph"] == GRAPH)

    checks = stream_overlaps()
    print("stream disjointness:", json.dumps(checks["overlaps"]), flush=True)
    if not checks["disjoint"]:
        raise SystemExit(f"stream overlap: {checks['overlaps']}")

    # ---------------------------------------------------------------- 1. fixed configuration
    graph = normalise_in_weights(load_raw(GRAPH).copy(), SUM_TO_ONE)
    n = len(graph.nodes())
    k = max(1, int(round(FRACTION * n)))
    contract = resolve_target_contract(graph, "degree-tail", TARGET_FRACTION, k)
    target = set(contract.objective.target_set)

    # read, then check -- never re-draw
    seeds = [int(s) for s in block["existing_seeds"]]
    candidates = [int(c) for c in block["candidates"]]
    frozen_config = {
        "target_set_matches": sorted(int(t) for t in target) == sorted(block["target_set"]),
        "seed_count_matches": len(seeds) == block["seed_size"],
        "candidate_count_matches": len(candidates) == block["candidate_count"],
        "target_size_matches": len(target) == block["target_size"],
    }
    if not all(frozen_config.values()):
        raise SystemExit(f"configuration could not be reproduced: {frozen_config}")

    # the pool is rebuilt only to confirm the stored seeds and candidates are the original ones
    rng = random.Random(BASE_SEED + 31 * k)
    eligible = contract.objective.legal_candidates(graph)
    pool = degree_stratified_pool(graph, eligible, min(120, len(eligible)), rng)
    rebuilt_seeds = sorted(pool, key=lambda v: -graph.out_degree(v))[:k]
    rebuilt_candidates = [v for v in pool if v not in set(rebuilt_seeds)][:len(candidates)]
    frozen_config["seeds_reproduced"] = [int(s) for s in rebuilt_seeds] == seeds
    frozen_config["candidates_reproduced"] = [int(c) for c in rebuilt_candidates] == candidates
    if not (frozen_config["seeds_reproduced"] and frozen_config["candidates_reproduced"]):
        raise SystemExit(f"stored configuration is not the original one: {frozen_config}")
    print(f"configuration fixed: n={n} |S|={k} |D|={len(target)} candidates={len(candidates)}",
          flush=True)

    # ---------------------------------------------------------------- 2. exposure reproduction
    oracle = TargetedMonteCarloOracle(graph, contract, mc_runs=STATE_TRIALS,
                                      random_seed=BASE_SEED + STATE_NS)
    state_current = oracle.state(seeds, step=0)
    state_reads = oracle.stats.state_cascades
    state_empty = oracle.state([], step=1)
    empty_max_abs = max(abs(v) for v in state_empty.values())
    if empty_max_abs > 0.0:
        raise SystemExit(f"the empty-state exposure is not all-zero (max |delta| = {empty_max_abs})")
    zero_exposure = {node: 0.0 for node in graph.nodes()}

    # ---------------------------------------------------------------- 3. the four versions
    scores = {
        "A": exposure_scores_delta(graph, candidates, zero_exposure, set(), target_set=target),
        "B": exposure_scores_delta(graph, candidates, zero_exposure, set(seeds), target_set=target),
        "C": exposure_scores_delta(graph, candidates, state_current, set(), target_set=target),
        "D": exposure_scores_delta(graph, candidates, state_current, set(seeds), target_set=target),
    }
    if any(len(scores[v]) != len(candidates) for v in VERSIONS):
        raise SystemExit("a score list is not one-score-per-candidate; alignment is not guaranteed")

    by_id = {c: i for i, c in enumerate(candidates)}
    old_static = {r["candidate"]: r["static_delta2"] for r in block["candidate_rows"]}
    old_state = {r["candidate"]: r["state_delta2"] for r in block["candidate_rows"]}

    # Is the seed mask LIVE, or is the argument being ignored?  A and B come out byte-identical
    # below, so the identity has to be explained rather than assumed.  Two facts plus one control:
    # every seed is outside D by construction, so no summed term can be dropped by the mask; and a
    # deliberately widened mask must move the scores, which proves the mask is actually applied.
    seeds_in_target = len(set(seeds) & target)
    reaches_seed = 0
    for candidate in candidates:
        one_hop = {t for t, _ in out_edges(graph, candidate)}
        two_hop = set(one_hop)
        for mid in one_hop:
            two_hop |= {t for t, _ in out_edges(graph, mid)}
        reaches_seed += 1 if (two_hop & set(seeds)) else 0
    widened = set(seeds)
    for candidate in candidates:
        widened |= {t for t, _ in out_edges(graph, candidate) if t in target}
    widened_scores = exposure_scores_delta(graph, candidates, zero_exposure, widened,
                                           target_set=target)
    liveness = {
        "seeds_outside_target": seeds_in_target == 0,
        "seeds_in_target": seeds_in_target,
        "candidates_whose_two_hop_reach_touches_a_seed": reaches_seed,
        "candidate_count": len(candidates),
        "control_widened_mask_max_abs_score_change":
            max(abs(a - b) for a, b in zip(scores["A"], widened_scores)),
        "control_meaning": "nonzero proves the mask reaches the score; the A=B identity below is "
                           "therefore a property of this configuration, not a dead argument",
    }

    def max_abs_diff(a: list[float], b: dict[int, float]) -> float:
        return max(abs(a[by_id[c]] - b[c]) for c in candidates)

    repro = {
        "A_vs_old_static_max_abs_diff": max_abs_diff(scores["A"], old_static),
        "D_vs_old_state_max_abs_diff": max_abs_diff(scores["D"], old_state),
        "old_static_top8_matches": [int(c) for c in rank_by_score(candidates, scores["A"])[:SHORTLIST]]
                                  == [int(c) for c in block["methods"]["static_delta2"]["top8"]],
        "old_state_top8_matches": [int(c) for c in rank_by_score(candidates, scores["D"])[:SHORTLIST]]
                                 == [int(c) for c in block["methods"]["state_delta2"]["top8"]],
    }
    ref = int(block["reference_candidate"]["candidate"])
    ranks = {v: rank_by_score(candidates, scores[v]).index(ref) + 1 for v in VERSIONS}
    repro["reference_rank_reproduced"] = {
        "static_old": block["reference_candidate"]["static_delta2_rank"],
        "static_new": ranks["A"],
        "state_old": block["reference_candidate"]["state_delta2_rank"],
        "state_new": ranks["D"],
    }
    reproduced = (repro["A_vs_old_static_max_abs_diff"] == 0.0
                  and repro["D_vs_old_state_max_abs_diff"] == 0.0
                  and repro["old_static_top8_matches"] and repro["old_state_top8_matches"]
                  and ranks["A"] == block["reference_candidate"]["static_delta2_rank"]
                  and ranks["D"] == block["reference_candidate"]["state_delta2_rank"])
    print(f"reproduction: A diff={repro['A_vs_old_static_max_abs_diff']:.2e} "
          f"D diff={repro['D_vs_old_state_max_abs_diff']:.2e} "
          f"top8 A={repro['old_static_top8_matches']} D={repro['old_state_top8_matches']} "
          f"ranks A={ranks['A']} D={ranks['D']} -> {'EXACT' if reproduced else 'MISMATCH'}",
          flush=True)
    if not reproduced:
        (ROOT / "docs" / "results" / "grqc_rank_reversal_reproduction_failure.json").write_text(
            json.dumps({"repro": repro, "ranks": ranks}, indent=2), encoding="utf-8")
        raise SystemExit("A/D did not reproduce the original scores; stopping before any evaluation")

    # ---------------------------------------------------------------- 4. freeze the choices
    b1 = {r["candidate"]: r["batch1_paired"] for r in block["candidate_rows"]}

    def b1_mean(c: int) -> float:
        return sum(b1[c]) / len(b1[c])

    degree_scores = [float(graph.out_degree(v)) for v in candidates]
    all_scores = dict(scores)
    all_scores["degree"] = degree_scores

    choices: dict[str, dict] = {}
    for label, score_list in all_scores.items():
        short = rank_by_score(candidates, score_list)[:SHORTLIST]
        chosen = max(short, key=lambda c: (b1_mean(c), -c))
        choices[label] = {"top8": [int(c) for c in short], "chosen": int(chosen)}

    # the degree choice and the MC reference must match the original run
    if choices["degree"]["chosen"] != int(block["methods"]["degree"]["chosen"]):
        raise SystemExit("recomputed degree choice differs from the original run")

    required = sorted({choices[v]["chosen"] for v in VERSIONS}
                      | {choices["degree"]["chosen"], ref})
    # evaluation order is frozen here and never changed afterwards
    frozen_order = [int(c) for c in required]
    payload = {
        "graph": GRAPH, "fraction": FRACTION,
        "seed_size": k, "target_size": len(target),
        "reference_candidate": ref,
        "note": "every candidate below is evaluated from the same actual seed set S",
        "versions": {v: choices[v] for v in VERSIONS},
        "degree": choices["degree"],
        "frozen_evaluation_order": frozen_order,
        "chosen_by_version": {v: choices[v]["chosen"] for v in VERSIONS},
        "same_choice_pairs": sorted(f"{a}={b}" for i, a in enumerate(VERSIONS)
                                    for b in VERSIONS[i + 1:]
                                    if choices[a]["chosen"] == choices[b]["chosen"]),
        "nodes_of_interest": {str(node): {"in_frozen_evaluation": node in frozen_order,
                                          "why": why}
                              for node, why in NODES_OF_INTEREST.items()},
        "frozen_before_evaluation": True,
        "select_trials": SELECT_TRIALS,
        "confirm_trials": args.trials,
        "code_version": code_version(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["payload_sha256"] = digest
    args.candidate_file.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"frozen choices: " + "  ".join(f"{v}={choices[v]['chosen']}" for v in VERSIONS)
          + f"  degree={choices['degree']['chosen']}  reference={ref}", flush=True)
    print(f"  same-choice pairs: {payload['same_choice_pairs']}", flush=True)
    print(f"  evaluating {frozen_order} -> {args.candidate_file.name} sha256={digest[:16]}",
          flush=True)
    if args.check_only:
        print(f"check-only: no evaluation run.  wall {time.time() - t_start:.0f}s")
        return 0

    # ---------------------------------------------------------------- 5. batch 3, fixed 6000
    nodes = list(graph.nodes())
    mode = contract.diffusion.activation_mode
    law = contract.diffusion.threshold_law
    lo = contract.diffusion.window_lo
    per_candidate: dict[int, dict[str, list[float]]] = {
        c: {"marginal": [], "newly_positive": [], "lost_positive": []} for c in frozen_order}
    base_counts: list[float] = []

    t0 = time.time()
    for i, trial_seed in enumerate(trial_seeds(BASE_SEED + BATCH3_NS, args.trials)):
        rng = random.Random(trial_seed)
        windows = oe.sample_threshold_windows(nodes, rng, threshold_law=law, window_lo=lo)
        base_run = oe.run_overexposure(graph, seeds, windows, rng, activation_mode=mode)
        base_pos = {int(v) for v in base_run.positive} & target
        base_counts.append(float(len(base_pos)))
        for candidate in frozen_order:
            run = oe.run_overexposure(graph, [*seeds, candidate], windows, rng,
                                      activation_mode=mode)
            pos = {int(v) for v in run.positive} & target
            gained = len(pos - base_pos)
            lost = len(base_pos - pos)
            delta = float(len(pos) - len(base_pos))
            if delta != float(gained - lost):
                raise SystemExit(f"decomposition broken at trial {i}, candidate {candidate}: "
                                 f"{delta} != {gained} - {lost}")
            per_candidate[candidate]["marginal"].append(delta)
            per_candidate[candidate]["newly_positive"].append(float(gained))
            per_candidate[candidate]["lost_positive"].append(float(lost))
        if (i + 1) % 1000 == 0:
            print(f"    batch 3: {i + 1}/{args.trials} trials  ({time.time() - t0:.0f}s)",
                  flush=True)
    confirm_seconds = time.time() - t0

    # ---------------------------------------------------------------- 6. decomposition + contrasts
    decomposition = {}
    for candidate in frozen_order:
        d = per_candidate[candidate]
        gained = sum(d["newly_positive"])
        lost = sum(d["lost_positive"])
        total = sum(d["marginal"])
        decomposition[str(candidate)] = {
            "marginal": describe(d["marginal"]),
            "newly_positive_mean": gained / args.trials,
            "lost_positive_mean": lost / args.trials,
            "identity_holds": abs(total - (gained - lost)) < 1e-9,
            "identity_residual": total - (gained - lost),
            "any_loss_at_all": any(v > 0 for v in d["lost_positive"]),
            "trials_with_a_lost_positive": sum(1 for v in d["lost_positive"] if v > 0),
        }
    if not all(v["identity_holds"] for v in decomposition.values()):
        raise SystemExit("marginal != newly_positive - lost_positive for some candidate")

    def comparison(a: str, b: str) -> dict:
        ca, cb = choices[a]["chosen"], choices[b]["chosen"]
        if ca == cb:
            return {"same_choice": True, "chosen": int(ca),
                    "note": "both versions selected the SAME candidate; the contrast is exactly "
                            "zero by construction and is NOT independent evidence"}
        res = paired(per_candidate[ca]["marginal"], per_candidate[cb]["marginal"])
        res.update({"same_choice": False, "chosen_a": int(ca), "chosen_b": int(cb)})
        return {"same_choice": False, "chosen_a": int(ca), "chosen_b": int(cb),
                "difference_a_minus_b": res}

    primary = comparison(*PRIMARY)
    secondary = {f"{a}-{b}": comparison(a, b) for a, b in SECONDARY}

    # Collapse: two versions that chose the same candidate give a contrast of exactly zero, and two
    # contrasts over the same candidate pair are the same measurement restated, not a second test.
    def pair_key(res: dict) -> tuple | None:
        if res["same_choice"]:
            return None
        return tuple(sorted((res["chosen_a"], res["chosen_b"])))

    informative = {"primary_A-D": primary}
    informative.update(secondary)
    by_pair: dict[tuple, list[str]] = {}
    for name, res in informative.items():
        key = pair_key(res)
        if key is not None:
            by_pair.setdefault(key, []).append(name)
    distinct = {f"{k[0]}_vs_{k[1]}": names for k, names in by_pair.items()}
    secondary_p = {k: v["difference_a_minus_b"] for k, v in secondary.items()
                   if not v["same_choice"]}
    if len(distinct) <= 1:
        multiplicity = {
            "primary": "A-D, pre-registered",
            "same_choice_contrasts": sorted(k for k, v in informative.items() if v["same_choice"]),
            "distinct_candidate_pairs": distinct,
            "correction_applied": False,
            "reason": "every informative contrast reduces to the same candidate pair, so there is "
                      "exactly one independent measurement.  The A-C and B-D entries are that same "
                      "measurement restated, not extra evidence, and no multiplicity correction is "
                      "meaningful or applied.",
        }
    else:
        multiplicity = holm(secondary_p) if secondary_p else {
            "correction_applied": False, "reason": "no informative secondary contrast"}

    # score side, for the node the original report singled out
    def score_view(node: int) -> dict:
        out = {}
        for label, score_list in all_scores.items():
            if node not in by_id:
                out[label] = None
                continue
            order = rank_by_score(candidates, score_list)
            out[label] = {"score": score_list[by_id[node]],
                          "rank": order.index(node) + 1,
                          "in_top8": node in order[:SHORTLIST]}
        return out

    score_side = {str(node): score_view(node) for node in NODES_OF_INTEREST}

    # Which factor actually moves the score?  This is the only place the two inputs can be told
    # apart: at the level of the CHOICE they are confounded, because either one alone flips it.
    def spearman(xs: list[float], ys: list[float]) -> float:
        def ranks(vs: list[float]) -> list[float]:
            order = sorted(range(len(vs)), key=lambda i: vs[i])
            out = [0.0] * len(vs)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for t in range(i, j + 1):
                    out[order[t]] = avg
                i = j + 1
            return out

        rx, ry = ranks(xs), ranks(ys)
        mx = sum(rx) / len(rx)
        my = sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else float("nan")

    def rank_positions(score_list: list[float]) -> dict[int, int]:
        return {c: i + 1 for i, c in enumerate(rank_by_score(candidates, score_list))}

    factor_effects: dict[str, dict] = {}
    for x, y, changed in (("A", "B", "seed mask only, exposure held at zero"),
                          ("C", "D", "seed mask only, exposure held at current"),
                          ("A", "C", "exposure input only, seed mask held empty"),
                          ("B", "D", "exposure input only, seed mask held at S"),
                          ("A", "D", "both inputs changed (static vs state)")):
        px, py = rank_positions(all_scores[x]), rank_positions(all_scores[y])
        factor_effects[f"{x}_vs_{y}"] = {
            "changed": changed,
            "max_abs_score_difference": max(abs(all_scores[x][by_id[c]] - all_scores[y][by_id[c]])
                                            for c in candidates),
            "spearman_between_scores": spearman(all_scores[x], all_scores[y]),
            "identical_ranking": px == py,
            "identical_top8": ([c for c, p in sorted(px.items(), key=lambda kv: kv[1]) if p <= 8]
                               == [c for c, p in sorted(py.items(), key=lambda kv: kv[1]) if p <= 8]),
            "candidates_changing_position": sum(1 for c in candidates if px[c] != py[c]),
            "reference_rank_from": px[ref], "reference_rank_to": py[ref],
        }

    # ---------------------------------------------------------------- write
    artifact = {
        "script": Path(__file__).name,
        "code_version": code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "objective": "Delta(v|S) = |P_final(S u {v}) cap D| - |P_final(S) cap D|",
        "graph": GRAPH, "fraction": FRACTION,
        "source_artifact": str(args.source.relative_to(ROOT)),
        "source_code_version": source.get("code_version"),
        "configuration": {"n": n, "seed_size": k, "target_size": len(target),
                          "candidate_count": len(candidates), "shortlist_size": SHORTLIST,
                          "seed_set": [int(s) for s in seeds],
                          "verified_against_source": frozen_config},
        "ablation": {
            "definition": {
                "A": "zero exposure, empty seed mask (reproduces the original static score)",
                "B": "zero exposure, current seed set as mask",
                "C": "current mean exposure, empty seed mask",
                "D": "current mean exposure, current seed set as mask (original state score)",
            },
            "held_fixed": ["two-hop depth", "score function", "candidate list", "target set D",
                           "existing seed set S", "actual simulation always starts from S"],
            "exposure_reproduced_from_stream": f"base_seed + {STATE_NS}, "
                                               f"{STATE_TRIALS} trials, step=0",
            "state_cascades": state_reads,
            "empty_state_all_zero": True,
            "reproduction": repro,
            "reproduced_exactly": reproduced,
            "state_exposure_sha256": hashlib.sha256(
                json.dumps({str(v): state_current[v] for v in sorted(state_current)},
                           sort_keys=True).encode("utf-8")).hexdigest(),
            "seed_mask_liveness_control": liveness,
        },
        "stream_check": checks,
        "frozen_candidates": {
            "file": str(args.candidate_file.relative_to(ROOT)),
            "payload_sha256": digest,
            "written_before_evaluation": True,
            "evaluation_order": frozen_order,
            "choices": {label: choices[label] for label in (*VERSIONS, "degree")},
            "reference_candidate": ref,
            "same_choice_pairs": payload["same_choice_pairs"],
        },
        "confirmation_batch": {
            "trials": args.trials, "fixed_in_advance": True,
            "stopped_early": False,
            "namespace": BATCH3_NS,
            "cascades": (1 + len(frozen_order)) * args.trials + state_reads,
            "base_counts": describe(base_counts),
            "wall_seconds": confirm_seconds,
            "shared_windows_within_trial": True,
            "coin_stream_note": "activation flips are drawn sequentially from the per-trial stream, "
                                "so results depend on the frozen evaluation order",
        },
        "gain_decomposition": decomposition,
        # The per-trial arrays themselves, so the paired intervals can be recomputed from the file
        # alone.  They are a deterministic REPLAY: the first version of this run computed these
        # arrays in memory and wrote only the summaries.  Re-running the same frozen streams
        # reproduces them bit-for-bit, which is a backfill of the same evidence, NOT new evidence.
        "per_trial_data": {
            "replay_of_identical_streams": True,
            "not_new_evidence": True,
            "note": "recomputed from the same frozen streams as the published summaries; the "
                    "summaries are asserted unchanged against the previously committed artifact",
            "graph": GRAPH, "trials": args.trials,
            "stream_namespace": BATCH3_NS,
            "candidates": {str(c): per_candidate[c] for c in frozen_order},
        },
        "effect_size_bound": {
            "difference_definition": "A - D = (static choice) - (state choice), paired per trial",
            "point": primary.get("difference_a_minus_b", {}).get("mean"),
            "ci95": [primary.get("difference_a_minus_b", {}).get("ci95_low"),
                     primary.get("difference_a_minus_b", {}).get("ci95_high")],
            "upper_bound_on_state_method_loss": primary.get("difference_a_minus_b", {}).get("ci95_high"),
            "upper_bound_on_static_method_loss":
                -primary.get("difference_a_minus_b", {}).get("ci95_low", 0.0)
                if primary.get("difference_a_minus_b") else None,
            "reading": "with the difference defined as static minus state, the interval's UPPER end "
                       "is the largest state-method loss the data leave open, and its LOWER end is "
                       "the largest static-method loss.  Neither direction is established.",
            "no_threshold_applied": "no gate tolerance is applied to this interval, and none is "
                                    "claimed: being a small fraction of |D| is not evidence of "
                                    "equivalence and is not used as one here",
        },
        "contrasts": {
            "primary_A_minus_D": primary,
            "secondary": secondary,
            "distinct_candidate_pairs": distinct,
            "multiplicity": multiplicity,
            "reading": "the primary contrast is the only pre-registered one; the four secondary "
                       "contrasts are mechanism exploration",
        },
        "score_side": score_side,
        "factor_effects_on_the_ranking": factor_effects,
        "confounding": {
            "same_choice_pairs": payload["same_choice_pairs"],
            "note": "CORRECTED.  The two input factors are NOT both able to change the choice.  "
                    "Versions A and B chose the same candidate and C and D chose the same candidate, "
                    "so the seed mask changes nothing here; the exposure input alone produces the "
                    "reversal, and the factor-effect block below measures that directly.",
        },
        "score_versus_gain_warning":
            "a negative score is not evidence of a negative true gain; the confirmation batch is the "
            "only thing that measures the gain",
        "nodes_of_interest": {str(k): v for k, v in NODES_OF_INTEREST.items()},
        "wall_seconds": time.time() - t_start,
        "complete": True,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- console summary
    print()
    print("=" * 100)
    print(f"  ca-GrQc 1%  |S|={k}  |D|={len(target)}  batch 3 = {args.trials} trials  "
          f"({confirm_seconds:.0f}s, {artifact['confirmation_batch']['cascades']} cascades)")
    print("=" * 100)
    print(f"  base |P_final(S) cap D|: {artifact['confirmation_batch']['base_counts']['mean']:.3f}"
          f" +-{artifact['confirmation_batch']['base_counts']['se']:.3f}")
    print()
    print(f"  {'cand':>7}{'marginal':>11}{'95% CI':>22}{'new':>8}{'lost':>8}  role")
    roles = {choices[v]["chosen"]: f"{v} choice" for v in VERSIONS}
    roles[choices["degree"]["chosen"]] = roles.get(choices["degree"]["chosen"], "") + " degree"
    roles[ref] = roles.get(ref, "") + " MC reference"
    for candidate in frozen_order:
        d = decomposition[str(candidate)]["marginal"]
        ci = f"[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]"
        print(f"  {candidate:>7}{d['mean']:>11.3f}{ci:>22}"
              f"{decomposition[str(candidate)]['newly_positive_mean']:>8.3f}"
              f"{decomposition[str(candidate)]['lost_positive_mean']:>8.3f}  {roles.get(candidate, '')}")
    print()
    print("  PRIMARY  A - D")
    if primary["same_choice"]:
        print(f"    same choice ({primary['chosen']}) -- no contrast exists")
    else:
        r = primary["difference_a_minus_b"]
        print(f"    {primary['chosen_a']} - {primary['chosen_b']} = {r['mean']:+.3f} "
              f"+-{r['se']:.3f}  95% CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
              f"p={r['p_two_sided']:.4g}")
    print()
    print("  SECONDARY (mechanism exploration)")
    for name, res in secondary.items():
        if res["same_choice"]:
            print(f"    {name:<7} same choice ({res['chosen']}) -- not independent evidence")
        else:
            r = res["difference_a_minus_b"]
            print(f"    {name:<7} {res['chosen_a']} - {res['chosen_b']} = {r['mean']:+.3f} "
                  f"+-{r['se']:.3f}  CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
                  f"p={r['p_two_sided']:.4g}")
    print(f"    distinct candidate pairs across all contrasts: {distinct}")
    print(f"    {multiplicity.get('reason', multiplicity.get('method', ''))}")
    print()
    print("  SCORE SIDE for the nodes the original report singled out")
    for node in NODES_OF_INTEREST:
        row = score_side[str(node)]
        if row["A"] is None:
            print(f"    {node}: not in the candidate pool")
            continue
        print(f"    {node}: " + "  ".join(
            f"{lab} {row[lab]['score']:+.4f} (rank {row[lab]['rank']}, "
            f"{'in' if row[lab]['in_top8'] else 'out'})" for lab in ("A", "B", "C", "D")))
    print()
    print("  WHICH INPUT MOVES THE RANKING")
    for name, fe in factor_effects.items():
        print(f"    {name:<7} {fe['changed']:<42} "
              f"rank of {ref}: {fe['reference_rank_from']:>2} -> {fe['reference_rank_to']:<2} "
              f"moved={fe['candidates_changing_position']:>2}/50 "
              f"rho={fe['spearman_between_scores']:+.3f} top8same={fe['identical_top8']}")
    print(f"    seed-mask liveness: seeds in D = {liveness['seeds_in_target']}, "
          f"candidates touching a seed in 2 hops = "
          f"{liveness['candidates_whose_two_hop_reach_touches_a_seed']}/{len(candidates)}, "
          f"widened-mask control change = "
          f"{liveness['control_widened_mask_max_abs_score_change']:.6f}")
    print()
    print(f"  wrote {args.output}")
    print(f"  wall {artifact['wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
