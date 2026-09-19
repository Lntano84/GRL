"""Selection-budget curve: does more selection simulation improve which candidate is chosen?

The question
------------
``docs/GRQC_NEW_STATES.md`` found that at a 1000-trial selection budget an expensive Monte-Carlo
screen never beat ``degree``, and recorded a plausible explanation: the selection-batch argmax over 50
noisy estimates may be largely the argmax of the noise.  That explanation is **not** established --
the observed selection-versus-evaluation gap is a difference between two finite batches, and the
evaluation batch has error of its own.  This experiment tests the explanation instead of assuming it:

> If the selection budget is raised, does the *choice* improve?

What is frozen (read from disk, never regenerated)
--------------------------------------------------
The four configurations' ``S``, their 50 candidates in their original order, the same target set ``D``,
the diffusion parameters, the weight handling, and the original ``degree`` top-8.  No new graph, no
model training, no re-estimation of the state.  Nothing here uses the state-conditioned score.

The two selection methods
-------------------------
=================  =====================================================================
``full50``         the highest selection-batch mean over all 50 candidates
``degree8``        the highest selection-batch mean **within the fixed degree top-8** -- degree
                   filtering plus Monte-Carlo verification, which is not a zero-simulation method
=================  =====================================================================

Ties are broken by ascending node id, fixed in advance and never adjusted to a result.  Both methods
read the **same paired data**: one window draw per trial, shared by all 50 candidates.

The three budgets are nested prefixes of one stream
---------------------------------------------------
1000, 3000 and 10000 trials per candidate.  Trial ``t`` is computed from its own seed and does not
depend on any other trial, so the 3000-trial selection is the 10000-trial selection restricted to its
first 3000 trials.  **All three budgets are computed and frozen before any confirmation batch exists**,
and no result is inspected to decide whether to go on to 10000; the chunk workers report progress
counts only.  Sharing a prefix across budgets is deliberate and is not a stream overlap -- different
*purposes* must not share a stream, and here the confirmation stream is entirely separate.

The confirmation batch
----------------------
A new 6000-trial stream (``CONFIRM_NS``) that overlaps no old batch and no part of the selection
stream, checked over the actual stream lengths rather than an assumed 1000.  Only the union of the
distinct candidates the six selections picked is evaluated -- two methods times three budgets is at
most six candidates, not fifty.  Per trial: ``marginal``, ``newly_positive`` and ``lost_positive``,
with ``marginal = newly_positive - lost_positive`` asserted on every trial.

The old evaluation batches are **historical only** and are not mixed into this confirmation.  The
6000 trials are a fixed budget and do not guarantee significance.
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

from go_no_go import load_raw  # noqa: E402
from grl.data.weights import SUM_TO_ONE, normalise_in_weights  # noqa: E402
from grl.diffusion import overexposure as oe  # noqa: E402
from grl.diffusion.contract import resolve_target_contract  # noqa: E402
from grl.oracle import trial_seeds  # noqa: E402
from grl.scoring import rank_by_score  # noqa: E402

GRAPH = "ca_grqc"
FRACTION = 0.01
TARGET_FRACTION = 0.2
SHORTLIST = 8

NEW_SEEDS = [20261001, 20261002, 20261003, 20261004]
BUDGETS = (1000, 3000, 10000)
MAX_TRIALS = max(BUDGETS)
CONFIRM_TRIALS = 6000
#: Selection keeps the original namespace, because the first 1000 trials must be the stored ones.
SELECT_NS = 1_100_000
#: The confirmation stream is new.  It must not touch any old batch or the selection stream.
CONFIRM_NS = 1_700_000
#: The original configuration's streams, for the overlap check.
ORIGINAL_SEED = 20260917
ORIGINAL_NS = {"state": 900_000, "batch1": 1_100_000, "batch2": 1_300_000, "batch3": 1_500_000}
ORIGINAL_CONFIRM_TRIALS = 6000

#: Selection trial chunks.  Trials are independent, so a chunk is exactly the corresponding slice of
#: one sequential run; the boundaries are fixed here and recorded in the artifact.
CHUNKS = ((0, 3334), (3334, 6668), (6668, 10000))

SOURCE_CONFIG = ROOT / "docs" / "results" / "grqc_new_states_cfg{index}.json"
#: The degree top-8 lives in the frozen-choices file, not in the per-configuration artifact.
SOURCE_CHOICES = ROOT / "docs" / "results" / "grqc_new_states_choices_cfg{index}.json"
TARGETS = ROOT / "docs" / "results" / "shortlist_headroom.json"
SEL_CHUNK = ROOT / "docs" / "results" / "sbc_sel_cfg{index}_{start:05d}_{end:05d}.json"
CHOICES = ROOT / "docs" / "results" / "sbc_choices_cfg{index}.json"
CONFIRM = ROOT / "docs" / "results" / "sbc_confirm_cfg{index}.json"
OUTPUT = ROOT / "docs" / "results" / "selection_budget_curve.json"

METHODS = ("full50", "degree8")


# ------------------------------------------------------------------------------------ utilities
def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def payload_digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "payload_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def two_sided_p(mean: float, se: float) -> float:
    if not math.isfinite(se) or se <= 0:
        return 1.0
    return 2.0 * (1.0 - 0.5 * math.erfc(-abs(mean) / se / math.sqrt(2.0)))


def describe(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else float("inf")
    return {"mean": mean, "sd": math.sqrt(var), "se": se, "n": n,
            "ci95_low": mean - 1.96 * se if math.isfinite(se) else float("-inf"),
            "ci95_high": mean + 1.96 * se if math.isfinite(se) else float("inf")}


def paired(a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    out = describe(diffs)
    out["p_two_sided"] = two_sided_p(out["mean"], out["se"])
    return out


def holm(pairs: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted(pairs.items(), key=lambda kv: kv[1])
    m = len(ordered)
    running, out = 0.0, {}
    for i, (name, p) in enumerate(ordered):
        value = min(1.0, (m - i) * p)
        running = max(running, value)
        out[name] = {"p_raw": p, "factor": m - i, "p_adjusted": running,
                     "significant_at_0_05": bool(running < alpha)}
    return {"method": "Holm-Bonferroni", "alpha": alpha, "comparisons": m, "per_comparison": out}


def stream_overlap_check() -> dict:
    """Every stream this experiment touches, at its ACTUAL length, pairwise.

    The old check assumed 1000 trials for every stream and never examined the last 5000 seeds of the
    original 6000-trial confirmation stream.  Budgets within the selection purpose share a prefix by
    design; anything else overlapping would be a fault.
    """
    streams: dict[str, set[int]] = {}
    for name, ns in ORIGINAL_NS.items():
        n = ORIGINAL_CONFIRM_TRIALS if name == "batch3" else 1000
        streams[f"orig_{name}"] = set(trial_seeds(ORIGINAL_SEED + ns, n))
    for i, seed in enumerate(NEW_SEEDS):
        streams[f"cfg{i}_select_10000"] = set(trial_seeds(seed + SELECT_NS, MAX_TRIALS))
        streams[f"cfg{i}_confirm_6000"] = set(trial_seeds(seed + CONFIRM_NS, CONFIRM_TRIALS))
    names = sorted(streams)
    overlaps = {f"{a}|{b}": len(streams[a] & streams[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    bad = {k: v for k, v in overlaps.items() if v}
    return {
        "streams_checked": len(names), "pairs_checked": len(overlaps),
        "stream_lengths": {n: len(s) for n, s in streams.items()},
        "budgets": {"selection": MAX_TRIALS, "confirmation": CONFIRM_TRIALS,
                    "nested_prefixes": list(BUDGETS)},
        "nested_prefixes_are_intentional":
            "the 1000- and 3000-trial selections are prefixes of the 10000-trial selection, which is "
            "what makes the three budget levels comparable on identical data",
        "overlapping_pairs": bad, "disjoint": not bad,
    }


def load_configuration(index: int, graph, contract, k: int) -> dict:
    """Read S, the 50 candidates and the degree top-8 from disk.  Nothing is regenerated."""
    stored = json.loads(SOURCE_CONFIG.with_name(SOURCE_CONFIG.name.format(index=index))
                        .read_text(encoding="utf-8"))
    frozen_prev = json.loads(SOURCE_CHOICES.with_name(SOURCE_CHOICES.name.format(index=index))
                             .read_text(encoding="utf-8"))
    seeds = [int(s) for s in stored["configuration"]["existing_seeds"]]
    candidates = [int(c) for c in stored["configuration"]["candidates"]]
    degree_top8 = [int(c) for c in frozen_prev["choices"]["degree"]["top8"]]
    if sorted(int(t) for t in contract.objective.target_set) != sorted(
            int(t) for t in json.loads(TARGETS.read_text(encoding="utf-8"))["blocks"][2]["target_set"]):
        raise SystemExit("target set D differs from the original; it must be held fixed")
    # the degree top-8 is recomputed only to CHECK the stored list, never to replace it
    recomputed = rank_by_score(candidates, [float(graph.out_degree(v)) for v in candidates])[:SHORTLIST]
    if recomputed != degree_top8:
        raise SystemExit(f"stored degree top-8 for cfg{index} is not the degree top-8 of its pool")
    return {"seeds": seeds, "candidates": candidates, "degree_top8": degree_top8,
            "stored": stored}


def selection_chunk(index: int, start: int, end: int, checkpoint_every: int = 200) -> int:
    """Per-trial selection marginals for trials ``[start, end)``, all 50 candidates, one window draw.

    **Resumable.**  Trials are independent and each is derived from its own seed, so an interrupted
    chunk continues from where it stopped without changing a single number.  The arrays are written
    every ``checkpoint_every`` trials and validated by digest on restart.

    This exists because the first attempt lost roughly 1080 CPU-minutes: the workers wrote their
    output only at the very end, and a dropped connection killed all twelve with nothing on disk.
    """
    path = SEL_CHUNK.with_name(SEL_CHUNK.name.format(index=index, start=start, end=end))
    graph = normalise_in_weights(load_raw(GRAPH).copy(), SUM_TO_ONE)
    n = len(graph.nodes())
    k = max(1, int(round(FRACTION * n)))
    contract = resolve_target_contract(graph, "degree-tail", TARGET_FRACTION, k)
    cfg = load_configuration(index, graph, contract, k)
    candidates, seeds = cfg["candidates"], cfg["seeds"]

    per: dict[int, list[float]] = {c: [] for c in candidates}
    base_counts: list[float] = []
    done = 0
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        usable = (prev.get("payload_sha256") == payload_digest(prev)
                  and prev.get("index") == index
                  and prev.get("trial_start") == start and prev.get("trial_end") == end
                  and [int(c) for c in prev.get("candidates", [])] == candidates
                  and [int(s) for s in prev.get("existing_seeds", [])] == seeds)
        if not usable:
            raise SystemExit(f"{path.name} exists but does not match this run; refusing to reuse it")
        if prev.get("complete"):
            print(f"  cfg{index} selection chunk [{start},{end}) already complete; nothing to do",
                  flush=True)
            return 0
        for c in candidates:
            per[c] = [float(v) for v in prev["per_trial_marginals"][str(c)]]
        base_counts = [float(v) for v in prev["base_counts"]]
        done = len(base_counts)
        if any(len(per[c]) != done for c in candidates):
            raise SystemExit(f"{path.name} has ragged per-candidate arrays")
        print(f"  cfg{index} chunk [{start},{end}): resuming at trial {done}", flush=True)

    def write(complete: bool, seconds: float) -> str:
        payload = {
            "script": Path(__file__).name, "code_version": code_version(),
            "index": index, "random_seed": NEW_SEEDS[index],
            "trial_start": start, "trial_end": end, "trials_done": len(base_counts),
            "stream": NEW_SEEDS[index] + SELECT_NS,
            "existing_seeds": seeds, "candidates": candidates,
            "per_trial_marginals": {str(c): per[c] for c in candidates},
            "base_counts": base_counts,
            "resumable": True, "checkpoint_every": checkpoint_every,
            "complete": complete, "wall_seconds": seconds,
        }
        payload["payload_sha256"] = payload_digest(payload)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic, so a drop mid-write cannot leave a half-written file
        return payload["payload_sha256"]

    target = set(contract.objective.target_set)
    nodes = list(graph.nodes())
    mode = contract.diffusion.activation_mode
    law = contract.diffusion.threshold_law
    lo = contract.diffusion.window_lo

    all_seeds = trial_seeds(NEW_SEEDS[index] + SELECT_NS, end)
    t0 = time.time()
    for i in range(start + done, end):
        rng = random.Random(all_seeds[i])
        windows = oe.sample_threshold_windows(nodes, rng, threshold_law=law, window_lo=lo)
        base_run = oe.run_overexposure(graph, list(seeds), windows, rng, activation_mode=mode)
        base = float(len({int(v) for v in base_run.positive} & target))
        base_counts.append(base)
        # the candidate order is the stored one: it decides the order coin flips are consumed, so it
        # is part of the frozen configuration and not a free choice
        for candidate in candidates:
            run = oe.run_overexposure(graph, [*seeds, candidate], windows, rng, activation_mode=mode)
            per[candidate].append(float(len({int(v) for v in run.positive} & target)) - base)
        if (i - start + 1) % checkpoint_every == 0:
            write(False, time.time() - t0)
            print(f"    cfg{index} chunk [{start},{end}) {i + 1 - start}/{end - start} "
                  f"({time.time() - t0:.0f}s) checkpointed", flush=True)

    digest = write(True, time.time() - t0)
    print(f"  cfg{index} selection chunk [{start},{end}) complete in {time.time() - t0:.0f}s "
          f"-> {path.name} sha={digest[:12]}", flush=True)
    return 0


def assemble(index: int) -> int:
    """Concatenate the chunks, verify the first 1000 trials against the stored batch, freeze choices."""
    stored = json.loads(SOURCE_CONFIG.with_name(SOURCE_CONFIG.name.format(index=index))
                        .read_text(encoding="utf-8"))
    frozen_prev = json.loads(SOURCE_CHOICES.with_name(SOURCE_CHOICES.name.format(index=index))
                             .read_text(encoding="utf-8"))
    candidates = [int(c) for c in stored["configuration"]["candidates"]]
    degree_top8 = [int(c) for c in frozen_prev["choices"]["degree"]["top8"]]

    per = {c: [] for c in candidates}
    base_counts: list[float] = []
    chunk_hashes = {}
    for start, end in CHUNKS:
        path = SEL_CHUNK.with_name(SEL_CHUNK.name.format(index=index, start=start, end=end))
        if not path.exists():
            raise SystemExit(f"missing selection chunk {path.name}; refusing to assemble")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("payload_sha256") != payload_digest(payload):
            raise SystemExit(f"{path.name} changed after being written")
        if not payload.get("complete"):
            raise SystemExit(f"{path.name} is an incomplete checkpoint "
                             f"({payload.get('trials_done')} of {end - start} trials); resume it first")
        if [int(c) for c in payload["candidates"]] != candidates:
            raise SystemExit(f"{path.name} used a different candidate list")
        chunk_hashes[path.name] = payload["payload_sha256"]
        for c in candidates:
            per[c].extend(float(v) for v in payload["per_trial_marginals"][str(c)])
        base_counts.extend(float(v) for v in payload["base_counts"])
    if len(per[candidates[0]]) != MAX_TRIALS:
        raise SystemExit(f"assembled {len(per[candidates[0]])} trials, expected {MAX_TRIALS}")

    # ---- reproduce the stored 1000-trial selection exactly ----
    old = stored["per_trial_selection_marginals"]
    compare_trials = min(1000, len(per[candidates[0]]))
    max_diff = max(abs(a - float(b)) for c in candidates
                   for a, b in zip(per[c][:compare_trials], old[str(c)][:compare_trials]))
    # the stored base counts are a summary, not a per-trial array, so only the mean is comparable
    base_diff = abs(statistics.fmean(base_counts[:compare_trials])
                    - float(stored["selection_base_counts"]["mean"]))
    reproduction = {"trials_compared": compare_trials,
                    "max_abs_marginal_difference": max_diff,
                    "abs_base_count_mean_difference": base_diff,
                    "exact": max_diff == 0.0 and base_diff == 0.0}
    if not reproduction["exact"]:
        (ROOT / "docs" / "results" / "sbc_reproduction_failure.json").write_text(
            json.dumps({"index": index, "reproduction": reproduction}, indent=2), encoding="utf-8")
        raise SystemExit(f"cfg{index}: the first 1000 selection trials do not reproduce the stored "
                         f"batch (max diff {max_diff}); stopping before any choice is frozen")

    # ---- the six selections, frozen here, before any confirmation batch exists ----
    def pick(short: list[int], trials: int) -> int:
        means = {c: sum(per[c][:trials]) / trials for c in short}
        return max(short, key=lambda c: (means[c], -c))  # ties -> smallest node id, fixed in advance

    selection: dict[str, dict] = {}
    for budget in BUDGETS:
        full = pick(candidates, budget)
        deg = pick(degree_top8, budget)
        for method, chosen, pool in (("full50", full, candidates), ("degree8", deg, degree_top8)):
            means = {c: sum(per[c][:budget]) / budget for c in candidates}
            selection[f"{method}@{budget}"] = {
                "method": method, "budget": budget, "chosen": int(chosen),
                "candidate_pool": "all 50" if method == "full50" else "degree top-8",
                "pool": [int(c) for c in pool],
                "chosen_selection_mean": means[chosen],
                "best_in_pool_selection_mean": max(means[c] for c in pool),
            }

    # cross-checks against the previously stored choices, which are exactly these two methods at 1000
    stored_checks = {
        "full50@1000_equals_stored_mc_reference":
            selection["full50@1000"]["chosen"] == int(stored["methods"]["mc_reference"]["chosen"]),
        "degree8@1000_equals_stored_degree_choice":
            selection["degree8@1000"]["chosen"] == int(stored["methods"]["degree"]["chosen"]),
    }
    if not all(stored_checks.values()):
        raise SystemExit(f"cfg{index}: frozen 1000-trial choices disagree with the stored run: "
                         f"{stored_checks}")

    order = []
    for budget in BUDGETS:
        for method in METHODS:
            c = selection[f"{method}@{budget}"]["chosen"]
            if c not in order:
                order.append(c)

    frozen = {
        "index": index, "random_seed": NEW_SEEDS[index], "graph": GRAPH,
        "budgets": list(BUDGETS), "methods": list(METHODS),
        "tie_break": "highest selection-batch mean; ties broken by ascending node id",
        "selection": selection,
        "confirmation_candidates": order,
        "confirmation_candidate_count": len(order),
        "chunk_hashes": chunk_hashes,
        "reproduction_of_stored_1000": reproduction,
        "stored_cross_checks": stored_checks,
        "frozen_before_confirmation": True,
        "code_version": code_version(),
        "note": "all three budgets were computed and frozen before any confirmation batch existed",
    }
    digest = payload_digest(frozen)
    frozen["payload_sha256"] = digest
    path = CHOICES.with_name(CHOICES.name.format(index=index))
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"  cfg{index} frozen: " + "  ".join(
        f"{m}@{b}={selection[f'{m}@{b}']['chosen']}" for b in BUDGETS for m in METHODS)
        + f"  confirm-candidates={order}  sha={digest[:12]}", flush=True)
    print(f"  cfg{index} reproduced the stored 1000-trial selection exactly: "
          f"{reproduction['exact']}", flush=True)
    return 0


def confirm(index: int, trials: int) -> int:
    """6000-trial independent confirmation of the union of frozen choices, with per-trial arrays."""
    frozen = json.loads(CHOICES.with_name(CHOICES.name.format(index=index)).read_text(encoding="utf-8"))
    if frozen.get("payload_sha256") != payload_digest(frozen):
        raise SystemExit(f"cfg{index}: frozen choices changed after being written")
    order = [int(c) for c in frozen["confirmation_candidates"]]

    graph = normalise_in_weights(load_raw(GRAPH).copy(), SUM_TO_ONE)
    n = len(graph.nodes())
    k = max(1, int(round(FRACTION * n)))
    contract = resolve_target_contract(graph, "degree-tail", TARGET_FRACTION, k)
    cfg = load_configuration(index, graph, contract, k)
    seeds = cfg["seeds"]
    target = set(contract.objective.target_set)
    nodes = list(graph.nodes())
    mode = contract.diffusion.activation_mode
    law = contract.diffusion.threshold_law
    lo = contract.diffusion.window_lo

    per = {c: {"marginal": [], "newly_positive": [], "lost_positive": []} for c in order}
    base_counts: list[float] = []
    path = CONFIRM.with_name(CONFIRM.name.format(index=index))
    done = 0
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        ok = (prev.get("payload_sha256") == payload_digest(prev)
              and prev.get("index") == index and prev.get("trials") == trials
              and prev.get("frozen_choices_sha256") == frozen["payload_sha256"]
              and [int(c) for c in prev.get("candidates", [])] == order)
        if not ok:
            raise SystemExit(f"{path.name} exists but does not match this run; refusing to reuse it")
        if prev.get("complete"):
            print(f"  cfg{index} confirmation already complete; nothing to do", flush=True)
            return 0
        per = {int(c): {k: [float(x) for x in v] for k, v in prev["per_trial"][c].items()}
               for c in prev["per_trial"]}
        base_counts = [float(v) for v in prev["base_counts"]]
        done = len(base_counts)
        print(f"  cfg{index} confirmation: resuming at trial {done}", flush=True)

    def write(complete: bool, seconds: float) -> None:
        payload = {
            "script": Path(__file__).name, "code_version": code_version(),
            "index": index, "random_seed": NEW_SEEDS[index],
            "trials": trials, "trials_done": len(base_counts),
            "stream": NEW_SEEDS[index] + CONFIRM_NS,
            "candidates": order, "existing_seeds": seeds,
            "frozen_choices_sha256": frozen["payload_sha256"],
            "per_trial": {str(c): per[c] for c in order},
            "base_counts": base_counts,
            "resumable": True, "wall_seconds": seconds, "complete": complete,
        }
        payload["payload_sha256"] = payload_digest(payload)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    t0 = time.time()
    for i, seed in enumerate(trial_seeds(NEW_SEEDS[index] + CONFIRM_NS, trials)):
        if i < done:
            continue
        rng = random.Random(seed)
        windows = oe.sample_threshold_windows(nodes, rng, threshold_law=law, window_lo=lo)
        base_run = oe.run_overexposure(graph, list(seeds), windows, rng, activation_mode=mode)
        base_pos = {int(v) for v in base_run.positive} & target
        base_counts.append(float(len(base_pos)))
        for candidate in order:
            run = oe.run_overexposure(graph, [*seeds, candidate], windows, rng, activation_mode=mode)
            pos = {int(v) for v in run.positive} & target
            gained, lost = len(pos - base_pos), len(base_pos - pos)
            if float(len(pos) - len(base_pos)) != float(gained - lost):
                raise SystemExit(f"decomposition broken at trial {i}, candidate {candidate}")
            per[candidate]["marginal"].append(float(len(pos) - len(base_pos)))
            per[candidate]["newly_positive"].append(float(gained))
            per[candidate]["lost_positive"].append(float(lost))
        if (i + 1) % 1000 == 0:
            write(False, time.time() - t0)
            print(f"    cfg{index} confirmation {i + 1}/{trials} ({time.time() - t0:.0f}s) "
                  f"checkpointed", flush=True)

    write(True, time.time() - t0)
    print(f"  cfg{index} confirmation complete in {time.time() - t0:.0f}s -> {path.name}", flush=True)
    return 0


def merge() -> int:
    blocks, holm_inputs = [], {}
    for index in range(len(NEW_SEEDS)):
        frozen = json.loads(CHOICES.with_name(CHOICES.name.format(index=index))
                            .read_text(encoding="utf-8"))
        if frozen.get("payload_sha256") != payload_digest(frozen):
            raise SystemExit(f"cfg{index}: frozen choices changed after being written")
        conf = json.loads(CONFIRM.with_name(CONFIRM.name.format(index=index))
                          .read_text(encoding="utf-8"))
        if conf.get("payload_sha256") != payload_digest(conf):
            raise SystemExit(f"cfg{index}: confirmation file changed after being written")
        if conf["frozen_choices_sha256"] != frozen["payload_sha256"]:
            raise SystemExit(f"cfg{index}: confirmation was not produced from the frozen choices")
        if conf["trials"] != CONFIRM_TRIALS:
            raise SystemExit(f"cfg{index}: confirmation used {conf['trials']} trials")
        if not conf.get("complete"):
            raise SystemExit(f"cfg{index}: confirmation is an incomplete checkpoint "
                             f"({conf.get('trials_done')} of {CONFIRM_TRIALS} trials)")

        per = {int(c): conf["per_trial"][c] for c in conf["per_trial"]}
        order = [int(c) for c in conf["candidates"]]
        sel = frozen["selection"]

        rows = []
        for budget in BUDGETS:
            f, d = sel[f"full50@{budget}"]["chosen"], sel[f"degree8@{budget}"]["chosen"]
            row = {
                "configuration": index, "random_seed": NEW_SEEDS[index], "budget": budget,
                "full50_candidate": f, "degree8_candidate": d,
                "same_choice": f == d,
                "full50_confirmation": describe(per[f]["marginal"]),
                "degree8_confirmation": describe(per[d]["marginal"]),
                "selection_cascades": budget * (1 + 50),
            }
            if f == d:
                row["difference_full50_minus_degree8"] = {
                    "same_choice": True, "chosen": f,
                    "note": "both methods selected the SAME candidate at this budget; the contrast is "
                            "exactly zero by construction and is NOT independent evidence"}
            else:
                row["difference_full50_minus_degree8"] = dict(
                    paired(per[f]["marginal"], per[d]["marginal"]), same_choice=False)
            rows.append(row)
        primary = next(r for r in rows if r["budget"] == max(BUDGETS))
        if not primary["same_choice"]:
            holm_inputs[str(index)] = primary["difference_full50_minus_degree8"]["p_two_sided"]

        budget_effects = {}
        for method in METHODS:
            a = sel[f"{method}@{BUDGETS[0]}"]["chosen"]
            c = sel[f"{method}@{BUDGETS[-1]}"]["chosen"]
            budget_effects[method] = {
                "from_budget": BUDGETS[0], "to_budget": BUDGETS[-1],
                "from_candidate": a, "to_candidate": c, "same_choice": a == c,
            }
            if a != c:
                budget_effects[method].update(
                    dict(paired(per[c]["marginal"], per[a]["marginal"]), same_choice=False))
            else:
                budget_effects[method]["note"] = ("the higher budget selected the same candidate; "
                                                  "there is no budget effect to measure")

        blocks.append({
            "configuration": index, "random_seed": NEW_SEEDS[index],
            "existing_seeds": conf["existing_seeds"],
            "degree_top8": json.loads(SOURCE_CHOICES.with_name(
                SOURCE_CHOICES.name.format(index=index)).read_text(encoding="utf-8"))["choices"]["degree"]["top8"],
            "frozen_choices_sha256": frozen["payload_sha256"],
            "confirmation_sha256": conf["payload_sha256"],
            "reproduction_of_stored_1000": frozen["reproduction_of_stored_1000"],
            "stored_cross_checks": frozen["stored_cross_checks"],
            "confirmation_candidates": order,
            "confirmation_trials": conf["trials"],
            "confirmation_cascades": conf["trials"] * (1 + len(order)),
            "confirmation_base_counts": describe(conf["base_counts"]),
            "rows": rows,
            "budget_effect": budget_effects,
            "per_candidate_confirmation": {
                str(c): {"marginal": describe(per[c]["marginal"]),
                         "newly_positive_mean": sum(per[c]["newly_positive"]) / conf["trials"],
                         "lost_positive_mean": sum(per[c]["lost_positive"]) / conf["trials"],
                         "identity_holds": all(
                             abs(m - (g - l)) < 1e-9
                             for m, g, l in zip(per[c]["marginal"], per[c]["newly_positive"],
                                                per[c]["lost_positive"]))}
                for c in order},
        })

    multiplicity = holm(holm_inputs) if holm_inputs else {
        "method": "Holm-Bonferroni", "note": "every configuration's primary contrast was a same-choice "
                                             "tie, so there was nothing to correct"}
    sel_total = sum(r["selection_cascades"] for b in blocks for r in b["rows"] if r["budget"] == max(BUDGETS))
    conf_total = sum(b["confirmation_cascades"] for b in blocks)

    artifact = {
        "script": Path(__file__).name, "code_version": code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "question": "does raising the selection budget improve which candidate is chosen?",
        "objective": "Delta(v|S) = |P_final(S u {v}) cap D| - |P_final(S) cap D|",
        "frozen": ["the four configurations' S and 50 candidates in their original order",
                   "the same target set D", "diffusion parameters and weight handling",
                   "the original degree top-8", "no model training", "no state re-estimation"],
        "changed": "the selection budget alone: 1000, 3000 and 10000 trials per candidate",
        "methods": {"full50": "highest selection-batch mean over all 50 candidates",
                    "degree8": "highest selection-batch mean within the fixed degree top-8 "
                               "(degree filtering plus Monte-Carlo verification, not zero-cost)"},
        "tie_break": "highest selection-batch mean; ties broken by ascending node id",
        "stream_check": stream_overlap_check(),
        "confirmation": {"trials": CONFIRM_TRIALS, "namespace": CONFIRM_NS,
                         "new_stream": True,
                         "old_evaluation_batches": "historical only, not mixed into this confirmation",
                         "selection_cascades_total": sel_total,
                         "confirmation_cascades_total": conf_total,
                         "total_cascades": sel_total + conf_total},
        "primary_comparison": {"definition": "full50 minus degree8 at budget 10000, paired per trial",
                               "holm_over_configurations": multiplicity},
        "auxiliary_comparison": {
            "definition": "full50 at budget 10000 minus full50 at budget 1000, paired per trial",
            "status": "EXPLORATORY; no multiplicity correction and no confirmatory claim"},
        "configurations": blocks,
        "no_pooling": "the four configurations are separate states; nothing is averaged across them",
        "complete": all(b["rows"] for b in blocks),
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ the twelve-row table
    print("=" * 118)
    print("  SELECTION BUDGET CURVE -- all four configurations, three budgets")
    print("=" * 118)
    print(f"  streams checked {artifact['stream_check']['streams_checked']} "
          f"({artifact['stream_check']['pairs_checked']} pairs), disjoint "
          f"{artifact['stream_check']['disjoint']}")
    print(f"  selection cascades {sel_total:,} + confirmation cascades {conf_total:,} "
          f"= {sel_total + conf_total:,}")
    print()
    print(f"  {'cfg':>3}{'budget':>7}  {'full50':>7}{'gain':>9}{'95% CI':>20}  "
          f"{'degree8':>8}{'gain':>9}{'95% CI':>20}  {'diff & CI':>22}")
    for b in blocks:
        for r in b["rows"]:
            f, d = r["full50_confirmation"], r["degree8_confirmation"]
            if r["same_choice"]:
                diff = "same choice"
            else:
                x = r["difference_full50_minus_degree8"]
                diff = f"{x['mean']:+.3f} [{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}]"
            fci = f"[{f['ci95_low']:+.3f},{f['ci95_high']:+.3f}]"
            dci = f"[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]"
            print(f"  {b['configuration']:>3}{r['budget']:>7}  {r['full50_candidate']:>7}"
                  f"{f['mean']:>9.3f}{fci:>20}  "
                  f"{r['degree8_candidate']:>8}{d['mean']:>9.3f}{dci:>20}  {diff:>22}")
    print()
    print("  PRIMARY: full50 - degree8 at budget 10000")
    for b in blocks:
        r = b["rows"][-1]
        if r["same_choice"]:
            print(f"    cfg{b['configuration']}  same choice ({r['full50_candidate']})")
        else:
            x = r["difference_full50_minus_degree8"]
            print(f"    cfg{b['configuration']}  {r['full50_candidate']} - {r['degree8_candidate']} "
                  f"= {x['mean']:+.3f} +-{x['se']:.3f} [{x['ci95_low']:+.3f}, {x['ci95_high']:+.3f}]"
                  f"  p={x['p_two_sided']:.3f}")
    print(f"    {multiplicity.get('method', '')}: "
          + (json.dumps({k: round(v["p_adjusted"], 3)
                         for k, v in multiplicity.get("per_comparison", {}).items()})
             if "per_comparison" in multiplicity else multiplicity.get("note", "")))
    print()
    print("  AUXILIARY (exploratory): full50 at 10000 minus full50 at 1000")
    for b in blocks:
        e = b["budget_effect"]["full50"]
        if e["same_choice"]:
            print(f"    cfg{b['configuration']}  same choice ({e['from_candidate']})")
        else:
            print(f"    cfg{b['configuration']}  {e['to_candidate']} - {e['from_candidate']} "
                  f"= {e['mean']:+.3f} +-{e['se']:.3f} [{e['ci95_low']:+.3f}, {e['ci95_high']:+.3f}]"
                  f"  p={e['p_two_sided']:.3f}")
    print()
    print(f"  wrote {OUTPUT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-streams-only", action="store_true")
    parser.add_argument("--select-chunk", nargs=3, metavar=("INDEX", "START", "END"), type=int)
    parser.add_argument("--assemble", type=int, metavar="INDEX")
    parser.add_argument("--confirm", type=int, metavar="INDEX")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--confirm-trials", type=int, default=CONFIRM_TRIALS)
    args = parser.parse_args()

    if args.check_streams_only:
        check = stream_overlap_check()
        print(json.dumps(check, indent=2))
        return 0 if check["disjoint"] else 1
    if args.select_chunk:
        index, start, end = args.select_chunk
        check = stream_overlap_check()
        if not check["disjoint"]:
            raise SystemExit(f"stream overlap: {check['overlapping_pairs']}")
        return selection_chunk(index, start, end)
    if args.assemble is not None:
        return assemble(args.assemble)
    if args.confirm is not None:
        return confirm(args.confirm, args.confirm_trials)
    if args.merge:
        return merge()
    parser.error("give one of --check-streams-only, --select-chunk, --assemble, --confirm, --merge")


if __name__ == "__main__":
    raise SystemExit(main())
