"""The budget-limited learning pilot: does ranking a batch of 32 make the search better, or just cost more?

The one question
----------------
Given the same batch of randomly drawn swap candidates, can the learned model put the swaps worth asking
about at the front, and does that improve the quality-versus-time behaviour of the complete search?

What is frozen, and what is not
-------------------------------
Frozen: the graph version (sha256 recorded), ``K = 4``, ``T = 52``, the degree-52 initial set, the
``5,000``-query budget, the ``60`` s cap, the original acceptance rule (the true ``sigma``, first
**strictly** improving swap), the cache, and the restart rule.  Batch size is ``32`` for all three arms.
The learned model is the checkpoint selected by the pre-registered validation rule, hash recorded below;
it is not retrained, not re-selected, and no second checkpoint is tried.

Not frozen, because it must vary: which of the 32 candidates is asked about first.  That is the whole
intervention.

The three arms
--------------
``random``   keep the draw order                       -- the baseline
``degree``   sort by ``degree(v) - degree(u)``          -- the simple structural baseline
``learned``  sort by the model's predicted swap gain

Three **new** pre-fixed search seeds (100, 101, 102), disjoint from the training/validation seeds 0/1/2.
The methods diverge after an accepted swap, so later candidates are not required to be identical across
arms; only the initialisation and the randomisation protocol are shared, and every arm issues exactly the
same number of true diffusion queries.

Costs that are *not* part of the search timing
----------------------------------------------
Label generation (the replay in ``collect_swap_records.py``), training
(``train_swap_ranker.py``) and model loading are reported **separately** and are never treated as free.
If the online run is not faster at all, there is no basis on which to amortise them, and the report says
so instead of hiding the bill.

Pre-fixed verdicts
------------------
Read from the numbers by :func:`verdict`, which is written before the run:

* query efficiency **and** wall clock both improve -> worth a next-round independent validation; still not
  a paper contribution from one development instance;
* query efficiency improves but the inference cost cancels it -> the model learned a useful ordering but
  there is no acceleration value;
* only individual seeds improve, or no clear gain -> stop this version of the learned ranker; do not move
  to bigger models or more training.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ccim.batched_search import batched_search  # noqa: E402
from ccim.learned_rank import RankModel  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

K = 4
SEED_FRACTION = 0.01
PILOT_SEEDS = (100, 101, 102)          # new, pre-fixed, disjoint from the train/validate seeds 0/1/2
ARMS = ("random", "degree", "learned")
BATCH_SIZE = 32
QUERY_BUDGET = 5_000
TIME_LIMIT = 60.0
RESTARTS = 6
CHECKPOINTS = (100, 300, 1_000, 3_000, 5_000)
MODEL = ROOT / "results" / "swap_ranker.pt"
TRAINING = ROOT / "results" / "swap_ranker_training.json"
OUTPUT = ROOT / "results" / "learned_ranking_pilot.json"


def queries_to_reach(history, target: float) -> int | None:
    """First query count at which the running best reached ``target``; ``None`` if it never did."""
    for q, best, *_ in history:
        if best >= target:
            return int(q)
    return None


def seconds_to_reach(history, target: float) -> float | None:
    for q, best, _distinct, secs, *_ in history:
        if best >= target:
            return float(secs)
    return None


def verdict(summary: dict) -> dict:
    """Apply the pre-fixed decision rule to the measured summary.  No numbers are seen before it is run."""
    mean = {a: summary["aggregate"][a]["mean_sigma"] for a in ARMS}
    secs = {a: summary["aggregate"][a]["mean_total_seconds"] for a in ARMS}
    per_seed_better_than_random = [s for s in summary["per_seed"]
                                   if s["arms"]["learned"]["sigma"] > s["arms"]["random"]["sigma"]]
    per_seed_worse_than_random = [s for s in summary["per_seed"]
                                  if s["arms"]["learned"]["sigma"] < s["arms"]["random"]["sigma"]]
    quality_clear = (mean["learned"] > mean["random"] and mean["learned"] > mean["degree"]
                     and not per_seed_worse_than_random)
    time_clear = secs["learned"] < secs["random"] and secs["learned"] < secs["degree"]
    if quality_clear and time_clear:
        label = "worth a next-round independent validation"
        detail = ("query efficiency and wall clock both improved.  Still not a paper contribution: this "
                  "is one development instance with three seeds, and the next round must validate on an "
                  "independent instance/graph before any such claim.")
    elif quality_clear and not time_clear:
        label = "useful ranking, no acceleration"
        detail = ("query efficiency improved but the added feature/inference/sort cost cancels it at this "
                  "scale, so the model learned a useful ordering without delivering a time gain.")
    elif per_seed_better_than_random and per_seed_worse_than_random:
        label = "no clear gain"
        detail = "the learned arm wins on some seeds and loses on others, which is seed noise, not a gain."
    else:
        label = "no clear gain"
        detail = "no consistent improvement over the two baselines."
    stop = label in ("no clear gain", "useful ranking, no acceleration")
    return {"label": label, "detail": detail,
            "quality_clear": quality_clear, "time_clear": time_clear,
            "mean_sigma": mean, "mean_total_seconds": secs,
            "learned_beats_random_on_seeds": per_seed_better_than_random,
            "learned_loses_to_random_on_seeds": per_seed_worse_than_random,
            "stop_this_version": stop,
            "next_step_if_stopped": ("stop this version of the learned ranker; do not move to bigger "
                                     "models or more training") if stop else
                                    "independent validation on another instance/graph"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PILOT_SEEDS))
    parser.add_argument("--budget", type=int, default=QUERY_BUDGET)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rows, manifest = parse_raw(FILE)
    graph = build_undirected(rows)
    graph_seconds = time.perf_counter() - t0
    n = graph.number_of_nodes()
    T = int((n * SEED_FRACTION) // 1)

    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    t_load = time.perf_counter()
    model = RankModel.load(str(MODEL))
    load_seconds = time.perf_counter() - t_load

    print("=" * 112)
    print("  BUDGET-LIMITED LEARNING PILOT -- rank a batch of 32, then verify in order with true sigma")
    print("=" * 112)
    print(f"    graph        : ca-GrQc, sha256 {manifest['sha256'][:16]}..., n={n:,}, "
          f"m={graph.number_of_edges():,}")
    print(f"    K / T        : {K} / {T}      initial set: degree top-{T}      batch size: {BATCH_SIZE}")
    print(f"    seeds        : {args.seeds}   (new; train/validate used 0/1/2)")
    print(f"    budget       : {args.budget:,} queries or {args.time_limit:.0f} s, whichever first")
    print(f"    acceptance   : first strictly improving swap under the true sigma -- never the model's call")
    print(f"    model        : {MODEL.name}  sha256 {training['checkpoint']['sha256'][:16]}...")
    print(f"                   epoch {training['checkpoint']['selected_epoch']}, "
          f"selected by the pre-registered validation rule (not proven-optimal)")
    print(f"    graph read {graph_seconds:.3f} s   model load {load_seconds:.3f} s  "
          f"(both excluded from search timing)")

    results: list[dict] = []
    for seed in args.seeds:
        print(f"\n  seed {seed}")
        per_arm = {}
        for arm in ARMS:
            res = batched_search(graph, ranker=arm, T=T, K=K, budget=args.budget, seed=seed,
                                 batch_size=BATCH_SIZE, restarts=RESTARTS,
                                 checkpoints=CHECKPOINTS, time_limit=args.time_limit,
                                 timer=time.perf_counter, collect_logs=True, trace_hash=True,
                                 return_history=True,
                                 model=model if arm == "learned" else None)
            tb = res.extra["timing_buckets"]
            ss = res.extra["scan_summary"]
            per_arm[arm] = {
                "sigma": res.sigma, "normalized": res.sigma / n,
                "queries": res.cost["decision_evaluations"],
                "distinct_queries": res.cost["distinct_queries"],
                "accepts": res.extra["accepts"], "restarts": res.extra["restarts_used"],
                "checkpoints": {f"@{c['checkpoint']}": c["best_sigma"]
                                for c in res.extra["checkpoint_curve"]},
                "total_seconds": tb["total_seconds"],
                "diffusion_seconds": tb["1_diffusion_seconds"],
                "cache_seconds": tb["2_cache_and_key_seconds"],
                "draw_seconds": tb["3a_candidate_draw_seconds"],
                "features_inference_sort_seconds": tb["3b_features_inference_sort_seconds"],
                "control_seconds": tb["4_search_control_seconds"],
                "unaccounted_seconds": tb["5_logging_and_unaccounted_seconds"],
                "reconciliation_error_seconds": tb["reconciliation_error_seconds"],
                "diffusion_share_of_total": tb["diffusion_share_of_total"],
                "scans": ss["scans"], "mean_first_improvement_position": ss["mean_first_improvement_position"],
                "candidates_scanned_total": ss["candidates_scanned_total"],
                "candidates_drawn_total": ss["candidates_drawn_total"],
                "pool_exhausted_count": ss["pool_exhausted_count"],
                "stopped_by_time_limit": tb["stopped_by_time_limit"],
                "accept_sequence": res.extra["accept_sequence"],
                "history": res.extra["history"],
            }
            print(f"    {arm:<8}: sigma {res.sigma:>4}  ({100 * res.sigma / n:.2f}%)  "
                  f"queries {res.cost['decision_evaluations']:,}  distinct {res.cost['distinct_queries']:,}"
                  f"  accepts {res.extra['accepts']:>3}  total {tb['total_seconds']:6.2f} s"
                  f"  diffusion {tb['1_diffusion_seconds']:6.2f} s"
                  f"  feat+inf {tb['3b_features_inference_sort_seconds']:.4f} s")
        results.append({"seed": seed, "arms": per_arm})

    # ------------------------------------------------------------------ aggregates
    print("\n" + "=" * 112)
    print("  (1) QUALITY AT THE SAME QUERY BUDGET")
    print("=" * 112)
    cps = [f"@{c}" for c in CHECKPOINTS]
    header = f"    {'seed':<6}" + "".join(f"{a:>34}" for a in ARMS)
    print(header)
    print(f"    {'':<6}" + "".join(f"{c:>11}" for _ in ARMS for c in cps))
    for entry in results:
        line = f"    {entry['seed']:<6}"
        for a in ARMS:
            line += "".join(f"{entry['arms'][a]['checkpoints'][c]:>11}" for c in cps)
        print(line)
    print(f"\n    final sigma      " + "".join(
        f"{a}={statistics.mean([e['arms'][a]['sigma'] for e in results]):.1f}  " for a in ARMS))
    print(f"    mean normalized  " + "".join(
        f"{a}={statistics.mean([e['arms'][a]['normalized'] for e in results]):.4f}  " for a in ARMS))
    print(f"    mean accepts     " + "".join(
        f"{a}={statistics.mean([e['arms'][a]['accepts'] for e in results]):.1f}  " for a in ARMS))
    print(f"    mean 1st-improve " + "".join(
        f"{a}={statistics.mean([e['arms'][a]['mean_first_improvement_position'] for e in results]):.1f}  "
        for a in ARMS))

    print("\n" + "=" * 112)
    print("  (2) WALL CLOCK, ALL OVERHEAD INCLUDED")
    print("=" * 112)
    print(f"    {'arm':<9}{'total s':>10}{'diffusion':>11}{'cache':>9}{'draw':>9}"
          f"{'feat+inf+sort':>15}{'control':>10}{'unacct':>9}")
    for a in ARMS:
        tot = statistics.mean([e['arms'][a]['total_seconds'] for e in results])
        dif = statistics.mean([e['arms'][a]['diffusion_seconds'] for e in results])
        cac = statistics.mean([e['arms'][a]['cache_seconds'] for e in results])
        drw = statistics.mean([e['arms'][a]['draw_seconds'] for e in results])
        fea = statistics.mean([e['arms'][a]['features_inference_sort_seconds'] for e in results])
        ctl = statistics.mean([e['arms'][a]['control_seconds'] for e in results])
        una = statistics.mean([e['arms'][a]['unaccounted_seconds'] for e in results])
        print(f"    {a:<9}{tot:>10.3f}{dif:>11.3f}{cac:>9.4f}{drw:>9.4f}{fea:>15.4f}"
              f"{ctl:>10.4f}{una:>9.4f}")
    print(f"\n    learned vs random : "
          f"{statistics.mean([e['arms']['learned']['total_seconds'] for e in results]) -
             statistics.mean([e['arms']['random']['total_seconds'] for e in results]):+.3f} s")
    print(f"    learned vs degree : "
          f"{statistics.mean([e['arms']['learned']['total_seconds'] for e in results]) -
             statistics.mean([e['arms']['degree']['total_seconds'] for e in results]):+.3f} s")
    print(f"    the model's own cost (features + inference + sort) is the 'feat+inf+sort' column;")
    print(f"    everything else in the row is present in all three arms.")

    print("\n  costs NOT included in the table above, listed separately because they are not free:")
    print(f"    label generation (3-trajectory replay) : "
          f"{training['cost']['label_generation_seconds']:.2f} s")
    print(f"    training (40 epochs, 1 config)         : "
          f"{training['cost']['training_seconds']:.2f} s")
    print(f"    model loading, this run                : {load_seconds:.3f} s")
    print(f"    graph read + preprocess, this run      : {graph_seconds:.3f} s")

    # ------------------------------------------------------------------ supplementary quality-time
    print("\n" + "=" * 112)
    print("  SUPPLEMENTARY: queries and seconds needed to reach the random arm's own final sigma")
    print("=" * 112)
    print(f"    {'seed':<6}{'target':>9}" + "".join(f"{a + ' queries':>16}{a + ' s':>11}" for a in ARMS))
    target_rows = []
    for entry in results:
        tgt = entry["arms"]["random"]["sigma"]
        row = {"seed": entry["seed"], "target_sigma": tgt, "arms": {}}
        line = f"    {entry['seed']:<6}{tgt:>9}"
        for a in ARMS:
            q = queries_to_reach(entry["arms"][a]["history"], tgt)
            s = seconds_to_reach(entry["arms"][a]["history"], tgt)
            row["arms"][a] = {"queries": q, "seconds": s}
            line += f"{(str(q) if q is not None else 'never'):>16}" \
                    f"{(('%.2f' % s) if s is not None else 'never'):>11}"
        print(line)
        target_rows.append(row)

    # ------------------------------------------------------------------ verdict
    summary = {
        "per_seed": [{"seed": e["seed"],
                      "arms": {a: {k: v for k, v in e["arms"][a].items()
                                   if k not in ("history", "accept_sequence")}
                               for a in ARMS}} for e in results],
        "aggregate": {a: {
            "mean_sigma": statistics.mean([e["arms"][a]["sigma"] for e in results]),
            "mean_normalized": statistics.mean([e["arms"][a]["normalized"] for e in results]),
            "mean_total_seconds": statistics.mean([e["arms"][a]["total_seconds"] for e in results]),
            "mean_diffusion_seconds": statistics.mean([e["arms"][a]["diffusion_seconds"]
                                                       for e in results]),
            "mean_features_inference_sort_seconds":
                statistics.mean([e["arms"][a]["features_inference_sort_seconds"] for e in results]),
            "mean_accepts": statistics.mean([e["arms"][a]["accepts"] for e in results]),
            "mean_first_improvement_position":
                statistics.mean([e["arms"][a]["mean_first_improvement_position"] for e in results]),
        } for a in ARMS},
    }
    v = verdict(summary)
    summary["verdict"] = v

    print("\n" + "=" * 112)
    print("  PRE-FIXED VERDICT")
    print("=" * 112)
    print(f"    {v['label'].upper()}")
    print(f"    {v['detail']}")
    print(f"    quality improved clearly : {v['quality_clear']}")
    print(f"    wall clock improved      : {v['time_clear']}")
    for k, val in v["mean_sigma"].items():
        print(f"      mean sigma {k:<8}: {val:.1f}")
    for k, val in v["mean_total_seconds"].items():
        print(f"      mean total {k:<8}: {val:.3f} s")

    artifact = {
        "script": Path(__file__).name,
        "scope": ("rank a batch of 32 candidate swaps, verify in order with the true diffuser, accept the "
                  "first strict improvement; the model never decides acceptance"),
        "frozen": {"graph": "ca-GrQc", "file_sha256": manifest["sha256"], "n": n,
                   "m": graph.number_of_edges(), "K": K, "T": T,
                   "initial_set": f"degree top-{T}", "batch_size": BATCH_SIZE,
                   "search_seeds": list(args.seeds), "query_budget": args.budget,
                   "time_limit_seconds": args.time_limit, "restarts": RESTARTS,
                   "acceptance_rule": "first strictly improving swap (true sigma)",
                   "candidate_pool_per_state": T * (n - T)},
        "model": {"file": MODEL.name, "sha256": training["checkpoint"]["sha256"],
                  "selected_epoch": training["checkpoint"]["selected_epoch"],
                  "selection_rule": training["config"]["checkpoint_rule"],
                  "label": training["checkpoint"]["label"]},
        "arms": list(ARMS),
        "costs_not_in_search_timing": {
            "label_generation_seconds": training["cost"]["label_generation_seconds"],
            "training_seconds": training["cost"]["training_seconds"],
            "model_load_seconds": load_seconds,
            "graph_read_seconds": graph_seconds,
            "note": ("reported separately on purpose; a learner with fast inference and expensive labels "
                     "has not demonstrated an acceleration"),
        },
        "quality_time_to_random_target": target_rows,
        "results": results,
        "aggregate": summary["aggregate"],
        "verdict": v,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
