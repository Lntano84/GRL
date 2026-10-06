"""Search probe from the degree solution on ca-GrQc: quality versus cost, with a full time breakdown.

This is its own protocol and is not pooled with the earlier small-graph scan-order pilot.

Frozen
------
* graph: **the frozen ca-GrQc version of this round** -- simple undirected, self-loop rows dropped
  first, so nodes that appear only as a self-loop are excluded.  n = 5,241, m = 14,484, file sha256
  recorded below.  Self-loops are removed **without** forcing the node out in general; this version
  simply excludes the only-self-loop node, and the label says so.
* diffusion: deterministic CCIM, ``p0=1, p1=0``, ``K = 4``;
* seed budget ``T = 52``, initial set = degree top-52 with the fixed tie-break (degree desc, id asc);
* search: uniformly random scan order, accept the **first strictly improving** swap;
* restarts: the existing rule, recorded rather than adjusted (degree start, then 6 random restarts);
* three pre-fixed search seeds;
* per run: 5,000 full-reward query requests **or** 60 seconds, whichever comes first.  Initialisation
  and restart queries count toward the budget.  The cache is per run.  The best solution is kept across
  restarts.  If the time limit fires, the **actual** query count is recorded and is not topped up.

Not done here
-------------
No optimisation while measuring: the existing implementation is measured as it is, including the
269,828-candidate pool that a 52-seed set at n=5,241 implies.  No extra diffuser call is made to
"confirm" a checkpoint: this is a deterministic model and the saved values are used as they are.
No model is trained.
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

from ccim.scan_order import ordered_search  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

K = 4
SEED_FRACTION = 0.01
SEARCH_SEEDS = (0, 1, 2)
QUERY_BUDGET = 5_000
TIME_LIMIT = 60.0
CHECKPOINTS = (100, 300, 1_000, 3_000, 5_000)
RESTARTS = 6
OUTPUT = ROOT / "results" / "ca_grqc_search_probe.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEARCH_SEEDS))
    parser.add_argument("--budget", type=int, default=QUERY_BUDGET)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    # ---- graph reading and one-time preprocessing, timed SEPARATELY from the search ----
    t0 = time.perf_counter()
    rows, manifest = parse_raw(FILE)
    graph = build_undirected(rows)
    load_seconds = time.perf_counter() - t0
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    T = int((n * SEED_FRACTION) // 1)
    by_degree = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))
    initial = by_degree[:T]

    print("=" * 104)
    print("  FROZEN SETTING")
    print("=" * 104)
    print(f"    graph version      : ca-GrQc, simple undirected, only-self-loop nodes EXCLUDED")
    print(f"    file sha256        : {manifest['sha256']}")
    print(f"    n / m              : {n:,} / {m:,}      (excluded only-self-loop nodes: "
          f"{manifest['nodes_only_present_as_self_loops']})")
    print(f"    K / T              : {K} / {T}     (T = floor(0.01 n))")
    print(f"    initial set        : degree top-{T}, tie-break (degree desc, node id asc)")
    print(f"    budget / time cap  : {args.budget:,} queries OR {args.time_limit:.0f} s")
    print(f"    restarts           : {RESTARTS} (existing rule: degree start + random restarts)")
    print(f"    graph read + preprocess (NOT in search timing): {load_seconds:.3f} s")

    # The degree-52 sigma was measured and saved by the earlier probe; this is a deterministic model,
    # so it is read from that artifact rather than recomputed with an extra diffuser call.
    prior = json.loads((ROOT / "results" / "ca_grqc_probe.json").read_text(encoding="utf-8"))
    initial_sigma = next(r["final_activated"] for r in prior["results"]
                         if r["kind"] == "degree_top_T")
    assert T == next(r["T"] for r in prior["results"] if r["kind"] == "degree_top_T"), \
        "the frozen T no longer matches the saved measurement"

    runs = []
    for seed in args.seeds:
        res = ordered_search(graph, method="random", T=T, K=K, budget=args.budget,
                             seed=seed, restarts=RESTARTS, checkpoints=CHECKPOINTS,
                             time_limit=args.time_limit, timer=time.perf_counter)
        buckets = res.extra["timing_buckets"]
        curve = {c["checkpoint"]: c for c in res.extra["checkpoint_curve"]}
        scans = res.extra["scan_log"]
        runs.append({
            "search_seed": seed,
            "initial_sigma": initial_sigma,
            "best_sigma": res.sigma, "best_normalized": res.normalized,
            "best_seeds": res.seeds,
            "improvement_over_initial": res.sigma - initial_sigma,
            "improvement_over_initial_fraction": (res.sigma - initial_sigma) / initial_sigma,
            "queries_used": res.cost["decision_evaluations"],
            "distinct_queries": res.cost["distinct_queries"],
            "cache_hits": res.cost["cached_repeats"],
            "accepts": res.extra["accepts"], "restarts_used": res.extra["restarts_used"],
            "scans": res.extra["scan_summary"]["scans"],
            "candidate_generation_seconds_per_scan":
                (sum(s["scoring_seconds"] for s in scans) / len(scans)) if scans else None,
            "candidates_per_scan": scans[0]["candidates_available"] if scans else None,
            "stopped_by_time_limit": buckets["stopped_by_time_limit"],
            "checkpoints": {str(c): curve[c] for c in CHECKPOINTS},
            "timing_buckets": buckets,
        })
        print(f"\n  seed {seed}: best sigma {res.sigma} ({res.normalized:.4f})  "
              f"queries {res.cost['decision_evaluations']:,}  accepts {res.extra['accepts']}  "
              f"restarts {res.extra['restarts_used']}  scans {len(scans)}  "
              f"total {buckets['total_seconds']:.2f} s"
              + ("   STOPPED BY TIME LIMIT" if buckets["stopped_by_time_limit"] else ""))

    artifact = {
        "script": Path(__file__).name,
        "scope": "search probe from the degree solution; no model training",
        "protocol_note": "own protocol; not pooled with the earlier small-graph scan-order pilot",
        "initial_sigma_source":
            "results/ca_grqc_probe.json -> degree_top_T; read, not recomputed (deterministic model)",
        "initial_sigma": initial_sigma,
        "frozen": {
            "graph_version": "ca-GrQc simple undirected, only-self-loop nodes excluded",
            "file_sha256": manifest["sha256"], "n": n, "m": m,
            "excluded_only_self_loop_nodes": manifest["nodes_only_present_as_self_loops"],
            "self_loop_handling": "self-loop ROWS dropped; this excludes the one node that has no "
                                  "other edge.  Removing a self-loop does not by itself force the "
                                  "node out -- the version label states the exclusion explicitly.",
            "K": K, "T": T, "seed_budget_rule": f"T = floor({SEED_FRACTION} * n) = {T}",
            "initial_set": "degree top-T, tie-break (degree desc, node id asc)",
            "search": "uniformly random scan order, first strictly improving swap accepted",
            "restart_rule": f"degree start, then {RESTARTS} random restarts (existing implementation)",
            "search_seeds": args.seeds,
            "query_budget": args.budget, "time_limit_seconds": args.time_limit,
            "budget_note": "initialisation and restart queries are counted; the cache is per run; the "
                           "best solution is kept across restarts; a time-limit stop records the "
                           "actual query count and is never topped up",
        },
        "candidate_pool_size": T * (n - T),
        "graph_read_and_preprocess_seconds": load_seconds,
        "timing_protocol": {
            "total_region": "from starting to construct the initial set to returning the best solution",
            "excluded": ["graph reading/parsing", "one-time preprocessing (degree ranking)"],
            "buckets_are_disjoint": True,
            "instrumentation_note": "the timer adds a few perf_counter calls per query; that cost "
                                    "lands inside bucket 2",
        },
        "runs": runs,
        "complete": True,
    }

    # ------------------------------------------------------------------ table 1: quality vs cost
    print("\n" + "=" * 104)
    print(f"  TABLE 1  QUALITY vs COST   (per search seed; initial degree-52 sigma = {initial_sigma})")
    print("=" * 104)
    for r in runs:
        print(f"\n  seed {r['search_seed']}"
              + ("   [stopped by the time limit]" if r["stopped_by_time_limit"] else ""))
        print(f"    {'queries':>8}{'best':>8}{'vs init':>9}{'wall s':>9}{'misses':>9}"
              f"{'accepts':>9}{'restarts':>10}")
        for c in CHECKPOINTS:
            e = r["checkpoints"][str(c)]
            if not e["reached"]:
                print(f"    {c:>8}{'MISSING':>8}")
                continue
            print(f"    {e['queries']:>8}{e['best_sigma']:>8}"
                  f"{e['best_sigma'] - initial_sigma:>+9}{e['cumulative_seconds']:>9.2f}"
                  f"{e['distinct_queries']:>9}{e['accepts']:>9}{e['restarts']:>10}")

    # ------------------------------------------------------------------ table 2: time breakdown
    print("\n" + "=" * 104)
    print("  TABLE 2  TIME BREAKDOWN (seconds; buckets disjoint; reconciled against the total)")
    print("=" * 104)
    print(f"  {'seed':>5}{'diffusion':>11}{'cache/key':>11}{'candidate':>11}{'control':>10}"
          f"{'log/other':>11}{'sum':>9}{'total':>9}{'err':>8}{'diff share':>12}")
    for r in runs:
        b = r["timing_buckets"]
        print(f"  {r['search_seed']:>5}{b['1_diffusion_seconds']:>11.3f}"
              f"{b['2_cache_and_key_seconds']:>11.3f}{b['3_candidate_generation_seconds']:>11.3f}"
              f"{b['4_search_control_seconds']:>10.3f}"
              f"{b['5_logging_and_unaccounted_seconds']:>11.3f}"
              f"{b['sum_of_buckets']:>9.3f}{b['total_seconds']:>9.3f}"
              f"{b['reconciliation_error_seconds']:>8.4f}"
              f"{b['diffusion_share_of_total'] * 100:>11.1f}%")

    print("\n  graph read + preprocessing (excluded from the above): "
          f"{load_seconds:.3f} s")
    print(f"\n  wrote {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
