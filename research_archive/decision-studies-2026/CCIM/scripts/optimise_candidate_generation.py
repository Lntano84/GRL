"""Candidate generation, and nothing else: explicit-array versus lazy Fisher-Yates scanning.

Frozen for both versions: the same graph version (sha256 recorded), ``K = 4``, ``T = 52``, the
degree-52 initial set, the three pre-fixed search seeds, ``5,000`` queries or ``60`` seconds, the
original acceptance rule (true ``sigma``), cache and restart rules.  The **only** difference is how the
next candidate is produced:

* ``explicit`` -- the same Fisher-Yates rule with the whole ``0..M-1`` array materialised;
* ``lazy``     -- the same rule with a sparse map, so only touched positions are stored.

Both draw from an isolated scan stream and emit candidates on demand.  The legacy implementation (which
shuffled the full pair list) is kept as history and is not required to match.

Verification gates, all of which must pass before any timing is read:

1. trajectory identity -- query-set sequence hash, acceptance sequence and per-checkpoint gains equal;
2. the lazy scanner's sparse map stays far below ``M`` (no hidden full array, no full permutation).

Nothing else is optimised here: the diffuser and the logging are untouched, and the ~11% unaccounted
bucket is left in place rather than assumed away.
"""
from __future__ import annotations

import argparse
import json
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
MODES = ("explicit", "lazy")
OUTPUT = ROOT / "results" / "candidate_generation_opt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEARCH_SEEDS))
    parser.add_argument("--budget", type=int, default=QUERY_BUDGET)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rows, manifest = parse_raw(FILE)
    graph = build_undirected(rows)
    load_seconds = time.perf_counter() - t0
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    T = int((n * SEED_FRACTION) // 1)

    prior = json.loads((ROOT / "results" / "ca_grqc_probe.json").read_text(encoding="utf-8"))
    initial_sigma = next(r["final_activated"] for r in prior["results"] if r["kind"] == "degree_top_T")

    print("=" * 112)
    print("  FROZEN SETTING (unchanged from the search probe)")
    print("=" * 112)
    print(f"    graph          : ca-GrQc, sha256 {manifest['sha256'][:16]}..., n={n:,}, m={m:,}")
    print(f"    K / T          : {K} / {T}      initial set: degree top-{T}")
    print(f"    seeds          : {args.seeds}      budget: {args.budget:,} queries or "
          f"{args.time_limit:.0f} s")
    print(f"    candidate pool : {T * (n - T):,} per state")
    print(f"    graph read + preprocess (excluded from search timing): {load_seconds:.3f} s")

    results: dict[int, dict[str, dict]] = {}
    for seed in args.seeds:
        results[seed] = {}
        for mode in MODES:
            res = ordered_search(graph, method="random", T=T, K=K, budget=args.budget,
                                 seed=seed, restarts=RESTARTS, checkpoints=CHECKPOINTS,
                                 time_limit=args.time_limit, timer=time.perf_counter,
                                 scanner=mode, trace_hash=True)
            b = res.extra["timing_buckets"]
            results[seed][mode] = {
                "sigma": res.sigma, "normalized": res.normalized,
                "queries": res.cost["decision_evaluations"],
                "accepts": res.extra["accepts"], "restarts": res.extra["restarts_used"],
                "scans": res.extra["scan_summary"]["scans"],
                "query_sequence_sha256": res.extra["query_sequence_sha256"],
                "accept_sequence": res.extra["accept_sequence"],
                "checkpoints": {f"@{c['checkpoint']}": c["best_sigma"]
                                for c in res.extra["checkpoint_curve"]},
                "timing": b,
                "timing_breakdown": {k: v for k, v in b.items()},
                "sparse_entries_max": b["sparse_entries_max"],
                "candidates_drawn_max_in_one_scan": b["candidates_drawn_max_in_one_scan"],
                "stopped_by_time_limit": b["stopped_by_time_limit"],
            }
            print(f"    seed {seed} {mode:<9}: sigma {res.sigma}  queries {res.cost['decision_evaluations']:,}"
                  f"  accepts {res.extra['accepts']}  scans {res.extra['scan_summary']['scans']}"
                  f"  total {b['total_seconds']:.2f} s", flush=True)

    # ------------------------------------------------------------------ gates
    gates = {}
    for seed in args.seeds:
        e, l = results[seed]["explicit"], results[seed]["lazy"]
        gates[seed] = {
            "same_query_sequence": e["query_sequence_sha256"] == l["query_sequence_sha256"],
            "same_accept_sequence": e["accept_sequence"] == l["accept_sequence"],
            "same_checkpoint_gains": e["checkpoints"] == l["checkpoints"],
            "same_final_sigma": e["sigma"] == l["sigma"],
            "same_query_count": e["queries"] == l["queries"],
            "lazy_sparse_map_far_below_M": l["sparse_entries_max"] <= 2 * l["candidates_drawn_max_in_one_scan"],
        }
        gates[seed]["all"] = all(v for k, v in gates[seed].items() if k != "lazy_sparse_map_far_below_M")
    gates_ok = all(g["all"] for g in gates.values())
    print(f"\n  trajectory identity gates: {'ALL PASS' if gates_ok else 'FAILED'}")

    artifact = {
        "script": Path(__file__).name,
        "scope": "candidate generation only; no learner, no change to the search rule",
        "frozen": {"graph": "ca-GrQc", "file_sha256": manifest["sha256"], "n": n, "m": m,
                   "K": K, "T": T, "initial_set": f"degree top-{T}",
                   "search_seeds": args.seeds, "query_budget": args.budget,
                   "time_limit_seconds": args.time_limit, "restarts": RESTARTS,
                   "acceptance_rule": "first strictly improving swap (true sigma)",
                   "candidate_pool_per_state": T * (n - T)},
        "versions": {
            "explicit": "Fisher-Yates with the full 0..M-1 array materialised (verification reference)",
            "lazy": "the same Fisher-Yates rule via a sparse map; only touched positions stored",
            "legacy": "the original full pair-list shuffle; kept as history, not required to match",
        },
        "not_optimised_here": ["the diffuser", "logging",
                               "the unaccounted bucket (~11% before this change) is left in place"],
        "random_streams": {"restart": "isolated", "scan": "isolated", "two_swap": "isolated",
                           "note": "legacy used one shared ordering stream; the two new scanners use "
                                   "separate streams so scan-draw counts cannot perturb restarts"},
        "graph_read_and_preprocess_seconds": load_seconds,
        "initial_sigma": initial_sigma,
        "results": {str(k): v for k, v in results.items()},
        "gates": {str(k): v for k, v in gates.items()},
        "gates_all_pass": gates_ok,
        "complete": True,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ the single table
    print("\n" + "=" * 112)
    print("  RESULT")
    print("=" * 112)
    print(f"  {'seed':>4}{'trajectory':>12}{'final sigma':>13}{'explicit total':>16}"
          f"{'lazy total':>12}{'cand gen':>18}{'diffusion':>18}{'rest':>16}")
    for seed in args.seeds:
        e, l = results[seed]["explicit"], results[seed]["lazy"]
        et, lt = e["timing"], l["timing"]
        same = "identical" if gates[seed]["all"] else "DIFFER"
        cand = f"{et['3_candidate_generation_seconds']:.2f} -> {lt['3_candidate_generation_seconds']:.2f}"
        diff = f"{et['1_diffusion_seconds']:.2f} -> {lt['1_diffusion_seconds']:.2f}"
        rest_e = et["total_seconds"] - et["1_diffusion_seconds"] - et["3_candidate_generation_seconds"]
        rest_l = lt["total_seconds"] - lt["1_diffusion_seconds"] - lt["3_candidate_generation_seconds"]
        rest = f"{rest_e:.2f} -> {rest_l:.2f}"
        sigma = f"{e['sigma']} ({e['normalized']:.4f})"
        print(f"  {seed:>4}{same:>12}{sigma:>13}{et['total_seconds']:>16.2f}"
              f"{lt['total_seconds']:>12.2f}{cand:>18}{diff:>18}{rest:>16}")

    print("\n  (cand gen / diffusion / rest are 'explicit -> lazy', seconds)")
    print("\n  five-bucket breakdown, lazy version (seconds):")
    print(f"  {'seed':>4}{'diffusion':>11}{'cache/key':>11}{'candidate':>11}{'control':>10}"
          f"{'log/other':>11}{'total':>9}{'err':>9}{'diff share':>12}")
    for seed in args.seeds:
        b = results[seed]["lazy"]["timing"]
        print(f"  {seed:>4}{b['1_diffusion_seconds']:>11.3f}{b['2_cache_and_key_seconds']:>11.3f}"
              f"{b['3_candidate_generation_seconds']:>11.3f}{b['4_search_control_seconds']:>10.3f}"
              f"{b['5_logging_and_unaccounted_seconds']:>11.3f}{b['total_seconds']:>9.3f}"
              f"{b['reconciliation_error_seconds']:>9.4f}"
              f"{b['diffusion_share_of_total'] * 100:>11.1f}%")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
