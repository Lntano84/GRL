"""Generate the per-query training rows by replaying the three accepted trajectories.

The problem this solves
-----------------------
The three ca-GrQc trajectories were stored only as *per-scan* summaries
(``results/candidate_generation_opt.json``).  They contain no ``(S, u, v, before, after)`` rows, and no
such rows exist anywhere on disk for this graph.  The learner cannot be trained on per-scan summaries.

The route taken, and why it is not new labels
---------------------------------------------
The three runs are replayed under the *identical frozen setting* (same graph sha256, ``K = 4``,
``T = 52``, degree-52 initial set, seeds 0/1/2, ``5,000`` queries, ``60`` s, ``6`` restarts, first
strictly improving swap, true ``sigma``) and each query is now **logged** instead of merely counted.
Nothing is re-ordered and no additional candidate is evaluated: the run is over when it was over, at the
same query count, on the same seed sets, with the same accept sequence.

That claim is not asserted, it is checked: every replayed run must reproduce the
``query_sequence_sha256`` and the ``accept_sequence`` recorded in ``results/candidate_generation_opt.json``
(the ``lazy`` column, which is the accepted implementation).  A run that fails either check is not
written to the artifact.

The replay is driven by ``batched_search(batch_size=1)``, which
``tests/test_batched_search.py::test_batch_size_one_reproduces_the_sequential_lazy_scan`` shows to be an
exact re-parameterisation of ``ordered_search(method="random", scanner="lazy")`` -- the replay therefore
also re-verifies that equivalence at the real budget on the real graph.

Cost accounting
---------------
The wall-clock time of this script is the **label-generation cost**.  It is reported here and repeated in
the pilot report, and it is *not* part of the online search timing: training-side costs are listed
separately on purpose, because a learner whose inference is fast but whose labels are expensive has not
shown an acceleration.
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

from ccim.batched_search import batched_search  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

K = 4
SEED_FRACTION = 0.01
SEARCH_SEEDS = (0, 1, 2)
QUERY_BUDGET = 5_000
TIME_LIMIT = 60.0
RESTARTS = 6
REFERENCE = ROOT / "results" / "candidate_generation_opt.json"
OUTPUT = ROOT / "results" / "swap_records.json"


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
    graph_seconds = time.perf_counter() - t0
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    T = int((n * SEED_FRACTION) // 1)

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    ref_lazy = {int(k): v["lazy"] for k, v in reference["results"].items()}

    print("=" * 112)
    print("  LABEL GENERATION -- replay of the three accepted trajectories, now logged per query")
    print("=" * 112)
    print(f"    graph          : ca-GrQc, sha256 {manifest['sha256'][:16]}..., n={n:,}, m={m:,}")
    print(f"    K / T          : {K} / {T}      initial set: degree top-{T}")
    print(f"    seeds          : {args.seeds}   budget {args.budget:,} queries or "
          f"{args.time_limit:.0f} s   restarts {RESTARTS}")
    print(f"    reference      : {REFERENCE.name} (lazy column)")
    print(f"    graph read + preprocess (excluded from replay timing): {graph_seconds:.3f} s")

    out: dict[str, dict] = {}
    gates: dict[str, dict] = {}
    replay_seconds_total = 0.0

    for seed in args.seeds:
        t_seed = time.perf_counter()
        res = batched_search(graph, ranker="random", T=T, K=K, budget=args.budget, seed=seed,
                             batch_size=1, restarts=RESTARTS, time_limit=args.time_limit,
                             timer=time.perf_counter, collect_logs=False, record_rows=True,
                             trace_hash=True)
        seconds = time.perf_counter() - t_seed
        replay_seconds_total += seconds
        ref = ref_lazy[seed]
        gate = {
            "same_query_sequence": res.extra["query_sequence_sha256"] == ref["query_sequence_sha256"],
            "same_accept_sequence": res.extra["accept_sequence"] == ref["accept_sequence"],
            "same_final_sigma": res.sigma == ref["sigma"],
            "same_query_count": res.cost["decision_evaluations"] == ref["queries"],
            "query_count_is_the_full_budget": res.cost["decision_evaluations"] == args.budget,
            "stopped_by_time_limit": res.extra["timing_buckets"]["stopped_by_time_limit"],
        }
        gate["all"] = all(v for k, v in gate.items() if k != "stopped_by_time_limit")
        gates[str(seed)] = gate

        n_rows = len(res.extra["rows"])
        n_states = len(res.extra["state_sets"])
        positives = sum(1 for r in res.extra["rows"] if r["after"] > r["before"])
        print(f"\n    seed {seed}: sigma {res.sigma}  queries {res.cost['decision_evaluations']:,}"
              f"  accepts {res.extra['accepts']}  states {n_states}  rows {n_rows:,}"
              f"  improving rows {positives}")
        print(f"      query-sequence hash reproduces : {gate['same_query_sequence']}")
        print(f"      accept sequence reproduces     : {gate['same_accept_sequence']}")
        print(f"      replay wall clock              : {seconds:.2f} s")

        out[str(seed)] = {
            "sigma": res.sigma, "queries": res.cost["decision_evaluations"],
            "accepts": res.extra["accepts"], "restarts": res.extra["restarts_used"],
            "rows": res.extra["rows"], "state_sets": res.extra["state_sets"],
            "query_sequence_sha256": res.extra["query_sequence_sha256"],
            "accept_sequence": res.extra["accept_sequence"],
            "replay_seconds": seconds,
            "positive_rows": positives,
            "scan_summary": res.extra["scan_summary"],
        }

    ok = all(g["all"] for g in gates.values())
    print("\n" + "=" * 112)
    print(f"  REPLAY IDENTITY GATES: {'ALL PASS' if ok else 'FAILED'}")
    print("=" * 112)
    if not ok:
        for seed, g in gates.items():
            for k, v in g.items():
                if k != "all" and not v:
                    print(f"    seed {seed}: FAILED {k}")
        print("  artifact NOT written -- a replayed trajectory that does not reproduce is not the same run")
        return 1

    artifact = {
        "script": Path(__file__).name,
        "scope": ("per-query training rows obtained by replaying the three accepted trajectories; "
                  "no new queries, no reordering"),
        "frozen": {"graph": "ca-GrQc", "file_sha256": manifest["sha256"], "n": n, "m": m,
                   "K": K, "T": T, "initial_set": f"degree top-{T}", "restarts": RESTARTS,
                   "query_budget": args.budget, "time_limit_seconds": args.time_limit,
                   "acceptance_rule": "first strictly improving swap (true sigma)",
                   "search_seeds": args.seeds},
        "reference": {"file": REFERENCE.name, "column": "lazy"},
        "gates": gates,
        "label_generation": {
            "graph_read_and_preprocess_seconds": graph_seconds,
            "replay_seconds_per_seed": {s: out[s]["replay_seconds"] for s in out},
            "replay_seconds_total": replay_seconds_total,
            "note": ("this is the cost of producing the labels; it is reported separately from the "
                     "online search timing and is never treated as free"),
        },
        "seeds": out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    total_rows = sum(len(v["rows"]) for v in out.values())
    print(f"  rows written      : {total_rows:,} across {len(out)} trajectories")
    print(f"  label gen. cost   : {replay_seconds_total:.2f} s of replay "
          f"(+ {graph_seconds:.2f} s graph read)")
    print(f"  artifact          : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
