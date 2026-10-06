"""Minimal single-graph measurement on ca-GrQc: can these pre-fixed seed sets propagate, and what does
one diffusion actually cost?

Scope, fixed in advance
-----------------------
* graph: ca-GrQc, after verifying the load semantics and writing a manifest;
* model: deterministic CCIM, ``p0 = 1, p1 = 0, K = 4``;
* seed budget: ``T = floor(0.01 * n)`` -- a pre-fixed 1%-of-nodes workload, **not** a reproduction of
  the paper's ratio and not claimed to be the most reasonable or the hardest setting.  ``K`` and ``T``
  are **not** adjusted after seeing any propagation result;
* 21 fixed seed sets: 20 uniform random sets from pre-fixed seeds, plus one fixed-rule degree set.

Verification before measurement
-------------------------------
1. the optimised diffuser and an independent brute-force reference must agree on the final activated
   **set**, node for node -- not merely on the count.
2. ``K``-core is not used as a propagation gate anywhere; see ``tests/test_ccim_diffuser.py`` for the
   two counterexamples that falsify it.

Timing protocol
---------------
Only the diffusion call is timed.  Graph loading, the brute-force reference and all reporting are
outside the timed region.  Each fixed seed set is timed ``REPEATS`` times, and those repeats are
**repeated computation on the same input, not new propagation samples** -- the count is reported
separately so the two can never be confused.

Nothing else happens here: no training, no search, no parameter sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402

from ccim.model import cascade  # noqa: E402

FILE = Path(r"C:\Users\windows\Desktop\_grl_merge\merged\data\paper\ca-GrQc.txt")
K = 4
SEED_FRACTION = 0.01
RANDOM_SEED_BASE = 20260919          # pre-fixed; set i uses RANDOM_SEED_BASE + i
N_RANDOM_SETS = 20
REPEATS = 7                          # repeated TIMING of the same input, not new samples
OUTPUT = ROOT / "results" / "ca_grqc_probe.json"


def parse_raw(path: Path) -> tuple[list[tuple[int, int]], dict]:
    """Read the edge list with the rules stated in the manifest, counting everything."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows: list[tuple[int, int]] = []
    comment_lines = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            comment_lines += 1
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    self_loop_rows = [r for r in rows if r[0] == r[1]]
    non_loop = [r for r in rows if r[0] != r[1]]
    as_directed = set(rows)
    unordered = {tuple(sorted(r)) for r in non_loop}
    both_directions = sum(1 for (u, v) in unordered if (u, v) in as_directed and (v, u) in as_directed)
    all_ids = {x for r in rows for x in r}
    non_loop_ids = {x for r in non_loop for x in r}
    manifest = {
        "file": path.name, "sha256": digest.upper(),
        "comment_lines": comment_lines, "data_rows": len(rows),
        "duplicate_rows_in_same_direction": len(rows) - len(as_directed),
        "self_loop_rows": len(self_loop_rows),
        "self_loop_nodes": sorted({r[0] for r in self_loop_rows}),
        "distinct_node_ids": len(all_ids),
        "node_id_min": min(all_ids),
        "node_id_max": max(all_ids),
        "as_directed_arcs": len(as_directed),
        "unordered_pairs": len(unordered),
        "pairs_with_both_directions_present": both_directions,
        "pairs_with_only_one_direction_present": len(unordered) - both_directions,
        # A node that appears ONLY in a self-loop row survives in the file's node count but has no
        # neighbour, so it is absent from the simple undirected graph.  Recording it keeps the two
        # node counts from looking like a version mismatch.
        "nodes_only_present_as_self_loops": sorted(all_ids - non_loop_ids),
        "nodes_in_simple_undirected_graph": len(non_loop_ids),
    }
    return non_loop, manifest


def build_undirected(rows: list[tuple[int, int]]) -> nx.Graph:
    """Undirected simple graph: CCIM's neighbourhood ``N(x)`` is undirected in the source paper.

    Self-loops are dropped (a node is never its own neighbour) and parallel edges collapse.  The file
    stores each unordered pair twice, so the symmetrised graph has half the arcs.
    """
    g = nx.Graph()
    g.add_edges_from(rows)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def brute_force(graph: nx.Graph, seeds, K: int) -> set:
    """Independent reference: rescan every node until a full pass changes nothing."""
    active = set(int(s) for s in seeds)
    changed = True
    while changed:
        changed = False
        for v in graph.nodes():
            if v in active:
                continue
            if sum(1 for u in graph.neighbors(v) if u in active) >= K:
                active.add(v)
                changed = True
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows, manifest = parse_raw(FILE)
    graph = build_undirected(rows)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    T = n * SEED_FRACTION
    T = int(T // 1)

    manifest.update({"n_undirected": n, "m_undirected": m,
                     "directed_as_stored": True,
                     "undirected_reading_used": True,
                     "symmetrised_edge_ratio": m / manifest["as_directed_arcs"],
                     "self_loops_dropped": True, "parallel_edges_collapsed": True})
    print("=" * 100)
    print("  GRAPH MANIFEST")
    print("=" * 100)
    for k, v in manifest.items():
        print(f"    {k:<45} {v}")
    print(f"\n  K = {K}, T = floor({SEED_FRACTION} * {n}) = {T}")

    # ---------------------------------------------------------------- fixed seed sets
    sets: list[dict] = []
    for i in range(N_RANDOM_SETS):
        rng = random.Random(RANDOM_SEED_BASE + i)
        seeds = sorted(rng.sample(list(graph.nodes()), T))
        sets.append({"kind": "uniform_random", "index": i,
                     "rng_seed": RANDOM_SEED_BASE + i, "seeds": seeds})
    degree_set = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))[:T]
    sets.append({"kind": "degree_top_T", "index": 0, "rng_seed": None, "seeds": sorted(degree_set)})
    assert len(sets) == 21

    # ---------------------------------------------------------------- verify, then measure
    results = []
    for entry in sets:
        seeds = entry["seeds"]
        reference = brute_force(graph, seeds, K)
        stats: dict = {}
        fast = cascade(graph, seeds, K, stats=stats)
        agree = fast == reference
        if not agree:
            raise SystemExit(f"diffuser and reference disagree on {entry['kind']} "
                             f"{entry['index']}: symmetric difference "
                             f"{sorted(fast ^ reference)[:10]}")

        timings_ns = []
        for _ in range(args.repeats):
            t0 = time.perf_counter_ns()
            cascade(graph, seeds, K)
            timings_ns.append(time.perf_counter_ns() - t0)
        timings_ms = [t / 1e6 for t in timings_ns]

        results.append({
            **{k: entry[k] for k in ("kind", "index", "rng_seed")},
            "seeds": seeds,
            "T": len(seeds),
            "final_activated": len(fast),
            "extra_activated": len(fast) - len(seeds),
            "normalized": len(fast) / n,
            "activating_rounds": stats["activating_rounds"],
            "waves_processed": stats["waves"],
            "nodes_initialised": stats["nodes_initialised"],
            "nodes_dequeued": stats["nodes_dequeued"],
            "edges_scanned": stats["edges_scanned"],
            "reference_agrees_on_the_set": agree,
            "timing_ms": {
                "repeats": args.repeats,
                "min": min(timings_ms), "median": statistics.median(timings_ms),
                "mean": statistics.fmean(timings_ms), "max": max(timings_ms),
                "all": timings_ms,
            },
        })
        unit = "ms"
        print(f"  {entry['kind']:<15} #{entry['index']:<2} |S|={len(seeds)}  "
              f"activated {len(fast):>5}  extra {len(fast) - len(seeds):>5}  "
              f"rounds {stats['activating_rounds']:>2}  edges {stats['edges_scanned']:>8}  "
              f"{statistics.median(timings_ms):.3f} {unit}", flush=True)

    # ---------------------------------------------------------------- summary
    extra = [r["extra_activated"] for r in results]
    med = [r["timing_ms"]["median"] for r in results]
    propagated = [r for r in results if r["extra_activated"] > 0]
    artifact = {
        "script": Path(__file__).name,
        "scope": "single-graph measurement; no training, no search, no parameter sweep",
        "settings": {"graph": "ca_grqc", "K": K, "p0": 1, "p1": 0,
                     "seed_budget_rule": f"T = floor({SEED_FRACTION} * n) = {T}",
                     "seed_budget_note": "a pre-fixed 1%-of-nodes workload; NOT a paper-ratio "
                                         "reproduction, and not claimed to be optimal or hardest.  "
                                         "K and T were not adjusted after seeing any result.",
                     "seed_sets": {"uniform_random": N_RANDOM_SETS,
                                   "degree_top_T": 1, "total": len(results)}},
        "manifest": manifest,
        "timing_protocol": {
            "timed_region": "the diffusion call only",
            "excluded": ["graph loading/parsing", "the brute-force reference", "reporting"],
            "repeats_per_seed_set": args.repeats,
            "repeats_are_not_new_samples":
                f"{args.repeats * len(results)} timed calls over {len(results)} fixed inputs; the "
                f"repeats are repeated computation on the SAME input and are not propagation samples",
        },
        "verification": {
            "reference": "independent brute-force scan-until-stable implementation",
            "comparison": "final activated SET, node for node",
            "all_sets_agree": all(r["reference_agrees_on_the_set"] for r in results),
            "k_core_used_as_a_gate": False,
            "note": "new graphs get an independent-reference comparison and boundary tests; the "
                    "paper's Table 1 values exist only for its own graphs and are not a requirement "
                    "for a graph the paper never used",
        },
        "results": results,
        "summary": {
            "sets_that_propagated": len(propagated),
            "sets_that_did_not_propagate": len(results) - len(propagated),
            "extra_activated_min": min(extra), "extra_activated_median": statistics.median(extra),
            "extra_activated_max": max(extra),
            "normalized_median": statistics.median([r["normalized"] for r in results]),
            "ms_per_diffusion_median_of_medians": statistics.median(med),
            "ms_per_diffusion_min": min(r["timing_ms"]["min"] for r in results),
            "ms_per_diffusion_max": max(r["timing_ms"]["max"] for r in results),
            "edges_scanned_median": statistics.median([r["edges_scanned"] for r in results]),
        },
        "what_this_does_not_show": [
            "that no effective seed set exists -- 21 fixed sets, none of them searched for",
            "whether a diffusion query is the dominant cost of a full search or training run; that "
            "needs the total runtime, not the per-call time",
            "anything about other graphs, other K, other T, or the stochastic setting",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    s = artifact["summary"]
    print(f"    seed sets that propagated : {s['sets_that_propagated']}/{len(results)}")
    print(f"    extra activations         : min {s['extra_activated_min']}, "
          f"median {s['extra_activated_median']}, max {s['extra_activated_max']}")
    print(f"    normalized spread (median): {s['normalized_median']:.4f}")
    print(f"    one diffusion             : median {s['ms_per_diffusion_median_of_medians']:.3f} ms "
          f"(range {s['ms_per_diffusion_min']:.3f} - {s['ms_per_diffusion_max']:.3f} ms)")
    print(f"    edges scanned (median)    : {s['edges_scanned_median']:.0f}")
    print(f"    reference agrees on the SET for all sets: "
          f"{artifact['verification']['all_sets_agree']}")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
