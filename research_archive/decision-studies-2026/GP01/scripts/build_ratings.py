#!/usr/bin/env python3
"""Build frozen K, AVG, CAL-TIE, and CAL-EXACT edge score files from teachers."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from gp01_common import GRAPHS, K_VALUES, read_metis, sha256


def stable_edge_hash(u: int, v: int) -> int:
    payload = f"20261001|{u}|{v}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def write_scores(path: Path, edges: list[tuple[int, int]], values: dict[tuple[int, int], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="ascii", newline="\n") as f:
        f.write("source,target,score\n")
        for u, v in edges:
            f.write(f"{u},{v},{values[(u, v)]:.9g}\n")


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "work/gp01/outputs"
    status = json.loads((out / "teacher_status.json").read_text(encoding="utf-8"))
    if not status.get("all_valid"):
        raise SystemExit("All 80 valid teacher partitions are required before rating construction")
    runs = json.loads((out / "teacher_results.json").read_text(encoding="utf-8"))
    by_key = {(r["graph"], r["k"], r["seed"]): r for r in runs}
    rating_dir = out / "ratings"
    graph_manifest = json.loads((out / "graph_manifest.json").read_text(encoding="utf-8"))
    graph_records = {r["name"]: r for r in graph_manifest["graphs"]}
    manifest = {"teacher_runs_used": 80, "target_k_values": list(K_VALUES),
                "edge_hash": "SHA-256(ASCII '20261001|u|v'), first 8 digest bytes, unsigned big endian",
                "higher_score_semantics": "higher teacher cut frequency; guided policy protects from contraction",
                "graphs": []}

    for graph in GRAPHS:
        graph_path = root / "work/gp01/graphs" / f"{graph}.graph"
        _, _, edges, _ = read_metis(graph_path)
        counts: dict[int, dict[tuple[int, int], int]] = {}
        for k in K_VALUES:
            edge_counts = {edge: 0 for edge in edges}
            for seed in range(10):
                rec = by_key[(graph, k, seed)]
                if rec.get("status") != "ok" or not rec.get("balanced"):
                    raise ValueError(f"invalid teacher partition {graph}, k={k}, seed={seed}")
                labels = [int(x.strip()) for x in Path(rec["partition_path"]).read_text(encoding="ascii").splitlines()
                          if x.strip()]
                if len(labels) != graph_records[graph]["nodes"]:
                    raise ValueError(f"teacher label count mismatch: {graph}, k={k}, seed={seed}")
                for edge in edges:
                    if labels[edge[0]] != labels[edge[1]]:
                        edge_counts[edge] += 1
            counts[k] = edge_counts

        q = {k: {edge: counts[k][edge] / 10.0 for edge in edges} for k in K_VALUES}
        avg_key = {edge: counts[4][edge] + counts[32][edge] for edge in edges}
        avg = {edge: avg_key[edge] / 20.0 for edge in edges}
        graph_record = {"name": graph, "nodes": graph_records[graph]["nodes"],
                        "edges": len(edges), "normalized_graph_sha256": sha256(graph_path),
                        "teacher_cut_count_ranges": {str(k): [min(counts[k].values()), max(counts[k].values())]
                                                      for k in K_VALUES},
                        "files": []}
        for target_k in K_VALUES:
            sorted_target = sorted(q[target_k].values())
            target_methods = {
                "K": dict(q[target_k]),
                "AVG": dict(avg),
            }

            groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for edge in edges:
                groups[avg_key[edge]].append(edge)
            tie_values: dict[tuple[int, int], float] = {}
            cursor = 0
            for key in sorted(groups):
                group_edges = groups[key]
                segment = sorted_target[cursor:cursor + len(group_edges)]
                if len(segment) != len(group_edges):
                    raise AssertionError("CAL-TIE segment length mismatch")
                mean_value = sum(segment) / len(segment)
                for edge in group_edges:
                    tie_values[edge] = mean_value
                cursor += len(group_edges)
            if cursor != len(edges):
                raise AssertionError("CAL-TIE did not consume the full q_k distribution")
            target_methods["CAL-TIE"] = tie_values

            exact_order = sorted(edges, key=lambda e: (avg_key[e], stable_edge_hash(*e)))
            target_methods["CAL-EXACT"] = {edge: value for edge, value in zip(exact_order, sorted_target)}

            for method, values in target_methods.items():
                path = rating_dir / f"{graph}_k{target_k}_{method}.csv"
                write_scores(path, edges, values)
                # Ordering preservation and distribution matching checks are deterministic assertions.
                if method in {"CAL-TIE", "CAL-EXACT"}:
                    order = sorted(edges, key=lambda e: (avg_key[e], stable_edge_hash(*e)))
                    for a, b in zip(order, order[1:]):
                        if values[a] > values[b] + 1e-12:
                            raise AssertionError(f"{method} reversed AVG order for {graph}, k={target_k}")
                if method == "CAL-TIE":
                    for key, group_edges in groups.items():
                        if len({values[e] for e in group_edges}) != 1:
                            raise AssertionError("CAL-TIE split an AVG tie group")
                if method == "CAL-EXACT" and sorted(values.values()) != sorted_target:
                    raise AssertionError("CAL-EXACT did not exactly match the q_k distribution")
                graph_record["files"].append({"target_k": target_k, "method": method,
                                               "path": str(path.resolve()), "sha256": sha256(path),
                                               "distinct_scores": len(set(values.values())),
                                               "score_min": min(values.values()),
                                               "score_max": max(values.values())})
        manifest["graphs"].append(graph_record)

    (out / "rating_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graphs": len(manifest["graphs"]),
                      "rating_files": sum(len(g["files"]) for g in manifest["graphs"]),
                      "status": "frozen"}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
