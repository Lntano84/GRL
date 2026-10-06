#!/usr/bin/env python3
"""Run the frozen 8-call score export/replay qualification for GP01."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


PAIRS = [("barth5", 4), ("barth5", 32), ("wing_nodal", 4), ("wing_nodal", 32)]
SEED = 100
EPSILON = 0.03
TIMEOUT_SECONDS = 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_metis(path: Path) -> tuple[int, int, list[tuple[int, int]], list[set[int]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    n, m = map(int, lines[0].split()[:2])
    if len(lines) != n + 1:
        raise ValueError(f"{path}: expected {n} adjacency rows, got {len(lines) - 1}")
    adjacency: list[set[int]] = []
    undirected: set[tuple[int, int]] = set()
    for u, line in enumerate(lines[1:]):
        listed = [int(x) - 1 for x in line.split()]
        ns = set(listed)
        if len(ns) != len(listed):
            raise ValueError(f"{path}: repeated neighbor at vertex {u}")
        if u in ns or any(v < 0 or v >= n for v in ns):
            raise ValueError(f"{path}: self-loop or out-of-range neighbor at vertex {u}")
        adjacency.append(ns)
        for v in ns:
            undirected.add((min(u, v), max(u, v)))
    if any(v not in adjacency[u] or u not in adjacency[v] for u, v in undirected):
        raise ValueError(f"{path}: adjacency is not symmetric")
    if len(undirected) != m:
        raise ValueError(f"{path}: header says {m} edges, parsed {len(undirected)}")
    if sum(map(len, adjacency)) != 2 * m:
        raise ValueError(f"{path}: directed adjacency count is not 2m")
    return n, m, sorted(undirected), adjacency


def parse_partition(path: Path, n: int, k: int, edges: list[tuple[int, int]]) -> dict:
    labels = [int(line.strip()) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(labels) != n or min(labels) < 0 or max(labels) >= k:
        raise ValueError(f"{path}: invalid partition label count or range")
    sizes = [labels.count(b) for b in range(k)]
    bound = math.floor((1.0 + EPSILON) * math.ceil(n / k))
    cut = sum(labels[u] != labels[v] for u, v in edges)
    printed = re.findall(r"cut\s*=\s*(\d+)\s*\(primary objective function\)",
                         path.with_name("stdout.log").read_text(encoding="utf-8", errors="replace"))
    if not printed or int(printed[-1]) != cut:
        raise ValueError(f"{path}: independent cut {cut} does not match solver result {printed[-1:]}")
    return {"cut": cut, "sizes": sizes, "max_block_size": max(sizes),
            "capacity_bound": bound, "balanced": max(sizes) <= bound}


def run_one(exe: Path, graph_path: Path, run_dir: Path, k: int, seed: int,
            env_base: dict[str, str], export_path: Path | None,
            replay_path: Path | None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    local_graph = run_dir / "input.graph"
    shutil.copyfile(graph_path, local_graph)
    audit = run_dir / "gp01_audit.csv"
    audit.write_text("event,context,pass,k,nodes,arcs,active_nodes,source\n", encoding="ascii")
    for key in ("MKH_GP01_EXPORT_SCORES", "MKH_GP01_SCORE_FILE", "MKH_GP00_DUMP_FEATURES"):
        env_base.pop(key, None)
    env = env_base.copy()
    env["MKH_GP01_AUDIT_LOG"] = str(audit.resolve())
    if export_path is not None:
        env["MKH_GP01_EXPORT_SCORES"] = str(export_path.resolve())
    if replay_path is not None:
        env["MKH_GP01_SCORE_FILE"] = str(replay_path.resolve())
    cmd = [str(exe), "-h", "input.graph", "--input-file-format=metis",
           "--instance-type=graph", "--preset-type=default", "-k", str(k),
           "-e", str(EPSILON), "-o", "cut", "-t", "1", "--seed", str(seed),
           "--c-t=160", "--write-partition-file=true", "--partition-output-folder", "."]
    (run_dir / "command.json").write_text(json.dumps({"cwd": str(run_dir.resolve()),
                                                       "argv": cmd,
                                                       "score_mode": "model" if export_path else "replay"},
                                                      indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=False)
        elapsed = time.perf_counter() - start
        output = result.stdout.decode("utf-8", errors="replace")
        (run_dir / "stdout.log").write_text(output, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"solver returned {result.returncode}")
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        partial = exc.stdout or b""
        (run_dir / "stdout.log").write_bytes(partial if isinstance(partial, bytes)
                                             else partial.encode("utf-8", errors="replace"))
        raise TimeoutError(f"solver exceeded {TIMEOUT_SECONDS}s")
    parts = list(run_dir.glob(f"input.graph.part{k}.epsilon{EPSILON}.seed{seed}.*"))
    if len(parts) != 1:
        raise RuntimeError(f"expected one partition output, found {len(parts)}")
    n, m, edges, adjacency = read_metis(local_graph)
    part = parse_partition(parts[0], n, k, edges)
    events = list(csv.DictReader(audit.open(encoding="ascii", newline="")))
    model_events = [x for x in events if x["event"] == "MODEL"]
    passes = [x for x in events if x["event"] == "PASS"]
    mode = "replay" if replay_path is not None else "model"
    if len(model_events) != 1 or model_events[0]["source"] != mode:
        raise RuntimeError(f"expected one active {mode} score-entry event, got {model_events}")
    score_rows = None
    score_hash = None
    if export_path is not None:
        score_hash = sha256(export_path)
        with export_path.open(encoding="ascii", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 2 * m:
            raise ValueError(f"expected {2*m} directed arc scores, got {len(rows)}")
        arc_keys = set()
        pair_scores: dict[tuple[int, int], list[float]] = {}
        for row in rows:
            edge_id, u, v = int(row["edge_id"]), int(row["source"]), int(row["target"])
            score = float(row["score"])
            if edge_id < 0 or edge_id >= len(rows) or (u, v) in arc_keys:
                raise ValueError("duplicate or out-of-range score arc")
            if v not in adjacency[u]:
                raise ValueError(f"score arc {(u, v)} is absent from normalized graph")
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("score is non-finite or outside [0,1]")
            arc_keys.add((u, v))
            pair_scores.setdefault((min(u, v), max(u, v)), []).append(score)
        if len(arc_keys) != 2 * m or len(pair_scores) != m:
            raise ValueError("score export does not cover the graph arcs exactly")
        if any(len(values) != 2 or values[0] != values[1] for values in pair_scores.values()):
            raise ValueError("opposite directed arcs do not have identical scores")
        score_rows = len(rows)
    return {"mode": mode, "returncode": result.returncode, "wall_seconds": elapsed,
            "graph_sha256": sha256(local_graph), "partition_sha256": sha256(parts[0]),
            "partition_file": parts[0].name, "score_rows": score_rows,
            "score_sha256": score_hash, "coarsening_pass_events": passes,
            "rating_entry_events": model_events, **part}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    exe = root / "work/gp00/mt-kahypar/build-gp00/mt-kahypar/application/MtKaHyPar.exe"
    graph_dir = root / "work/gp01/graphs"
    runs_dir = root / "work/gp01/runs/interface-audit"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runtime_bins = [r"C:\mingw64\mingw64\bin",
                    str(root / "work/gp00/mt-kahypar/build-gp00/gnu_14.2_cxx11_64_release"),
                    str(root / "work/vcpkg/installed/x64-mingw-dynamic/bin")]
    env_base = os.environ.copy()
    env_base["PATH"] = ";".join(runtime_bins + [env_base.get("PATH", "")])
    records = []
    try:
        for graph_name, k in PAIRS:
            graph = graph_dir / f"{graph_name}.graph"
            pair_dir = runs_dir / f"{graph_name}_k{k}_s{SEED}"
            model_dir, replay_dir = pair_dir / "model", pair_dir / "replay"
            scores = model_dir / "model_scores.csv"
            model = run_one(exe, graph, model_dir, k, SEED, env_base, scores, None)
            replay = run_one(exe, graph, replay_dir, k, SEED, env_base, None, scores)
            pair = {"graph": graph_name, "k": k, "seed": SEED,
                    "input_sha256": sha256(graph), "model": model, "replay": replay,
                    "partition_hash_equal": model["partition_sha256"] == replay["partition_sha256"],
                    "cut_equal": model["cut"] == replay["cut"],
                    "balanced_both": model["balanced"] and replay["balanced"],
                    "score_hash": model["score_sha256"],
                    "score_rows": model["score_rows"]}
            records.append(pair)
            (runs_dir / "interface_audit_results.json").write_text(
                json.dumps({"pairs": records}, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"graph": graph_name, "k": k,
                              "partition_hash_equal": pair["partition_hash_equal"],
                              "cut": [model["cut"], replay["cut"]],
                              "balanced": pair["balanced_both"],
                              "score_rows": pair["score_rows"]}, separators=(",", ":")))
            if not (pair["partition_hash_equal"] and pair["cut_equal"] and pair["balanced_both"]):
                raise RuntimeError(f"score roundtrip qualification failed for {graph_name}, k={k}")
    except Exception as exc:
        (runs_dir / "interface_audit_failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    passed = all(x["partition_hash_equal"] and x["cut_equal"] and x["balanced_both"]
                 for x in records) and len(records) == len(PAIRS)
    (runs_dir / "interface_audit_status.json").write_text(
        json.dumps({"passed": passed, "calls": 2 * sum(1 for _ in records),
                    "expected_calls": 8, "pairs_completed": len(records)}, indent=2) + "\n",
        encoding="utf-8")
    if not passed:
        raise SystemExit("interface audit did not pass; do not generate teachers")
    print(f"PASS: {2 * len(records)} calls across {len(records)} graph/k pairs")


if __name__ == "__main__":
    main()
