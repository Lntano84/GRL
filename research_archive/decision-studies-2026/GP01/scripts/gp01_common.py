"""Shared execution and independent verification helpers for GP01."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

EPSILON = 0.03
TIMEOUT_SECONDS = 60
METHODS = ("BASE", "AUTHOR", "K", "AVG", "CAL-TIE", "CAL-EXACT")
GRAPHS = ("barth5", "wing_nodal", "as-22july06", "PGPgiantcompo")
K_VALUES = (4, 32)
EVAL_SEEDS = (100, 101, 102)


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
    edges: set[tuple[int, int]] = set()
    for u, line in enumerate(lines[1:]):
        listed = [int(x) - 1 for x in line.split()]
        ns = set(listed)
        if len(ns) != len(listed) or u in ns or any(v < 0 or v >= n for v in ns):
            raise ValueError(f"{path}: duplicate, self-loop, or invalid neighbor at vertex {u}")
        adjacency.append(ns)
        for v in ns:
            edges.add((min(u, v), max(u, v)))
    if any(v not in adjacency[u] or u not in adjacency[v] for u, v in edges):
        raise ValueError(f"{path}: adjacency is not symmetric")
    if len(edges) != m or sum(map(len, adjacency)) != 2 * m:
        raise ValueError(f"{path}: edge count does not agree with the METIS header")
    return n, m, sorted(edges), adjacency


def parse_partition(run_dir: Path, k: int, seed: int, n: int, edges: list[tuple[int, int]]) -> dict:
    pattern = f"input.graph.part{k}.epsilon{EPSILON}.seed{seed}.*"
    parts = list(run_dir.glob(pattern))
    if len(parts) != 1:
        raise ValueError(f"{run_dir}: expected one partition file matching {pattern}, found {len(parts)}")
    labels = [int(line.strip()) for line in parts[0].read_text(encoding="ascii").splitlines()
              if line.strip()]
    if len(labels) != n or not labels or min(labels) < 0 or max(labels) >= k:
        raise ValueError(f"{parts[0]}: invalid partition size or label range")
    sizes = [0] * k
    for label in labels:
        sizes[label] += 1
    bound = math.floor((1.0 + EPSILON) * math.ceil(n / k))
    cut = sum(labels[u] != labels[v] for u, v in edges)
    log = (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
    printed = re.findall(r"cut\s*=\s*(\d+)\s*\(primary objective function\)", log)
    if not printed or int(printed[-1]) != cut:
        raise ValueError(f"{parts[0]}: independent cut {cut} disagrees with solver output {printed[-1:]}")
    if max(sizes) > bound:
        raise ValueError(f"{parts[0]}: maximum block size {max(sizes)} exceeds {bound}")
    return {"partition_path": str(parts[0]), "partition_sha256": sha256(parts[0]),
            "cut": cut, "sizes": sizes, "max_block_size": max(sizes),
            "capacity_bound": bound, "balanced": True}


def run_solver(root: Path, graph: str, k: int, seed: int, method: str,
               score_file: Path | None = None) -> dict:
    graph_path = root / "work/gp01/graphs" / f"{graph}.graph"
    run_dir = root / "work/gp01/runs/formal" / graph / f"k{k}" / method / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.graph"
    shutil.copyfile(graph_path, input_path)
    audit = run_dir / "gp01_audit.csv"
    audit.write_text("event,context,pass,k,nodes,arcs,active_nodes,source\n", encoding="ascii")
    exe = root / "work/gp00/mt-kahypar/build-gp00/mt-kahypar/application/MtKaHyPar.exe"
    cmd = [str(exe), "-h", "input.graph", "--input-file-format=metis",
           "--instance-type=graph", "--preset-type=default", "-k", str(k),
           "-e", str(EPSILON), "-o", "cut", "-t", "1", "--seed", str(seed),
           "--c-t=160", "--write-partition-file=true", "--partition-output-folder", "."]
    if method == "BASE":
        cmd.extend(["--c-guiding-by-integrated-model=false",
                    "--c-rating-degree-similarity-policy=always_accept"])
    env = os.environ.copy()
    bins = [r"C:\mingw64\mingw64\bin",
            str(root / "work/gp00/mt-kahypar/build-gp00/gnu_14.2_cxx11_64_release"),
            str(root / "work/vcpkg/installed/x64-mingw-dynamic/bin")]
    env["PATH"] = ";".join(bins + [env.get("PATH", "")])
    for name in ("MKH_GP00_DUMP_FEATURES", "MKH_GP01_EXPORT_SCORES", "MKH_GP01_SCORE_FILE"):
        env.pop(name, None)
    env["MKH_GP01_AUDIT_LOG"] = str(audit.resolve())
    if score_file is not None:
        env["MKH_GP01_SCORE_FILE"] = str(score_file.resolve())
    (run_dir / "command.json").write_text(json.dumps({"argv": cmd, "cwd": str(run_dir.resolve()),
                                                       "graph_sha256": sha256(graph_path),
                                                       "score_file": str(score_file.resolve()) if score_file else None},
                                                      indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    status = "ok"
    returncode = None
    try:
        result = subprocess.run(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=False)
        returncode = result.returncode
        (run_dir / "stdout.log").write_text(result.stdout.decode("utf-8", errors="replace"),
                                            encoding="utf-8")
        if returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        partial = exc.stdout or b""
        (run_dir / "stdout.log").write_bytes(partial if isinstance(partial, bytes)
                                             else partial.encode("utf-8", errors="replace"))
    elapsed = time.perf_counter() - start
    rec = {"graph": graph, "k": k, "seed": seed, "method": method, "status": status,
           "returncode": returncode, "wall_seconds": elapsed, "run_dir": str(run_dir.resolve()),
           "graph_sha256": sha256(graph_path), "input_sha256": sha256(input_path),
           "audit_log": str(audit.resolve()), "score_file": str(score_file.resolve()) if score_file else None}
    if status == "ok":
        n, _, edges, _ = read_metis(graph_path)
        rec.update(parse_partition(run_dir, k, seed, n, edges))
        events = list(csv_rows(audit))
        rec["rating_entry_events"] = [e for e in events if e["event"] == "MODEL"]
        rec["coarsening_pass_events"] = [e for e in events if e["event"] == "PASS"]
        if method in {"AUTHOR", "K", "AVG", "CAL-TIE", "CAL-EXACT"}:
            expected_source = "model" if method == "AUTHOR" else "replay"
            if len(rec["rating_entry_events"]) != 1 or rec["rating_entry_events"][0]["source"] != expected_source:
                raise ValueError(f"{method} expected one model scoring call, got {rec['rating_entry_events']}")
        elif method == "BASE" and rec["rating_entry_events"]:
            raise ValueError("BASE unexpectedly invoked the integrated model")
    return rec


def csv_rows(path: Path):
    import csv
    with path.open(encoding="ascii", newline="") as f:
        yield from csv.DictReader(f)


def load_clock(root: Path) -> dict:
    path = root / "work/gp01/outputs/gp01_experiment_clock.json"
    if not path.exists():
        from datetime import datetime, timedelta, timezone
        start = datetime.now(timezone.utc)
        state = {"started_utc": start.isoformat(), "deadline_utc": (start + timedelta(hours=2)).isoformat(),
                 "budget_seconds": 7200, "per_call_timeout_seconds": TIMEOUT_SECONDS}
        path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def seconds_left(clock: dict) -> float:
    from datetime import datetime, timezone
    deadline = datetime.fromisoformat(clock["deadline_utc"])
    return (deadline - datetime.now(timezone.utc)).total_seconds()
