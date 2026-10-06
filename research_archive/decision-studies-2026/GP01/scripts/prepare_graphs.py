#!/usr/bin/env python3
"""Convert the four frozen SuiteSparse Matrix Market inputs to simple METIS graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "barth5": (15606, "Pothen/barth5"),
    "wing_nodal": (10937, "DIMACS10/wing_nodal"),
    "as-22july06": (22963, "Newman/as-22july06"),
    "PGPgiantcompo": (10680, "Arenas/PGPgiantcompo"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_mm(path: Path) -> tuple[int, int, list[tuple[int, int]], str, int]:
    with path.open("rt", encoding="ascii", errors="strict") as f:
        header = f.readline().strip().split()
        if len(header) != 5 or header[0].lower() != "%%matrixmarket":
            raise ValueError(f"not a Matrix Market file: {path}")
        _, object_type, storage, field, symmetry = [x.lower() for x in header]
        if object_type != "matrix" or storage != "coordinate":
            raise ValueError(f"expected a coordinate matrix, got {header}")
        dimensions = None
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                parts = stripped.split()
                if len(parts) != 3:
                    raise ValueError("invalid Matrix Market dimension row")
                dimensions = tuple(map(int, parts))
                break
        if dimensions is None:
            raise ValueError("missing Matrix Market dimensions")
        nrows, ncols, declared_entries = dimensions
        if nrows != ncols:
            raise ValueError(f"graph matrix must be square: {nrows}x{ncols}")
        edges: list[tuple[int, int]] = []
        self_loops = 0
        rows = 0
        for line_no, line in enumerate(f, start=3):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"invalid coordinate at line {line_no}")
            u, v = int(parts[0]) - 1, int(parts[1]) - 1
            rows += 1
            if not (0 <= u < nrows and 0 <= v < nrows):
                raise ValueError(f"coordinate out of range at line {line_no}")
            if u != v:
                edges.append((min(u, v), max(u, v)))
            else:
                self_loops += 1
        if rows != declared_entries:
            raise ValueError(f"declared {declared_entries} entries but read {rows}")
        if symmetry not in {"symmetric", "general", "hermitian", "skew-symmetric"}:
            raise ValueError(f"unsupported symmetry flag: {symmetry}")
        return nrows, declared_entries, edges, f"{field}/{symmetry}", self_loops


def convert(name: str, source: Path, output_dir: Path, archive: Path | None) -> dict:
    expected_n, collection_id = EXPECTED[name]
    n, declared_entries, raw_edges, mm_semantics, self_loops = parse_mm(source)
    if n != expected_n:
        raise ValueError(f"{name}: expected {expected_n} vertices, got {n}")
    edges = sorted(set(raw_edges))
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"{name}.graph"
    with graph_path.open("wt", encoding="ascii", newline="\n") as f:
        f.write(f"{n} {len(edges)}\n")
        for neighbors in adjacency:
            f.write(" ".join(str(v + 1) for v in neighbors) + "\n")

    return {
        "name": name,
        "collection_id": collection_id,
        "expected_nodes": expected_n,
        "nodes": n,
        "matrix_market": str(source),
        "matrix_market_sha256": sha256(source),
        "archive": str(archive) if archive else None,
        "archive_sha256": sha256(archive) if archive and archive.exists() else None,
        "matrix_market_semantics": mm_semantics,
        "declared_coordinate_entries": declared_entries,
        "unique_simple_undirected_edges": len(edges),
        "self_loops_dropped": self_loops,
        "duplicate_or_reverse_entries_dropped": len(raw_edges) - len(edges),
        "edge_weights": "discarded; output is an unweighted structural graph",
        "graph_file": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "graph_bytes": graph_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory containing extracted <name>/<name>.mtx files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for name in EXPECTED:
        source = args.data_dir / name / f"{name}.mtx"
        archive = args.data_dir.parent / "data_raw" / f"{name}.tar.gz"
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(convert(name, source, args.output_dir, archive))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"graphs": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
