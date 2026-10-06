"""Q01 step 1a: data qualification of the packaged LimeQO matrices.

Answers, from the four shipped CSVs only:
  * shape, whether rows are unique query filenames, whether the first column is a "default" plan;
  * units and value ranges, and whether any cell is missing / infinite / zero;
  * the initial mask: how many cells are already observed before exploration starts;
  * columns that are numerically identical (candidate duplicate-plan equivalence classes).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "limeqo_mirror" / "dataset"


def read_matrix(path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    names = [r[0] for r in rows[1:] if r]
    vals = [[float(x) for x in r[1:]] for r in rows[1:] if r]
    return header, names, vals


def main() -> int:
    out = {}
    print("=" * 100)
    print("  Q01 STEP 1a -- DATA QUALIFICATION")
    print("=" * 100)
    for name in ("ceb", "job", "stack", "dsb"):
        p = DATASET / f"{name}-matrix.csv"
        if not p.exists():
            print(f"\n  {name}: MISSING {p}")
            continue
        header, names, vals = read_matrix(p)
        n, m = len(vals), len(vals[0])
        flat = [v for row in vals for v in row]
        nan = sum(1 for v in flat if v != v)
        inf = sum(1 for v in flat if v in (float("inf"), float("-inf")))
        zero = sum(1 for v in flat if v == 0.0)
        neg = sum(1 for v in flat if v < 0)
        # identical columns
        dup_groups = []
        seen = {}
        for j in range(m):
            key = tuple(vals[i][j] for i in range(n))
            if key in seen:
                dup_groups.append((seen[key], j))
            else:
                seen[key] = j
        # undefined hints: a column that is one constant value for every query
        const_cols = [j for j in range(m)
                      if len({vals[i][j] for i in range(n)}) == 1]
        # is column 0 the per-query minimum?
        col0_is_min = sum(1 for i in range(n)
                          if vals[i][0] == min(vals[i]))
        print(f"\n  === {name} ===")
        print(f"    file {p.name} ({p.stat().st_size:,} bytes)")
        print(f"    header[0] = {header[0]!r}; columns = {m} ({header[1]} .. {header[-1]})")
        print(f"    rows = {n}; unique row labels = {len(set(names))}")
        print(f"    values: min {min(flat):.6g}  max {max(flat):.6g}  "
              f"median {sorted(flat)[len(flat)//2]:.6g}")
        print(f"    NaN {nan}  inf {inf}  exact-zero {zero}  negative {neg}")
        print(f"    identical column pairs (same vector over all rows): {len(dup_groups)}")
        if dup_groups[:5]:
            print(f"      examples: {[(a+1, b+1) for a, b in dup_groups[:5]]}")
        print(f"    columns constant across all queries: {len(const_cols)} "
              f"{[c+1 for c in const_cols[:8]]}")
        print(f"    column 0 equals the row minimum for {col0_is_min}/{n} queries "
              f"({col0_is_min/n:.2%})")
        out[name] = {"file": p.name, "rows": n, "cols": m, "unique_row_labels": len(set(names)),
                     "min": min(flat), "max": max(flat),
                     "median": sorted(flat)[len(flat) // 2],
                     "nan": nan, "inf": inf,
                     "zero": zero, "negative": neg,
                     "identical_column_pairs": len(dup_groups),
                     "constant_columns": len(const_cols),
                     "col0_is_row_min": col0_is_min}

    (HERE / "q01_data_profile.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01_data_profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
