"""Compact join-capability audit: which identifiers actually overlap across the three tables?

This decides whether job-level service times can be rebuilt from the raw trace at all.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
csv.field_size_limit(10 ** 9)


def ids(path, col, limit=None):
    """Return (set of ids, Counter of lengths) from one column."""
    s = set()
    lens = Counter()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if len(row) <= col:
                continue
            v = row[col].strip()
            if v:
                s.add(v)
                lens[len(v)] += 1
            if limit and i >= limit:
                break
    return s, lens


def main() -> int:
    print("=" * 96)
    print("  JOIN-CAPABILITY AUDIT")
    print("=" * 96)

    job0, job0l = ids(DATA / "pai_job_table.csv", 0)
    job1, job1l = ids(DATA / "pai_job_table.csv", 1)
    job2, job2l = ids(DATA / "pai_job_table.csv", 2)
    print(f"\n  job table")
    print(f"    col0 (job_name)  : {len(job0):,} distinct, lengths {dict(job0l.most_common(3))}")
    print(f"    col1 (inst_id)   : {len(job1):,} distinct, lengths {dict(job1l.most_common(3))}")
    print(f"    col2 (user)      : {len(job2):,} distinct, lengths {dict(job2l.most_common(3))}")

    task0, task0l = ids(DATA / "pai_task_table.csv", 0)
    task1, task1l = ids(DATA / "pai_task_table.csv", 1)
    print(f"\n  task table")
    print(f"    col0 (task_name) : {len(task0):,} distinct, lengths {dict(task0l.most_common(3))}")
    print(f"    col1 (framework) : {len(task1):,} distinct, lengths {dict(task1l.most_common(3))}")

    gt0, gt0l = ids(DATA / "pai_group_tag_table.csv", 0)
    gt1, gt1l = ids(DATA / "pai_group_tag_table.csv", 1)
    gt3, gt3l = ids(DATA / "pai_group_tag_table.csv", 3)
    print(f"\n  group-tag table")
    print(f"    col0             : {len(gt0):,} distinct, lengths {dict(gt0l.most_common(3))}")
    print(f"    col1             : {len(gt1):,} distinct, lengths {dict(gt1l.most_common(3))}")
    print(f"    col3 (tag)       : {len(gt3):,} distinct, lengths {dict(gt3l.most_common(3))}")

    print(f"\n  OVERLAPS THAT WOULD BE NEEDED FOR THE BENCHMARK JOIN")
    checks = [
        ("task.col0  vs job.col0 (job_name)", task0, job0),
        ("task.col0  vs job.col1 (inst_id)", task0, job1),
        ("gtag.col0  vs job.col1 (inst_id)", gt0, job1),
        ("gtag.col0  vs task.col0", gt0, task0),
        ("gtag.col1  vs job.col2 (user)", gt1, job2),
        ("gtag.col0  vs job.col0", gt0, job0),
    ]
    for name, a, b in checks:
        inter = len(a & b)
        print(f"    {name:<38} |A|={len(a):>9,} |B|={len(b):>9,} overlap={inter:>9,}"
              f"  {'<-- JOINABLE' if inter else ''}")

    print(f"\n  VERDICT")
    joinable = [n for n, a, b in checks if a & b]
    if not joinable:
        print("    No tested identifier pair overlaps.  The job table and the task table do not")
        print("    share a key, so a JOB-level processing time cannot be rebuilt from the raw trace")
        print("    by joining job -> task as the benchmark code assumes.")
    else:
        print(f"    joinable pairs: {joinable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
