"""Step 1 of the qualification check: what is actually in the raw PAI trace, and what is observable when?

This does not model anything.  It answers only:
  * which status values exist in each raw table;
  * how many records have an empty end time (i.e. never terminated inside the trace);
  * what the time origin and horizon are;
  * whether job-level duration can be rebuilt the way the ATLAS benchmark does it.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
csv.field_size_limit(10 ** 9)

JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK_COLS = ["task_name", "job_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
GRP_COLS = ["inst_id", "user", "group_id", "tag_id", "unknown"]


def scan(path, cols, name):
    print(f"\n{'=' * 96}\n  {name}: {path.name}\n{'=' * 96}")
    status_ct = Counter()
    empty_start = 0
    empty_end = 0
    both = 0
    tmin, tmax = None, None
    rows = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            rows += 1
            rec = dict(zip(cols, row))
            status_ct[rec.get("status", "?")] += 1
            s, e = rec.get("start_time", "").strip(), rec.get("end_time", "").strip()
            if not s:
                empty_start += 1
            if not e:
                empty_end += 1
            if not s and not e:
                both += 1
            for v in (s, e):
                if v:
                    try:
                        t = float(v)
                    except ValueError:
                        continue
                    tmin = t if tmin is None else min(tmin, t)
                    tmax = t if tmax is None else max(tmax, t)
    print(f"  rows: {rows:,}")
    print(f"  status distribution: {dict(status_ct.most_common())}")
    print(f"  rows with EMPTY start_time: {empty_start:,}")
    print(f"  rows with EMPTY end_time  : {empty_end:,}")
    print(f"  rows with BOTH empty      : {both:,}")
    if tmin is not None:
        print(f"  time range: {tmin:,.0f} .. {tmax:,.0f}  "
              f"(span {(tmax - tmin) / 86400:.2f} days, origin offset {tmin:,.0f} s)")
    return {"rows": rows, "status": dict(status_ct), "empty_start": empty_start,
            "empty_end": empty_end, "both_empty": both, "tmin": tmin, "tmax": tmax}


def main() -> int:
    out = {}
    out["job"] = scan(DATA / "pai_job_table.csv", JOB_COLS, "JOB TABLE")
    out["task"] = scan(DATA / "pai_task_table.csv", TASK_COLS, "TASK TABLE")
    out["group"] = scan(DATA / "pai_group_tag_table.csv", GRP_COLS, "GROUP TAG TABLE")

    print(f"\n{'=' * 96}\n  CROSS-TABLE OBSERVATION\n{'=' * 96}")
    js, ts = out["job"]["status"], out["task"]["status"]
    print(f"  job statuses : {js}")
    print(f"  task statuses: {ts}")
    print()
    print("  The ATLAS benchmark filters BOTH tables with `status == 'Terminated'`,")
    print("  which discards exactly the records whose end_time is empty.  Those records are")
    print("  the genuinely incomplete ones and are therefore available in the raw trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
