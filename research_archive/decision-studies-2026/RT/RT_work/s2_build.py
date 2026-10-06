"""Step 2: build the derived job table and audit the timing / feedback semantics.

Mirrors the ATLAS recipe (job processing time = latest task end - earliest task start, submission time
from the job table) but KEEPS every status instead of filtering to ``Terminated``, so that the
incomplete records survive.

Writes ``rt_jobs.csv`` and prints the audit needed before any estimator is fitted.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
csv.field_size_limit(10 ** 9)

JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
GRP_COLS = ["inst_id", "user_hash", "group_id", "tag_id", "pad"]
# Field order VERIFIED against the raw files (see s1b_join.py):
#   task.col0 = job_name  -> joins to job.col0 (overlap 1,055,501 / 1,055,501)
#   task.col1 = framework (not a task id; the notebook's own header labels this wrong)
#   group_tag.col0 = inst_id -> joins to job.col1 (overlap 1,053,971)
#   group_tag.col3 = tag_id (32-hex workload tag)
TASK_COLS = ["job_name", "framework", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]


def fnum(v):
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    print("=" * 100)
    print("  STEP 2 -- derived job table + timing/feedback audit")
    print("=" * 100)

    # ---- group tags: inst_id -> tag_id (job type) -------------------------------------------
    tag_of_inst = {}
    with (DATA / "pai_group_tag_table.csv").open(newline="", encoding="utf-8",
                                                  errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            inst = row[0].strip()
            user_hash = row[1].strip()
            tag = row[3].strip()
            if inst:
                tag_of_inst[inst] = (tag, user_hash)
    print(f"  group-tag rows: {len(tag_of_inst):,}")

    # ---- tasks: per job, earliest start and latest end --------------------------------------
    job_task_start = {}
    job_task_end = {}
    job_ntask = Counter()
    job_task_status = defaultdict(Counter)
    job_framework = defaultdict(Counter)
    task_rows = 0
    with (DATA / "pai_task_table.csv").open(newline="", encoding="utf-8",
                                            errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            task_rows += 1
            rec = dict(zip(TASK_COLS, row))
            jn = rec["job_name"].strip()
            if not jn:
                continue
            job_ntask[jn] += 1
            job_task_status[jn][rec["status"].strip()] += 1
            job_framework[jn][rec["framework"].strip()] += 1
            s, e = fnum(rec["start_time"]), fnum(rec["end_time"])
            if s is not None:
                if jn not in job_task_start or s < job_task_start[jn]:
                    job_task_start[jn] = s
            if e is not None:
                if jn not in job_task_end or e > job_task_end[jn]:
                    job_task_end[jn] = e
    print(f"  task rows: {task_rows:,}; distinct jobs with tasks: {len(job_ntask):,}")

    # ---- jobs -------------------------------------------------------------------------------
    jobs = []
    status_ct = Counter()
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            rec = dict(zip(JOB_COLS, row))
            jn = rec["job_name"].strip()
            inst = rec["inst_id"].strip()
            status = rec["status"].strip()
            status_ct[status] += 1
            tag, user_gt = tag_of_inst.get(inst, ("", ""))
            js, je = fnum(rec["start_time"]), fnum(rec["end_time"])
            fw = job_framework.get(jn)
            jobs.append({
                "job_name": jn, "inst_id": inst, "user": rec["user"].strip(),
                "status": status, "job_start": js, "job_end": je,
                "task_start": job_task_start.get(jn), "task_end": job_task_end.get(jn),
                "n_tasks": job_ntask.get(jn, 0),
                "framework": fw.most_common(1)[0][0] if fw else "",
                "tag_id": tag, "group_id": user_gt,
            })
    print(f"  job rows: {len(jobs):,}   statuses: {dict(status_ct.most_common())}")

    # ---- timing semantics -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  TIMING SEMANTICS AUDIT")
    print("=" * 100)
    no_task = sum(1 for j in jobs if j["task_start"] is None)
    has_jobstart = sum(1 for j in jobs if j["job_start"] is not None)
    print(f"  jobs with a task-level start: {len(jobs) - no_task:,} "
          f"({(len(jobs)-no_task)/len(jobs):.2%})")
    print(f"  jobs with a job-level start : {has_jobstart:,} ({has_jobstart/len(jobs):.2%})")
    print(f"  jobs with NO task-level start (never launched inside trace): {no_task:,}")

    # submit vs first task start: how long is the queue delay?
    qd = [j["task_start"] - j["job_start"] for j in jobs
          if j["job_start"] is not None and j["task_start"] is not None
          and j["task_start"] >= j["job_start"]]
    qd.sort()
    if qd:
        print(f"  queue delay (first task start - job submit), n={len(qd):,}: "
              f"p50={qd[len(qd)//2]:,.0f}s  p90={qd[int(0.9*len(qd))]:,.0f}s  "
              f"p99={qd[int(0.99*len(qd))]:,.0f}s  max={qd[-1]:,.0f}s")
        print(f"    fraction with queue delay == 0: {sum(1 for x in qd if x == 0)/len(qd):.2%}")

    # duration definitions
    dur_span = [(j["task_end"] - j["task_start"]) for j in jobs
                if j["task_start"] is not None and j["task_end"] is not None]
    dur_span = [d for d in dur_span if d > 0]
    dur_span.sort()
    print(f"\n  p* = max task end - min task start (the benchmark definition), n={len(dur_span):,}")
    print(f"    p50={dur_span[len(dur_span)//2]:,.0f}s  p90={dur_span[int(0.9*len(dur_span))]:,.0f}s  "
          f"p99={dur_span[int(0.99*len(dur_span))]:,.0f}s  max={dur_span[-1]:,.0f}s  "
          f"mean={statistics.fmean(dur_span):,.0f}s")
    print(f"    jobs with p* <= 0 (excluded by the benchmark): "
          f"{sum(1 for j in jobs if j['task_start'] is not None and j['task_end'] is not None and j['task_end'] - j['task_start'] <= 0):,}")

    # do failed / running jobs have a start?
    print("\n  BY STATUS -- is the record launched, and does it have an end?")
    print(f"    {'status':<12} {'n':>9} {'has task_start':>15} {'has task_end':>13} "
          f"{'has job_end':>12}")
    for st, _ in status_ct.most_common():
        sub = [j for j in jobs if j["status"] == st]
        print(f"    {st:<12} {len(sub):>9,} "
              f"{sum(1 for j in sub if j['task_start'] is not None):>15,} "
              f"{sum(1 for j in sub if j['task_end'] is not None):>13,} "
              f"{sum(1 for j in sub if j['job_end'] is not None):>12,}")

    tmin = min(j["job_start"] for j in jobs if j["job_start"] is not None)
    tmax = max([j["job_end"] for j in jobs if j["job_end"] is not None]
               + [j["task_end"] for j in jobs if j["task_end"] is not None])
    print(f"\n  time origin: {tmin:,.0f}; latest observed end: {tmax:,.0f}; "
          f"span {(tmax - tmin)/86400:.2f} days")

    # ---- write ------------------------------------------------------------------------------
    out = HERE / "rt_jobs.csv"
    cols = ["job_name", "status", "user", "framework", "tag_id", "group_id", "n_tasks",
            "job_start", "job_end", "task_start", "task_end"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for j in jobs:
            w.writerow(j)
    print(f"\n  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    meta = {"job_rows": len(jobs), "task_rows": task_rows,
            "status_counts": dict(status_ct), "time_origin": tmin, "time_max": tmax,
            "jobs_no_task_start": no_task,
            "p_star_median": dur_span[len(dur_span) // 2],
            "p_star_p90": dur_span[int(0.9 * len(dur_span))],
            "p_star_max": dur_span[-1],
            "queue_delay_median": qd[len(qd) // 2] if qd else None}
    (HERE / "s2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
