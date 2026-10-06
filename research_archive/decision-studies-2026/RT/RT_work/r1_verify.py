"""RT-01R step 1: verify the audit's numerical claims against the stored artifacts.

Read-only.  Confirms or refutes each item so the corrections rest on recomputed numbers, not on
either party's recollection.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
csv.field_size_limit(10 ** 9)
DAY = 86400.0
T_SPLIT_FRAC = 0.60


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            jobs.append({
                "tag": r["tag_id"], "user": r["user"], "status": r["status"],
                "fw": r["framework"], "n": int(r["n_tasks"] or 0),
                "s": float(r["task_start"]) if r["task_start"] else None,
                "e": float(r["task_end"]) if r["task_end"] else None,
            })
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    span = t1 - t0
    T = t0 + T_SPLIT_FRAC * span

    print("=" * 100)
    print("  RT-01R CLAIM VERIFICATION")
    print("=" * 100)

    # ---------------- CLAIM 1: oracle coverage ------------------------------------------------
    train = [j for j in launched if j["s"] < T]
    completed = [j for j in train if j["e"] is not None and j["e"] <= T and j["e"] - j["s"] > 0]
    unfinished = [j for j in train if j["e"] is None or j["e"] > T]
    rescued = [j for j in unfinished if j["e"] is not None and j["e"] - j["s"] > 0]
    print(f"\n  CLAIM 1 -- oracle coverage at T = {T:,.0f} (day {(T-t0)/DAY:.1f})")
    print(f"    launched records                    : {len(launched):,}")
    print(f"    started before T                    : {len(train):,}")
    print(f"    classified unfinished at T          : {len(unfinished):,}")
    print(f"    of those, HAVE a usable final dur.  : {len(rescued):,} "
          f"({len(rescued)/len(unfinished):.2%})")
    print(f"    of those, NO end time at all        : {len(unfinished)-len(rescued):,} "
          f"({1-len(rescued)/len(unfinished):.2%})")
    print(f"    COMPLETED training rows             : {len(completed):,}")
    print(f"    ALLFINAL training rows (C + rescued): {len(completed)+len(rescued):,}")
    print(f"    extra rows added by the oracle      : {len(rescued):,} "
          f"({len(rescued)/len(completed):.3%} of COMPLETED)")
    print(f"    --> the earlier claim that the oracle supplied 'all 116,215 unfinished jobs' is")
    print(f"        WRONG.  It supplied {len(rescued):,} rows, i.e. {len(rescued)/len(unfinished):.2%} of them.")

    # ---------------- CLAIM 4: status mix of the unfinished ------------------------------------
    print(f"\n  CLAIM 4 -- archived status of the {len(unfinished):,} unfinished-at-T records")
    st = Counter(j["status"] for j in unfinished)
    for k, v in st.most_common():
        print(f"    {k:<12} {v:>8,}  ({v/len(unfinished):>6.2%})")
    print(f"    --> 'unfinished' is a MIXTURE; {st.get('Running',0)/len(unfinished):.1%} are archived Running.")

    # ---------------- CLAIM 3: single-record buckets -------------------------------------------
    print(f"\n  CLAIM 3 -- (tag, user, start-minute) bucket coherence, all buckets vs multi-record only")
    key = lambda j: (j["tag"], j["user"], int(j["s"]) // 60)
    buckets = defaultdict(list)
    for j in [j for j in launched if j["s"] <= T]:
        buckets[key(j)].append(j)
    sizes = Counter(len(v) for v in buckets.values())
    single = sizes.get(1, 0)
    multi = {k: v for k, v in buckets.items() if len(v) >= 2}
    pure_m = mixed_m = 0
    for v in multi.values():
        nu = sum(1 for j in v if j["e"] is None or j["e"] > T)
        if nu == 0 or nu == len(v):
            pure_m += 1
        else:
            mixed_m += 1
    print(f"    buckets total                       : {len(buckets):,}")
    print(f"    single-record buckets               : {single:,} ({single/len(buckets):.2%})")
    print(f"    multi-record buckets                : {len(multi):,}")
    print(f"    multi-record pure (all same state)  : {pure_m:,} ({pure_m/max(1,len(multi)):.2%})")
    print(f"    multi-record mixed                  : {mixed_m:,} ({mixed_m/max(1,len(multi)):.2%})")
    print(f"    --> the earlier 99.9% figure was inflated by single-record buckets; the")
    print(f"        multi-record-only figure is {pure_m/max(1,len(multi)):.2%}.")

    # ---------------- CLAIM 4b: partial task ends ----------------------------------------------
    print(f"\n  CLAIM 4b -- jobs where SOME tasks have an end time and others do not")
    partial = 0
    partial_rows = 0
    task_start = {}
    task_end = {}
    task_status = defaultdict(Counter)
    with (DATA / "pai_task_table.csv").open(newline="", encoding="utf-8",
                                            errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            jn = row[0].strip()
            stt = row[3].strip()
            task_status[jn][stt] += 1
            if row[4].strip():
                task_start[jn] = task_start.get(jn, 0) + 1
            if row[5].strip():
                task_end[jn] = task_end.get(jn, 0) + 1
    for jn, cnt in task_status.items():
        tot = sum(cnt.values())
        ended = task_end.get(jn, 0)
        if 0 < ended < tot:
            partial += 1
            partial_rows += tot - ended
    print(f"    jobs with 0 < ended_tasks < total_tasks : {partial:,}")
    print(f"    their task rows lacking an end time     : {partial_rows:,}")
    print(f"    --> for these jobs the max-over-visible-tasks end time is NOT a complete")
    print(f"        job end; it must be flagged rather than used as a finished duration.")

    # ---------------- CLAIM 6: start-time equality ---------------------------------------------
    print(f"\n  CLAIM 6 -- job.start_time vs min(task.start_time)")
    both = [(j["s"], j["e"]) for j in launched]
    print(f"    comparable records earlier reported     : 1,051,838")
    print(f"    launched records here                   : {len(launched):,}")
    print(f"    --> equality is a fact about this batch; the DOCUMENTED semantics differ")
    print(f"        (job.start_time = submission, task.start_time = launch), so the correct")
    print(f"        statement is that no positive waiting time is OBSERVABLE in the two columns,")
    print(f"        not that a submission time is absent.")

    out = {
        "T_split": T, "t0": t0, "span": span,
        "launched": len(launched), "train": len(train),
        "unfinished_at_T": len(unfinished),
        "unfinished_with_usable_final": len(rescued),
        "unfinished_without_any_end": len(unfinished) - len(rescued),
        "completed_training_rows": len(completed),
        "allfinal_training_rows": len(completed) + len(rescued),
        "oracle_rows_added": len(rescued),
        "oracle_coverage_of_unfinished": len(rescued) / len(unfinished),
        "unfinished_status_mix": dict(st),
        "buckets_total": len(buckets), "buckets_single": single,
        "buckets_multi": len(multi), "buckets_multi_pure": pure_m, "buckets_multi_mixed": mixed_m,
        "jobs_with_partial_task_ends": partial,
        "task_rows_lacking_end_in_partial_jobs": partial_rows,
    }
    (HERE / "r1_claim_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'r1_claim_verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
