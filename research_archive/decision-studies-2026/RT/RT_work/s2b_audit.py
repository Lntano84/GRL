"""Step 2b: precise audit of the timing semantics and of who is unfinished."""
from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 9)


def main() -> int:
    rows = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    print("=" * 100)
    print(f"  STEP 2b -- timing semantics on {len(rows):,} jobs")
    print("=" * 100)

    def num(v):
        return float(v) if v not in ("", None) else None

    # ---- job_start vs task_start ------------------------------------------------------------
    both = [(num(r["job_start"]), num(r["task_start"])) for r in rows]
    both = [(a, b) for a, b in both if a is not None and b is not None]
    diff = Counter(b - a for a, b in both)
    print("\n  (A) job.start_time  vs  min(task start_time)")
    print(f"      pairs: {len(both):,}")
    print(f"      difference distribution (task_start - job_start): "
          f"{dict(list(diff.most_common(6)))}")
    print(f"      exactly equal: {diff.get(0.0, 0):,} / {len(both):,} "
          f"({diff.get(0.0,0)/len(both):.2%})")
    pos = [d for d in diff.elements() if d > 0]
    print(f"      task starts LATER than job start in {len(pos):,} jobs"
          + (f"; max delay {max(pos):,.0f}s" if pos else ""))
    print("      -> the trace does NOT record a submission time separate from the start of")
    print("         execution; 'submit_time' in the ATLAS recipe is this same start_time.")

    # ---- who is unfinished, and are they launched? ------------------------------------------
    print("\n  (B) status x launched x has end")
    print(f"      {'status':<12} {'n':>9} {'launched':>10} {'has end':>9} {'never end':>11}")
    for st in ("Terminated", "Failed", "Running", "Waiting"):
        sub = [r for r in rows if r["status"] == st]
        launched = sum(1 for r in sub if num(r["task_start"]) is not None)
        hasend = sum(1 for r in sub if num(r["task_end"]) is not None)
        print(f"      {st:<12} {len(sub):>9,} {launched:>10,} {hasend:>9,} "
              f"{len(sub)-hasend:>11,}")

    # ---- censoring over time -----------------------------------------------------------------
    print("\n  (C) censoring at frozen update points")
    starts = [num(r["task_start"]) for r in rows if num(r["task_start"]) is not None]
    ends = [num(r["task_end"]) for r in rows if num(r["task_end"]) is not None]
    t0, t1 = min(starts), max(ends)
    print(f"      trace window: {t0:,.0f} .. {t1:,.0f}  ({(t1-t0)/86400:.2f} days)")

    launched = [r for r in rows if num(r["task_start"]) is not None]
    print(f"      launched jobs: {len(launched):,}")
    print(f"\n      {'T (day)':>8} {'arrived':>9} {'finished':>9} {'still running':>14} "
          f"{'% unfinished':>13}")
    for frac in (0.2, 0.4, 0.6, 0.8, 0.95):
        T = t0 + frac * (t1 - t0)
        arrived = [r for r in launched if num(r["task_start"]) <= T]
        finished = [r for r in arrived if (num(r["task_end"]) or 1e30) <= T]
        n_arr = len(arrived)
        n_fin = len(finished)
        print(f"      {frac*68.95:>8.1f} {n_arr:>9,} {n_fin:>9,} {n_arr-n_fin:>14,} "
              f"{(n_arr-n_fin)/max(1,n_arr):>12.2%}")

    # ---- are the unfinished concentrated by job type? ---------------------------------------
    print("\n  (D) is 'unfinished' concentrated by job type (tag_id)?")
    T = t0 + 0.6 * (t1 - t0)
    arrived = [r for r in launched if num(r["task_start"]) <= T]
    fin = [r for r in arrived if (num(r["task_end"]) or 1e30) <= T]
    unf = [r for r in arrived if (num(r["task_end"]) or 1e30) > T]
    print(f"      at T = {T:,.0f}: arrived {len(arrived):,}, finished {len(fin):,}, "
          f"unfinished {len(unf):,} ({len(unf)/len(arrived):.2%})")
    tag_ct_fin = Counter(r["tag_id"] for r in fin)
    tag_ct_unf = Counter(r["tag_id"] for r in unf)
    print(f"      distinct tags: finished {len(tag_ct_fin):,}, unfinished {len(tag_ct_unf):,}")
    # top tags by unfinished share among tags with >= 200 arrived
    tot = Counter(r["tag_id"] for r in arrived)
    items = []
    for tag, n in tot.items():
        if n >= 200:
            items.append((tag_ct_unf.get(tag, 0) / n, n, tag))
    items.sort(reverse=True)
    print(f"      tags with >=200 arrived: {len(items):,}")
    print(f"      {'tag':<34} {'arrived':>9} {'unfinished':>11} {'share':>8}")
    for share, n, tag in items[:8]:
        print(f"      {tag:<34} {n:>9,} {tag_ct_unf.get(tag,0):>11,} {share:>7.2%}")
    print("      ...")
    for share, n, tag in items[-4:]:
        print(f"      {tag:<34} {n:>9,} {tag_ct_unf.get(tag,0):>11,} {share:>7.2%}")
    overall = len(unf) / len(arrived)
    print(f"      overall unfinished share: {overall:.2%}")
    tail = [i for i in items if i[0] >= 2 * overall]
    print(f"      tags whose unfinished share is >= 2x overall: {len(tail):,}")

    # ---- run-time distribution of unfinished vs finished ------------------------------------
    print("\n  (E) run time of finished jobs vs attained service of unfinished jobs at T")
    d_fin = [num(r["task_end"]) - num(r["task_start"]) for r in fin]
    d_fin = [d for d in d_fin if d and d > 0]
    d_unf = [T - num(r["task_start"]) for r in unf]
    d_unf = [d for d in d_unf if d and d > 0]
    for name, arr in (("finished (completed duration)", d_fin),
                      ("unfinished (attained so far)", d_unf)):
        arr = sorted(arr)
        print(f"      {name:<32} n={len(arr):>8,}  p50={arr[len(arr)//2]:>9,.0f}s  "
              f"p90={arr[int(0.9*len(arr))]:>9,.0f}s  mean={statistics.fmean(arr):>10,.0f}s")
    print("      -> unfinished jobs have already run longer than the median finished job,")
    print("         so dropping them from a training window biases the estimated mean downward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
