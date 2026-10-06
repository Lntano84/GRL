"""Step 5: why the unfinished class is not a feedback-latency effect.

Tests whether "never terminated inside the trace" is explained by batch composition (all instances of
a job being stuck together) rather than by jobs simply being long.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 9)
DAY = 86400.0


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            jobs.append({
                "tag": r["tag_id"], "fw": r["framework"], "user": r["user"],
                "status": r["status"], "n": int(r["n_tasks"] or 0),
                "s": float(r["task_start"]) if r["task_start"] else None,
                "e": float(r["task_end"]) if r["task_end"] else None,
            })
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    span = t1 - t0
    T = t0 + 0.6 * span
    arrived = [j for j in launched if j["s"] <= T]
    unf = [j for j in arrived if j["e"] is None or j["e"] > T]
    comp = [j for j in arrived if j["e"] is not None and j["e"] <= T]

    print("=" * 100)
    print("  STEP 5 -- what kind of record is 'unfinished'?")
    print("=" * 100)
    print(f"  T = {T:,.0f} (day {(T-t0)/DAY:.1f});  arrived {len(arrived):,};  "
          f"unfinished {len(unf):,} ({len(unf)/len(arrived):.2%})")

    # ---- age of records ---------------------------------------------------------------------
    print("\n  (A) how long ago did unfinished records START?")
    ages = sorted(T - j["s"] for j in unf)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
        i = min(len(ages) - 1, int(q * (len(ages) - 1)))
        print(f"      p{int(q*100):<3} = {ages[i]/DAY:>7.1f} days")
    print(f"      started in the last 1 day : {sum(1 for a in ages if a <= DAY):>7,} "
          f"({sum(1 for a in ages if a <= DAY)/len(ages):.1%})")
    print(f"      started in the last 7 days: {sum(1 for a in ages if a <= 7*DAY):>7,} "
          f"({sum(1 for a in ages if a <= 7*DAY)/len(ages):.1%})")
    print(f"      started > 30 days ago     : {sum(1 for a in ages if a > 30*DAY):>7,} "
          f"({sum(1 for a in ages if a > 30*DAY)/len(ages):.1%})")

    # ---- batch composition ------------------------------------------------------------------
    print("\n  (B) are unfinished jobs explained by batch composition (same tag + user + start time)?")
    key = lambda j: (j["tag"], j["user"], int(j["s"]) // 60)
    for name, group in (("unfinished", unf), ("completed", comp)):
        buckets = defaultdict(list)
        for j in group:
            buckets[key(j)].append(j)
        sizes = sorted((len(v) for v in buckets.values()), reverse=True)
        mixed = 0
        for v in buckets.values():
            pass
        print(f"      {name:<11} distinct (tag,user,start-minute) buckets: {len(buckets):>7,}  "
              f"median bucket size {sizes[len(sizes)//2] if sizes else 0}  max {sizes[0] if sizes else 0}")

    # all-or-nothing: for buckets that contain unfinished jobs, what share of the bucket is unfinished?
    buckets = defaultdict(list)
    for j in arrived:
        buckets[key(j)].append(j)
    pure_unf = pure_comp = mixed_b = 0
    for v in buckets.values():
        nu = sum(1 for j in v if j["e"] is None or j["e"] > T)
        if nu == len(v):
            pure_unf += 1
        elif nu == 0:
            pure_comp += 1
        else:
            mixed_b += 1
    print(f"      buckets entirely unfinished : {pure_unf:>7,}")
    print(f"      buckets entirely completed  : {pure_comp:>7,}")
    print(f"      buckets mixed               : {mixed_b:>7,}")
    print(f"      -> share of buckets that are all-or-nothing: "
          f"{(pure_unf+pure_comp)/len(buckets):.1%}")

    # ---- how long do unfinished jobs eventually take? ---------------------------------------
    print("\n  (C) for unfinished-at-T jobs that DO have a recorded end, when does it arrive?")
    have_end = [j for j in unf if j["e"] is not None]
    print(f"      n with a recorded end: {len(have_end):,} of {len(unf):,} "
          f"({len(have_end)/len(unf):.1%})")
    if have_end:
        delay = sorted(j["e"] - T for j in have_end)
        print(f"      end time is AFTER T by: p50={delay[len(delay)//2]/DAY:.2f} days  "
              f"p90={delay[int(0.9*len(delay))]/DAY:.2f} days  max={delay[-1]/DAY:.2f} days")
        cont = sorted(j["e"] - j["s"] for j in have_end)
        real = sorted(j["e"] - j["s"] for j in comp)
        print(f"      their TOTAL duration: p50={cont[len(cont)//2]/DAY:.2f} days  "
              f"p90={cont[int(0.9*len(cont))]/DAY:.2f} days")
        print(f"      completed jobs' total duration: p50={real[len(real)//2]/60:.1f} min  "
              f"p90={real[int(0.9*len(real))]/60:.1f} min")
        print(f"      ratio of medians: "
              f"{(cont[len(cont)//2])/max(1e-9, real[len(real)//2]):,.0f}x")

    # ---- requested size of unfinished vs completed ------------------------------------------
    print("\n  (D) requested size: are unfinished jobs bigger, or just older?")
    for name, group in (("unfinished", unf), ("completed", comp)):
        ns = sorted(j["n"] for j in group)
        print(f"      {name:<11} tasks per job: p50={ns[len(ns)//2]}  p90={ns[int(0.9*len(ns))]}  "
              f"mean={sum(ns)/len(ns):.2f}")
    # age-matched comparison: restrict to jobs that started in the last 2 days before T
    recent = [j for j in arrived if (T - j["s"]) <= 2 * DAY]
    ru = [j for j in recent if j["e"] is None or j["e"] > T]
    rc = [j for j in recent if j["e"] is not None and j["e"] <= T]
    print(f"\n      AGE-MATCHED (started within 2 days of T): arrived {len(recent):,}")
    print(f"        unfinished {len(ru):,} ({len(ru)/max(1,len(recent)):.1%})  "
          f"completed {len(rc):,}")
    if ru and rc:
        nu = sorted(T - j["s"] for j in ru)
        dc = sorted(j["e"] - j["s"] for j in rc)
        print(f"        unfinished attained: p50={nu[len(nu)//2]/3600:.1f}h")
        print(f"        completed duration : p50={dc[len(dc)//2]/3600:.1f}h  "
              f"p90={dc[int(0.9*len(dc))]/3600:.1f}h")
        frac_beyond = sum(1 for a in nu if a > dc[int(0.9 * len(dc))]) / len(nu)
        print(f"        unfinished already past the completed p90: {frac_beyond:.1%}")
        print("        -> with age matching, the two classes are directly comparable; this is the")
        print("           version of the comparison that a feedback-latency story would predict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
