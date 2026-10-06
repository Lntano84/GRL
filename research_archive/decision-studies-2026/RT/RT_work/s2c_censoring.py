"""Step 2c: separate genuinely-running (cleanly censored) jobs from stale never-terminated records.

At a frozen update point T:
  * a job with a recorded end <= T is COMPLETE;
  * a job with no recorded end that was LAUNCHED before T is UNFINISHED;
  * among the unfinished, only those whose status is ``Running`` are plausibly still executing.

The ATLAS/PAI release keeps rows that never received an end even when their status is ``Failed``;
those have huge attained ages and are stale bookkeeping rather than running jobs.  This script
quantifies both readings so the choice is explicit rather than hidden.
"""
from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 9)


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            jobs.append({
                "status": r["status"], "tag": r["tag_id"], "fw": r["framework"],
                "s": float(r["task_start"]) if r["task_start"] else None,
                "e": float(r["task_end"]) if r["task_end"] else None,
                "n": int(r["n_tasks"] or 0),
            })
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max([j["e"] for j in jobs if j["e"] is not None])
    span = t1 - t0
    print("=" * 104)
    print(f"  STEP 2c -- censoring, clean reading vs permissive reading")
    print(f"  window {t0:,.0f}..{t1:,.0f} ({span/86400:.2f} days), launched {len(launched):,}")
    print("=" * 104)

    print(f"\n  {'T/span':>8} {'T day':>7} | {'arrived':>9} {'complete':>9} "
          f"{'UNFIN(all)':>11} {'%':>7} | {'UNFIN(Running)':>15} {'%':>7} "
          f"{'attained p50':>13} {'attained p90':>13}")
    table = []
    for frac in (0.2, 0.4, 0.6, 0.8, 0.95):
        T = t0 + frac * span
        arrived = [j for j in launched if j["s"] <= T]
        comp = [j for j in arrived if j["e"] is not None and j["e"] <= T]
        unf_all = [j for j in arrived if j["e"] is None or j["e"] > T]
        unf_run = [j for j in unf_all if j["status"] == "Running"]
        att = sorted(T - j["s"] for j in unf_run)
        p50 = att[len(att) // 2] if att else 0
        p90 = att[int(0.9 * len(att))] if att else 0
        print(f"  {frac:>8.2f} {frac*span/86400:>7.1f} | {len(arrived):>9,} {len(comp):>9,} "
              f"{len(unf_all):>11,} {len(unf_all)/len(arrived):>6.2%} | "
              f"{len(unf_run):>15,} {len(unf_run)/len(arrived):>6.2%} "
              f"{p50:>12,.0f}s {p90:>12,.0f}s")
        table.append({"frac": frac, "T": T, "arrived": len(arrived), "complete": len(comp),
                      "unfinished_all": len(unf_all), "unfinished_running": len(unf_run),
                      "attained_p50_running": p50, "attained_p90_running": p90})

    print(f"\n  attained-service percentiles of the two unfinished readings at T/span = 0.6")
    T = t0 + 0.6 * span
    arrived = [j for j in launched if j["s"] <= T]
    unf_all = [j for j in arrived if j["e"] is None or j["e"] > T]
    for name, sub in (("all never-ended", unf_all),
                      ("status=Running", [j for j in unf_all if j["status"] == "Running"]),
                      ("status=Failed", [j for j in unf_all if j["status"] == "Failed"])):
        att = sorted(T - j["s"] for j in sub)
        if not att:
            print(f"    {name:<22} n=0")
            continue
        print(f"    {name:<22} n={len(att):>7,}  p10={att[int(0.1*len(att))]:>12,.0f}s "
              f"p50={att[len(att)//2]:>12,.0f}s p90={att[int(0.9*len(att))]:>12,.0f}s "
              f"max={att[-1]:>12,.0f}s")

    print(f"\n  long-lived unfinished jobs at T/span = 0.6 (attained > 30 days)")
    stale = [j for j in unf_all if (T - j["s"]) > 30 * 86400]
    print(f"    n = {len(stale):,} ({len(stale)/max(1,len(unf_all)):.1%} of unfinished)")
    print(f"    statuses: {dict(Counter(j['status'] for j in stale).most_common())}")
    print(f"    -> these are stale bookkeeping rows; treating them as still-running would be wrong.")

    print(f"\n  CLEAN reading at T/span = 0.6 (unfinished = status Running)")
    unf_run = [j for j in unf_all if j["status"] == "Running"]
    comp = [j for j in arrived if j["e"] is not None and j["e"] <= T]
    d_comp = sorted(j["e"] - j["s"] for j in comp)
    att_run = sorted(T - j["s"] for j in unf_run)
    print(f"    complete  n={len(d_comp):>8,}  mean={statistics.fmean(d_comp):>10,.0f}s  "
          f"p50={d_comp[len(d_comp)//2]:>9,.0f}s  p90={d_comp[int(0.9*len(d_comp))]:>10,.0f}s")
    print(f"    unfinished n={len(att_run):>7,}  mean attained>={statistics.fmean(att_run):>9,.0f}s  "
          f"p50={att_run[len(att_run)//2]:>9,.0f}s  p90={att_run[int(0.9*len(att_run))]:>10,.0f}s")
    print(f"    a completed-only estimator sees mean {statistics.fmean(d_comp):,.0f}s; the")
    print(f"    unfinished jobs have already exceeded that for "
          f"{sum(1 for a in att_run if a > statistics.fmean(d_comp))/len(att_run):.1%} of them")

    # concentration by type, clean reading
    print(f"\n  concentration at T/span = 0.6, clean reading")
    tot = Counter(j["tag"] for j in arrived)
    unf = Counter(j["tag"] for j in unf_run)
    items = sorted(((unf.get(t, 0) / n, n, t) for t, n in tot.items() if n >= 200), reverse=True)
    overall = len(unf_run) / len(arrived)
    print(f"    overall unfinished share {overall:.2%}; tags with >=200 arrived: {len(items)}")
    print(f"    top-5 tags by share:")
    for share, n, t in items[:5]:
        print(f"      {t}  arrived={n:>6,} unfinished={unf.get(t,0):>6,} share={share:>7.2%}")
    print(f"    tags with share >= 2x overall: {sum(1 for s,_,_ in items if s >= 2*overall)}")
    print(f"    share of all unfinished jobs sitting in the top 10 tags: "
          f"{sum(unf.get(t,0) for _,_,t in items[:10])/max(1,len(unf_run)):.1%}")

    # framework view
    print(f"\n  concentration by FRAMEWORK at T/span = 0.6, clean reading")
    totf = Counter(j["fw"] for j in arrived)
    unf_f = Counter(j["fw"] for j in unf_run)
    for f, n in totf.most_common(8):
        print(f"    {f:<18} arrived={n:>7,} unfinished={unf_f.get(f,0):>6,} "
              f"share={unf_f.get(f,0)/n:>7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
