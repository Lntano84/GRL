"""RT-01R step 5: group-level view with the CORRECT field semantics.

The superseded report called column 3 of the group_tag table a "workload tag".  The official schema
is ``inst_id, user, gpu_type_spec, group, workload``, so column 3 is ``group`` -- documented as "a
semantic tag that indicates some instances have similar customized inputs ... instances with the same
group tag are considered as repeated instances".  Column 4 is ``workload`` and is sparse.

This recomputes the censoring concentration using ``group`` and reports ``workload`` separately.
It also reports the batch-coherence figure restricted to multi-record buckets, which is the only
version that carries information.
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


def main() -> int:
    grp, wl = {}, {}
    with (DATA / "pai_group_tag_table.csv").open(newline="", encoding="utf-8",
                                                 errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            inst = row[0].strip()
            if inst:
                grp[inst] = row[3].strip()      # group
                wl[inst] = row[4].strip()       # workload (sparse)
    jobs = []
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            inst = row[1].strip()
            jobs.append({"inst": inst, "user": row[2].strip(), "status": row[3].strip(),
                         "s": float(row[4]) if row[4].strip() else None,
                         "e": float(row[5]) if row[5].strip() else None,
                         "grp": grp.get(inst, ""), "wl": wl.get(inst, "")})
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    T = t0 + 0.60 * (t1 - t0)
    arrived = [j for j in launched if j["s"] <= T]
    unf = [j for j in arrived if j["e"] is None or j["e"] > T]
    print("=" * 100)
    print(f"  RT-01R STEP 5 -- concentration by GROUP (official field semantics)")
    print(f"  T = {T:,.0f} (day {(T-t0)/DAY:.1f});  arrived {len(arrived):,};  "
          f"unfinished {len(unf):,} ({len(unf)/len(arrived):.2%})")
    print("=" * 100)

    print(f"\n  group field non-empty : {sum(1 for j in arrived if j['grp']):,} / {len(arrived):,}")
    print(f"  workload field non-empty: {sum(1 for j in arrived if j['wl']):,} / {len(arrived):,} "
          f"({sum(1 for j in arrived if j['wl'])/len(arrived):.2%})  <- officially ~9%")

    for field, label in (("grp", "GROUP"), ("wl", "WORKLOAD")):
        tot = Counter(j[field] for j in arrived if j[field])
        uf = Counter(j[field] for j in unf if j[field])
        items = sorted(((uf.get(k, 0) / n, n, k) for k, n in tot.items() if n >= 200),
                       reverse=True)
        overall = len(unf) / len(arrived)
        print(f"\n  --- by {label} (>=200 arrived) ---")
        print(f"      distinct {label.lower()} values with >=200 arrived: {len(items):,}")
        print(f"      overall unfinished share: {overall:.2%}")
        for share, n, k in items[:5]:
            print(f"        {k[:34]:<34} arrived={n:>6,} unfinished={uf.get(k,0):>6,} "
                  f"share={share:>7.2%}")
        tail = [i for i in items if i[0] >= 2 * overall]
        print(f"      values with share >= 2x overall: {len(tail):,}")
        if len(items) >= 10:
            print(f"      unfinished jobs sitting in the top-10 {label.lower()} values: "
                  f"{sum(uf.get(k,0) for _,_,k in items[:10])/max(1,len(unf)):.1%}")
        vals = [s for s, _, _ in items]
        if vals:
            print(f"      share distribution over those values: p50={sorted(vals)[len(vals)//2]:.2%}  "
                  f"p90={sorted(vals)[int(0.9*len(vals))]:.2%}  max={max(vals):.2%}")
        print(f"      NOTE: instances sharing a group tag are documented as REPEATED instances,")
        print(f"            so a whole group sharing an outcome is expected by construction.")

    # ---- batch coherence, multi-record buckets only -------------------------------------------
    print("\n" + "=" * 100)
    print("  BATCH COHERENCE -- all buckets vs multi-record buckets only")
    print("=" * 100)
    for name, keyf in (("(group, user, start-minute)", lambda j: (j["grp"], j["user"],
                                                                 int(j["s"]) // 60)),
                       ("(group, start-minute)", lambda j: (j["grp"], int(j["s"]) // 60))):
        buckets = defaultdict(list)
        for j in arrived:
            buckets[keyf(j)].append(j)
        single = sum(1 for v in buckets.values() if len(v) == 1)
        multi = [v for v in buckets.values() if len(v) >= 2]
        pure = sum(1 for v in multi
                   if (lambda nu: nu == 0 or nu == len(v))(
                       sum(1 for j in v if j["e"] is None or j["e"] > T)))
        print(f"\n  {name}")
        print(f"    buckets total {len(buckets):,} | single-record {single:,} "
              f"({single/len(buckets):.2%}) | multi-record {len(multi):,}")
        print(f"    coherence on ALL buckets      : "
              f"{(single+pure)/len(buckets):.2%}   <- inflated by single-record buckets")
        print(f"    coherence on MULTI-RECORD only: {pure/max(1,len(multi)):.2%}   <- the honest figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
