"""RT-01R follow-up: audit the 956 "rescued" records and correct the class-A coverage claim.

The superseded code admitted any unfinished record that had an end time and a positive duration,
without re-checking class A.  This script re-derives the eligibility of those records from the raw
task table so the corrected count is on the record.
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
    tot = defaultdict(int)
    nstart = defaultdict(int)
    nend = defaultdict(int)
    with (DATA / "pai_task_table.csv").open(newline="", encoding="utf-8",
                                            errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            jn = row[0].strip()
            tot[jn] += 1
            if row[4].strip():
                nstart[jn] += 1
            if row[5].strip():
                nend[jn] += 1

    jobs = []
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            jn = row[0].strip()
            jobs.append({"job": jn, "status": row[3].strip(),
                         "s": float(row[4]) if row[4].strip() else None,
                         "e": float(row[5]) if row[5].strip() else None,
                         "nt": tot.get(jn, 0), "nstart": nstart.get(jn, 0),
                         "nend": nend.get(jn, 0)})

    launched = [j for j in jobs if j["nstart"] > 0 and j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    T = t0 + 0.60 * (t1 - t0)

    def classA(j):
        return (j["nstart"] > 0 and j["nend"] == j["nt"] and j["status"] == "Terminated"
                and j["e"] is not None and j["e"] - j["s"] > 0)

    train = [j for j in launched if j["s"] < T]
    A_train = [j for j in train if classA(j) and j["e"] <= T]
    unf_train = [j for j in train if j["e"] is None or j["e"] > T]
    weak = [j for j in unf_train if j["e"] is not None and j["e"] - j["s"] > 0]      # old test
    strict = [j for j in weak if classA(j)]                                          # corrected test
    inadmissible = [j for j in weak if not classA(j)]

    print("=" * 100)
    print("  RT-01R -- eligibility of the 'rescued' records")
    print("=" * 100)
    print(f"  class-A training rows finished by T        : {len(A_train):,}")
    print(f"  unfinished at T                            : {len(unf_train):,}")
    print(f"  of those, end time + positive duration (OLD weak test): {len(weak):,}")
    print(f"  of those, ALSO class A (CORRECTED test)    : {len(strict):,}")
    print(f"  inadmissible under the corrected test      : {len(inadmissible):,}")

    print(f"\n  breakdown of the {len(inadmissible):,} inadmissible records:")
    c = Counter(j["status"] for j in inadmissible)
    for k, v in c.most_common():
        print(f"    {k:<12} {v:>5,}")
    partial = sum(1 for j in inadmissible if j["nend"] < j["nt"])
    print(f"    (of which have only SOME tasks ended: {partial:,})")
    print(f"\n  archived status of the {len(strict):,} admissible records:")
    for k, v in Counter(j["status"] for j in strict).most_common():
        print(f"    {k:<12} {v:>5,}")

    corrected_total = len(A_train) + len(strict)
    print(f"\n  CORRECTED class-A oracle training rows: {len(A_train):,} + {len(strict):,} = "
          f"**{corrected_total:,}**")
    print(f"  (the superseded report stated {len(A_train)+len(weak):,}, which included the "
          f"{len(inadmissible):,} ineligible rows)")

    out = {
        "classA_train": len(A_train),
        "unfinished_at_T": len(unf_train),
        "weak_test_admitted": len(weak),
        "corrected_test_admitted": len(strict),
        "inadmissible": len(inadmissible),
        "inadmissible_status": dict(c),
        "inadmissible_partial_tasks": partial,
        "corrected_oracle_train_rows": corrected_total,
        "superseded_oracle_train_rows": len(A_train) + len(weak),
    }
    (HERE / "r7_rescue_eligibility.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'r7_rescue_eligibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
