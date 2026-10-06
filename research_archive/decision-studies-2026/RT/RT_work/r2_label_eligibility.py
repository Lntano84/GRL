"""RT-01R step 2: label-eligibility table.

For every record, classify exactly what timing information it carries, then state which analytical
target each class can support.  No modelling.  This is the artifact that decides whether any
prediction comparison is admissible at all.

Classes (job level, from the raw task rows):
  A COMPLETE        every task has an end time, and the job's archived status is Terminated
  B COMPLETE_NONOK  every task has an end time, but the job is not Terminated (Failed/Running/Waiting)
  C PARTIAL         some tasks ended and some did not  -> max(end) is NOT a job completion time
  D NO_END          no task has an end time
  E NEVER_LAUNCHED  no task has a start time either
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
    print("=" * 104)
    print("  RT-01R STEP 2 -- label eligibility of every record")
    print("=" * 104)

    # ---- task-level roll-up -----------------------------------------------------------------
    tot_tasks = defaultdict(int)
    started_tasks = defaultdict(int)
    ended_tasks = defaultdict(int)
    with (DATA / "pai_task_table.csv").open(newline="", encoding="utf-8",
                                            errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            jn = row[0].strip()
            tot_tasks[jn] += 1
            if row[4].strip():
                started_tasks[jn] += 1
            if row[5].strip():
                ended_tasks[jn] += 1

    # ---- group_tag: inst_id -> group (semantic tag) ------------------------------------------
    # Official order: inst_id, user, gpu_type_spec, group, workload
    grp_of_inst = {}
    wl_of_inst = {}
    with (DATA / "pai_group_tag_table.csv").open(newline="", encoding="utf-8",
                                                 errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            inst = row[0].strip()
            if inst:
                grp_of_inst[inst] = row[3].strip()      # group
                wl_of_inst[inst] = row[4].strip()       # workload (sparse)

    rows = []
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            jn, inst, user, status = (row[0].strip(), row[1].strip(),
                                      row[2].strip(), row[3].strip())
            rows.append({
                "job": jn, "inst": inst, "user": user, "status": status,
                "s": row[4].strip(), "e": row[5].strip(),
                "grp": grp_of_inst.get(inst, ""), "wl": wl_of_inst.get(inst, ""),
                "nt": tot_tasks.get(jn, 0), "nstart": started_tasks.get(jn, 0),
                "nend": ended_tasks.get(jn, 0),
            })
    print(f"  job records: {len(rows):,}   group tags resolved: "
          f"{sum(1 for r in rows if r['grp'])}   workload tags non-empty: "
          f"{sum(1 for r in rows if r['wl'])}")

    def classify(r):
        if r["nstart"] == 0:
            return "E_NEVER_LAUNCHED"
        if r["nend"] == 0:
            return "D_NO_END"
        if r["nend"] < r["nt"]:
            return "C_PARTIAL"
        return "A_COMPLETE" if r["status"] == "Terminated" else "B_COMPLETE_NONOK"

    cls = Counter()
    per_status = defaultdict(Counter)
    for r in rows:
        c = classify(r)
        cls[c] += 1
        per_status[c][r["status"]] += 1

    order = ["A_COMPLETE", "B_COMPLETE_NONOK", "C_PARTIAL", "D_NO_END", "E_NEVER_LAUNCHED"]
    print(f"\n  {'class':<20} {'n':>10} {'share':>8}   statuses inside")
    for c in order:
        n = cls.get(c, 0)
        inner = ", ".join(f"{k}:{v:,}" for k, v in per_status[c].most_common())
        print(f"  {c:<20} {n:>10,} {n/len(rows):>7.2%}   {inner}")

    # ---- what each class can support --------------------------------------------------------
    print("\n" + "=" * 104)
    print("  WHAT EACH CLASS CAN SUPPORT")
    print("=" * 104)
    support = {
        "A_COMPLETE":
            "YES for a completion-time target. Every task ended and the job is archived "
            "Terminated, so max(end) is a genuine completion time. This is the only class that "
            "supports a supervised run-time label without further assumptions.",
        "B_COMPLETE_NONOK":
            "MIXED. Every task ended, so an end timestamp exists, but the job did not succeed. "
            "Usable for a 'time until this attempt stopped' target; NOT usable as a successful "
            "service duration without stating that assumption, because failure can truncate.",
        "C_PARTIAL":
            "NO for a completion-time target. max(end over the ended tasks) is a lower bound on the "
            "job end only; the remaining tasks' end times are missing. Usable only as a censored "
            "lower bound, and only after deciding why those tasks lack an end.",
        "D_NO_END":
            "NO label at all. Observation time is a lower bound; whether the job was still "
            "executing, had failed silently, or was never recorded cannot be told from the tables.",
        "E_NEVER_LAUNCHED":
            "NO label and NO start. These are submissions that never began, so they are queue "
            "records rather than service records.",
    }
    for c in order:
        print(f"\n  {c}  (n={cls.get(c,0):,})")
        print(f"    {support[c]}")

    # ---- consequence for the earlier comparison ---------------------------------------------
    print("\n" + "=" * 104)
    print("  CONSEQUENCE FOR THE EARLIER PREDICTION COMPARISON")
    print("=" * 104)
    usable = cls.get("A_COMPLETE", 0)
    print(f"  records with an unambiguous completion label (A)     : {usable:,} "
          f"({usable/len(rows):.2%})")
    print(f"  records whose 'duration' is a bound, not a completion: "
          f"{cls.get('B_COMPLETE_NONOK',0)+cls.get('C_PARTIAL',0):,}")
    print(f"  records with no label at all                         : "
          f"{cls.get('D_NO_END',0)+cls.get('E_NEVER_LAUNCHED',0):,}")
    print()
    print("  The earlier RT-01 comparison built its training labels from max(end) over whatever")
    print("  tasks happened to end.  That silently mixed A, B and C, and it treated D as")
    print("  'yields no label'.  A comparison of that kind cannot be read as an information-value")
    print("  bound until the label classes are separated.")

    # ---- how much of the training window is clean? ------------------------------------------
    launched = [r for r in rows if r["nstart"] > 0]
    st = [float(r["s"]) for r in launched if r["s"]]
    en = [float(r["e"]) for r in rows if r["e"]]
    t0, t1 = min(st), max(en)
    T = t0 + 0.60 * (t1 - t0)
    print(f"\n  AT THE FROZEN UPDATE POINT T (day {(T-t0)/DAY:.1f})")
    for label, cond in (("A_COMPLETE finished by T",
                         lambda r: classify(r) == "A_COMPLETE" and r["e"] and float(r["e"]) <= T),
                        ("B_COMPLETE_NONOK finished by T",
                         lambda r: classify(r) == "B_COMPLETE_NONOK" and r["e"]
                         and float(r["e"]) <= T),
                        ("C_PARTIAL with max(end) <= T",
                         lambda r: classify(r) == "C_PARTIAL" and r["e"] and float(r["e"]) <= T),
                        ("not finished by T (any class)",
                         lambda r: r["s"] and float(r["s"]) <= T
                         and (not r["e"] or float(r["e"]) > T))):
        n = sum(1 for r in launched if r["s"] and float(r["s"]) <= T and cond(r))
        print(f"    {label:<34} {n:>9,}")

    out = {
        "classes": {c: cls.get(c, 0) for c in order},
        "class_status_breakdown": {c: dict(per_status[c]) for c in order},
        "support": support,
        "workload_tag_non_empty": sum(1 for r in rows if r["wl"]),
        "group_tag_non_empty": sum(1 for r in rows if r["grp"]),
    }
    (HERE / "r2_label_eligibility.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'r2_label_eligibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
