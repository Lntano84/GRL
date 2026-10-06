"""RT-01R step 4: corrected regression, restricted to labels that are eligible.

Changes versus the superseded s4:

  * LABELS.  Training labels come only from class A (every task ended AND the job is archived
    Terminated), so ``max(end)`` is a genuine completion time.  Classes B (ended but not successful)
    and C (only some tasks ended) are excluded rather than silently folded in.
  * FEATURES.  One ``Design`` object builds both the fit and the predict matrices, so the
    group-censoring switch cannot differ between the two paths.  The old code trained with the
    feature and predicted with it zeroed for two of three variants.
  * RANK STATISTIC.  Tie-corrected Spearman.
  * ORACLE.  The oracle variant is retained only to make its IMPOSSIBILITY explicit: it can only be
    built for records that eventually received an end time, which is 956 of 116,215.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r3_estimator_lib import Design, fit_ridge, predict, rmsle, spearman  # noqa: E402

DATA = HERE / "data"
csv.field_size_limit(10 ** 9)
DAY = 86400.0


def main() -> int:
    # ---- roll-ups ---------------------------------------------------------------------------
    tot = defaultdict(int)
    nstart = defaultdict(int)
    nend = defaultdict(int)
    role = defaultdict(Counter)
    with (DATA / "pai_task_table.csv").open(newline="", encoding="utf-8",
                                            errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 10:
                continue
            jn = row[0].strip()
            tot[jn] += 1
            role[jn][row[1].strip()] += 1          # task role, per the official task_table docs
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
            jobs.append({"job": jn, "tag": row[1].strip()[:0] or jn, "user": row[2].strip(),
                         "status": row[3].strip(),
                         "s": float(row[4]) if row[4].strip() else None,
                         "e": float(row[5]) if row[5].strip() else None,
                         "nt": tot.get(jn, 0), "nstart": nstart.get(jn, 0),
                         "nend": nend.get(jn, 0)})
    # ---- real group tag via inst_id -----------------------------------------------------------
    grp = {}
    with (DATA / "pai_group_tag_table.csv").open(newline="", encoding="utf-8",
                                                 errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[0].strip():
                grp[row[0].strip()] = row[3].strip()
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        insts = [r[1].strip() for r in csv.reader(fh) if len(r) >= 6]
    for j, inst in zip(jobs, insts):
        j["tag"] = grp.get(inst, "")
        rl = role.get(j["job"])
        j["fw"] = rl.most_common(1)[0][0] if rl else "unknown"
        j["n"] = j["nt"]                    # Design() expects the task count under "n"

    launched = [j for j in jobs if j["nstart"] > 0 and j["s"] is not None]
    ends = [j["e"] for j in jobs if j["e"] is not None]
    t0, t1 = min(j["s"] for j in launched), max(ends)
    span = t1 - t0
    T = t0 + 0.60 * span
    T_END = T + 0.10 * span

    def classA(j):
        return (j["nstart"] > 0 and j["nend"] == j["nt"] and j["status"] == "Terminated"
                and j["e"] is not None and j["e"] - j["s"] > 0)

    train = [j for j in launched if j["s"] < T]
    test = [j for j in launched if T <= j["s"] < T_END and classA(j)]
    A_train = [j for j in train if classA(j) and j["e"] <= T]
    unf_train = [j for j in train if j["e"] is None or j["e"] > T]
    print("=" * 100)
    print("  RT-01R CORRECTED REGRESSION (class-A labels only)")
    print("=" * 100)
    print(f"  T = {T:,.0f} (day {(T-t0)/DAY:.1f});  test window n = {len(test):,} (all class A)")
    print(f"  class-A training rows finished by T : {len(A_train):,}")
    print(f"  unfinished at T (all classes)       : {len(unf_train):,}")
    unrescuable = sum(1 for j in unf_train if j["e"] is None)
    print(f"    of which NEVER have an end time   : {unrescuable:,} "
          f"({unrescuable/len(unf_train):.2%})  -> no oracle is constructible for these")
    # The rescued set must be re-filtered through classA(): having an end time and positive duration
    # is NOT sufficient.  An earlier version used that weaker test and admitted 61 records that are
    # class C (59 Failed + 2 Running with only some tasks ended), so the row below was not a
    # class-A-only comparison.  See r7_rescue_eligibility.py for the audit of those 61.
    resc = [j for j in unf_train if classA(j) and j["e"] is not None and j["e"] - j["s"] > 0]
    resc_all = [j for j in unf_train if j["e"] is not None and j["e"] - j["s"] > 0]
    print(f"    of which do have a final duration : {len(resc_all):,}")
    print(f"    of which are ALSO class A         : {len(resc):,}  <- the only admissible additions")
    print(f"    inadmissible (end time but not class A): {len(resc_all)-len(resc):,}")

    # ---- design -------------------------------------------------------------------------------
    d = Design()
    g_tot, g_unf = defaultdict(int), defaultdict(int)
    for j in train:
        g_tot[j["tag"]] += 1
        if j["e"] is None or j["e"] > T:
            g_unf[j["tag"]] += 1
    d.share = {t: g_unf[t] / g_tot[t] for t in g_tot}

    y_A = [math.log1p(j["e"] - j["s"]) for j in A_train]
    rows = {
        "COMPLETED_only": (A_train, False),
        "COMPLETED_plus_groupcensoring": (A_train, True),
    }
    # the oracle variant: class A plus the few rescued records.  Kept to show its coverage.
    resc_ok = [j for j in resc if j["e"] is not None and j["e"] - j["s"] > 0]
    if resc_ok:
        rows["A_plus_rescued_oracle"] = (A_train + resc_ok, False)

    results = {}
    for name, (tr, use_share) in rows.items():
        # FIT and PREDICT use the SAME switch -- this is the fixed defect.
        Xtr = d.matrix(tr, use_share)
        ytr = [math.log1p(j["e"] - j["s"]) for j in tr]
        w = fit_ridge(Xtr, ytr, 1.0)
        Xte = d.matrix(test, use_share)
        pred = predict(w, Xte)
        true = [j["e"] - j["s"] for j in test]
        r = rmsle(pred, true)
        rho = spearman(pred, true)
        results[name] = {"n_train": len(tr), "use_share": use_share,
                         "rmsle": r, "spearman": rho,
                         "median_pred": statistics.median(pred),
                         "median_true": statistics.median(true)}
        print(f"\n  {name:<32} n_train={len(tr):>8,}  share={'on' if use_share else 'off':<3} "
              f"RMSLE={r:.4f}  Spearman(tie-corr)={rho:+.4f}  "
              f"median pred={statistics.median(pred):,.0f}s true={statistics.median(true):,.0f}s")

    # ---- what the OLD mismatched configuration produced, for comparison -----------------------
    print("\n" + "=" * 100)
    print("  EFFECT OF THE FEATURE-TOGGLE DEFECT (same training rows, switch mismatched at predict)")
    print("=" * 100)
    for name in ("COMPLETED_only", "A_plus_rescued_oracle"):
        if name not in rows:
            continue
        tr, _ = rows[name]
        for fit_share in (True, False):
            w = fit_ridge(d.matrix(tr, fit_share),
                          [math.log1p(j["e"] - j["s"]) for j in tr], 1.0)
            for pred_share in (True, False):
                pred = predict(w, d.matrix(test, pred_share))
                r = rmsle(pred, [j["e"] - j["s"] for j in test])
                tag = "CONSISTENT" if fit_share == pred_share else "MISMATCH <-- old bug"
                print(f"  {name:<26} fit_share={str(fit_share):<5} pred_share={str(pred_share):<5} "
                      f"RMSLE={r:.4f}  {tag}")

    (HERE / "r4_corrected_regression.json").write_text(
        json.dumps({"T": T, "n_test": len(test), "n_classA_train": len(A_train),
                    "n_unfinished": len(unf_train), "n_unrescuable": unrescuable,
                    "n_rescued": len(resc), "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'r4_corrected_regression.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
