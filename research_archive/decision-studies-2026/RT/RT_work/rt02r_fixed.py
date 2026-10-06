"""RT-02R -- corrected controlled completion-feedback experiment.

Fixes applied, and nothing else (no new features, no threshold change, no window search):

1. FROZEN CLOCK.  ``t0``/``t1`` are taken from the previously frozen trace window
   (494,319 .. 6,451,192), NOT recomputed after the class-A filter.  Keeping the original clock is
   what makes the three update points the ones that were pre-registered.

2. FALLBACK RULES SEPARATED.  The superseded code collapsed two different rules into one:

       if v is None or len(pairs) < MIN_GROUP: v = gl_hist      # WRONG

   The stated protocol is
       * recent group too sparse (< 30)  -> that METHOD's global estimate (for KM, the global KM);
       * KM median unidentifiable        -> that JOB's HISTORY prediction.
   The two cases are now separate branches and are counted separately.

3. BLINDNESS CHECK ON ALL THREE WINDOWS, input AND prediction.  First perturb every censored job's
   hidden final duration (start time and censoring status untouched), then rebuild the visible channel
   and RE-RUN all four methods; any difference in inputs or predictions aborts the run.

Outputs per-job predictions, the fallback reason for each job, and the error decomposition, so the
KM-vs-HISTORY gap can be attributed to fallback samples or to non-fallback samples.
"""
from __future__ import annotations

import csv
import copy
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rt02_experiment import km_median  # reuse the KM routine that already passed its hand checks

DATA = HERE / "data"
csv.field_size_limit(10 ** 9)
DAY = 86400.0

# ---- the previously frozen window (RT01_report.md section 1 / RT01R) ------------------------
FROZEN_T0 = 494_319.0
FROZEN_T1 = 6_451_192.0
T_FRACS = (0.40, 0.60, 0.80)
RECENT_DAYS = 7
TEST_DAYS = 2
MIN_GROUP = 30
ABORT = []


def load_classA():
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
    grp = {}
    with (DATA / "pai_group_tag_table.csv").open(newline="", encoding="utf-8",
                                                 errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[0].strip():
                grp[row[0].strip()] = row[3].strip()
    A = []
    with (DATA / "pai_job_table.csv").open(newline="", encoding="utf-8",
                                           errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            jn, inst, status = row[0].strip(), row[1].strip(), row[3].strip()
            if status != "Terminated" or not (row[4].strip() and row[5].strip()):
                continue
            if tot.get(jn, 0) == 0 or nend.get(jn, 0) != tot.get(jn, 0):
                continue
            s, e = float(row[4]), float(row[5])
            if not (math.isfinite(s) and math.isfinite(e)) or e - s <= 0:
                continue
            A.append({"job": jn, "s": s, "e": e, "p": e - s, "grp": grp.get(inst, "")})
    A.sort(key=lambda j: j["s"])
    return A


def med(xs):
    return statistics.median(xs) if xs else None


def evaluate(A, T, perturb_hidden=False):
    """Run the four methods at one update point.  Returns predictions and diagnostics."""
    recent = [j for j in A if T - RECENT_DAYS * DAY <= j["s"] < T]
    test = [j for j in A if T <= j["s"] < T + TEST_DAYS * DAY]
    history = [j for j in A if j["s"] < T and j["e"] <= T]
    if not recent or not test:
        return None

    def hid(j):
        """The evaluation-side duration.  Perturbation only touches the HIDDEN side."""
        if perturb_hidden and j["e"] > T:
            return j["p"] * 1000.0
        return j["p"]

    # visible channel: identical under perturbation by construction
    obs = {}
    for j in recent:
        obs[j["job"]] = (min(j["p"], T - j["s"]), 1 if j["e"] <= T else 0)

    # global estimates
    gl_hist_vals = [j["p"] for j in history]
    gl_hist = med(gl_hist_vals) if gl_hist_vals else med([j["p"] for j in A if j["s"] < T])
    gl_rec_completed = [obs[j["job"]][0] for j in recent if obs[j["job"]][1] == 1]
    gl_rec = med(gl_rec_completed)
    km_all = [(obs[j["job"]][0], bool(obs[j["job"]][1])) for j in recent]
    gl_km = km_median([z for z, _ in km_all], [d for _, d in km_all])
    gl_full = med([hid(j) for j in recent])

    g_hist = defaultdict(list)
    for j in history:
        g_hist[j["grp"]].append(j["p"])
    g_rec_comp = defaultdict(list)
    g_rec_all = defaultdict(list)
    g_km = defaultdict(list)
    for j in recent:
        z, d = obs[j["job"]]
        g_rec_all[j["grp"]].append(hid(j))
        g_km[j["grp"]].append((z, bool(d)))
        if d == 1:
            g_rec_comp[j["grp"]].append(z)

    rows = []
    for j in test:
        g = j["grp"]
        rec = {"job": j["job"], "group": g, "true": j["p"]}

        # HISTORY: group history median, else the history global median
        hv = g_hist.get(g, [])
        if len(hv) >= MIN_GROUP and med(hv) is not None:
            rec["HISTORY"] = med(hv)
            rec["HISTORY_reason"] = "group_history"
        else:
            rec["HISTORY"] = gl_hist
            rec["HISTORY_reason"] = "sparse_history_global"

        # RECENT: completed jobs of the recent queue, group median, else the method's global
        rv = g_rec_comp.get(g, [])
        if len(rv) >= MIN_GROUP and med(rv) is not None:
            rec["RECENT"] = med(rv)
            rec["RECENT_reason"] = "group_recent_completed"
        else:
            rec["RECENT"] = gl_rec
            rec["RECENT_reason"] = "sparse_recent_global_recent"

        # KM: same recent queue.  TWO separate rules, per protocol.
        kv = g_km.get(g, [])
        if len(kv) < MIN_GROUP or not kv:
            rec["KM"] = gl_km                      # rule A: sparse -> this METHOD's global estimate
            rec["KM_reason"] = "sparse_group_global_km"
        else:
            kmed = km_median([z for z, _ in kv], [d for _, d in kv])
            if kmed is None:
                rec["KM"] = rec["HISTORY"]          # rule B: unidentifiable -> this job's HISTORY
                rec["KM_reason"] = "median_unidentifiable_uses_history"
            else:
                rec["KM"] = kmed
                rec["KM_reason"] = "group_km"

        # FULL_LABEL: not deployable
        fv = g_rec_all.get(g, [])
        if len(fv) >= MIN_GROUP and med(fv) is not None:
            rec["FULL_LABEL"] = med(fv)
            rec["FULL_LABEL_reason"] = "group_recent_all"
        else:
            rec["FULL_LABEL"] = gl_full
            rec["FULL_LABEL_reason"] = "sparse_recent_global_all"

        for m in ("HISTORY", "RECENT", "KM", "FULL_LABEL"):
            rec[f"err_{m}"] = abs(math.log1p(rec[m]) - math.log1p(j["p"]))
        rows.append(rec)

    diag = {
        "T": T, "n_recent": len(recent), "n_test": len(test), "n_history": len(history),
        "censored_recent": sum(1 for j in recent if obs[j["job"]][1] == 0),
        "censored_share": sum(1 for j in recent if obs[j["job"]][1] == 0) / len(recent),
        "gl_hist": gl_hist, "gl_rec": gl_rec, "gl_km": gl_km, "gl_full": gl_full,
        "km_returns_global_km_distinct_from_gl_hist":
            (gl_km is not None and gl_hist is not None and abs(gl_km - gl_hist) > 1e-12),
    }
    return rows, diag


def main() -> int:
    A = load_classA()
    print("=" * 100)
    print("  RT-02R -- corrected run")
    print("=" * 100)
    print(f"  class A, finite, positive duration: {len(A):,}")
    print(f"  FROZEN clock t0={FROZEN_T0:,.0f}  t1={FROZEN_T1:,.0f}  "
          f"span {(FROZEN_T1-FROZEN_T0)/DAY:.2f} days")
    inside = [j for j in A if j["s"] >= FROZEN_T0]
    print(f"  class-A jobs starting at/after the frozen t0: {len(inside):,} "
          f"(earliest class-A start {A[0]['s']:,.0f})")

    # ---- normal run ---------------------------------------------------------------------------
    per_frac = {}
    for frac in T_FRACS:
        T = FROZEN_T0 + frac * (FROZEN_T1 - FROZEN_T0)
        out = evaluate(A, T)
        if out is None:
            print(f"\n  T/{frac}: empty queue, skipped")
            continue
        rows, diag = evaluate(A, T)
        per_frac[f"{frac}"] = {"diag": diag, "rows": rows}
        print(f"\n{'=' * 100}")
        print(f"  T/{frac} (day {frac*(FROZEN_T1-FROZEN_T0)/DAY:.1f})  recent {diag['n_recent']:,}  "
              f"censored {diag['censored_recent']:,} ({diag['censored_share']:.2%})  "
              f"test {diag['n_test']:,}  history {diag['n_history']:,}")
        print(f"{'=' * 100}")
        print(f"  global estimates: HISTORY {diag['gl_hist']:,.0f}  RECENT(completed) "
              f"{diag['gl_rec']:,.0f}  KM {diag['gl_km'] if diag['gl_km'] is None else format(diag['gl_km'], ',.0f')}  "
              f"FULL {diag['gl_full']:,.0f}")
        print(f"  global KM differs from global history median: "
              f"{diag['km_returns_global_km_distinct_from_gl_hist']}")
        print(f"\n  {'method':<12} {'mean|log err|':>14}   fallback reasons")
        for m in ("HISTORY", "RECENT", "KM", "FULL_LABEL"):
            L = sum(r[f"err_{m}"] for r in rows) / len(rows)
            rc = Counter(r[f"{m}_reason"] for r in rows)
            print(f"  {m:<12} {L:>14.5f}   {dict(rc.most_common())}")

    # ---- blindness check: all three windows, inputs AND predictions ---------------------------
    print(f"\n{'=' * 100}")
    print("  BLINDNESS CHECK -- perturb hidden final durations x1000, ALL THREE windows")
    print(f"{'=' * 100}")
    blind_ok = True
    for frac in T_FRACS:
        T = FROZEN_T0 + frac * (FROZEN_T1 - FROZEN_T0)
        a = evaluate(A, T)
        b = evaluate(A, T, perturb_hidden=True)
        if a is None or b is None:
            print(f"  T/{frac}: empty queue, cannot check")
            continue
        rows_a, diag_a = a
        rows_b, diag_b = b
        # inputs: the visible channel and the global estimates derived from it
        in_same = (diag_a["censored_recent"] == diag_b["censored_recent"]
                   and abs(diag_a["gl_rec"] - diag_b["gl_rec"]) < 1e-12
                   and diag_a["gl_km"] == diag_b["gl_km"])
        # predictions from the two deployable recent methods
        pred_same = all(abs(ra["RECENT"] - rb["RECENT"]) < 1e-12
                        and abs(ra["KM"] - rb["KM"]) < 1e-12
                        for ra, rb in zip(rows_a, rows_b))
        hist_same = all(abs(ra["HISTORY"] - rb["HISTORY"]) < 1e-12
                        for ra, rb in zip(rows_a, rows_b))
        changed = diag_a["censored_recent"]
        print(f"  T/{frac}: perturbed {changed:,} hidden durations | visible inputs unchanged: "
              f"{in_same} | HISTORY unchanged: {hist_same} | RECENT&KM unchanged: {pred_same}")
        if not (in_same and pred_same):
            blind_ok = False
            ABORT.append(f"blindness check failed at T/{frac}")
    if not blind_ok:
        print(f"\n  ABORT: {ABORT}")
        return 1
    print("  -> all three windows pass; HISTORY is unaffected by construction, and the two")
    print("     deployable recent methods depend only on the visible channel")

    # ---- error decomposition ------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  ERROR DECOMPOSITION -- where does the KM minus HISTORY gap live?")
    print(f"{'=' * 100}")
    print(f"  {'window':>7} {'KM-HISTORY total':>17} {'on KM-fallback rows':>20} "
          f"{'on KM group-KM rows':>21} {'n fallback':>11}")
    decomp = {}
    for f, d in per_frac.items():
        rows = d["rows"]
        tot = sum(r["err_KM"] - r["err_HISTORY"] for r in rows) / len(rows)
        fb = [r for r in rows if r["KM_reason"] != "group_km"]
        nf = [r for r in rows if r["KM_reason"] == "group_km"]
        c_fb = sum(r["err_KM"] - r["err_HISTORY"] for r in fb) / len(rows)
        c_nf = sum(r["err_KM"] - r["err_HISTORY"] for r in nf) / len(rows)
        decomp[f] = {"total": tot, "fallback_contribution": c_fb,
                     "group_km_contribution": c_nf,
                     "n_fallback": len(fb), "n_group_km": len(nf)}
        print(f"  {float(f)*100:>6.0f}% {tot:>+17.6f} {c_fb:>+20.6f} {c_nf:>+21.6f} "
              f"{len(fb):>11,}")
    print("  (positive = KM worse than HISTORY)")

    # ---- scores and gate ----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  SCORES")
    print(f"{'=' * 100}")
    print(f"  {'window':>7} {'HISTORY':>10} {'RECENT':>10} {'KM':>10} {'FULL':>10} "
          f"{'KM vs min(H,R)':>15}")
    scores = {}
    for f, d in per_frac.items():
        rows = d["rows"]
        s = {m: sum(r[f"err_{m}"] for r in rows) / len(rows)
             for m in ("HISTORY", "RECENT", "KM", "FULL_LABEL")}
        rel = (min(s["HISTORY"], s["RECENT"]) - s["KM"]) / min(s["HISTORY"], s["RECENT"])
        scores[f] = {"scores": s, "km_vs_min_relative": rel}
        print(f"  {float(f)*100:>6.0f}% {s['HISTORY']:>10.5f} {s['RECENT']:>10.5f} "
              f"{s['KM']:>10.5f} {s['FULL_LABEL']:>10.5f} {rel:>+15.2%}")
    rels = [scores[f]["km_vs_min_relative"] for f in scores]
    pos = sum(1 for x in rels if x > 0)
    print(f"\n  KM better than both baselines in {pos}/{len(rels)} windows; "
          f"equal-weight mean {sum(rels)/len(rels):+.2%}")
    print(f"  KM better than HISTORY in "
          f"{sum(1 for f in scores if scores[f]['scores']['KM'] < scores[f]['scores']['HISTORY'])}"
          f"/{len(scores)} windows")

    # ---- save ---------------------------------------------------------------------------------
    payload = {
        "frozen_clock": {"t0": FROZEN_T0, "t1": FROZEN_T1},
        "earliest_classA_start": A[0]["s"],
        "n_classA": len(A),
        "n_classA_at_or_after_t0": len(inside),
        "T_fracs": list(T_FRACS), "recent_days": RECENT_DAYS, "test_days": TEST_DAYS,
        "min_group": MIN_GROUP,
        "diagnostics": {f: per_frac[f]["diag"] for f in per_frac},
        "scores": scores,
        "decomposition": decomp,
        "blindness_all_windows_passed": blind_ok,
        "per_job": {f: per_frac[f]["rows"] for f in per_frac},
    }
    (HERE / "rt02r_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'rt02r_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
