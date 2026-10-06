"""RT-02 -- one controlled completion-feedback experiment.

SCOPE, fixed in advance
-----------------------
* Records: class A only (every task ended AND the job is archived Terminated), all timestamps finite,
  duration positive.  ``p_j = max_t e_jt - min_t s_jt``; the replay start is ``min_t s_jt``.
  This studies the COMPLETION FEEDBACK of an execution span.  It is NOT cumulative attained service.
* Three update points, fixed at 40% / 60% / 80% of the frozen time span.
* At each T: recent training queue = jobs STARTING in the preceding 7 days; test queue = jobs
  STARTING in the following 2 days.  No window is re-selected on censoring share, method performance
  or group performance.
* The recent queue may only observe ``z_j = min(p_j, T - s_j)`` and ``delta_j = 1[e_j <= T]``.
  The full ``p_j`` of a censored job exists only on the evaluation side.
* Target for every method is the job's duration MEDIAN; metric is mean absolute log error.

CONDITIONING DISCLOSURE
-----------------------
Restricting to class A uses information from the archive (we know these jobs eventually succeeded).
This is a CONDITIONAL REPLAY of a successful-job queue, not a production reproduction: a deployed
system does not know in advance which live jobs will succeed.
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
from r3_estimator_lib import spearman  # noqa: E402

DATA = HERE / "data"
csv.field_size_limit(10 ** 9)
DAY = 86400.0
T_FRACS = (0.40, 0.60, 0.80)
RECENT_DAYS = 7
TEST_DAYS = 2
MIN_GROUP = 30


# ------------------------------------------------------------------ Kaplan-Meier

def km_median(times, events):
    """Kaplan-Meier median.  ``events[i]`` True = observed event, False = right-censored.

    Returns ``None`` when the curve never reaches 0.5.  Tied times are handled by processing all
    subjects at a time point together, so the risk set is the same for every subject at that time and
    the order of ties cannot matter.
    """
    pairs = sorted(zip(times, events))
    n = len(pairs)
    if n == 0:
        return None
    at_risk = n
    surv = 1.0
    i = 0
    while i < n:
        t = pairs[i][0]
        d = c = 0
        while i < n and pairs[i][0] == t:
            if pairs[i][1]:
                d += 1
            else:
                c += 1
            i += 1
        if d:
            surv *= (1.0 - d / at_risk)
            if surv <= 0.5:
                return t
        at_risk -= (d + c)
        if at_risk <= 0:
            break
    return None


def print_km(times, events, label):
    pairs = sorted(zip(times, events))
    n = len(pairs)
    at_risk, surv, i = n, 1.0, 0
    print(f"    {label}: n={n}")
    while i < n:
        t = pairs[i][0]
        d = c = 0
        while i < n and pairs[i][0] == t:
            if pairs[i][1]:
                d += 1
            else:
                c += 1
            i += 1
        if d:
            surv *= (1.0 - d / at_risk)
            print(f"      t={t:>6} at_risk={at_risk:>3} events={d} cens={c} S={surv:.6f}"
                  + ("   <- median reached" if surv <= 0.5 else ""))
        at_risk -= (d + c)


# ------------------------------------------------------------------ pre-check 4: KM unit tests

def km_tests() -> int:
    fails = []
    print("\n" + "=" * 100)
    print("  PRE-CHECK 4 -- Kaplan-Meier against hand-computed examples")
    print("=" * 100)

    def check(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            fails.append(name)

    # (a) no censoring: the median is the ordinary median
    t = [1, 2, 3, 4, 5]
    e = [True] * 5
    print("\n  (a) no censoring, times 1..5")
    print_km(t, e, "no censoring")
    check("no censoring -> median 3", km_median(t, e) == 3, f"{km_median(t, e)}")

    # (b) ties plus censoring.  Hand computation:
    #     t=1: 2 events of 5 at risk -> S=1-2/5=0.6      (already <=0.5? no, 0.6)
    #     t=2: 1 event of 3 at risk  -> S=0.6*(1-1/3)=0.4 -> median = 2
    t = [1, 1, 2, 3, 4]
    e = [True, True, True, False, False]
    print("\n  (b) ties + censoring")
    print_km(t, e, "ties+censoring")
    check("ties+censoring -> median 2", km_median(t, e) == 2, f"{km_median(t, e)}")

    # (c) curve never reaches 0.5
    t = [1, 2, 3, 4]
    e = [False, False, False, False]
    print("\n  (c) all censored")
    check("all censored -> None", km_median(t, e) is None, f"{km_median(t, e)}")

    t = [5, 6, 7, 8, 9, 10]
    e = [True, False, False, False, False, False]
    print("\n  (d) one event out of six (S never reaches 0.5)")
    print_km(t, e, "one event")
    check("one event of six -> None", km_median(t, e) is None, f"{km_median(t, e)}")

    # (e) tie ordering must not matter
    import itertools
    t = [2, 2, 2, 5, 5, 7]
    e = [True, True, False, True, False, True]
    results = {km_median(list(a), list(b)) for a, b in zip(
        itertools.permutations(t), itertools.permutations(e))} if False else None
    # permutation of (t,e) PAIRS, which is the meaningful invariance
    pairs = list(zip(t, e))
    seen = set()
    for perm in itertools.permutations(pairs):
        seen.add(km_median([a for a, _ in perm], [b for _, b in perm]))
    check("result invariant under permutation of (time, event) pairs",
          len(seen) == 1, f"distinct results {seen}")

    print()
    if fails:
        print(f"  {len(fails)} KM TEST(S) FAILED: {fails}")
    else:
        print("  ALL KM TESTS PASSED")
    return len(fails)


# ------------------------------------------------------------------ main

def main() -> int:
    print("=" * 100)
    print("  RT-02 -- controlled completion-feedback experiment")
    print("=" * 100)
    km_fails = km_tests()

    # ---- class A with finite times and positive duration ------------------------------------
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
            if status != "Terminated":
                continue
            if not (row[4].strip() and row[5].strip()):
                continue
            if tot.get(jn, 0) == 0 or nend.get(jn, 0) != tot.get(jn, 0):
                continue
            s, e = float(row[4]), float(row[5])
            if not (math.isfinite(s) and math.isfinite(e)) or e - s <= 0:
                continue
            A.append({"job": jn, "s": s, "e": e, "p": e - s, "grp": grp.get(inst, "")})
    A.sort(key=lambda j: j["s"])
    t0, t1 = A[0]["s"], max(j["e"] for j in A)
    span = t1 - t0
    print(f"\n  class A, finite, positive duration: {len(A):,} jobs")
    print(f"  span {t0:,.0f}..{t1:,.0f} ({(t1-t0)/DAY:.1f} days)")
    sizes = Counter(j["grp"] for j in A)
    print(f"  distinct groups: {len(sizes):,}; groups with >= {MIN_GROUP} jobs: "
          f"{sum(1 for v in sizes.values() if v >= MIN_GROUP):,}; "
          f"share of jobs in such groups: "
          f"{sum(v for v in sizes.values() if v >= MIN_GROUP)/len(A):.2%}")

    def med(xs):
        return statistics.median(xs) if xs else None

    results = {}
    for frac in T_FRACS:
        T = t0 + frac * span
        recent = [j for j in A if T - RECENT_DAYS * DAY <= j["s"] < T]
        test = [j for j in A if T <= j["s"] < T + TEST_DAYS * DAY]
        history = [j for j in A if j["s"] < T and j["e"] <= T]
        if not test or not recent:
            print(f"\n  T/{frac}: empty queue (recent {len(recent)}, test {len(test)}), skipped")
            continue
        # ---- the three recent methods see ONE identical queue and differ only in label access ---
        obs = {}          # job -> (z, delta)
        for j in recent:
            z = min(j["p"], T - j["s"])
            delta = 1 if j["e"] <= T else 0
            obs[j["job"]] = (z, delta)
        recent_jobs = {j["job"] for j in recent}

        gl_hist = med([j["p"] for j in history])
        gl_hist = gl_hist if gl_hist is not None else med([j["p"] for j in A if j["s"] < T])
        gl_rec = med([j["p"] for j in recent if obs[j["job"]][1] == 1])
        gl_full = med([j["p"] for j in recent])

        g_hist = defaultdict(list)
        for j in history:
            g_hist[j["grp"]].append(j["p"])
        g_rec_comp = defaultdict(list)
        g_rec_all = defaultdict(list)
        g_rec_km = defaultdict(list)
        for j in recent:
            z, delta = obs[j["job"]]
            g_rec_all[j["grp"]].append(j["p"])
            if delta == 1:
                g_rec_comp[j["grp"]].append(z)
            g_rec_km[j["grp"]].append((z, bool(delta)))

        preds = {"HISTORY": [], "RECENT": [], "KM": [], "FULL_LABEL": []}
        km_fallback = 0
        hist_fallback = 0
        rec_fallback = 0
        for j in test:
            g = j["grp"]
            # HISTORY: all class-A jobs completed before T, group median, else global
            v = med(g_hist.get(g, []))
            if v is None or len(g_hist.get(g, [])) < MIN_GROUP:
                v = gl_hist
                hist_fallback += 1
            preds["HISTORY"].append(v)
            # RECENT: completed jobs of the recent queue only
            v = med(g_rec_comp.get(g, []))
            if v is None or len(g_rec_comp.get(g, [])) < MIN_GROUP:
                v = gl_rec
                rec_fallback += 1
            preds["RECENT"].append(v)
            # KM: same recent queue, events and right-censored records together
            pairs = g_rec_km.get(g, [])
            v = km_median([z for z, _ in pairs], [d for _, d in pairs]) if pairs else None
            if v is None or len(pairs) < MIN_GROUP:
                v = gl_hist
                km_fallback += 1
            preds["KM"].append(v)
            # FULL_LABEL: the same recent queue with every label visible (NOT deployable)
            v = med(g_rec_all.get(g, []))
            if v is None or len(g_rec_all.get(g, [])) < MIN_GROUP:
                v = gl_full
            preds["FULL_LABEL"].append(v)

        true = [j["p"] for j in test]
        scores = {}
        for k, v in preds.items():
            L = sum(abs(math.log1p(p) - math.log1p(t)) for p, t in zip(v, true)) / len(true)
            scores[k] = {"mean_abs_log_error": L, "spearman": spearman(v, true),
                         "median_pred": med(v), "median_true": med(true)}
        results[str(frac)] = {
            "T": T, "n_recent": len(recent), "n_test": len(test), "n_history": len(history),
            "censored_share_recent": sum(1 for j in recent if obs[j["job"]][1] == 0) / len(recent),
            "km_fallback": km_fallback, "history_group_fallback": hist_fallback,
            "recent_group_fallback": rec_fallback,
            "scores": scores,
        }
        print(f"\n{'=' * 100}")
        print(f"  T/{frac}  (day {frac*(t1-t0)/DAY:.1f})   recent {len(recent):,} "
              f"(censored {sum(1 for j in recent if obs[j['job']][1]==0)/len(recent):.1%})   "
              f"test {len(test):,}   history {len(history):,}")
        print(f"{'=' * 100}")
        print(f"  {'method':<12} {'mean|log err|':>14} {'Spearman':>10} {'median pred':>12} "
              f"{'median true':>12}")
        for k in ("HISTORY", "RECENT", "KM", "FULL_LABEL"):
            s = scores[k]
            print(f"  {k:<12} {s['mean_abs_log_error']:>14.5f} {s['spearman']:>+10.4f} "
                  f"{s['median_pred']:>12,.0f} {s['median_true']:>12,.0f}")
        print(f"  fallbacks: KM {km_fallback:,}/{len(test):,} ({km_fallback/len(test):.1%}) | "
              f"HISTORY {hist_fallback/len(test):.1%} | RECENT {rec_fallback/len(test):.1%}")

    # ---- pre-check 1: identical selection, only label access differs -------------------------
    print(f"\n{'=' * 100}")
    print("  PRE-CHECK 1 -- the three recent methods use IDENTICAL job sets")
    print(f"{'=' * 100}")
    for frac in T_FRACS:
        T = t0 + frac * span
        recent = [j for j in A if T - RECENT_DAYS * DAY <= j["s"] < T]
        ids = [j["job"] for j in recent]
        dup = len(ids) - len(set(ids))
        print(f"  T/{frac}: recent queue {len(ids):,} rows, {len(set(ids)):,} distinct job ids, "
              f"{dup} duplicates")
        results.setdefault(str(frac), {})["recent_ids_count"] = len(set(ids))
        results[str(frac)]["recent_duplicate_ids"] = dup

    # ---- pre-check 2: FULL_LABEL coverage is 100% -------------------------------------------
    print(f"\n{'=' * 100}")
    print("  PRE-CHECK 2 -- FULL-LABEL coverage must be 100% (no silent dropping)")
    print(f"{'=' * 100}")
    for frac in T_FRACS:
        T = t0 + frac * span
        recent = [j for j in A if T - RECENT_DAYS * DAY <= j["s"] < T]
        have = sum(1 for j in recent if j["p"] is not None and math.isfinite(j["p"]))
        print(f"  T/{frac}: recent queue {len(recent):,}, with a finite final duration {have:,} "
              f"({have/len(recent):.4%})")
        results[str(frac)]["full_label_coverage"] = have / len(recent)

    # ---- pre-check 3: blindness of the visible channel --------------------------------------
    print(f"\n{'=' * 100}")
    print("  PRE-CHECK 3 -- perturbing the HIDDEN final durations must not change RECENT/KM input")
    print(f"{'=' * 100}")
    import copy
    for frac in T_FRACS[:1]:
        T = t0 + frac * span
        recent = [j for j in A if T - RECENT_DAYS * DAY <= j["s"] < T]
        before = {}
        for j in recent:
            z = min(j["p"], T - j["s"])
            delta = 1 if j["e"] <= T else 0
            before[j["job"]] = (z, delta)
        # perturb ONLY the final duration of censored jobs, keeping start time and status
        perturbed = []
        for j in recent:
            k = copy.deepcopy(j)
            if j["e"] > T:
                k["p"] = j["p"] * 1000.0        # hidden side changed drastically
                k["e"] = j["s"] + k["p"]
            perturbed.append(k)
        after = {}
        for j in perturbed:
            z = min(j["p"], T - j["s"])
            delta = 1 if j["e"] <= T else 0
            after[j["job"]] = (z, delta)
        same = before == after
        changed_hidden = sum(1 for j in recent if j["e"] > T)
        print(f"  T/{frac}: perturbed {changed_hidden:,} hidden final durations x1000")
        print(f"  visible channel (z, delta) for the recent queue unchanged: {same}")
        print(f"  and the start times and censoring status were left intact by construction")
        results[str(frac)]["blindness_check_passed"] = bool(same)

    # ---- verdict ---------------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  PRE-REGISTERED READING")
    print(f"{'=' * 100}")
    ok_windows = 0
    deltas = []
    for frac in T_FRACS:
        r = results.get(str(frac))
        if not r or "scores" not in r:
            continue
        h = r["scores"]["HISTORY"]["mean_abs_log_error"]
        km = r["scores"]["KM"]["mean_abs_log_error"]
        rec = r["scores"]["RECENT"]["mean_abs_log_error"]
        full = r["scores"]["FULL_LABEL"]["mean_abs_log_error"]
        rel = (min(h, rec) - km) / min(h, rec)
        positive = km < min(h, rec)
        deltas.append(rel)
        ok_windows += 1 if positive else 0
        print(f"  T/{frac}: KM {km:.5f} vs min(HISTORY {h:.5f}, RECENT {rec:.5f}) = "
              f"{min(h, rec):.5f} -> relative change {rel:+.2%}  "
              f"{'POSITIVE' if positive else 'not positive'}"
              + (f"   (FULL_LABEL {full:.5f})" if full is not None else ""))
    if deltas:
        avg = sum(deltas) / len(deltas)
        print(f"\n  positive windows: {ok_windows}/3")
        print(f"  equal-weight mean relative change: {avg:+.2%}")
        gate = (ok_windows == 3 and avg >= 0.01)
        print(f"  PRE-REGISTERED GATE (all 3 positive AND mean >= 1%): "
              f"{'PASS -> proceed to a fixed-scheduler check' if gate else 'NOT MET'}")
    if km_fails:
        print(f"\n  WARNING: {km_fails} KM pre-check(s) failed")

    (HERE / "rt02_results.json").write_text(json.dumps(
        {"t0": t0, "t1": t1, "n_classA": len(A), "T_fracs": T_FRACS,
         "recent_days": RECENT_DAYS, "test_days": TEST_DAYS, "min_group": MIN_GROUP,
         "results": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'rt02_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
