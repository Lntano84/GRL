"""Step 3: does the incompleteness actually bias a completed-only runtime predictor?

Three estimators for the same quantity (a group's job run time), evaluated on the same records:

  A  COMPLETED-ONLY   : mean/median of durations of jobs finished by the update point T.
  B  RUNNING-AS-DATA  : treats each unfinished job's attained service as if it were a full duration.
  C  CENSOR-AWARE     : unfinished jobs enter only as lower bounds.  Reported as a Kaplan-Meier
                        median plus the "still running beyond the completed p90" share, which is the
                        quantity a scheduler can act on without extrapolating.
  D  ORACLE (NOT DEPLOYABLE) : uses the FINAL duration of jobs that were unfinished at T.  This is
                        the diagnostic control that says how much of the gap is recoverable at all.

The grouping is by workload tag (job type) and by framework.  The output that matters is whether the
ORDER of groups changes between A and C, because only an order change can alter a scheduling decision.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 9)
T_FRACS = (0.4, 0.6, 0.8)
MIN_GROUP = 200          # groups smaller than this are not reported


def km_median(durations, censored_at):
    """Kaplan-Meier median of right-censored data.

    ``durations`` is the observed time for every subject (event time if uncensored, censoring time if
    censored); ``censored_at`` is the matching boolean.  Returns ``None`` if the curve never reaches
    0.5, which is itself the informative outcome ("more than half are still running").
    """
    pairs = sorted(zip(durations, censored_at))
    n = len(pairs)
    if n == 0:
        return None
    at_risk = n
    surv = 1.0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        d = c = 0
        while i < len(pairs) and pairs[i][0] == t:
            if pairs[i][1]:
                c += 1
            else:
                d += 1
            i += 1
        if d:
            surv *= (1.0 - d / at_risk)
            if surv <= 0.5:
                return t
        at_risk -= (d + c)
        if at_risk <= 0:
            break
    return None


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = float(r["task_start"]) if r["task_start"] else None
            e = float(r["task_end"]) if r["task_end"] else None
            jobs.append({"status": r["status"], "tag": r["tag_id"], "fw": r["framework"],
                         "s": s, "e": e})
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    span = t1 - t0
    print("=" * 108)
    print("  STEP 3 -- completed-only vs censor-aware vs oracle, by group")
    print("=" * 108)

    results = {}
    for frac in T_FRACS:
        T = t0 + frac * span
        arrived = [j for j in launched if j["s"] <= T]
        groups = defaultdict(list)
        for j in arrived:
            groups[("tag", j["tag"])].append(j)
            groups[("framework", j["fw"])].append(j)

        rows = []
        for (kind, gname), members in groups.items():
            if len(members) < MIN_GROUP:
                continue
            complete = [j for j in members if j["e"] is not None and j["e"] <= T]
            unfinished = [j for j in members if j["e"] is None or j["e"] > T]
            if len(complete) < 20:
                continue
            d_comp = [j["e"] - j["s"] for j in complete]
            d_comp = [d for d in d_comp if d > 0]
            if not d_comp:
                continue
            att = [T - j["s"] for j in unfinished]
            # oracle: final duration of the jobs that were unfinished at T
            orc = [(j["e"] - j["s"]) for j in unfinished if j["e"] is not None]
            orc = [d for d in orc if d and d > 0]

            A_mean = statistics.fmean(d_comp)
            A_p50 = statistics.median(d_comp)
            B_mean = statistics.fmean(att) if att else None
            # C: KM over all arrived subjects; censoring time = T - s for unfinished
            dur = d_comp + att
            cens = [False] * len(d_comp) + [True] * len(att)
            C_med = km_median(dur, cens)
            # action-oriented scalar: share of the group already past the completed p90
            p90c = sorted(d_comp)[int(0.9 * len(d_comp)) - 1] if len(d_comp) >= 10 else None
            share_past = (sum(1 for a in att if p90c and a > p90c) / len(att)) if att else 0.0
            rows.append({
                "kind": kind, "group": gname, "n_arrived": len(members),
                "n_complete": len(d_comp), "n_unfinished": len(att),
                "A_mean": A_mean, "A_p50": A_p50,
                "B_mean": B_mean,
                "C_km_median": C_med,
                "oracle_p50": statistics.median(orc) if orc else None,
                "share_unfinished_past_p90": share_past,
            })
        results[frac] = rows

        print(f"\n{'=' * 108}")
        print(f"  T/span = {frac}  (day {frac*span/86400:.1f});  "
              f"arrived {len(arrived):,};  groups>= {MIN_GROUP} arrived: {len(rows)}")
        print(f"{'=' * 108}")

        for kind in ("framework", "tag"):
            sub = [r for r in rows if r["kind"] == kind]
            if not sub:
                continue
            sub.sort(key=lambda r: -r["A_mean"])
            top = sub[:8] if kind == "framework" else sub[:10]
            print(f"\n  --- by {kind}: top by COMPLETED-ONLY mean (A) ---")
            print(f"  {'group':<34} {'arrived':>8} {'comp':>7} {'unfin':>7} "
                  f"{'A mean(s)':>11} {'A p50':>9} {'B mean':>13} {'C KM med':>10} "
                  f"{'oracle p50':>11} {'>p90':>7}")
            for r in top:
                g = r["group"][:32]
                cm = f"{r['C_km_median']:,.0f}" if r["C_km_median"] is not None else "never<0.5"
                om = f"{r['oracle_p50']:,.0f}" if r["oracle_p50"] is not None else "-"
                bm = f"{r['B_mean']:,.0f}" if r["B_mean"] is not None else "-"
                print(f"  {g:<34} {r['n_arrived']:>8,} {r['n_complete']:>7,} "
                      f"{r['n_unfinished']:>7,} {r['A_mean']:>11,.0f} {r['A_p50']:>9,.0f} "
                      f"{bm:>13} {cm:>10} {om:>11} {r['share_unfinished_past_p90']:>6.1%}")

        # ---- the question that decides everything: does the ORDER change? -------------------
        print(f"\n  --- ORDER COMPARISON at T/span = {frac} ---")
        for kind in ("framework", "tag"):
            sub = [r for r in rows if r["kind"] == kind]
            if len(sub) < 3:
                continue
            # A ranking uses completed-only mean; KM ranking puts "never reaches 0.5" last
            a_order = [r["group"] for r in sorted(sub, key=lambda r: -r["A_mean"])]
            c_order = [r["group"] for r in sorted(
                sub, key=lambda r: (r["C_km_median"] is None,
                                    -(r["C_km_median"] or 0)))]
            # pair concordance
            idx_c = {g: i for i, g in enumerate(c_order)}
            conc = disc = 0
            for i in range(len(a_order)):
                for j in range(i + 1, len(a_order)):
                    if idx_c[a_order[i]] < idx_c[a_order[j]]:
                        conc += 1
                    else:
                        disc += 1
            tot = conc + disc
            tau = (conc - disc) / tot if tot else float("nan")
            # how many pairs are out of order, and does it move any group across the midpoint?
            moved = sum(1 for g, i in ((g, i) for i, g in enumerate(a_order))
                        if abs(i - idx_c[g]) > 0)
            top_moved = (a_order[0] != c_order[0])
            print(f"    {kind:<10} n={len(sub):<4} Kendall tau={tau:+.3f}  "
                  f"rank positions changed for {moved}/{len(sub)} groups  "
                  f"top-1 changed: {top_moved}")
            if top_moved:
                print(f"      A top-1 = {a_order[0]}   C top-1 = {c_order[0]}")
            kend = max(5, len(sub) // 4)
            print(f"      A top-{kend}: {[g[:12] for g in a_order[:kend]]}")
            print(f"      C top-{kend}: {[g[:12] for g in c_order[:kend]]}")

    (HERE / "s3_estimators.json").write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 's3_estimators.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
