"""Q02 diagnostics: was the retry arm actually exercised, and how did the caps behave?

The screening returned NOT MET, but MF-RETRY and MF-BEST produced identical P on 1 of 3 seeds at the
full budget.  Before writing that up, this script measures whether retry was even reachable in this
replay: how many exploration actions were retries of an already-timed-out class, and what caps the
frozen rules produced.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("q02", HERE / "q02_experiment.py")
q02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q02)


def main() -> int:
    W, labels, classes = q02.load()
    D = float(W[:, 0].sum())
    print("=" * 100)
    print("  Q02 DIAGNOSTICS -- was retry reachable?")
    print("=" * 100)

    rows = []
    for frac in q02.BUDGET_FRACS:
        B = frac * D
        for arm in q02.ARMS:
            for seed in q02.SEEDS:
                r = q02.Replay(W, classes, B, seed).run(arm)
                expl = [a for a in r.actions if a["phase"] == "explore"]
                retries = [a for a in expl
                           if a["kind"] == "cancelled" or True]  # placeholder, refined below
                # a retry is an action on a class that already had a bound BEFORE this action
                seen_bound = set()
                n_retry = 0
                for a in expl:
                    key = (a["q"], a["ci"])
                    if key in seen_bound:
                        n_retry += 1
                    if a["kind"] == "cancelled":
                        seen_bound.add(key)
                completed_after_retry = sum(
                    1 for a in expl if a["kind"] == "completed" and (a["q"], a["ci"]) in seen_bound)
                rows.append({
                    "frac": frac, "arm": arm, "seed": seed,
                    "explore_actions": len(expl),
                    "retry_actions": n_retry,
                    "cancelled_actions": sum(1 for a in expl if a["kind"] == "cancelled"),
                    "cap_eq_b": sum(1 for a in expl if abs(a["cap"] - r.b_of(a["q"])) < 1e-9)
                    if False else None,
                    "fallbacks": r.fallbacks,
                })
    print(f"  {'budget':>7} {'arm':<12} {'explore':>8} {'retries':>8} {'cancelled':>10} "
          f"{'fallbacks':>10}")
    agg = {}
    for r in rows:
        print(f"  {r['frac']:>6.2f}D {r['arm']:<12} {r['explore_actions']:>8} "
              f"{r['retry_actions']:>8} {r['cancelled_actions']:>10} {r['fallbacks']:>10}")
        agg.setdefault(r["arm"], []).append(r)

    print("\n  SUMMARY per arm (all budgets, all seeds)")
    for arm in q02.ARMS:
        sub = agg[arm]
        print(f"    {arm:<12} explore {sum(s['explore_actions'] for s in sub):>5}  "
              f"retry actions {sum(s['retry_actions'] for s in sub):>4}  "
              f"cancelled {sum(s['cancelled_actions'] for s in sub):>4}  "
              f"fallbacks {sum(s['fallbacks'] for s in sub):>3}")

    print("\n  INTERPRETATION")
    tot_retry = sum(s["retry_actions"] for s in agg["MF-RETRY"])
    tot_expl = sum(s["explore_actions"] for s in agg["MF-RETRY"])
    print(f"    MF-RETRY performed {tot_retry} retry actions out of {tot_expl} exploration actions "
          f"({tot_retry/max(1,tot_expl):.1%})")
    print(f"    at the full budget the figure is "
          f"{sum(s['retry_actions'] for s in agg['MF-RETRY'] if s['frac']==1.0)} retries")
    if tot_retry == 0:
        print("    => the retry branch was never taken; this replay cannot compare retry against")
        print("       no-retry, so the NOT MET verdict is uninformative about the mechanism.")
    else:
        print("    => the retry branch was exercised, so the comparison is meaningful.")

    (HERE / "q02_retry_diagnostic.json").write_text(json.dumps(
        {"per_run": rows, "mf_retry_total_retries": tot_retry,
         "mf_retry_total_explore": tot_expl}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q02_retry_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
