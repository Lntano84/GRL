"""Q02: why does the retry branch almost never fire?

MF-RETRY took 1 retry action in 239 exploration actions.  A retry is only reachable when a class
timed out and its revealed bound still sits below the query's current best.  This script counts, at
every decision point, how many such classes exist and where they rank under the arm's own scoring.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q02", HERE / "q02_experiment.py")
q02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q02)


class Traced(q02.Replay):
    """Same loop, but records the retry-candidate count and rank at each decision."""

    def run(self, arm):
        self.decisions = []
        self.initialise()
        while self.remaining > 1e-12:
            allow_retry = (arm == "MF-RETRY")
            cands = self.feasible(allow_retry)
            if not cands:
                break
            n_retry_feasible = sum(1 for (q, ci) in cands
                                   if self.bound.get((q, ci), float("-inf")) > float("-inf"))
            pred = None
            chosen_was_retry = False
            if arm in ("RANDOM-BEST", "GREEDY-BEST"):
                q, ci = cands[0]
            else:
                pred = self.als()
                good, bad = [], []
                for (q, ci) in cands:
                    p = float(pred[q, self.classes[q][ci][0]])
                    b = self.b_of(q)
                    if not math.isfinite(p) or p <= 0:
                        bad.append((q, ci))
                        continue
                    if arm == "MF-BEST":
                        cost = max(1e-9, min(b, p))
                    else:
                        cost = max(1e-9, self.cap_for(q, ci, b, x_hat=p))
                    is_retry = self.bound.get((q, ci), float("-inf")) > float("-inf")
                    good.append(((b - p) / cost, q, ci, is_retry))
                good.sort(key=lambda t: (-t[0], t[1], t[2]))
                retry_ranks = [i for i, g in enumerate(good) if g[3]]
                self.decisions.append({
                    "n_cands": len(cands), "n_retry_feasible": n_retry_feasible,
                    "n_scored": len(good), "n_bad": len(bad),
                    "retry_ranks": retry_ranks[:5],
                    "top_is_retry": bool(good and good[0][3]),
                })
                if good:
                    q, ci = good[0][1], good[0][2]
                    chosen_was_retry = good[0][3]
                elif bad:
                    q, ci = bad[0]
                else:
                    break
            b = self.b_of(q)
            x_hat = None
            if pred is not None:
                p = float(pred[q, self.classes[q][ci][0]])
                x_hat = p if (math.isfinite(p) and p > 0) else None
            cap = self.cap_for(q, ci, b, x_hat)
            if cap is None or cap <= 0:
                break
            self.execute(q, ci, cap)
        return self


def main() -> int:
    W, labels, classes = q02.load()
    D = float(W[:, 0].sum())
    print("=" * 100)
    print("  Q02 -- WHY THE RETRY BRANCH DOES NOT FIRE")
    print("=" * 100)

    out = {}
    for arm in ("MF-BEST", "MF-RETRY"):
        B = D
        tot_dec = tot_retry_feas = tot_retry_present = 0
        top_retry = 0
        for seed in q02.SEEDS:
            r = Traced(W, classes, B, seed).run(arm)
            ds = r.decisions
            tr = [d for d in ds if d["n_retry_feasible"] > 0]
            tot_dec += len(ds)
            tot_retry_feas += sum(d["n_retry_feasible"] for d in ds)
            tot_retry_present += len(tr)
            top_retry += sum(1 for d in ds if d["top_is_retry"])
            print(f"\n  {arm} seed {seed}: {len(ds)} decisions, "
                  f"{len(tr)} with at least one retry-feasible class, "
                  f"{sum(1 for d in ds if d['top_is_retry'])} where the top choice was a retry")
            for i, d in enumerate(ds[:6]):
                print(f"      decision {i}: cands={d['n_cands']:>3} retry_feasible="
                      f"{d['n_retry_feasible']:>2} scored={d['n_scored']:>3} "
                      f"retry_ranks={d['retry_ranks']}")
            if len(ds) > 6:
                print(f"      ... {len(ds)-6} more decisions")
            out[f"{arm}_{seed}"] = ds
        print(f"\n  {arm} TOTAL: {tot_dec} decisions, {tot_retry_present} with a retry-feasible "
              f"class, {tot_retry_feas} retry-feasible (class, decision) pairs, "
              f"{top_retry} topped by a retry")

    # how often does a timeout leave a bound below the best?
    print("\n" + "=" * 100)
    print("  ROOT CAUSE")
    print("=" * 100)
    r = Traced(W, classes, D, 0).run("MF-RETRY")
    n_cancel = sum(1 for a in r.actions if a["kind"] == "cancelled" and a["phase"] == "explore")
    n_outside = 0
    for a in r.actions:
        if a["phase"] != "explore" or a["kind"] != "cancelled":
            continue
        key = (a["q"], a["ci"])
        L = r.bound.get(key, float("-inf"))
        b = r.b_of(a["q"])
        if b is not None and L >= b:
            n_outside += 1
    print(f"  at B = D, seed 0: {n_cancel} exploratory timeouts")
    print(f"    of these, {n_cancel - n_outside} left a bound still below the query best "
          f"(retry-feasible), {n_outside} reached the best (legally pruned)")
    print()
    print("  A timeout cannot produce a bound above the cap, and the cap is at most b_q.  So a")
    print("  timed-out class keeps a bound below b_q only when the cap was strictly below b_q.")
    print("  The frozen cap rules are")
    print("      fresh  cap = min(b_q, 15 x_hat)")
    print("      retry  cap = min(b_q, 2 L)")
    print("  so the retry branch is reachable only for those fresh candidates whose cap fell below")
    print("  b_q and that then timed out.  In this replay that is a very small set, which is why")
    print("  MF-RETRY and MF-BEST agree on 4 of 6 decision sequences.")

    (HERE / "q02_retry_why.json").write_text(json.dumps(out, indent=2)[:2000000], encoding="utf-8")
    print(f"\n  wrote {HERE / 'q02_retry_why.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
