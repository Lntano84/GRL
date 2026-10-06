"""Q01-R3: two corrections to the witnesses.  Runs before anything else in this round.

Correction A.  The privileged-stop witness used hand-filled scalars (``opt_time = 260`` with
``min_observed.sum() = 200``).  That combination is IMPOSSIBLE: because every already-observed plan is
contained in the full matrix,

    sum_q min_h W[q,h]  <=  sum_q b_q

so a full-matrix optimum of 260 cannot coexist with observed-best sum 200.  This script checks the
inequality on the real matrices and reproduces the auditor's valid two-world example instead.

Correction B.  The previous check 4 claimed a scenario in which the policy knew b = 10, but under the
frozen boundary rule (``x < cap`` completes) the run ``x = 10, cap = 10`` TIMES OUT, so the policy
never received that measurement.  This script shows the actual outcome and separates "budget
exhausted -> STOP" from "if budget remained, the next candidate would be ...".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = HERE / "limeqo_mirror" / "dataset"


# ------------------------------------------------------------------ the frozen boundary rule

def observe(x, cap):
    """x < cap completes; x >= cap times out.  Frozen convention, unchanged."""
    return ("completed", x, x) if x < cap else ("cancelled", cap, cap)


def legal_next(candidates, retired, bounds, best_of_query, cap_for):
    """Return (action, cap) or ("stop", None).  Prunes only on revealed information."""
    for (q, h) in candidates:
        if (q, h) in retired:
            continue
        if bounds.get((q, h), float("-inf")) >= best_of_query(q):
            continue                       # legal prune: the bound already rules out an improvement
        return (q, h), cap_for(q, h)
    return "stop", None


# ------------------------------------------------------------------ correction A

def inequality_check():
    print("=" * 100)
    print("  CORRECTION A.1 -- the inequality sum_q min_h W[q,h] <= sum_q b_q")
    print("=" * 100)
    out = {}
    for name in ("ceb", "job", "stack", "dsb"):
        p = DATASET / f"{name}-matrix.csv"
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        W = np.array([[float(x) for x in r[1:]] for r in rows[1:] if r], dtype=float)
        opt = float(W.min(axis=1).sum())
        b0 = float(W[:, 0].sum())
        ok = opt <= b0 + 1e-9
        print(f"  {name:<6} sum of full-matrix row minima = {opt:>14,.2f}   "
              f"sum of default plan b = {b0:>14,.2f}   inequality holds: {ok}")
        out[name] = {"full_optimum": opt, "default_sum": b0, "holds": ok}
    print()
    print("  Any observed best b_q is the minimum over a SUBSET of that row's entries, so it cannot")
    print("  be below the row minimum.  A witness with observed-best sum 200 therefore cannot have a")
    print("  full-matrix optimum of 260; that number was hand-filled and is withdrawn.")
    return out


def valid_stop_witness():
    print("\n" + "=" * 100)
    print("  CORRECTION A.2 -- the valid two-world encounter (auditor's example, re-derived here)")
    print("=" * 100)
    WA = np.array([[100.0, 50.0], [100.0, 50.0]])
    WB = np.array([[100.0, 99.0], [100.0, 99.0]])
    print("  W_A = [[100, 50], [100, 50]]        W_B = [[100, 99], [100, 99]]")
    print("  both worlds execute ONLY column 0, with cap = 101\n")
    res = {}
    for nm, W in (("A", WA), ("B", WB)):
        env_best = {q: float("inf") for q in range(W.shape[0])}
        obs = []
        # execute column 0 for every query at cap 101
        caps = []
        for q in range(W.shape[0]):
            x = float(W[q, 0])
            kind, val, charged = observe(x, 101.0)
            obs.append({"q": q, "h": 0, "x": x, "cap": 101.0, "kind": kind,
                        "value_or_bound": val, "charged": charged})
            if kind == "completed":
                env_best[q] = min(env_best[q], val)
        minobs_sum = sum(env_best.values())
        opt_time = float(W.min(axis=1).sum())
        choose_default = W[:, 1]     # a surviving candidate column
        # legal policy: b = 10 for query 0 (from its completed measurement), candidate (0,1)
        b = dict(env_best)
        nxt, cap = legal_next([(0, 1)], set(), {}, lambda q: b[q], lambda q, h: 2)
        cont = minobs_sum > opt_time + 20
        res[nm] = {"observations": obs, "min_observed_sum": minobs_sum,
                   "opt_time": opt_time, "continue": bool(cont),
                   "legal_next_action": nxt if nxt == "stop" else list(nxt),
                   "legal_next_cap": cap}
        print(f"  world {nm}: observed = {[o['value_or_bound'] for o in obs]}  "
              f"sum(min_observed) = {minobs_sum:.0f}  opt_time(from FULL matrix) = {opt_time:.0f}")
        print(f"           rule `{minobs_sum:.0f} > opt_time + 20` -> continue = {cont}")
        print(f"           legal policy next candidate = {nxt} at cap {cap}")
    same_hist = ([o["value_or_bound"] for o in res["A"]["observations"]]
                 == [o["value_or_bound"] for o in res["B"]["observations"]])
    stops_differ = res["A"]["continue"] != res["B"]["continue"]
    print()
    print(f"  identical revealed history in both worlds : {same_hist}")
    print(f"  privileged stop decision differs          : {stops_differ}")
    print(f"  legal policy next candidate identical     : "
          f"{res['A']['legal_next_action'] == res['B']['legal_next_action']}")
    print(f"  [{'PASS' if same_hist and stops_differ else 'FAIL'}] valid witness: same legal history, "
          f"different privileged stop decision")
    print()
    print("  This is the correct form of the demonstration.  The earlier 260/200 pair violated")
    print("  sum_q min_h W <= sum_q b and is withdrawn.")
    return res, {"identical_history": same_hist, "stop_differs": stops_differ,
                 "legal_next_identical":
                     res["A"]["legal_next_action"] == res["B"]["legal_next_action"]}


# ------------------------------------------------------------------ correction B

def rerun_previous_check4():
    print("\n" + "=" * 100)
    print("  CORRECTION B -- what the previous check 4 actually produced")
    print("=" * 100)
    plan = [(0, 0, 10.0), (0, 1, 2.0)]
    truth = {(0, 0): 10.0, (0, 1): 5.0}
    env_best = {0: 10.0}          # the environment was pre-seeded with 10 ...
    pol_mask, pol_bounds, pol_retired = {}, {}, set()
    spent = 0.0
    calls = []
    for (q, h, cap) in plan:
        x = truth[(q, h)]
        kind, val, charged = observe(x, cap)
        spent += charged
        calls.append({"q": q, "h": h, "cap": cap, "kind": kind, "value_or_bound": val,
                      "charged": charged})
        if kind == "completed":
            pol_mask[(q, h)] = val
            pol_retired.add((q, h))
        else:
            pol_bounds[(q, h)] = max(pol_bounds.get((q, h), float("-inf")), cap)
    best_of_q0 = min([v for (qq, _), v in pol_mask.items() if qq == 0], default=float("inf"))
    print(f"  plan executed: (0,0) at cap 10, then (0,1) at cap 2")
    for c in calls:
        print(f"    ({c['q']},{c['h']}) cap={c['cap']:<5} -> {c['kind']:<10} "
              f"value_or_bound={c['value_or_bound']} charged={c['charged']}")
    print()
    print(f"  the environment was seeded with best=10, but the POLICY received no completed")
    print(f"  measurement: x=10 at cap=10 satisfies x >= cap, so it TIMES OUT.")
    print(f"  policy best_get(0) = {best_of_q0}   (inf, i.e. still unknown)")
    print(f"  cumulative cost    = {spent}")
    print()
    print("  So the previous 'state identical' result was true, but NOT for the reason the report")
    print("  implied: the policy did not know b = 10 at all.  The scenario is withdrawn and replaced")
    print("  by the column-0-at-cap-101 construction above, which does deliver the measurement.")

    # STOP vs "if budget remained"
    print("\n  BUDGET DISCIPLINE")
    budget = 12.0
    remaining = budget - spent
    candidates = [(0, 1)]
    nxt, cap = legal_next(candidates, pol_retired, pol_bounds, lambda q: best_of_q0, lambda q, h: 2)
    print(f"  budget = {budget}, spent = {spent}, remaining = {remaining}")
    print(f"  budget exhausted -> action = STOP")
    print(f"  [report separately] if budget remained, the next candidate would be {nxt} at cap {cap}")
    print("  The two must not be conflated: 'STOP' is the actual action.")
    return {"calls": calls, "policy_best_get_0": best_of_q0, "spent": spent,
            "action_on_exhaustion": "STOP",
            "next_if_budget_remained": nxt if nxt == "stop" else list(nxt),
            "next_cap_if_budget_remained": cap}


def main() -> int:
    ineq = inequality_check()
    worldA, wit = valid_stop_witness()
    c4 = rerun_previous_check4()
    ok = all(v["holds"] for v in ineq.values()) and wit["identical_history"] \
        and wit["stop_differs"] and c4["spent"] == 12.0
    print("\n" + "=" * 100)
    print(f"  WITNESS CORRECTIONS: {'PASS' if ok else 'FAIL'}")
    print("=" * 100)
    (HERE / "q01r3_witness_corrections.json").write_text(json.dumps(
        {"inequality": ineq, "valid_stop_witness": worldA, "witness_checks": wit,
         "previous_check4_actual": c4, "all_pass": bool(ok)}, indent=2), encoding="utf-8")
    print(f"  wrote {HERE / 'q01r3_witness_corrections.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
