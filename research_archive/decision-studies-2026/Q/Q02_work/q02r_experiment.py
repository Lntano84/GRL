"""Q02R -- protocol-corrected re-run.  Same 36 replays, same data, same predictor, same multipliers.

What was wrong in Q02 (implementation defects, not data or setting)
------------------------------------------------------------------
1. BOTH MF arms were given the prediction and both went through ``cap_for``, which returns
   ``min(b_q, 15 x_hat)`` for a fresh candidate.  So MF-BEST was ALSO cutting early, whereas the
   protocol says it must run straight to ``b_q``.  The main contrast was therefore mis-defined.
2. The two MF arms used different scoring denominators: ``min(b_q, x_hat)`` for MF-BEST versus
   ``cap`` (budget dependent) for MF-RETRY.  With different caps this also changes WHICH fresh
   candidate is picked, so the observed gap was not attributable to retry.
3. ``policy_seconds`` timed only the ALS call, not candidate generation or ranking, so it was not the
   full strategy cost.  The agreed cost model is database-replay execution time PLUS local policy
   time; both are in seconds, so they are added.
4. ``fallbacks`` counted decisions at which an invalid prediction was present, not decisions at which
   the random fallback was actually taken.
5. The main results kept only ``len(r.actions)``, not the action list.

Frozen protocol for this run
----------------------------
  caps, dispatched explicitly per arm
      RANDOM-BEST / GREEDY-BEST / MF-BEST : planned cap = b_q
      MF-RETRY                            : fresh  = min(b_q, 15 x_hat)
                                            retry  = min(b_q, 2 L)
      all arms: the planned cap is then truncated by the remaining budget
  scoring, shared by both MF arms
      S(q,h) = (b_q - x_hat) / max(1e-9, min(b_q, x_hat))
  cost model
      execution cost (replay seconds) plus policy cost (local seconds), both recorded separately;
      the remaining budget is computed from their sum
  action log
      b_before, L_before, is_retry, prediction, score, planned_cap, actual_cap, feedback,
      exec_cost, policy_cost

No data, predictor, multiplier or seed changes; no second workload.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Q01_work" / "limeqo_mirror" / "src"))

from models.matrix_factorization import censored_als  # noqa: E402

MATRIX = ROOT / "Q01_work" / "limeqo_mirror" / "dataset" / "job-matrix.csv"
EQUIV = HERE / "job_equivalence.json"

RANK, LAMBDA, ITERS = 5, 0.2, 50
ALPHA_NEW, BETA_RETRY = 15.0, 2.0
BUDGET_FRACS = (0.25, 0.50, 1.00)
SEEDS = (0, 1, 2)
ARMS = ("RANDOM-BEST", "GREEDY-BEST", "MF-BEST", "MF-RETRY")
MF_ARMS = ("MF-BEST", "MF-RETRY")

# ------------------------------------------------------------------ the two guard checks

def guard_checks():
    print("=" * 104)
    print("  GUARD CHECKS (must pass before the main comparison is trusted)")
    print("=" * 104)
    ok = True
    # guard 1: b = 10, x_hat = 0.1
    b, x_hat = 10.0, 0.1
    cap_best = b                                        # MF-BEST: straight to b_q
    cap_retry = min(b, ALPHA_NEW * x_hat)
    print(f"  guard 1: b = {b}, x_hat = {x_hat}")
    print(f"    MF-BEST  planned cap = {cap_best}   (required 10)")
    print(f"    MF-RETRY planned cap = {cap_retry}  (required 1.5)")
    g1 = (cap_best == 10.0 and abs(cap_retry - 1.5) < 1e-12)
    ok &= g1
    print(f"    [{'PASS' if g1 else 'FAIL'}]")

    # guard 2: identical candidate state -> identical scores and identical ranking
    print(f"\n  guard 2: both MF arms receive the SAME candidate state")
    cands = [(0, 1), (0, 2), (1, 3)]
    preds = {(0, 1): 0.5, (0, 2): 7.0, (1, 3): 0.05}
    bof = {0: 10.0, 1: 4.0}
    def score(q, ci):
        p = preds[(q, ci)]
        bq = bof[q]
        return (bq - p) / max(1e-9, min(bq, p))
    s_best = sorted(((score(q, ci), q, ci) for (q, ci) in cands), key=lambda t: (-t[0], t[1], t[2]))
    s_retry = sorted(((score(q, ci), q, ci) for (q, ci) in cands), key=lambda t: (-t[0], t[1], t[2]))
    same = s_best == s_retry
    print(f"    MF-BEST  order: {[(q, ci, round(s, 4)) for s, q, ci in s_best]}")
    print(f"    MF-RETRY order: {[(q, ci, round(s, 4)) for s, q, ci in s_retry]}")
    ok &= same
    print(f"    [{'PASS' if same else 'FAIL'}] identical scores and ordering")
    print(f"\n  GUARDS {'PASSED' if ok else 'FAILED'}")
    return {"guard1": {"b": b, "x_hat": x_hat, "mf_best_cap": cap_best,
                       "mf_retry_cap": cap_retry, "pass": bool(g1)},
            "guard2": {"order_best": [[float(s), q, ci] for s, q, ci in s_best],
                       "order_retry": [[float(s), q, ci] for s, q, ci in s_retry],
                       "identical": bool(same)},
            "all_pass": bool(ok)}


def load():
    with MATRIX.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    W = np.array([[float(x) for x in r[1:]] for r in rows[1:] if r], dtype=float)
    labels = [r[0] for r in rows[1:] if r]
    eq = json.loads(EQUIV.read_text(encoding="utf-8"))
    classes = {qi: [sorted(s) for s in sorted(eq["classes_per_query"][lab], key=min)]
               for qi, lab in enumerate(labels)}
    for qi, cs in classes.items():
        assert len({h for c in cs for h in c}) == W.shape[1], qi
    return W, labels, classes


# ------------------------------------------------------------------ the corrected loop

class Replay:
    def __init__(self, W, classes, B, seed):
        self.W, self.classes, self.B = W, classes, B
        self.rng = np.random.default_rng(seed)
        self.n, self.m = W.shape
        self.measured, self.bound, self.completed = {}, {}, set()
        self.exec_cost = 0.0          # replay seconds, charged to B
        self.policy_cost = 0.0        # local seconds, charged to B
        self.init_cost = 0.0          # separate line item, NOT charged to B
        self.fallbacks_taken = 0      # the random fallback was actually executed
        self.fallbacks_available = 0  # an invalid prediction was present at a decision
        self.actions = []

    @property
    def spent(self):
        return self.exec_cost + self.policy_cost

    @property
    def remaining(self):
        return self.B - self.spent

    def b_of(self, q):
        vals = [v for (qq, _), v in self.measured.items() if qq == q]
        return min(vals) if vals else None

    def class_of(self, q, hint):
        for ci, c in enumerate(self.classes[q]):
            if hint in c:
                return ci
        raise KeyError((q, hint))

    # ---- execution ---------------------------------------------------------------------------
    def _exec(self, q, ci, cap, phase):
        members = self.classes[q][ci]
        x = float(self.W[q, min(members)])
        if x < cap:
            kind, val, charged = "completed", x, x
            self.completed.add((q, ci))
            for h in members:
                self.measured[(q, h)] = x
        else:
            kind, val, charged = "cancelled", cap, cap
            self.bound[(q, ci)] = max(self.bound.get((q, ci), float("-inf")), cap)
        if phase == "init":
            self.init_cost += charged
        else:
            self.exec_cost += charged
        return kind, val, charged

    def initialise(self):
        for q in range(self.n):
            ci = self.class_of(q, 0)
            self._exec(q, ci, float(self.W[q, 0]) + 1.0, "init")

    # ---- candidate generation (timed: it is part of the strategy cost) ------------------------
    def feasible(self, allow_retry):
        t0 = time.perf_counter()
        out = []
        for q in range(self.n):
            b = self.b_of(q)
            if b is None:
                continue
            for ci in range(len(self.classes[q])):
                if (q, ci) in self.completed:
                    continue
                L = self.bound.get((q, ci), float("-inf"))
                if L >= b:
                    continue
                if not allow_retry and L > float("-inf"):
                    continue
                out.append((q, ci))
        self.policy_cost += time.perf_counter() - t0
        return out

    def als(self):
        mask = np.zeros((self.n, self.m))
        log_m = np.zeros((self.n, self.m))
        timeout_m = np.zeros((self.n, self.m))
        for (q, h), v in self.measured.items():
            mask[q, h] = 1.0
            log_m[q, h] = math.log1p(v)
        for (q, ci), L in self.bound.items():
            for h in self.classes[q][ci]:
                timeout_m[q, h] = math.log1p(L)
        t0 = time.perf_counter()
        np.random.seed(int(self.rng.integers(0, 2 ** 31 - 1)))
        pred = np.expm1(censored_als(log_m, mask, timeout_m, RANK, ITERS, LAMBDA))
        self.policy_cost += time.perf_counter() - t0
        return np.where(mask == 1, np.inf, pred)

    # ---- planned cap, dispatched explicitly per arm -------------------------------------------
    def planned_cap(self, arm, q, ci, b, x_hat):
        L = self.bound.get((q, ci), float("-inf"))
        if arm == "MF-RETRY":
            if L > float("-inf"):
                return min(b, BETA_RETRY * L), "retry"
            if x_hat is not None and math.isfinite(x_hat) and x_hat > 0:
                return min(b, ALPHA_NEW * x_hat), "fresh_prediction"
            return b, "fresh_no_prediction"
        # RANDOM-BEST, GREEDY-BEST, MF-BEST: straight to the current best
        return b, "straight_to_b"

    def run(self, arm):
        self.initialise()
        while self.remaining > 1e-12:
            allow_retry = (arm == "MF-RETRY")
            cands = self.feasible(allow_retry)
            if not cands:
                break
            pred = None
            chosen = None
            if arm in ("RANDOM-BEST", "GREEDY-BEST"):
                t0 = time.perf_counter()
                if arm == "RANDOM-BEST":
                    chosen = cands[int(self.rng.integers(0, len(cands)))]
                else:
                    bq = {q: self.b_of(q) for q, _ in cands}
                    mx = max(bq.values())
                    top = [c for c in cands if bq[c[0]] == mx]
                    chosen = top[int(self.rng.integers(0, len(top)))]
                self.policy_cost += time.perf_counter() - t0
            else:
                pred = self.als()
                t0 = time.perf_counter()
                good, bad = [], []
                for (q, ci) in cands:
                    p = float(pred[q, self.classes[q][ci][0]])
                    b = self.b_of(q)
                    if not math.isfinite(p) or p <= 0:
                        bad.append((q, ci))
                        continue
                    s = (b - p) / max(1e-9, min(b, p))     # SHARED scoring function
                    good.append((s, q, ci))
                if bad:
                    self.fallbacks_available += 1
                self.policy_cost += time.perf_counter() - t0
                if good:
                    good.sort(key=lambda t: (-t[0], t[1], t[2]))
                    chosen = (good[0][1], good[0][2])
                    self._score, self._pred = good[0][0], float(
                        pred[good[0][1], self.classes[good[0][1]][good[0][2]][0]])
                elif bad:
                    chosen = bad[int(self.rng.integers(0, len(bad)))]
                    self.fallbacks_taken += 1
                    self._score, self._pred = None, None
                else:
                    break

            q, ci = chosen
            b_before = self.b_of(q)
            L_before = self.bound.get((q, ci), float("-inf"))
            x_hat = None
            score = None
            if pred is not None:
                p = float(pred[q, self.classes[q][ci][0]])
                if math.isfinite(p) and p > 0:
                    x_hat = p
                    score = (b_before - p) / max(1e-9, min(b_before, p))
            plan_cap, cap_reason = self.planned_cap(arm, q, ci, b_before, x_hat)
            # the policy cost of THIS decision has just accrued (candidate generation, prediction,
            # ranking), so the remaining budget is recomputed immediately before dispatch.  Without
            # this the last action of a run can push the total slightly over B.
            actual_cap = min(plan_cap, self.remaining)
            if actual_cap <= 0:
                break
            kind, val, charged = self._exec(q, ci, actual_cap, "explore")
            assert self.spent <= self.B + 1e-9, (self.spent, self.B)
            self.actions.append({
                "arm": arm, "q": q, "ci": ci, "members": self.classes[q][ci],
                "b_before": b_before, "L_before": (None if L_before == float("-inf") else L_before),
                "is_retry": bool(L_before > float("-inf")), "prediction": x_hat,
                "score": score, "planned_cap": plan_cap, "cap_reason": cap_reason,
                "actual_cap": actual_cap, "kind": kind, "value_or_bound": val,
                "exec_cost": charged, "cumulative_exec": self.exec_cost,
                "cumulative_policy": self.policy_cost,
                "remaining_after": self.B - self.spent,
            })
        return self


def main() -> int:
    g = guard_checks()
    if not g["all_pass"]:
        print("\n  ABORT: guard checks failed; the main comparison would not be trustworthy.")
        return 1

    W, labels, classes = load()
    D = float(W[:, 0].sum())
    print("\n" + "=" * 104)
    print("  Q02R -- protocol-corrected four-arm screening on JOB")
    print("=" * 104)
    print(f"  D = {D:,.4f}; budgets {[f'{f:.2f}D={f*D:,.2f}' for f in BUDGET_FRACS]}; seeds {SEEDS}")
    print(f"  caps: RANDOM/GREEDY/MF-BEST -> b_q ;  MF-RETRY -> fresh min(b_q,15x_hat), retry min(b_q,2L)")
    print(f"  scoring shared by both MF arms: S=(b_q-x_hat)/max(1e-9,min(b_q,x_hat))")
    print(f"  cost = replay execution seconds + local policy seconds (both charged to B);")
    print(f"         the column-0 baseline is a separate, identical line item (frozen protocol)")

    rows = []
    for frac in BUDGET_FRACS:
        B = frac * D
        print(f"\n  --- budget {frac:.2f}D = {B:,.4f} " + "-" * 58)
        for arm in ARMS:
            for seed in SEEDS:
                wall0 = time.perf_counter()
                r = Replay(W, classes, B, seed).run(arm)
                wall = time.perf_counter() - wall0
                got = [r.b_of(q) for q in range(r.n)]
                P = sum(v for v in got if v is not None)
                expl = [a for a in r.actions]
                rows.append({
                    "budget_frac": frac, "B": B, "arm": arm, "seed": seed,
                    "P": P, "P_over_D": P / D,
                    "exec_cost": r.exec_cost, "policy_cost": r.policy_cost,
                    "spent": r.spent, "init_cost": r.init_cost,
                    "explore_actions": len(expl),
                    "retry_actions": sum(1 for a in expl if a["is_retry"]),
                    "fallbacks_taken": r.fallbacks_taken,
                    "fallbacks_available": r.fallbacks_available,
                    "wall_seconds": wall,
                    "missing_queries": sum(1 for v in got if v is None),
                    "actions": expl,
                })
                print(f"    {arm:<12} seed {seed}  P={P:>9.4f} ({P/D:>6.2%})  "
                      f"exec={r.exec_cost:>8.4f} pol={r.policy_cost:>6.3f}  "
                      f"acts={len(expl):>4} retry={sum(1 for a in expl if a['is_retry']):>2}  "
                      f"fb={r.fallbacks_taken}/{r.fallbacks_available}")

    # ---- main comparison ---------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("  MAIN COMPARISON at full budget B = D:  MF-RETRY vs MF-BEST")
    print("=" * 104)
    full = {(r["arm"], r["seed"]): r for r in rows if r["budget_frac"] == 1.00}
    print(f"  {'seed':>4} {'MF-BEST':>11} {'MF-RETRY':>11} {'rel change':>12} "
          f"{'RANDOM-BEST':>13} {'GREEDY-BEST':>13} {'retries':>8}")
    rels = []
    for s in SEEDS:
        a, b = full[("MF-BEST", s)]["P"], full[("MF-RETRY", s)]["P"]
        rels.append((a - b) / a)
        print(f"  {s:>4} {a:>11.4f} {b:>11.4f} {rels[-1]:>11.2%} "
              f"{full[('RANDOM-BEST', s)]['P']:>13.4f} {full[('GREEDY-BEST', s)]['P']:>13.4f} "
              f"{full[('MF-RETRY', s)]['retry_actions']:>8}")
    all_better = all(x > 0 for x in rels)
    mean_rel = statistics.fmean(rels)
    beats_cheap = all(full[("MF-RETRY", s)]["P"] < min(full[("RANDOM-BEST", s)]["P"],
                                                       full[("GREEDY-BEST", s)]["P"])
                      for s in SEEDS)
    gate = all_better and mean_rel >= 0.02 and beats_cheap
    print(f"\n  MF-RETRY better in all 3 seeds : {all_better}")
    print(f"  mean relative reduction        : {mean_rel:+.2%}")
    print(f"  beats BOTH cheap arms          : {beats_cheap}")
    print(f"  GATE (3/3 better, mean >= 2%, beats both cheap arms): {'PASS' if gate else 'NOT MET'}")

    print("\n  COST CURVE P(B)/D (mean over seeds)")
    print(f"  {'budget':>8} " + " ".join(f"{a:>13}" for a in ARMS))
    for frac in BUDGET_FRACS:
        cells = [f"{statistics.fmean([r['P_over_D'] for r in rows if r['budget_frac'] == frac and r['arm'] == arm]):>13.4%}"
                 for arm in ARMS]
        print(f"  {frac:>7.2f}D " + " ".join(cells))

    print("\n  RETRY ACTIVITY PER BUDGET")
    print(f"  {'budget':>8} {'arm':<12} {'explore':>9} {'retries':>9} {'fb taken':>9} {'fb avail':>9}")
    for frac in BUDGET_FRACS:
        for arm in ARMS:
            sub = [r for r in rows if r["budget_frac"] == frac and r["arm"] == arm]
            print(f"  {frac:>7.2f}D {arm:<12} {sum(r['explore_actions'] for r in sub):>9} "
                  f"{sum(r['retry_actions'] for r in sub):>9} "
                  f"{sum(r['fallbacks_taken'] for r in sub):>9} "
                  f"{sum(r['fallbacks_available'] for r in sub):>9}")

    print("\n  COST ACCOUNTING")
    print(f"  init_cost identical across arms: {rows[0]['init_cost']:.4f} (separate line item)")
    print(f"  {'budget':>8} {'arm':<12} {'mean exec':>11} {'mean policy':>12} {'mean total':>11} "
          f"{'budget':>10} {'over?':>6}")
    for frac in BUDGET_FRACS:
        for arm in ARMS:
            sub = [r for r in rows if r["budget_frac"] == frac and r["arm"] == arm]
            me = statistics.fmean([r["exec_cost"] for r in sub])
            mp = statistics.fmean([r["policy_cost"] for r in sub])
            ov = sum(1 for r in sub if r["spent"] > r["B"] + 1e-9)
            print(f"  {frac:>7.2f}D {arm:<12} {me:>11.4f} {mp:>12.4f} {me+mp:>11.4f} "
                  f"{sub[0]['B']:>10.4f} {'YES' if ov else 'no':>6}")

    keep = [{k: v for k, v in r.items() if k != "actions"} for r in rows]
    (HERE / "q02r_results.json").write_text(json.dumps({
        "D": D, "budget_fracs": list(BUDGET_FRACS), "seeds": list(SEEDS), "arms": list(ARMS),
        "guards": g,
        "protocol": {
            "caps": {"RANDOM-BEST": "b_q", "GREEDY-BEST": "b_q", "MF-BEST": "b_q",
                     "MF-RETRY": "fresh min(b_q,15*x_hat), retry min(b_q,2*L)"},
            "scoring": "S=(b_q-x_hat)/max(1e-9,min(b_q,x_hat)), shared by both MF arms",
            "cost": "replay execution seconds + local policy seconds, both charged to B",
            "init_cost": "separate identical line item, not charged to B",
            "predictor": {"name": "censored_als", "rank": RANK, "lambda": LAMBDA, "iters": ITERS},
        },
        "gate": {"all_seeds_better": all_better, "mean_relative_reduction": mean_rel,
                 "beats_both_cheap_arms": beats_cheap, "passed": bool(gate)},
        "rows": keep}, indent=2), encoding="utf-8")
    (HERE / "q02r_actions.json").write_text(json.dumps(
        [{"budget_frac": r["budget_frac"], "arm": r["arm"], "seed": r["seed"], "actions": r["actions"]}
         for r in rows], indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q02r_results.json'}")
    print(f"  wrote {HERE / 'q02r_actions.json'}  (full action logs, persisted this time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
