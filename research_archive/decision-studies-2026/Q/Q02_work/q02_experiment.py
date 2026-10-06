"""Q02 -- four-arm cost-quality screening on JOB, with a budget-controlled search loop.

Frozen configuration
--------------------
  workload      JOB (113 queries x 49 hints); one hint equivalence class = one execution object,
                one observation propagates to the whole class and is charged once
  init          ONLY column 0 and its equivalence class.  The unexplained init_mask is not used.
  budget B      0.25 D, 0.5 D, D   with D = sum_q W[q, 0]
  seeds         0, 1, 2            -> 4 arms x 3 budgets x 3 seeds = 36 replays
  arms          RANDOM-BEST, GREEDY-BEST, MF-BEST, MF-RETRY
  predictor     censored_als(rank=5, lambda=0.2, iters=50) -- the existing implementation, no tuning
  caps          fresh : cap = min(b_q, 15 * x_hat)
                retry : cap = min(b_q, 2 * L),  L = strongest known lower bound
  exclusions    completed candidates, and candidates with L >= b_q (legal prune)
  fallback      a non-positive / non-finite prediction may not produce a zero-cost action; the arm
                draws a uniformly random feasible candidate instead and the count is logged
  dispatch      remaining budget is checked before every action; the cap never exceeds the remaining
                budget; true runtimes are never consulted to decide whether a run "would fit"

Budget accounting
-----------------
Simulated execution seconds are the currency.  Policy computation is wall-clock and is reported
alongside rather than added to it, because adding wall-clock seconds to simulated seconds would mix
units.  This is stated rather than hidden.
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


def load():
    with MATRIX.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    W = np.array([[float(x) for x in r[1:]] for r in rows[1:] if r], dtype=float)
    labels = [r[0] for r in rows[1:] if r]
    eq = json.loads(EQUIV.read_text(encoding="utf-8"))
    classes = {}
    for qi, lab in enumerate(labels):
        classes[qi] = [sorted(s) for s in sorted(eq["classes_per_query"][lab], key=min)]
    for qi, cs in classes.items():
        assert len({h for c in cs for h in c}) == W.shape[1], qi
    return W, labels, classes


class Replay:
    """Budget-controlled loop.  The policy side sees only feedback, never the matrix."""

    def __init__(self, W, classes, B, seed):
        self.W, self.classes, self.B = W, classes, B
        self.rng = np.random.default_rng(seed)
        self.n, self.m = W.shape
        self.measured = {}          # (q,h) -> revealed completed runtime
        self.bound = {}             # (q,ci) -> strongest revealed lower bound
        self.completed = set()      # (q,ci)
        self.spent = 0.0            # exploration spend, charged against B
        self.init_cost = 0.0        # initial measurement cost, a separate and identical line item
        self.policy_seconds = 0.0
        self.fallbacks = 0
        self.actions = []

    # ---- revealed state ----------------------------------------------------------------------
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

    def initialise(self):
        """Observe column 0 (and its class) for every query.

        IDENTICAL for every arm and charged as a separate line item ``init_cost``, NOT against the
        exploration budget B.  At B = 0.25 D the column-0 baseline alone costs exactly D, so charging
        it to B would leave nothing to explore and make all four arms degenerate to the same run.
        Aggregate simulated spend is therefore init_cost + B; the report states this.
        """
        for q in range(self.n):
            ci = self.class_of(q, 0)
            self._execute(q, ci, float(self.W[q, 0]) + 1.0, charge_init=True)

    def execute(self, q, ci, cap):
        return self._execute(q, ci, cap, charge_init=False)

    def _execute(self, q, ci, cap, charge_init):
        """One execution of one class at ``cap``.  The truth only produces the feedback."""
        members = self.classes[q][ci]
        if not charge_init:
            cap = min(cap, self.remaining)
        if cap <= 0:
            return None
        x = float(self.W[q, min(members)])
        if x < cap:
            kind, val, charged = "completed", x, x
            self.completed.add((q, ci))
            for h in members:
                self.measured[(q, h)] = x
        else:
            kind, val, charged = "cancelled", cap, cap
            self.bound[(q, ci)] = max(self.bound.get((q, ci), float("-inf")), cap)
        if charge_init:
            self.init_cost += charged
        else:
            self.spent += charged
        self.actions.append({"q": q, "ci": ci, "members": members, "cap": cap, "kind": kind,
                             "value_or_bound": val, "charged": charged,
                             "phase": "init" if charge_init else "explore",
                             "cumulative": self.spent})

    # ---- candidates --------------------------------------------------------------------------
    def feasible(self, allow_retry):
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
                cap = self.cap_for(q, ci, b, x_hat=None)
                if cap is not None and cap > 0:
                    out.append((q, ci))
        return out

    def cap_for(self, q, ci, b, x_hat):
        L = self.bound.get((q, ci), float("-inf"))
        if L > float("-inf"):
            cap = min(b, BETA_RETRY * L)                   # retry: doubling
        elif x_hat is not None and math.isfinite(x_hat) and x_hat > 0:
            cap = min(b, ALPHA_NEW * x_hat)                # fresh: prediction-driven
        else:
            cap = b                                        # no usable prediction -> run to b
        return min(cap, self.remaining)

    # ---- predictor ---------------------------------------------------------------------------
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
        self.policy_seconds += time.perf_counter() - t0
        return np.where(mask == 1, np.inf, pred)

    # ---- the loop ----------------------------------------------------------------------------
    def run(self, arm):
        self.initialise()
        while self.remaining > 1e-12:
            allow_retry = (arm == "MF-RETRY")
            cands = self.feasible(allow_retry)
            if not cands:
                break
            pred = None
            if arm in ("RANDOM-BEST", "GREEDY-BEST"):
                if arm == "RANDOM-BEST":
                    q, ci = cands[int(self.rng.integers(0, len(cands)))]
                else:
                    bq = {q: self.b_of(q) for q, _ in cands}
                    mx = max(bq.values())
                    top = [c for c in cands if bq[c[0]] == mx]
                    q, ci = top[int(self.rng.integers(0, len(top)))]
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
                    good.append(((b - p) / cost, q, ci))
                if bad:
                    self.fallbacks += 1
                if good:
                    good.sort(key=lambda t: (-t[0], t[1], t[2]))
                    q, ci = good[0][1], good[0][2]
                elif bad:
                    q, ci = bad[int(self.rng.integers(0, len(bad)))]
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
    W, labels, classes = load()
    D = float(W[:, 0].sum())
    print("=" * 104)
    print("  Q02 -- four-arm cost-quality screening on JOB")
    print("=" * 104)
    print(f"  queries {W.shape[0]}  hints {W.shape[1]}  classes/query "
          f"{min(len(c) for c in classes.values())}-{max(len(c) for c in classes.values())}")
    print(f"  D = sum_q W[q,0] = {D:,.4f}   budgets "
          f"{[f'{f:.2f}D={f*D:,.2f}' for f in BUDGET_FRACS]}")
    print(f"  seeds {SEEDS}; arms {ARMS}; replays {len(ARMS)*len(BUDGET_FRACS)*len(SEEDS)}")
    print(f"  predictor censored_als(rank={RANK}, lambda={LAMBDA}, iters={ITERS}); "
          f"caps new=min(b,{ALPHA_NEW:.0f}x_hat) retry=min(b,{BETA_RETRY:.0f}L)")
    print(f"  NOTE: the column-0 baseline costs exactly D = {D:,.4f} and is charged as a separate,")
    print(f"        identical line item (init_cost); the budget B applies to exploration on top of it.")

    results = []
    for frac in BUDGET_FRACS:
        B = frac * D
        print(f"\n  --- budget {frac:.2f}D = {B:,.4f} " + "-" * 60)
        for arm in ARMS:
            for seed in SEEDS:
                t0 = time.perf_counter()
                r = Replay(W, classes, B, seed).run(arm)
                wall = time.perf_counter() - t0
                got = [r.b_of(q) for q in range(r.n)]
                P = sum(v for v in got if v is not None)
                results.append({
                    "budget_frac": frac, "B": B, "arm": arm, "seed": seed, "P": P,
                    "P_over_D": P / D, "exec_seconds": r.spent, "init_cost": r.init_cost,
                    "actions": len(r.actions), "explore_actions":
                        sum(1 for a in r.actions if a["phase"] == "explore"),
                    "wall_seconds": wall, "policy_seconds": r.policy_seconds,
                    "fallbacks": r.fallbacks,
                    "missing_queries": sum(1 for v in got if v is None),
                })
                print(f"    {arm:<12} seed {seed}  P={P:>9.4f} ({P/D:>6.2%})  "
                      f"exec={r.spent:>8.4f}  acts={sum(1 for a in r.actions if a['phase']=='explore'):>4}  "
                      f"fb={r.fallbacks:>3}  wall={wall:>5.1f}s")

    # ---- main comparison ---------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("  MAIN COMPARISON at full budget B = D:  MF-RETRY vs MF-BEST")
    print("=" * 104)
    full = {(r["arm"], r["seed"]): r for r in results if r["budget_frac"] == 1.00}
    print(f"  {'seed':>4} {'MF-BEST':>11} {'MF-RETRY':>11} {'rel change':>12} "
          f"{'RANDOM-BEST':>13} {'GREEDY-BEST':>13}")
    rels = []
    for s in SEEDS:
        a, b = full[("MF-BEST", s)]["P"], full[("MF-RETRY", s)]["P"]
        rels.append((a - b) / a)
        print(f"  {s:>4} {a:>11.4f} {b:>11.4f} {rels[-1]:>11.2%} "
              f"{full[('RANDOM-BEST', s)]['P']:>13.4f} {full[('GREEDY-BEST', s)]['P']:>13.4f}")
    all_better = all(x > 0 for x in rels)
    mean_rel = statistics.fmean(rels)
    beats_cheap = all(full[("MF-RETRY", s)]["P"] < min(full[("RANDOM-BEST", s)]["P"],
                                                       full[("GREEDY-BEST", s)]["P"])
                      for s in SEEDS)
    gate = all_better and mean_rel >= 0.02 and beats_cheap
    print(f"\n  MF-RETRY better in all 3 seeds : {all_better}")
    print(f"  mean relative reduction        : {mean_rel:+.2%}")
    print(f"  beats BOTH cheap arms on every seed : {beats_cheap}")
    print(f"  PRE-REGISTERED GATE (3/3 better, mean >= 2%, beats both cheap arms): "
          f"{'PASS' if gate else 'NOT MET'}")

    print("\n  COST CURVE  P(B)/D  (mean over seeds)")
    print(f"  {'budget':>8} " + " ".join(f"{a:>13}" for a in ARMS))
    for frac in BUDGET_FRACS:
        cells = [f"{statistics.fmean([r['P_over_D'] for r in results if r['budget_frac'] == frac and r['arm'] == arm]):>13.4%}"
                 for arm in ARMS]
        print(f"  {frac:>7.2f}D " + " ".join(cells))

    print("\n  BUDGET ACCOUNTING  (exec seconds are the currency; policy time is separate)")
    print(f"  init_cost is identical across arms: "
          f"{results[0]['init_cost']:.4f} (a separate line item, not charged to B)")
    print(f"  {'budget':>8} {'arm':<12} {'mean exec':>11} {'budget':>11} {'over?':>6} "
          f"{'mean policy s':>14} {'mean wall s':>12} {'fallbacks':>10}")
    for frac in BUDGET_FRACS:
        for arm in ARMS:
            sub = [r for r in results if r["budget_frac"] == frac and r["arm"] == arm]
            over = sum(1 for r in sub if r["exec_seconds"] > sub[0]["B"] + 1e-9)
            print(f"  {frac:>7.2f}D {arm:<12} "
                  f"{statistics.fmean([r['exec_seconds'] for r in sub]):>11.4f} "
                  f"{sub[0]['B']:>11.4f} {str(over):>6} "
                  f"{statistics.fmean([r['policy_seconds'] for r in sub]):>14.2f} "
                  f"{statistics.fmean([r['wall_seconds'] for r in sub]):>12.2f} "
                  f"{sum(r['fallbacks'] for r in sub):>10}")

    (HERE / "q02_results.json").write_text(json.dumps({
        "D": D, "budget_fracs": list(BUDGET_FRACS), "seeds": list(SEEDS), "arms": list(ARMS),
        "predictor": {"name": "censored_als", "rank": RANK, "lambda": LAMBDA, "iters": ITERS},
        "caps": {"new": f"min(b_q, {ALPHA_NEW} * x_hat)", "retry": f"min(b_q, {BETA_RETRY} * L)"},
        "gate": {"all_seeds_better": all_better, "mean_relative_reduction": mean_rel,
                 "beats_both_cheap_arms": beats_cheap, "passed": bool(gate)},
        "rows": results}, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'q02_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
