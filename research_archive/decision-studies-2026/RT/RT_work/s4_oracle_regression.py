"""Step 4: the decisive test -- does the incompleteness cost the completed-only predictor any
accuracy that is actually RECOVERABLE by using the unfinished records?

Discipline:
  * fit on the window strictly before a split time T0; evaluate on jobs that START in [T0, T0+E];
  * a target job's own outcome is never used before it is observed;
  * OUTCOME-BLIND features only: workload tag, framework, user, requested GPU count and task count.
    No time-since-start, no attained service, no feature that encodes the outcome -- this is a
    deliberate oracle regression, and it is labelled as such;
  * the label is log1p(final duration).

Variants of the training set, all fitted identically:
  COMPLETED   : only jobs that had finished by T0 (what a completed-only pipeline uses)
  ALLFINAL    : every job that had STARTED before T0, using its final duration -- this is the
                NON-DEPLOYABLE oracle that says how much is recoverable at all
  CENSORED    : only jobs completed by T0 (the deployable set); the unmatched jobs enter solely as a
                dummy feature indicating whether their GROUP had many still-unfinished members
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

# ---- tiny ridge regression on standardised features (no third-party libraries) ---------------

def fit_ridge(X, y, lam=1.0):
    """Solve (X'X + lam I) w = X'y with plain Gaussian elimination."""
    n, d = len(X), len(X[0])
    A = [[0.0] * (d + 1) for _ in range(d)]
    for i in range(n):
        xi, yi = X[i], y[i]
        for a in range(d):
            xa = xi[a]
            if xa == 0.0:
                continue
            row = A[a]
            for b in range(d):
                row[b] += xa * xi[b]
            row[d] += xa * yi
    for a in range(d):
        A[a][a] += lam
    # Gaussian elimination with partial pivoting
    for col in range(d):
        piv = max(range(col, d), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        for b in range(col, d + 1):
            A[col][b] /= pv
        for r in range(d):
            if r != col and A[r][col] != 0.0:
                f = A[r][col]
                for b in range(col, d + 1):
                    A[r][b] -= f * A[col][b]
    return [A[a][d] for a in range(d)]


def rmsle(pred, true):
    n = len(true)
    return math.sqrt(sum((math.log1p(max(p, 0.0)) - math.log1p(max(t, 0.0))) ** 2
                         for p, t in zip(pred, true)) / n)


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = float(r["task_start"]) if r["task_start"] else None
            e = float(r["task_end"]) if r["task_end"] else None
            jobs.append({"tag": r["tag_id"], "fw": r["framework"], "user": r["user"],
                         "n": int(r["n_tasks"] or 0), "s": s, "e": e})
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    span = t1 - t0

    T_SPLIT = t0 + 0.60 * span          # fit strictly before this
    EVAL_LEN = 0.10 * span              # evaluate on jobs starting in the next window
    T_EVAL_END = T_SPLIT + EVAL_LEN

    train_all = [j for j in launched if j["s"] < T_SPLIT]
    test = [j for j in launched if T_SPLIT <= j["s"] < T_EVAL_END and j["e"] is not None
            and j["e"] - j["s"] > 0]
    print("=" * 104)
    print("  STEP 4 -- oracle regression: how much accuracy does dropping unfinished records cost?")
    print("=" * 104)
    print(f"  fit window : starts before {T_SPLIT:,.0f} (day {(T_SPLIT-t0)/86400:.1f})")
    print(f"  test window: starts in [{T_SPLIT:,.0f}, {T_EVAL_END:,.0f})  "
          f"n={len(test):,} (completed only, so labels are real)")

    completed = [j for j in train_all if j["e"] is not None and j["e"] <= T_SPLIT]
    unfinished = [j for j in train_all if j["e"] is None or j["e"] > T_SPLIT]
    unf_launched = [j for j in unfinished if j["s"] is not None]
    print(f"  train rows : total launched {len(train_all):,} | completed by split "
          f"{len(completed):,} | unfinished at split {len(unfinished):,} "
          f"({len(unfinished)/len(train_all):.2%})")

    # ---- features ---------------------------------------------------------------------------
    # Categoricals are hashed into a FIXED number of buckets.  One-hot over 1,433 users x ~500k rows
    # is a dense matrix and blows up memory for no benefit; bucketing keeps the design matrix bounded
    # and is blinding to the outcome either way.
    import hashlib

    N_TAG_B, N_FW_B, N_USER_B = 128, 32, 256

    def bucket(s, k):
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big") % k

    DIM = N_TAG_B + N_FW_B + N_USER_B + 3

    def feats(j, group_unfin_share=None):
        x = [0.0] * DIM
        x[bucket(j["tag"], N_TAG_B)] += 1.0
        x[N_TAG_B + bucket(j["fw"], N_FW_B)] += 1.0
        x[N_TAG_B + N_FW_B + bucket(j["user"], N_USER_B)] += 1.0
        base = N_TAG_B + N_FW_B + N_USER_B
        x[base] = math.log1p(max(0, j["n"]))
        x[base + 1] = 1.0
        x[base + 2] = group_unfin_share if group_unfin_share is not None else 0.0
        return x

    # group unfinished share, computed from training window only
    g_tot = defaultdict(int)
    g_unf = defaultdict(int)
    for j in train_all:
        g_tot[j["tag"]] += 1
        if j["e"] is None or j["e"] > T_SPLIT:
            g_unf[j["tag"]] += 1
    share = {t: (g_unf[t] / g_tot[t] if g_tot[t] else 0.0) for t in g_tot}

    def build(rows, labels_from_final):
        X, y = [], []
        for j in rows:
            d = (j["e"] - j["s"]) if j["e"] is not None else None
            if labels_from_final:
                if d is None or d <= 0:
                    continue
            else:
                if j["e"] is None or j["e"] > T_SPLIT:
                    continue
                if d is None or d <= 0:
                    continue
            X.append(feats(j, share.get(j["tag"], 0.0)))
            y.append(math.log1p(d))
        return X, y

    results = {}
    for name, rows, from_final, use_share in (
            ("COMPLETED", train_all, False, False),
            ("ALLFINAL_oracle", train_all, True, False),
            ("COMPLETED_plus_groupcensoring", train_all, False, True)):
        X, y = build(rows, from_final)
        w = fit_ridge(X, y, lam=1.0)
        Xt = [feats(j, share.get(j["tag"], 0.0) if use_share else None) for j in test]
        pred = [math.expm1(max(0.0, sum(wi * xi for wi, xi in zip(w, x)))) for x in Xt]
        true = [j["e"] - j["s"] for j in test]
        r = rmsle(pred, true)
        # rank correlation on the test set (Spearman via ranks)
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            for pos, i in enumerate(order):
                rk[i] = pos
            return rk
        rp, rt = rank(pred), rank(true)
        mp, mt = statistics.fmean(rp), statistics.fmean(rt)
        num = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
        den = math.sqrt(sum((a - mp) ** 2 for a in rp) * sum((b - mt) ** 2 for b in rt))
        rho = num / den if den else float("nan")
        results[name] = {"n_train": len(X), "rmsle": r, "spearman": rho,
                         "median_pred": statistics.median(pred),
                         "median_true": statistics.median(true)}
        print(f"\n  {name:<32} n_train={len(X):>8,}  RMSLE={r:.4f}  Spearman={rho:+.4f}  "
              f"median pred={statistics.median(pred):,.0f}s  median true={statistics.median(true):,.0f}s")

    print("\n" + "=" * 104)
    print("  READING")
    print("=" * 104)
    base = results["COMPLETED"]["rmsle"]
    orc = results["ALLFINAL_oracle"]["rmsle"]
    shr = results["COMPLETED_plus_groupcensoring"]["rmsle"]
    print(f"  completed-only RMSLE                 : {base:.4f}")
    print(f"  oracle (unfinished final labels) RMSLE: {orc:.4f}")
    print(f"  completed + group-censoring RMSLE     : {shr:.4f}")
    print(f"  recoverable accuracy at best          : {(base-orc)/base:.2%} of completed-only RMSLE")
    print(f"  actually recovered by the group signal: {(base-shr)/base:.2%}")

    (HERE / "s4_oracle_regression.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 's4_oracle_regression.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
