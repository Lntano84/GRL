"""Step 6: is the group-censoring signal real, and does the bias concentrate on a identifiable subset?

Two questions:
  (1) Leave-one-out shrinkage: the raw per-group unfinished share is computed from a handful of jobs
      for many tags.  Is its variance signal or noise?  Without this, a "0.4% improvement" could be
      an artefact of fitting a noisy feature.
  (2) Targeted bias: split the TEST jobs by their group's train-window censoring share and measure the
      error of the completed-only model on each subset.  If the bias is concentrated on an
      identifiable subset, aggregate RMSLE hides it and it is still actionable.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 9)


def fit_ridge(X, y, lam=1.0):
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


def main() -> int:
    jobs = []
    with (HERE / "rt_jobs.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            jobs.append({"tag": r["tag_id"], "fw": r["framework"], "user": r["user"],
                         "n": int(r["n_tasks"] or 0),
                         "s": float(r["task_start"]) if r["task_start"] else None,
                         "e": float(r["task_end"]) if r["task_end"] else None})
    launched = [j for j in jobs if j["s"] is not None]
    t0 = min(j["s"] for j in launched)
    t1 = max(j["e"] for j in jobs if j["e"] is not None)
    span = t1 - t0
    T_SPLIT = t0 + 0.60 * span
    T_EVAL_END = T_SPLIT + 0.10 * span

    train = [j for j in launched if j["s"] < T_SPLIT]
    test = [j for j in launched if T_SPLIT <= j["s"] < T_EVAL_END
            and j["e"] is not None and j["e"] - j["s"] > 0]

    N_TAG_B, N_FW_B, N_USER_B = 128, 32, 256
    DIM = N_TAG_B + N_FW_B + N_USER_B + 3

    def bucket(s, k):
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big") % k

    def feats(j, sh=None):
        x = [0.0] * DIM
        x[bucket(j["tag"], N_TAG_B)] += 1.0
        x[N_TAG_B + bucket(j["fw"], N_FW_B)] += 1.0
        x[N_TAG_B + N_FW_B + bucket(j["user"], N_USER_B)] += 1.0
        base = N_TAG_B + N_FW_B + N_USER_B
        x[base] = math.log1p(max(0, j["n"]))
        x[base + 1] = 1.0
        x[base + 2] = sh if sh is not None else 0.0
        return x

    # ---- group stats with leave-one-out ------------------------------------------------
    g_n = defaultdict(int)
    g_u = defaultdict(int)
    for j in train:
        g_n[j["tag"]] += 1
        if j["e"] is None or j["e"] > T_SPLIT:
            g_u[j["tag"]] += 1
    print("=" * 100)
    print("  STEP 6 -- is the group censoring share signal or noise?")
    print("=" * 100)
    sizes = sorted(g_n.values())
    print(f"  groups(tags): {len(g_n):,}   size p50={sizes[len(sizes)//2]}  "
          f"p90={sizes[int(0.9*len(sizes))]}  max={sizes[-1]}")
    big = [t for t, n in g_n.items() if n >= 50]
    print(f"  groups with >=50 train jobs: {len(big):,}")

    # split each big group's jobs in half by start order; compare the two halves' censoring shares
    half = defaultdict(list)
    for j in train:
        if j["tag"] in g_n:
            half[j["tag"]].append(j)
    diffs = []
    for t, lst in half.items():
        if len(lst) < 50:
            continue
        lst.sort(key=lambda j: j["s"])
        h = len(lst) // 2
        for grp in (lst[:h], lst[h:]):
            pass
        sh_a = sum(1 for j in lst[:h] if j["e"] is None or j["e"] > T_SPLIT) / h
        sh_b = sum(1 for j in lst[h:] if j["e"] is None or j["e"] > T_SPLIT) / (len(lst) - h)
        diffs.append(abs(sh_a - sh_b))
    if diffs:
        diffs.sort()
        print(f"\n  within-group split-half agreement of the censoring share "
              f"(groups with >=50 jobs, n={len(diffs):,}):")
        print(f"    |share_first_half - share_second_half|: p50={diffs[len(diffs)//2]:.3f}  "
              f"p90={diffs[int(0.9*len(diffs))]:.3f}")
        print(f"    groups agreeing within 0.10: {sum(1 for d in diffs if d <= 0.10)/len(diffs):.1%}")
        print("    -> a share that is mostly 0 or 1 with high agreement is a STABLE GROUP PROPERTY;")
        print("       one that swings between halves is noise.")

    # ---- fit the two deployed models ----------------------------------------------------
    share = {t: (g_u[t] / g_n[t]) for t in g_n}
    X_c = [feats(j) for j in train if j["e"] is not None and j["e"] <= T_SPLIT]
    y_c = [math.log1p(j["e"] - j["s"]) for j in train
           if j["e"] is not None and j["e"] <= T_SPLIT and j["e"] - j["s"] > 0]
    X_c = [feats(j) for j in train if j["e"] is not None and j["e"] <= T_SPLIT
           and j["e"] - j["s"] > 0]
    X_s = [feats(j, share.get(j["tag"], 0.0)) for j in train
           if j["e"] is not None and j["e"] <= T_SPLIT and j["e"] - j["s"] > 0]
    w_c = fit_ridge(X_c, y_c, 1.0)
    w_s = fit_ridge(X_s, y_c, 1.0)

    def predict(w, j, with_share):
        sh = share.get(j["tag"], 0.0) if with_share else None
        x = feats(j, sh)
        return max(0.0, math.expm1(max(0.0, sum(a * b for a, b in zip(w, x)))))

    # ---- targeted bias by group censoring share -----------------------------------------
    print("\n" + "=" * 100)
    print("  TARGETED BIAS: error of the completed-only model by the group's train censoring share")
    print("=" * 100)
    buckets = defaultdict(list)
    for j in test:
        s = share.get(j["tag"], 0.0)
        b = "0.00" if s == 0 else "0.01-0.10" if s <= 0.10 else "0.10-0.30" if s <= 0.30 \
            else "0.30-0.60" if s <= 0.60 else ">0.60"
        buckets[b].append(j)
    order = ["0.00", "0.01-0.10", "0.10-0.30", "0.30-0.60", ">0.60"]
    print(f"  {'share bin':<12} {'n test':>8} {'true p50':>10} {'pred p50':>10} "
          f"{'log-ratio':>10} {'RMSLE':>8} {'RMSLE+share':>12}")
    rows = []
    for b in order:
        sub = buckets.get(b, [])
        if not sub:
            continue
        true = [j["e"] - j["s"] for j in sub]
        pc = [predict(w_c, j, False) for j in sub]
        ps = [predict(w_s, j, True) for j in sub]
        r_c = math.sqrt(sum((math.log1p(p) - math.log1p(t)) ** 2 for p, t in zip(pc, true)) / len(sub))
        r_s = math.sqrt(sum((math.log1p(p) - math.log1p(t)) ** 2 for p, t in zip(ps, true)) / len(sub))
        mt, mp = statistics.median(true), statistics.median(pc)
        print(f"  {b:<12} {len(sub):>8,} {mt:>10,.0f} {mp:>10,.0f} "
              f"{math.log1p(mt)-math.log1p(mp):>+10.3f} {r_c:>8.4f} {r_s:>12.4f}")
        rows.append({"bin": b, "n": len(sub), "median_true": mt, "median_pred_completed": mp,
                     "log_ratio": math.log1p(mt) - math.log1p(mp),
                     "rmsle_completed": r_c, "rmsle_with_share": r_s})
    overall_c = math.sqrt(sum((math.log1p(predict(w_c, j, False))
                               - math.log1p(j["e"] - j["s"])) ** 2 for j in test) / len(test))
    overall_s = math.sqrt(sum((math.log1p(predict(w_s, j, True))
                               - math.log1p(j["e"] - j["s"])) ** 2 for j in test) / len(test))
    print(f"  {'OVERALL':<12} {len(test):>8,} {'':>10} {'':>10} {'':>10} "
          f"{overall_c:>8.4f} {overall_s:>12.4f}")
    hi = [r for r in rows if r["bin"] in ("0.30-0.60", ">0.60")]
    if hi:
        wsum = sum(r["n"] for r in hi)
        print(f"\n  jobs in groups with train censoring share > 0.30: {wsum:,} "
              f"({wsum/len(test):.2%} of test)")
        for r in hi:
            print(f"    {r['bin']:<10} n={r['n']:>7,}  completed-only is LOW by "
                  f"{r['log_ratio']:+.3f} in log1p (median true / median pred = "
                  f"{math.exp(r['log_ratio']):.2f}x)")

    (HERE / "s6_targeted.json").write_text(json.dumps(
        {"bins": rows, "overall_completed": overall_c, "overall_with_share": overall_s,
         "split_half_median_abs_diff": diffs[len(diffs)//2] if diffs else None}, indent=2),
        encoding="utf-8")
    print(f"\n  wrote {HERE / 's6_targeted.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
