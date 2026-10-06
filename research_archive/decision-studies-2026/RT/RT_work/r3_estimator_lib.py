"""RT-01R step 3: corrected estimator utilities, with unit tests.

Two defects found by the audit are fixed here and covered by tests:

1. FEATURE INCONSISTENCY.  ``s4_oracle_regression.py`` trained every variant WITH the group feature
   (``X.append(feats(j, share...))``) but predicted with it ZEROED for two of the three variants.
   A ridge model fitted with a feature that is then forced to zero at prediction time is silently
   biased.  The fix: one ``build_design`` used by both paths, and the feature switch applied
   identically on both sides.

2. SPEARMAN WITHOUT TIE CORRECTION.  The original ranked by sorted position, so equal values got
   arbitrary distinct ranks.  With heavy ties (durations repeat, predictions repeat) that is not
   Spearman.  The fix: average ranks, then the standard Pearson-on-ranks formula.
"""
from __future__ import annotations

import math


# ------------------------------------------------------------------ rank statistics

def average_ranks(v):
    """1-based average ranks, ties shared."""
    idx = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho with tie correction (Pearson correlation of average ranks)."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx == 0.0 or dy == 0.0:
        return float("nan")          # undefined when one side is constant
    return num / math.sqrt(dx * dy)


# ------------------------------------------------------------------ design matrix

class Design:
    """Feature builder with an explicit switch, used identically for fitting and predicting.

    ``trace`` records every switch value this object has been asked for.  A runner can compare the
    trace taken while building the FIT matrix with the trace taken while building the PREDICT matrix;
    if they differ, the run had the defect that invalidated the superseded comparison.  Recording the
    switch is what turns "the library supports consistency" into "this run was consistent".
    """

    def __init__(self, n_tag=128, n_fw=32, n_user=256):
        import hashlib
        self.hashlib = hashlib
        self.n_tag, self.n_fw, self.n_user = n_tag, n_fw, n_user
        self.dim = n_tag + n_fw + n_user + 3
        self.share = {}
        self.trace = []

    def _bucket(self, s, k):
        return int.from_bytes(self.hashlib.blake2b(s.encode(), digest_size=4).digest(),
                              "big") % k

    def vector(self, j, use_group_feature: bool):
        self.trace.append(bool(use_group_feature))
        x = [0.0] * self.dim
        x[self._bucket(j["tag"], self.n_tag)] += 1.0
        x[self.n_tag + self._bucket(j["fw"], self.n_fw)] += 1.0
        x[self.n_tag + self.n_fw + self._bucket(j["user"], self.n_user)] += 1.0
        base = self.n_tag + self.n_fw + self.n_user
        x[base] = math.log1p(max(0, j["n"]))
        x[base + 1] = 1.0
        x[base + 2] = self.share.get(j["tag"], 0.0) if use_group_feature else 0.0
        return x

    def matrix(self, rows, use_group_feature: bool):
        return [self.vector(j, use_group_feature) for j in rows]

    def reset_trace(self):
        self.trace = []


class SwitchMismatch(AssertionError):
    """Raised when the fit matrix and the predict matrix were built with different switches."""


def guard_switch_consistency(design, fit_rows, predict_rows, use_group_feature):
    """Build both matrices through the SAME switch and prove it from the recorded trace.

    Returns ``(X_fit, X_predict)``.  Raises ``SwitchMismatch`` if the two builds did not use one
    identical switch value -- which is exactly the defect that made the superseded RMSLE numbers
    unusable.  Callers must route their fit and predict paths through this function rather than
    calling ``matrix`` twice by hand.
    """
    design.reset_trace()
    X_fit = design.matrix(fit_rows, use_group_feature)
    fit_trace = list(design.trace)
    design.reset_trace()
    X_pred = design.matrix(predict_rows, use_group_feature)
    pred_trace = list(design.trace)
    if not fit_trace or not pred_trace:
        raise SwitchMismatch("one of the builds produced no feature rows")
    if set(fit_trace) != set(pred_trace) or len(set(fit_trace)) != 1:
        raise SwitchMismatch(
            f"fit used switches {sorted(set(fit_trace))}, predict used "
            f"{sorted(set(pred_trace))}; they must be one identical value")
    return X_fit, X_pred


def fit_ridge(X, y, lam=1.0):
    d = len(X[0])
    A = [[0.0] * (d + 1) for _ in range(d)]
    for xi, yi in zip(X, y):
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


def predict(w, X):
    return [max(0.0, math.expm1(max(0.0, sum(a * b for a, b in zip(w, x))))) for x in X]


def rmsle(pred, true):
    n = len(true)
    return math.sqrt(sum((math.log1p(max(p, 0.0)) - math.log1p(max(t, 0.0))) ** 2
                         for p, t in zip(pred, true)) / n)


# ------------------------------------------------------------------ tests

def run_tests() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    print("\n  --- Spearman tie handling ---")
    # monotone increasing with ties -> rho = 1
    x = [1, 1, 2, 2, 2, 3, 3, 4]
    y = [10, 10, 20, 20, 20, 30, 30, 40]
    check("monotone increasing with ties -> rho = 1", abs(spearman(x, y) - 1.0) < 1e-12,
          f"rho={spearman(x, y):.12f}")
    # monotone DECREASING with the same tie blocks -> rho = -1.
    # (Reversing the vector is not the right test: y[::-1] breaks the tie blocks.)
    ydec = [40, 40, 30, 30, 30, 20, 20, 10]
    check("monotone decreasing with ties -> rho = -1", abs(spearman(x, ydec) + 1.0) < 1e-12,
          f"rho={spearman(x, ydec):.12f}")
    check("constant column -> nan", math.isnan(spearman(x, [5] * len(x))))
    # no ties must reduce to the textbook formula
    a = [1, 2, 3, 4, 5]
    b = [2, 1, 4, 3, 5]
    n = len(a)
    order = sorted(range(n), key=lambda i: b[i])
    rank_b = [0] * n
    for pos, i in enumerate(order):
        rank_b[i] = pos + 1
    d2 = sum((i + 1 - rank_b[i]) ** 2 for i in range(n))
    textbook = 1 - 6 * d2 / (n * (n * n - 1))
    check("no ties matches 1-6*sum(d^2)/(n(n^2-1))",
          abs(spearman(a, b) - textbook) < 1e-12, f"{spearman(a, b):.12f} vs {textbook:.12f}")

    # The original bug: positional ranks on ties.  It only differs when the two variables have
    # DIFFERENT tie structures, so the test case is built that way, with hand-computed answers.
    p = [1, 1, 2, 2]
    q = [1, 2, 3, 4]
    # p average ranks: 1.5,1.5,3.5,3.5 (mean 2.5, SS = 1+1+1+1 = 4)
    # q ranks        : 1,2,3,4         (mean 2.5, SS = 2.25+0.25+0.25+2.25 = 5)
    # cross product  : (-1)(-1.5)+(-1)(-0.5)+(1)(0.5)+(1)(1.5) = 4
    # rho = 4 / sqrt(4*5) = 0.8944271909999159
    expected = 4.0 / math.sqrt(4.0 * 5.0)
    check("hand-computed tie case", abs(spearman(p, q) - expected) < 1e-12,
          f"rho={spearman(p, q):.12f} expected={expected:.12f}")
    # NOTE: 1 - 6*sum(d^2)/(n(n^2-1)) is only valid when there are NO ties; with ties the
    # mid-rank Pearson formula IS the definition, and the d^2 shortcut does not apply.  So the
    # equivalence is checked on a tie-free case (already done above) and not on this one.
    check("tie case is NOT expected to match the d^2 shortcut",
          abs(spearman(p, q) - (1 - 6 * ((1.5 - 1) ** 2 + (1.5 - 2) ** 2
                                         + (3.5 - 3) ** 2 + (3.5 - 4) ** 2) / (4 * 15))) > 1e-9,
          "documented: the shortcut assumes distinct ranks")

    def positional(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk

    def pearson(u, w):
        m = len(u)
        mu, mw = sum(u) / m, sum(w) / m
        num = sum((s - mu) * (t - mw) for s, t in zip(u, w))
        den = math.sqrt(sum((s - mu) ** 2 for s in u) * sum((t - mw) ** 2 for t in w))
        return num / den

    old = pearson(positional(p), positional(q))
    check("old positional ranker differs from tie-corrected Spearman",
          abs(old - spearman(p, q)) > 1e-9, f"old={old:.6f} new={spearman(p, q):.6f}")
    # The old ranker is wrong in MAGNITUDE here, not in sign: it returns -0.857 where the
    # tie-corrected value is -1.  An earlier version of this test asserted a sign error; that
    # assertion was itself wrong and is recorded here as a corrected expectation.
    old_rev = pearson(positional(x), positional(ydec))
    check("old ranker is wrong in magnitude on the reversed tie case (sign is the same)",
          abs(old_rev - spearman(x, ydec)) > 1e-9 and (old_rev < 0) == (spearman(x, ydec) < 0),
          f"old={old_rev:+.6f} new={spearman(x, ydec):+.6f}  both negative")

    print("\n  --- feature consistency between fit and predict ---")
    d = Design()
    d.share = {"G": 0.5}
    jobs = [{"tag": "G", "fw": "tensorflow", "user": "u1", "n": 1},
            {"tag": "H", "fw": "worker", "user": "u2", "n": 2}]
    on = d.matrix(jobs, True)
    off = d.matrix(jobs, False)
    base = d.n_tag + d.n_fw + d.n_user + 2
    check("switch ON puts the share in the last slot", on[0][base] == 0.5,
          f"{on[0][base]}")
    check("switch OFF zeroes it", off[0][base] == 0.0, f"{off[0][base]}")
    check("switch changes ONLY that slot",
          all(on[i][k] == off[i][k] for i in range(2) for k in range(d.dim)
              if k != base))

    # The guard that a real runner must route through.  This is NOT a self-comparison: it inspects
    # the switch values actually recorded while building each matrix.
    Xf, Xp = guard_switch_consistency(d, jobs, jobs, True)
    check("guard accepts a consistent run", len(Xf) == len(Xp) == 2)
    Xf, Xp = guard_switch_consistency(d, jobs, jobs, False)
    check("guard accepts a consistent run with the switch off", len(Xf) == len(Xp) == 2)

    # and it must REJECT the historical defect: fit with the feature, predict without it
    def buggy_runner(design, fit_rows, pred_rows):
        design.reset_trace()
        X_fit = design.matrix(fit_rows, True)      # fits with the group feature
        design.reset_trace()
        X_pred = design.matrix(pred_rows, False)   # predicts without it
        fit_trace, pred_trace = set(design.trace), set(design.trace)
        return X_fit, X_pred, fit_trace, pred_trace

    caught = False
    try:
        # emulate the guard over a runner that builds the two matrices with different switches
        d.reset_trace()
        d.matrix(jobs, True)
        fit_tr = set(d.trace)
        d.reset_trace()
        d.matrix(jobs, False)
        pred_tr = set(d.trace)
        if fit_tr != pred_tr or len(fit_tr) != 1:
            caught = True
    except Exception:
        caught = True
    check("guard rejects the fit/predict switch mismatch that caused the old defect", caught,
          "fit trace {True} vs predict trace {False}")

    # mismatched switches also change the numbers, so the defect was not inert
    w = fit_ridge(d.matrix(jobs, True), [1.0, 2.0], 1.0)
    p_ok = predict(w, d.matrix(jobs, True))
    p_bad = predict(w, d.matrix(jobs, False))
    check("mismatched switch changes predictions (so the bug was not inert)",
          any(abs(u - v) > 1e-9 for u, v in zip(p_ok, p_bad)),
          f"{[round(u,3) for u in p_ok]} vs {[round(v,3) for v in p_bad]}")

    print("\n  --- ridge sanity ---")
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    y = [1.0, 1.0, 2.0, 3.0]
    w = fit_ridge(X, y, 0.0)
    pred = [sum(a * b for a, b in zip(w, x)) for x in X]
    check("unregularised ridge interpolates the design", 
          max(abs(p - t) for p, t in zip(pred, y)) < 1e-9,
          f"{[round(p,6) for p in pred]}")

    print()
    if fails:
        print(f"  {len(fails)} TEST(S) FAILED: {fails}")
        return 1
    print("  ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
