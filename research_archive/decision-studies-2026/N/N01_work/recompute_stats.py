"""N01-R: recompute all statistics offline from the stored per-world data with the CORRECT quantile.

No diffusion is re-run.  Inputs are N01_worlds.json.gz and N01S_worlds.json.gz, which hold the
per-world paired differences ``X_c`` and ``X_d`` for both the development and confirmation batches.

Corrections applied here
------------------------
1. Family size.  With ``N`` comparisons there are ``m = 2N`` interval estimates.  A simultaneous 95%
   family puts each two-sided interval at confidence ``1 - 0.05/m``, i.e. the quantile at level
   ``1 - 0.05/(2m) = 1 - 0.05/(4N)``.  The earlier version used ``1 - 0.05/(2N)``, one factor of two
   short, giving t(2047) = 3.2317 instead of 3.4258.  Both are computed and reported here.
2. The ``identical_by_construction`` label is gone; verdicts are purely descriptive.
3. Both batches are recomputed, so the development batch is no longer quoted from the stale run.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parent


# ------------------------------------------------------------------ t quantiles (self-contained)

def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        if abs(d * c - 1.0) < 1e-16:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    fr = math.exp(lb + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return fr * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_cdf(df, t):
    z = df / (df + t * t)
    if t <= 0:
        return 0.5 * _betai(df / 2.0, 0.5, z)
    return 1.0 - 0.5 * _betai(df / 2.0, 0.5, z)


def t_quantile(df, level):
    a, b = 0.0, 100.0
    for _ in range(200):
        m = 0.5 * (a + b)
        if t_cdf(df, m) > level:
            b = m
        else:
            a = m
        if b - a < 1e-15:
            break
    return 0.5 * (a + b)


T95_DF = {255: None, 2047: None}


def t95(n):
    df = n - 1
    return t_quantile(df, 0.975)


def t_bonf(n, n_comp):
    """Quantile for a simultaneous 95% family over m = 2N intervals."""
    return t_quantile(n - 1, 1.0 - 0.05 / (4 * n_comp))


def t_bonf_superseded(n, n_comp):
    """The level the earlier version used, kept only to show what changed."""
    return t_quantile(n - 1, 1.0 - 0.05 / (2 * n_comp))


# ------------------------------------------------------------------ statistics

def stats(xs, t_ord, t_adj):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n, "mean": m, "sd": sd, "se": se,
        "ci": (m - t_ord * se, m + t_ord * se),
        "bonf": (m - t_adj * se, m + t_adj * se),
    }


def classify(a, b):
    """Purely descriptive: which side of zero each adjusted interval falls on."""
    def side(iv):
        if iv[1] < 0:
            return "neg"
        if iv[0] > 0:
            return "pos"
        return "zero"
    sa, sb = side(a["bonf"]), side(b["bonf"])
    if {sa, sb} == {"neg", "pos"}:
        return "confirmed_reversal"
    if sa != "zero" and sb != "zero":
        return "both_significant_same_side"
    if sa != "zero" or sb != "zero":
        return "one_sided_significant"
    return "not_significant"


def run(label, results_csv, worlds_gz, cases_json):
    rows = list(csv.DictReader((WORK / results_csv).open(encoding="utf-8-sig")))
    ws = json.loads(gzip.open(WORK / worlds_gz, "rt", encoding="utf-8").read())
    cases = json.loads((WORK / cases_json).read_text(encoding="utf-8"))["cases"]
    n_comp = len(rows)
    out = {"label": label, "n_comparisons": n_comp,
           "family_size": 2 * n_comp,
           "quantile_level": 1.0 - 0.05 / (4 * n_comp),
           "batches": {}}
    for batch in ("dev", "confirm"):
        recs = []
        for r, pc, cs in zip(rows, ws["per_case"], cases):
            xc, xd = pc[batch]["X_c"], pc[batch]["X_d"]
            n = len(xc)
            to, ta = t95(n), t_bonf(n, n_comp)
            sc = stats(xc, to, ta)
            sd = stats(xd, to, ta)
            m = min(abs(sc["mean"]), abs(sd["mean"]))
            recs.append({
                "index": int(r["index"]), "group": r["group"],
                "a": int(r["a"]), "b": int(r["b"]), "c": int(r["c"]), "d": int(r["d"]),
                "U": list(cs["U"]),
                "delta_c_mean": sc["mean"], "delta_c_se": sc["se"],
                "delta_c_ci_low": sc["ci"][0], "delta_c_ci_high": sc["ci"][1],
                "delta_c_bonf_low": sc["bonf"][0], "delta_c_bonf_high": sc["bonf"][1],
                "delta_d_mean": sd["mean"], "delta_d_se": sd["se"],
                "delta_d_ci_low": sd["ci"][0], "delta_d_ci_high": sd["ci"][1],
                "delta_d_bonf_low": sd["bonf"][0], "delta_d_bonf_high": sd["bonf"][1],
                "reversal": classify(sc, sd),
                "m": m, "m_over_n": m / 5241.0,
                "X_c_eq_X_d_worlds": sum(1 for u, v in zip(xc, xd) if u == v),
                "n_worlds": n,
                "max_abs_X_c_minus_X_d": max(abs(u - v) for u, v in zip(xc, xd)),
                "point_estimate_reversal": bool(sc["mean"] * sd["mean"] < 0),
            })
        out["batches"][batch] = {
            "n_worlds": len(ws["per_case"][0][batch]["X_c"]),
            "t95": to, "t_bonferroni": ta,
            "t_bonferroni_superseded": t_bonf_superseded(
                len(ws["per_case"][0][batch]["X_c"]), n_comp),
            "verdict_counts": _counts(recs),
            "point_estimate_reversals": sum(1 for r in recs if r["point_estimate_reversal"]),
            "cases_with_any_nonzero_Xc_minus_Xd": sum(
                1 for r in recs if r["X_c_eq_X_d_worlds"] < r["n_worlds"]),
            "records": recs,
        }
    return out


def _counts(recs):
    c: dict[str, int] = {}
    for r in recs:
        c[r["reversal"]] = c.get(r["reversal"], 0) + 1
    return c


def main() -> int:
    main_run = run("N01 (frozen)", "N01_results.csv", "N01_worlds.json.gz", "N01_cases.json")
    probe_run = run("N01-S (probe)", "N01S_results.csv", "N01S_worlds.json.gz", "N01S_cases.json")

    print("=" * 104)
    print("  N01-R OFFLINE STATISTICS RECOMPUTE  (no diffusion re-run; from stored per-world data)")
    print("=" * 104)
    for r in (main_run, probe_run):
        print(f"\n  {r['label']}: N={r['n_comparisons']} comparisons, family m={r['family_size']} "
              f"intervals")
        print(f"    correct quantile level 1 - 0.05/(4N) = {r['quantile_level']:.7f}")
        for batch, d in r["batches"].items():
            print(f"    {batch:<8} worlds {d['n_worlds']:<5} "
                  f"t95 {d['t95']:.6f}  t_bonf(CORRECT) {d['t_bonferroni']:.6f}  "
                  f"t_bonf(superseded) {d['t_bonferroni_superseded']:.6f}")
            print(f"             verdicts (CORRECT)      : {d['verdict_counts']}")
            print(f"             point-estimate reversals: {d['point_estimate_reversals']}")
            print(f"             cases with any nonzero X_c-X_d: "
                  f"{d['cases_with_any_nonzero_Xc_minus_Xd']}/{r['n_comparisons']}")

    # what changed versus the superseded level
    print("\n" + "=" * 104)
    print("  WHAT CHANGED versus the superseded quantile")
    print("=" * 104)
    for r in (main_run, probe_run):
        conf = r["batches"]["confirm"]
        n_comp = r["n_comparisons"]
        n_worlds = conf["n_worlds"]
        t_old = conf["t_bonferroni_superseded"]
        t_new = conf["t_bonferroni"]
        old_counts, new_counts = {}, conf["verdict_counts"]
        changed = []
        ws = json.loads(gzip.open(WORK / ("N01_worlds.json.gz" if r["label"].startswith("N01 (")
                                          else "N01S_worlds.json.gz"), "rt", encoding="utf-8").read())
        for rec, pc in zip(conf["records"], ws["per_case"]):
            def side(iv):
                return "neg" if iv[1] < 0 else "pos" if iv[0] > 0 else "zero"
            old = {}
            for tag, xs in (("c", pc["confirm"]["X_c"]), ("d", pc["confirm"]["X_d"])):
                m = sum(xs) / len(xs)
                sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                se = sd / math.sqrt(len(xs))
                old[tag] = (m - t_old * se, m + t_old * se)
            so, sn = side(old["c"]), side(old["d"])
            lab_old = ("confirmed_reversal" if {so, sn} == {"neg", "pos"}
                       else "both_significant_same_side" if so != "zero" and sn != "zero"
                       else "one_sided_significant" if so != "zero" or sn != "zero"
                       else "not_significant")
            old_counts[lab_old] = old_counts.get(lab_old, 0) + 1
            if lab_old != rec["reversal"]:
                changed.append((rec["index"], lab_old, rec["reversal"],
                                round(rec["delta_c_mean"], 4), round(rec["delta_d_mean"], 4)))
        print(f"\n  {r['label']} confirm batch")
        print(f"    superseded t={t_old:.6f} -> {old_counts}")
        print(f"    CORRECT    t={t_new:.6f} -> {new_counts}")
        print(f"    verdict changes: {changed}")
        print(f"    reversals under CORRECT quantile: "
              f"{new_counts.get('confirmed_reversal', 0)}")

    (WORK / "N01R_statistics.json").write_text(
        json.dumps({"main": main_run, "probe": probe_run}, indent=2), encoding="utf-8")
    print(f"\n  wrote {WORK / 'N01R_statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
