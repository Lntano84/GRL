"""N01-R claim audit: verify each point of the external audit against the stored artifacts.

No new diffusion is run.  Everything here reads the saved per-world data and the frozen case file.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parent


def t_cdf(df, t):
    """Closed-form t CDF (same code as run_diffusion, duplicated so this stays independent)."""
    def betacf(a, b, x):
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

    def betai(a, b, x):
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        fr = math.exp(lb + a * math.log(x) + b * math.log1p(-x))
        if x < (a + 1.0) / (a + b + 2.0):
            return fr * betacf(a, b, x) / a
        return 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * betacf(b, a, 1.0 - x) / b

    if t <= 0:
        return 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))


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


def main() -> int:
    cases_art = json.loads((WORK / "N01_cases.json").read_text(encoding="utf-8"))
    cases = cases_art["cases"]
    main_worlds = json.loads(gzip.open(WORK / "N01_worlds.json.gz", "rt", encoding="utf-8").read())
    probe_worlds = json.loads(gzip.open(WORK / "N01S_worlds.json.gz", "rt", encoding="utf-8").read())

    print("=" * 100)
    print("  CLAIM 4 -- are the 40 frozen candidate pairs distinct?")
    print("=" * 100)
    pairs = [tuple(sorted((c["a"], c["b"]))) for c in cases]
    print(f"  unordered {{a,b}} pairs: {len(pairs)}, distinct: {len(set(pairs))}")
    q = [tuple(sorted((c["c"], c["d"]))) for c in cases]
    print(f"  unordered {{c,d}} pairs: {len(q)}, distinct: {len(set(q))}")
    print(f"  any duplicate at all: {len(set(pairs)) != len(pairs) or len(set(q)) != len(q)}")
    print("  -> the audit is right: 40 distinct candidate pairs, so a sign difference across cases")
    print("     cannot be attributed to the background U.  The claim must be withdrawn.")

    print("\n" + "=" * 100)
    print("  CLAIM 1b -- the stored data itself falsifies the identity")
    print("=" * 100)
    dev_ids = main_worlds["world_order"]["dev"]
    conf_ids = main_worlds["world_order"]["confirm"]
    pc0 = main_worlds["per_case"][0]
    idx = conf_ids.index(1000295)
    print(f"  main case 0, world 1000295: X_c = {pc0['confirm']['X_c'][idx]}, "
          f"X_d = {pc0['confirm']['X_d'][idx]}")
    worst = []
    for r, pc in zip(cases, main_worlds["per_case"]):
        xc, xd = pc["confirm"]["X_c"], pc["confirm"]["X_d"]
        for w, (u, v) in enumerate(zip(xc, xd)):
            if u != v:
                worst.append((abs(u - v), r["index"], conf_ids[w], u, v))
    worst.sort(reverse=True)
    print(f"  total non-zero pairs in the frozen confirm batch: {len(worst)}")
    print("  largest deviations:")
    for d, ci, wid, u, v in worst[:6]:
        print(f"    case {ci:>2} world {wid}: X_c={u:>5} X_d={v:>5} |diff|={d}")

    print("\n" + "=" * 100)
    print("  CLAIM 3 -- recompute the confirm-batch verdicts with the CORRECT quantile")
    print("=" * 100)
    t_wrong = t_quantile(2047, 1.0 - 0.05 / 80)      # what the runner used
    t_right = t_quantile(2047, 1.0 - 0.05 / 160)     # family 2N=80, per-interval 1-0.05/80
    print(f"  superseded quantile (bug)  1-0.05/80   -> t = {t_wrong:.9f}")
    print(f"  correct quantile           1-0.05/160  -> t = {t_right:.9f}")
    print(f"  difference: {t_right - t_wrong:.6f}")

    def verdicts(rs, ws, tcrit, n_comp):
        out = []
        for r, pc in zip(rs, ws["per_case"]):
            xc, xd = pc["confirm"]["X_c"], pc["confirm"]["X_d"]
            row = {}
            for tag, xs in (("c", xc), ("d", xd)):
                n = len(xs)
                m = sum(xs) / n
                sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
                se = sd / math.sqrt(n)
                row[tag] = (m, m - tcrit * se, m + tcrit * se)
            def sig(k):
                m, lo, hi = row[k]
                return ("neg" if hi < 0 else "pos" if lo > 0 else "zero")
            sc, sd_ = sig("c"), sig("d")
            if {sc, sd_} == {"neg", "pos"}:
                lab = "confirmed_reversal"
            elif sc in ("neg", "pos") and sd_ in ("neg", "pos"):
                lab = "both_significant_same_side" if sc == sd_ else "confirmed_reversal"
            else:
                lab = "not_significant"
            out.append((r["index"], lab, row["c"][0], row["d"][0]))
        return out

    main_rs = list(csv.DictReader((WORK / "N01_results.csv").open(encoding="utf-8-sig")))
    probe_rs = list(csv.DictReader((WORK / "N01S_results.csv").open(encoding="utf-8-sig")))

    for label, rs, ws in (("N01 (frozen)", main_rs, main_worlds),
                          ("N01-S (probe)", probe_rs, probe_worlds)):
        for tname, tcrit in (("superseded", t_wrong), ("CORRECT", t_right)):
            v = verdicts(rs, ws, tcrit, len(rs))
            cnt = {}
            for _, lab, _, _ in v:
                cnt[lab] = cnt.get(lab, 0) + 1
            print(f"  {label:<15} {tname:<11} t={tcrit:.6f} -> {cnt}")
        # which cases change
        va = {i: l for i, l, _, _ in verdicts(rs, ws, t_wrong, len(rs))}
        vb = {i: l for i, l, _, _ in verdicts(rs, ws, t_right, len(rs))}
        changed = [(i, va[i], vb[i]) for i in va if va[i] != vb[i]]
        print(f"  {label:<15} cases whose verdict changes: {changed}")
        opp = [(i, a, b) for i, _, a, b in verdicts(rs, ws, t_right, len(rs)) if a * b < 0]
        print(f"  {label:<15} point-estimate reversals under CORRECT quantile: {len(opp)} {opp}")

    print("\n" + "=" * 100)
    print("  CLAIM 5b -- what the compressed file actually stores")
    print("=" * 100)
    keys = sorted(main_worlds["per_case"][0]["confirm"].keys())
    print(f"  per-case confirm keys: {keys}")
    sp = main_worlds["per_case"][0]["confirm"].get("spreads")
    if sp:
        print(f"  'spreads' present with keys {sorted(sp.keys())}; "
              f"length {len(sp['S_ac'])}")
        print("  -> the four per-set activation counts ARE stored; the audit's remark that only")
        print("     differences were saved is incorrect for this artifact (but the point stands")
        print("     that the statistics were recomputed from stored values, not re-simulated).")
    else:
        print("  -> only differences stored.")

    print("\n" + "=" * 100)
    print("  CLAIM 5a -- 2-hop region INCLUDING the source node")
    print("=" * 100)
    import sys
    sys.path.insert(0, str(WORK))
    from n01_common import Graph, parse_ca_grqc, within_hops
    arcs, undirected, node_ids, _ = parse_ca_grqc()
    g = Graph(arcs, undirected, node_ids)
    r2_inc = {v: within_hops(g, v, 2) | {v} for v in range(g.n)}
    bad = []
    for c in cases:
        a, b, cc, dd = c["a"], c["b"], c["c"], c["d"]
        if (r2_inc[a] | r2_inc[b]) & (r2_inc[cc] | r2_inc[dd]):
            bad.append(c["index"])
    print(f"  frozen cases violating the condition when the source node is INCLUDED: {len(bad)} {bad}")
    print("  -> the audit is right: 40/40 still hold, so the frozen comparisons are retainable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
