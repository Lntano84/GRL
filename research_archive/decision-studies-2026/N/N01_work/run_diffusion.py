"""N01 diffusion evaluation + statistics on ca-GrQc.

Two batches, frozen in advance
------------------------------
* development batch : 256 common random worlds  (world seeds 0 .. 255)
* confirmation batch: 2048 further independent worlds (world seeds 1_000_000 .. 1_002_047)

The world-seed ranges are disjoint by construction, so no world is shared between the two batches.
Every frozen comparison is evaluated in both batches -- the confirmation batch covers all 40, not a
selected subset.

Common random numbers inside a world
------------------------------------
For one comparison and one world the live/dead state of every directed arc is drawn once from
``random.Random(world_seed)``.  The four seed sets are then evaluated by deterministic reachability in
that same state, so ``spread(S_ac)``, ``spread(S_bc)``, ``spread(S_ad)`` and ``spread(S_bd)`` all see
identical edge outcomes.  The four differences are therefore paired, and ``X_c`` and ``X_d`` are
computed inside the world rather than from separately re-seeded simulations.

Statistics
----------
Per comparison and per difference: mean, standard error, and an ordinary two-sided 95% interval using
the Student-t quantile.  Then a Bonferroni-adjusted simultaneous interval over all ``2N`` means, where
``N`` is the number of comparisons actually run, so each interval is built at level
``1 - 0.05/(2N)``.  These are reported as **approximate Monte Carlo intervals** -- a t interval on
simulation means is not a finite-sample mathematical certificate.

A comparison is marked as a confirmed ranking reversal only when both adjusted intervals lie strictly
on opposite sides of zero.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import (  # noqa: E402
    CONFIRM_WORLD_IDS, DEV_WORLD_IDS, Graph, build_live_adjacency, draw_live_arcs,
    ic_spread, parse_ca_grqc,
)

# Two-sided Student-t quantiles, computed and verified independently (see verify_t.py and
# t_critical_values.md).  They are NOT trusted as literals: main() recomputes each one at startup
# from the closed-form t CDF and refuses to run if a constant disagrees.
T95 = {
    19: 2.093024054408307, 39: 2.022690920036760, 99: 1.984216951508683,
    255: 1.969310569849874, 2047: 1.961123559863062,
}
# Bonferroni over a family of m = 2N two-sided intervals that must hold SIMULTANEOUSLY at 95%.
# Each interval gets alpha = 0.05/m, so a two-sided interval at confidence 1 - 0.05/m needs the
# quantile at level 1 - 0.05/(2m) = 1 - 0.05/(4N).  With N = 40 (m = 80) that is the 0.9996875
# point, t(2047) = 3.425839557.
#
# SUPERSEDED BUG (kept deliberately, and checked against below): an earlier version of this file
# used the quantile at 1 - 0.05/m = 1 - 0.05/80 = 0.999375, giving t(2047) = 3.231723228.  That is
# one factor of two short and made the adjusted intervals too narrow.  It was caught only by an
# external audit, because the self-check compared the table against the same wrong level.
T_BONF_80 = {  # family size m = 2N = 80 intervals; per-interval alpha = 0.05/80
    39: 3.720669712860826, 99: 3.533475944075475,
    255: 3.463598989782222, 2047: 3.425839556652871,
}
T_BONF_SUPERSEDED = {  # the WRONG level, retained so the guard below can detect a regression
    39: 3.479924403320799, 99: 3.322736366563675,
    255: 3.263705552933068, 2047: 3.231723227703862,
}


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _t_cdf(df, t):
    if t <= 0:
        return 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(df, level, lo=0.0, hi=100.0):
    """Inverse t CDF by bisection; used to check the table above rather than to look it up."""
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if _t_cdf(df, m) > level:
            b = m
        else:
            a = m
        if b - a < 1e-15:
            break
    return 0.5 * (a + b)


def check_t_tables(n_cases: int) -> None:
    """Recompute every t used and abort if a hard-coded value is wrong.

    The previous version of this guard compared the table against the SAME wrong level it was built
    from, so it passed while the intervals were too narrow.  It now checks both the correct level and
    the superseded one, so a silent regression back to the old constant is impossible.
    """
    problems = []
    for df, val in T95.items():
        got = t_quantile(df, 0.975)
        if abs(got - val) > 1e-9:
            problems.append(f"T95 df={df}: table {val!r} vs computed {got!r}")
    for df, val in T_BONF_80.items():
        got = t_quantile(df, 1.0 - 0.05 / (4 * n_cases))
        if abs(got - val) > 1e-9:
            problems.append(f"T_BONF_80 df={df}: table {val!r} vs computed {got!r} "
                            f"(level 1 - 0.05/(4N) with N={n_cases})")
    for df, val in T_BONF_SUPERSEDED.items():
        got = t_quantile(df, 1.0 - 0.05 / (2 * n_cases))
        if abs(got - val) > 1e-9:
            problems.append(f"T_BONF_SUPERSEDED df={df}: table {val!r} vs computed {got!r}")
    # guard against the exact regression that happened: using the superseded constants as if correct
    for df, val in T_BONF_SUPERSEDED.items():
        if abs(T_BONF_80[df] - val) < 1e-9:
            problems.append(f"T_BONF_80 df={df} equals the superseded value -- wrong alpha level")
    if problems:
        raise SystemExit("t-table self-check failed:\n  " + "\n  ".join(problems))
    print(f"  t-table self-check passed (N={n_cases} comparisons, family m=2N={2*n_cases} intervals; "
          f"per-interval alpha 0.05/{2*n_cases}; quantile level "
          f"1 - 0.05/{4*n_cases} = {1.0 - 0.05/(4*n_cases):.7f}; "
          f"t(2047) = {T_BONF_80[2047]:.6f})")


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return m, sd / math.sqrt(n), sd


def interval(m, se, t):
    return m - t * se, m + t * se


def run_case(graph, case, world_ids, out_path, log_every=256):
    """Evaluate the four sets of one comparison over ``world_ids``; return paired differences."""
    keys = ("S_ac", "S_bc", "S_ad", "S_bd")
    seeds = {k: case[k] for k in keys}
    xc, xd = [], []
    spreads = {k: [] for k in keys}
    max_depth = 0
    t0 = time.perf_counter()
    for i, wid in enumerate(world_ids):
        live = draw_live_arcs(graph, wid)
        adj = build_live_adjacency(graph, live)
        val = {}
        for k in keys:
            c, d = ic_spread(adj, seeds[k])
            val[k] = c
            spreads[k].append(c)
            if d > max_depth:
                max_depth = d
        xc.append(val["S_ac"] - val["S_bc"])
        xd.append(val["S_ad"] - val["S_bd"])
        if log_every and (i + 1) % log_every == 0:
            print(f"      {i+1}/{len(world_ids)} worlds "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return {"X_c": xc, "X_d": xd, "spreads": spreads, "max_bfs_depth": max_depth,
            "seconds": time.perf_counter() - t0}


def summarize(xs, df_expected, bonf_table):
    m, se, sd = mean_se(xs)
    n = len(xs)
    t_ord = T95.get(n - 1)
    t_bonf = bonf_table.get(n - 1)
    ordinary = interval(m, se, t_ord) if t_ord else (float("nan"), float("nan"))
    adjusted = interval(m, se, t_bonf) if t_bonf else (float("nan"), float("nan"))
    return {
        "n_worlds": n, "mean": m, "sd": sd, "se": se,
        "ci95_ordinary": {"low": ordinary[0], "high": ordinary[1], "t": t_ord},
        "ci95_bonferroni": {"low": adjusted[0], "high": adjusted[1], "t": t_bonf},
        "excludes_zero_ordinary": bool(ordinary[0] > 0 or ordinary[1] < 0),
        "excludes_zero_bonferroni": bool(adjusted[0] > 0 or adjusted[1] < 0),
    }


def classify(sc, sd_):
    """Descriptive label only.  Labels never assert a mathematical identity.

    An earlier version returned ``identical_by_construction`` whenever the two sample means happened
    to be equal.  Equal SAMPLE means do not establish that the two quantities are identically equal --
    they are one draw of a statistic, and the underlying per-world values can and do differ.  The only
    thing that may be reported is what was observed: both adjusted intervals exclude zero on the same
    side, on opposite sides, or not both exclude zero.
    """
    c_neg = sc["ci95_bonferroni"]["high"] < 0
    c_pos = sc["ci95_bonferroni"]["low"] > 0
    d_neg = sd_["ci95_bonferroni"]["high"] < 0
    d_pos = sd_["ci95_bonferroni"]["low"] > 0
    if (c_neg and d_pos) or (c_pos and d_neg):
        return "confirmed_reversal"
    if (c_neg or c_pos) and (d_neg or d_pos):
        return "both_significant_same_side"
    if sc["mean"] * sd_["mean"] < 0 and (c_neg or c_pos or d_neg or d_pos):
        return "one_sided_with_opposite_point_estimates"
    return "not_significant"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=HERE / "N01_cases.json")
    ap.add_argument("--dev", type=int, default=256)
    ap.add_argument("--confirm", type=int, default=2048)
    ap.add_argument("--out-prefix", type=Path, default=HERE / "N01")
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()

    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    cases_art = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = cases_art["cases"]
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    n_cases = len(cases)
    print(f"  n={graph.n} m={graph.m} arcs={graph.n_arcs} | {n_cases} comparisons")
    check_t_tables(n_cases)
    print(f"  dev worlds {len(DEV_WORLD_IDS[:args.dev])}, "
          f"confirm worlds {len(CONFIRM_WORLD_IDS[:args.confirm])}")
    print(f"  world-seed ranges disjoint: "
          f"{set(DEV_WORLD_IDS[:args.dev]) & set(CONFIRM_WORLD_IDS[:args.confirm]) == set()}")

    dev_ids = list(DEV_WORLD_IDS[:args.dev])
    conf_ids = list(CONFIRM_WORLD_IDS[:args.confirm])

    all_rows = []
    per_case = []
    depth_dev, depth_conf = [], []
    for i, case in enumerate(cases):
        print(f"\n  [{i+1}/{n_cases}] group={case['group']} "
              f"a={case['a']} b={case['b']} c={case['c']} d={case['d']}", flush=True)
        print("    dev batch", flush=True)
        dev = run_case(graph, case, dev_ids, None)
        print(f"    dev done in {dev['seconds']:.1f}s; max BFS depth {dev['max_bfs_depth']}; "
              f"mean X_c={sum(dev['X_c'])/len(dev['X_c']):+.4f} "
              f"mean X_d={sum(dev['X_d'])/len(dev['X_d']):+.4f}", flush=True)
        print("    confirm batch", flush=True)
        conf = run_case(graph, case, conf_ids, None)
        print(f"    confirm done in {conf['seconds']:.1f}s; max BFS depth {conf['max_bfs_depth']}; "
              f"mean X_c={sum(conf['X_c'])/len(conf['X_c']):+.4f} "
              f"mean X_d={sum(conf['X_d'])/len(conf['X_d']):+.4f}", flush=True)

        depth_dev.append(dev["max_bfs_depth"])
        depth_conf.append(conf["max_bfs_depth"])

        sc = summarize(conf["X_c"], len(conf_ids) - 1, T_BONF_80)
        sd_ = summarize(conf["X_d"], len(conf_ids) - 1, T_BONF_80)
        label = classify(sc, sd_)
        m = min(abs(sc["mean"]), abs(sd_["mean"]))

        sc_dev = summarize(dev["X_c"], len(dev_ids) - 1, T_BONF_80)
        sd_dev = summarize(dev["X_d"], len(dev_ids) - 1, T_BONF_80)

        row = {
            "index": i, "group": case["group"],
            "a": case["a"], "b": case["b"], "c": case["c"], "d": case["d"],
            "U": case["U"],
            "degree_a": case["degrees"]["a"], "degree_b": case["degrees"]["b"],
            "degree_c": case["degrees"]["c"], "degree_d": case["degrees"]["d"],
            "delta_c_mean": sc["mean"], "delta_c_se": sc["se"],
            "delta_c_ci_low": sc["ci95_ordinary"]["low"],
            "delta_c_ci_high": sc["ci95_ordinary"]["high"],
            "delta_c_bonf_low": sc["ci95_bonferroni"]["low"],
            "delta_c_bonf_high": sc["ci95_bonferroni"]["high"],
            "delta_d_mean": sd_["mean"], "delta_d_se": sd_["se"],
            "delta_d_ci_low": sd_["ci95_ordinary"]["low"],
            "delta_d_ci_high": sd_["ci95_ordinary"]["high"],
            "delta_d_bonf_low": sd_["ci95_bonferroni"]["low"],
            "delta_d_bonf_high": sd_["ci95_bonferroni"]["high"],
            "reversal": label,
            "m": m, "m_over_n": m / graph.n,
            "delta_c_mean_dev": sc_dev["mean"], "delta_d_mean_dev": sd_dev["mean"],
            "sign_agrees_dev_confirm_c": bool(sc_dev["mean"] * sc["mean"] > 0),
            "sign_agrees_dev_confirm_d": bool(sd_dev["mean"] * sd_["mean"] > 0),
        }
        all_rows.append(row)
        per_case.append({
            "index": i, "group": case["group"],
            # 'spreads' holds the four per-set activation counts for every world, so the set-algebra
            # decomposition (A, B, C, D and the background W = R(U)) can be verified afterwards without
            # re-simulating.  The first N01 export omitted these; that omission is why a later audit
            # could not check the dominant-term claim.
            "dev": {"X_c": dev["X_c"], "X_d": dev["X_d"],
                    "spreads": dev["spreads"], "seeds": {k: case[k] for k in
                                                          ("S_ac", "S_bc", "S_ad", "S_bd")},
                    "summaries": {"delta_c": sc_dev, "delta_d": sd_dev},
                    "max_bfs_depth": dev["max_bfs_depth"]},
            "confirm": {"X_c": conf["X_c"], "X_d": conf["X_d"],
                        "spreads": conf["spreads"],
                        "seeds": {k: case[k] for k in ("S_ac", "S_bc", "S_ad", "S_bd")},
                        "summaries": {"delta_c": sc, "delta_d": sd_},
                        "max_bfs_depth": conf["max_bfs_depth"]},
        })

    # ---------------------------------------------------------------- per-world identity audit
    # The structural condition does NOT force X_c == X_d: it constrains the deterministic two-hop
    # regions R2, while live-edge reachability extends further.  These counters measure how often the
    # two happen to coincide, which is a property of the sampling, not a mathematical identity.
    for row, pc in zip(all_rows, per_case):
        for batch in ("dev", "confirm"):
            xc, xd = pc[batch]["X_c"], pc[batch]["X_d"]
            row[f"{batch}_worlds_with_X_c_eq_X_d"] = sum(1 for u, v in zip(xc, xd) if u == v)
            row[f"{batch}_n_worlds"] = len(xc)
            row[f"{batch}_max_abs_X_c_minus_X_d"] = max((abs(u - v) for u, v in zip(xc, xd)),
                                                        default=0)

    # ---------------------------------------------------------------- per-world compressed store
    out_prefix = args.out_prefix
    world_path = out_prefix.with_name(out_prefix.name + "_worlds.json.gz")
    payload = {
        "world_order": {"dev": dev_ids, "confirm": conf_ids},
        "cases": [{"index": r["index"], "group": r["group"], "a": r["a"], "b": r["b"],
                   "c": r["c"], "d": r["d"]} for r in all_rows],
        "per_case": per_case,
    }
    with gzip.open(world_path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(payload, fh)
    print(f"\n  wrote {world_path} ({world_path.stat().st_size/1e6:.2f} MB)")

    # ---------------------------------------------------------------- results table
    fields = list(all_rows[0].keys())
    import csv
    csv_path = out_prefix.with_name(out_prefix.name + "_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  wrote {csv_path}")

    # ---------------------------------------------------------------- console summary
    print("\n" + "=" * 118)
    print("  CONFIRMATION BATCH -- per comparison (Bonferroni over all "
          f"{2*n_cases} means, each interval at 1 - 0.05/{2*n_cases})")
    print("=" * 118)
    hdr = (f"  {'#':>3} {'group':<10} {'delta_c':>9} {'bonf lo':>9} {'bonf hi':>9} "
           f"{'delta_d':>9} {'bonf lo':>9} {'bonf hi':>9} {'m':>6} {'m/n':>7}  verdict")
    print(hdr)
    for r in all_rows:
        print(f"  {r['index']:>3} {r['group']:<10} {r['delta_c_mean']:>+9.4f} "
              f"{r['delta_c_bonf_low']:>+9.4f} {r['delta_c_bonf_high']:>+9.4f} "
              f"{r['delta_d_mean']:>+9.4f} {r['delta_d_bonf_low']:>+9.4f} "
              f"{r['delta_d_bonf_high']:>+9.4f} {r['m']:>6.3f} {r['m_over_n']:>7.5f}  "
              f"{r['reversal']}")

    from collections import Counter
    counts = Counter(r["reversal"] for r in all_rows)
    print("\n  verdict counts (all):", dict(counts))
    for g in ("uniform_U", "degree_U"):
        sub = [r for r in all_rows if r["group"] == g]
        if not sub:
            print(f"    {g:<10} (no comparisons in this run)")
            continue
        print(f"    {g:<10} n={len(sub):<3} mean delta_c "
              f"{sum(r['delta_c_mean'] for r in sub)/len(sub):+.4f}  mean delta_d "
              f"{sum(r['delta_d_mean'] for r in sub)/len(sub):+.4f}  "
              f"reversals {sum(1 for r in sub if r['reversal']=='confirmed_reversal')}")
    agree_c = sum(1 for r in all_rows if r["sign_agrees_dev_confirm_c"])
    agree_d = sum(1 for r in all_rows if r["sign_agrees_dev_confirm_d"])
    print(f"\n  dev/confirm sign agreement: delta_c {agree_c}/{n_cases}, delta_d {agree_d}/{n_cases}")
    print(f"  max BFS depth: dev {max(depth_dev)}, confirm {max(depth_conf)} (rounds cap 100)")

    meta_out = {
        "n_cases": n_cases,
        "n_dev_worlds": len(dev_ids), "n_confirm_worlds": len(conf_ids),
        "dev_world_seed_range": [dev_ids[0], dev_ids[-1]],
        "confirm_world_seed_range": [conf_ids[0], conf_ids[-1]],
        "world_seed_overlap": 0,
        "verdict_counts": dict(counts),
        "max_bfs_depth": {"dev": max(depth_dev), "confirm": max(depth_conf)},
        "t_tables": {"T95": T95, "T_BONF_80": T_BONF_80},
        "seconds_total": None,
    }
    (out_prefix.with_name(out_prefix.name + "_run_meta.json")).write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
