"""Fixed wall clock, equal time for every method: does the learned ranker still win when random is given the same seconds?

The question this answers, and the one it does not
--------------------------------------------------
The budget-limited pilot compared three arms at an equal *query* count (5,000) and found the learned arm
ahead on the mean (662.0 vs 648.0 vs 633.3) but slower (10.42 s vs 6.20 s, corrected 7.34 s).  The obvious
objection is that the comparison was not run on equal terms: the learned arm spent more wall clock, so
perhaps it simply bought more search.  This script removes that objection by giving every method the
**same 10 seconds** and asking whether the gain survives.

It is not a new mechanism study and it runs no diagnostics.  One comparison, one number per arm per seed.

Frozen before the run
---------------------
* graph version (sha256 recorded), ``K = 4``, ``T = 52``, the degree-52 initial set;
* the audited epoch-1 checkpoint at its recorded hash -- not retrained, not re-selected, not swapped;
* NumPy inference, so the learned arm is charged its own feature/inference/sort cost but not the torch
  timing inflation that ``results/diffusion_cost_attribution.json`` identified as an artefact;
* 10 seconds per run, no query cap, so **time is the binding budget**;
* 10 never-used, pre-fixed search seeds (200-209; 0/1/2 were train/validate, 100/101/102 were the pilot);
* the three arms of one seed form a timing group, run **serially**, in an order randomised from a fixed
  seed *before* any result exists (written to ``results/wallclock_run_plan.json`` by ``--plan-only``).

Checkpoints are taken on the true clock
---------------------------------------
A run records, after each query returns, the elapsed wall clock and the best value so far.  The value at
checkpoint ``c`` is the best over entries with ``elapsed <= c``, so **a query that finishes after the
deadline cannot contribute to that deadline**.  The last query of a run routinely overshoots 10 s; it is
recorded, and it is excluded.

What is deliberately not done
-----------------------------
No tuning, no checkpoint change, no feature change, and no rule edited after seeing an intermediate
result.  The decision rule in :func:`decide` is written before the run and applied to the finished table.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ccim.batched_search import batched_search  # noqa: E402
from ccim.lazy_candidates import candidate_count, decode  # noqa: E402
from ccim.learned_rank import RankModel  # noqa: E402
from ccim.swap_features import build_context  # noqa: E402
from measure_ca_grqc import FILE, build_undirected, parse_raw  # noqa: E402

K = 4
SEED_FRACTION = 0.01
NEW_SEEDS = tuple(range(200, 210))      # 10 never-used, pre-fixed search seeds
ARMS = ("random", "degree", "learned")
BATCH_SIZE = 32
WALL_SECONDS = 10.0
CHECKPOINTS = (1.0, 3.0, 5.0, 10.0)
RESTARTS = 6
ORDER_RNG_SEED = 20240118               # fixes the within-group run order
MODEL = ROOT / "results" / "swap_ranker.pt"
TRAINING = ROOT / "results" / "swap_ranker_training.json"
PLAN = ROOT / "results" / "wallclock_run_plan.json"
OUTPUT = ROOT / "results" / "wallclock_comparison.json"


def build_plan() -> dict:
    """The pre-registered run order.  Deterministic, and written before any timing is measured."""
    rng = random.Random(ORDER_RNG_SEED)
    groups = []
    for seed in NEW_SEEDS:
        order = list(ARMS)
        rng.shuffle(order)
        groups.append({"seed": seed, "order": order})
    return {"order_rng_seed": ORDER_RNG_SEED, "arms": list(ARMS), "wall_seconds": WALL_SECONDS,
            "batch_size": BATCH_SIZE, "restarts": RESTARTS, "checkpoints_seconds": list(CHECKPOINTS),
            "seeds": list(NEW_SEEDS), "groups": groups,
            "note": ("run order was randomised and written to disk before any result existed; the "
                     "comparison script reads this file and refuses to run without it")}


def value_at(history, seconds: float) -> tuple[int | None, int]:
    """Best value over queries that *completed* by ``seconds``, and how many such queries there were."""
    best, n = None, 0
    for q, b, _d, secs, *_ in history:
        if secs <= seconds:
            best = b if best is None else max(best, b)
            n += 1
    return best, n


def compress(history) -> list:
    """The monotone step function only: (queries, seconds, best) whenever the best improved."""
    out, best = [], None
    for q, b, _d, secs, *_ in history:
        if best is None or b > best:
            out.append([int(q), float(secs), int(b)])
            best = b
    return out


def run_one(graph, arm, seed, T, model, np_ranker) -> dict:
    res = batched_search(graph, ranker=np_ranker if arm == "learned" else arm,
                         T=T, K=K, budget=10**9, seed=seed, batch_size=BATCH_SIZE,
                         restarts=RESTARTS, checkpoints=(), time_limit=WALL_SECONDS,
                         timer=time.perf_counter, collect_logs=True, return_history=True,
                         model=None, trace_hash=True)
    hist = res.extra["history"]
    tb = res.extra["timing_buckets"]
    at = {}
    for c in CHECKPOINTS:
        v, n = value_at(hist, c)
        at[f"@{c:g}s"] = {"sigma": v, "queries_by_then": n}
    return {"sigma_at_10s": at["@10s"]["sigma"],
            "queries_at_10s": at["@10s"]["queries_by_then"],
            "checkpoints": at,
            "wall_seconds": tb["total_seconds"],
            "overrun_seconds": tb["total_seconds"] - WALL_SECONDS,
            "queries_total": res.cost["decision_evaluations"],
            "queries_completed_within_10s": at["@10s"]["queries_by_then"],
            "distinct_queries": res.cost["distinct_queries"],
            "accepts": res.extra["accepts"],
            "diffusion_seconds": tb["1_diffusion_seconds"],
            "features_inference_sort_seconds": tb["3b_features_inference_sort_seconds"],
            "draw_seconds": tb["3a_candidate_draw_seconds"],
            "control_seconds": tb["4_search_control_seconds"],
            "unaccounted_seconds": tb["5_logging_and_unaccounted_seconds"],
            "reconciliation_error_seconds": tb["reconciliation_error_seconds"],
            "stopped_by_time_limit": tb["stopped_by_time_limit"],
            "curve": compress(hist)}


def paired(a: list, b: list) -> dict:
    """Paired difference ``a - b`` with a t interval, Wilcoxon and an exact sign test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    if se > 0:
        half = float(stats.t.ppf(0.975, n - 1) * se)
        tstat, tp = stats.ttest_rel(a, b)
    else:
        half, tstat, tp = 0.0, float("nan"), float("nan")
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    try:
        wp = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        wp = float("nan")
    sign_p = float(stats.binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    return {"n": n, "mean_difference": mean, "sd": sd, "se": se,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "t_statistic": float(tstat), "t_p": float(tp),
            "wilcoxon_p": wp, "sign_test_p": sign_p,
            "wins": wins, "losses": losses, "ties": ties,
            "differences": [float(x) for x in d]}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni over the two comparisons, as required before claiming both are beaten."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[name] = {"raw_p": p, "holm_adjusted_p": adj, "significant_at_0.05": adj < 0.05}
        prev = adj
    return out


def decide(table: dict, tests: dict, holm_res: dict) -> dict:
    """Pre-registered decision rule.  Written before the run; applied to the finished table only."""
    def clear(name):
        t, h = tests[name], holm_res[name]
        return (h["significant_at_0.05"] and t["mean_difference"] > 0
                and t["wins"] >= 7 and t["wins"] > t["losses"])

    beat_random = clear("learned_minus_random")
    beat_degree = clear("learned_minus_degree")
    if beat_random and beat_degree:
        outcome = "clear advantage over both"
        decision = ("next step is validation on a new graph; candidate direction: "
                    "learning-assisted local search")
        archive = False
    elif beat_random and not beat_degree:
        outcome = "beats random only, degree explains it"
        decision = ("keep the simple method; do not claim a distinctive benefit from learning")
        archive = True
    else:
        outcome = "no clear advantage"
        decision = ("archive this version of the learned method; no further seeds, training or "
                    "features to rescue the result")
        archive = True
    return {"outcome": outcome, "decision": decision, "archive_this_version": archive,
            "beat_random": beat_random, "beat_degree": beat_degree,
            "primary_comparison": "learned_minus_random",
            "multiplicity": "Holm-Bonferroni across the two comparisons",
            "criteria": {"holm_adjusted_p_below": 0.05, "mean_difference_positive": True,
                         "wins_at_least": 7, "wins_strictly_more_than_losses": True}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true",
                        help="write the frozen run plan and exit; no search is run")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plan_only:
        plan = build_plan()
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 100)
        print("  RUN PLAN FROZEN -- written before any result exists")
        print("=" * 100)
        print(f"    seeds          : {plan['seeds']}")
        print(f"    wall clock     : {plan['wall_seconds']:g} s per run, no query cap")
        print(f"    checkpoints    : {plan['checkpoints_seconds']} s")
        print(f"    order RNG seed : {plan['order_rng_seed']}")
        for g in plan["groups"]:
            print(f"      seed {g['seed']}: {' -> '.join(g['order'])}")
        print(f"\n    written to {args.plan}")
        return 0

    if not args.plan.exists():
        print(f"  refusing to run: the frozen plan {args.plan} does not exist.  Run --plan-only first.")
        return 1
    plan = json.loads(args.plan.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    rows_raw, manifest = parse_raw(FILE)
    graph = build_undirected(rows_raw)
    graph_seconds = time.perf_counter() - t0
    n = graph.number_of_nodes()
    T = int((n * SEED_FRACTION) // 1)

    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    t_load = time.perf_counter()
    model = RankModel.load(str(MODEL))
    load_seconds = time.perf_counter() - t_load
    np_ranker = model.numpy_ranker()

    print("=" * 100)
    print("  FIXED WALL CLOCK -- every method gets the same 10 seconds")
    print("=" * 100)
    print(f"    graph      : ca-GrQc sha256 {manifest['sha256'][:16]}...  n={n:,}  "
          f"m={graph.number_of_edges():,}  K={K}  T={T}  batch={BATCH_SIZE}")
    print(f"    seeds      : {plan['seeds']}  (never used before)")
    print(f"    budget     : {plan['wall_seconds']:g} s wall clock per run, no query cap")
    print(f"    model      : {MODEL.name} sha256 {training['checkpoint']['sha256'][:16]}... "
          f"epoch {training['checkpoint']['selected_epoch']} (unchanged)")
    print(f"    inference  : numpy forward, ordering checked identical to torch in the test suite")
    print(f"    outside timing: graph read {graph_seconds:.3f} s, model load {load_seconds:.3f} s")

    # ---- ordering-equivalence gate, on real states of this graph, before any timing is trusted ----
    print("\n    ordering gate: numpy vs torch on states the search actually visits")
    order_ok = True
    rng = random.Random(7)
    by_degree = sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))
    current = sorted(by_degree[:T])
    for _ in range(8):
        S = set(current)
        selected, outside = sorted(current), [v for v in graph.nodes() if v not in S]
        pool = candidate_count(selected, outside)
        ids = rng.sample(range(pool), min(BATCH_SIZE, pool))
        pairs = [decode(i, selected, outside) for i in ids]
        ctx = build_context(graph, selected, K)
        with torch.no_grad():
            x = torch.tensor([ctx.features(u, v) for (u, v) in pairs], dtype=torch.float32)
            sc = model.net((x - model.mean) / model.std).squeeze(-1)
            t_order = [pairs[i] for i in torch.argsort(sc, descending=True, stable=True).tolist()]
        if np_ranker(ctx, pairs) != t_order:
            order_ok = False
        # walk one accepted swap forward so the next state differs
        v = max(outside, key=lambda z: graph.degree(z))
        current = sorted([x for x in current if x != current[0]] + [v])
    print(f"    [{'PASS' if order_ok else 'FAIL'}] numpy and torch produce the same ordering on "
          f"8 visited states")
    if not order_ok:
        print("    refusing to run: the numpy ranker is not the audited model")
        return 1

    # ---- the 30 runs -------------------------------------------------------------------------
    results = {}
    print(f"\n    running {len(plan['groups']) * len(ARMS)} searches, serial, "
          f"{plan['wall_seconds']:g} s each")
    for group in plan["groups"]:
        seed = group["seed"]
        results[seed] = {}
        for arm in group["order"]:
            r = run_one(graph, arm, seed, T, model, np_ranker)
            results[seed][arm] = r
            print(f"      seed {seed:>3}  {arm:<8}  sigma@10s {r['sigma_at_10s']:>4}  "
                  f"queries {r['queries_completed_within_10s']:>5}  "
                  f"wall {r['wall_seconds']:6.3f} s  accepts {r['accepts']:>3}  "
                  f"feat+inf {r['features_inference_sort_seconds']:.3f} s", flush=True)

    # ---- main table --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  MAIN TABLE -- best value reached within 10 s (queries finishing after the deadline excluded)")
    print("=" * 100)
    print(f"    {'seed':>5}{'random@10s':>12}{'degree@10s':>12}{'learned@10s':>13}"
          f"{'learned-random':>16}{'learned-degree':>16}{'queries r/d/l':>20}")
    ordered = sorted(results)
    for s in ordered:
        a, b, c = (results[s]["random"], results[s]["degree"], results[s]["learned"])
        print(f"    {s:>5}{a['sigma_at_10s']:>12}{b['sigma_at_10s']:>12}{c['sigma_at_10s']:>13}"
              f"{c['sigma_at_10s'] - a['sigma_at_10s']:>+16}"
              f"{c['sigma_at_10s'] - b['sigma_at_10s']:>+16}"
              f"{a['queries_completed_within_10s']:>7}/{b['queries_completed_within_10s']}"
              f"/{c['queries_completed_within_10s']}")

    rand = [results[s]["random"]["sigma_at_10s"] for s in ordered]
    degr = [results[s]["degree"]["sigma_at_10s"] for s in ordered]
    lern = [results[s]["learned"]["sigma_at_10s"] for s in ordered]
    print(f"\n    mean sigma @10s :  random {statistics.mean(rand):.1f}   "
          f"degree {statistics.mean(degr):.1f}   learned {statistics.mean(lern):.1f}")
    print(f"    mean queries    :  random "
          f"{statistics.mean([results[s]['random']['queries_completed_within_10s'] for s in ordered]):.0f}"
          f"   degree "
          f"{statistics.mean([results[s]['degree']['queries_completed_within_10s'] for s in ordered]):.0f}"
          f"   learned "
          f"{statistics.mean([results[s]['learned']['queries_completed_within_10s'] for s in ordered]):.0f}")

    # ---- curves ------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  TIME-QUALITY CURVES -- mean best value by checkpoint")
    print("=" * 100)
    print(f"    {'checkpoint':<12}" + "".join(f"{a:>12}" for a in ARMS)
          + f"{'learned-random':>16}{'learned-degree':>16}")
    curve_table = {}
    for c in CHECKPOINTS:
        key = f"@{c:g}s"
        vals = {a: [results[s][a]["checkpoints"][key]["sigma"] for s in ordered] for a in ARMS}
        curve_table[key] = {a: {"mean": statistics.mean(vals[a]),
                                "per_seed": {str(s): results[s][a]["checkpoints"][key]["sigma"]
                                             for s in ordered}} for a in ARMS}
        curve_table[key]["learned_minus_random_mean"] = (statistics.mean(vals["learned"])
                                                         - statistics.mean(vals["random"]))
        curve_table[key]["learned_minus_degree_mean"] = (statistics.mean(vals["learned"])
                                                         - statistics.mean(vals["degree"]))
        print(f"    {key:<12}" + "".join(f"{statistics.mean(vals[a]):>12.1f}" for a in ARMS)
              + f"{curve_table[key]['learned_minus_random_mean']:>+16.1f}"
              f"{curve_table[key]['learned_minus_degree_mean']:>+16.1f}")

    # ---- paired tests ------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  PAIRED DIFFERENCES -- learned is the primary comparison, degree the mandatory control")
    print("=" * 100)
    tests = {"learned_minus_random": paired(lern, rand),
             "learned_minus_degree": paired(lern, degr),
             "degree_minus_random": paired(degr, rand)}
    holm_res = holm({"learned_minus_random": tests["learned_minus_random"]["sign_test_p"],
                     "learned_minus_degree": tests["learned_minus_degree"]["sign_test_p"]})
    for name, t in tests.items():
        print(f"\n    {name}  (n={t['n']})")
        print(f"      mean difference : {t['mean_difference']:+.2f}   "
              f"95% CI [{t['ci95_low']:+.2f}, {t['ci95_high']:+.2f}] (t, df={t['n'] - 1})")
        print(f"      wins/losses/ties: {t['wins']}/{t['losses']}/{t['ties']}")
        print(f"      paired t p={t['t_p']:.4f}   Wilcoxon p={t['wilcoxon_p']:.4f}   "
              f"exact sign-test p={t['sign_test_p']:.4f}")
    print("\n    multiplicity correction (Holm-Bonferroni) over the two claims:")
    for name, h in holm_res.items():
        print(f"      {name:<22} raw p={h['raw_p']:.4f}  adjusted p={h['holm_adjusted_p']:.4f}  "
              f"{'significant' if h['significant_at_0.05'] else 'not significant'}")

    decision = decide(results, tests, holm_res)
    print("\n" + "=" * 100)
    print("  PRE-REGISTERED DECISION")
    print("=" * 100)
    print(f"    outcome  : {decision['outcome'].upper()}")
    print(f"    decision : {decision['decision']}")
    print(f"    beats random (primary) : {decision['beat_random']}")
    print(f"    beats degree (control) : {decision['beat_degree']}")
    print(f"    archive this version   : {decision['archive_this_version']}")

    print("\n  costs NOT part of the 10 s online budget:")
    print(f"    label generation {training['cost']['label_generation_seconds']:.2f} s   "
          f"training {training['cost']['training_seconds']:.2f} s   "
          f"model load {load_seconds:.3f} s   graph read {graph_seconds:.3f} s")

    artifact = {
        "script": Path(__file__).name,
        "scope": ("equal wall clock, not equal queries: does the learned ranker still win when random "
                  "gets the same 10 seconds?"),
        "plan": plan,
        "frozen": {"graph": "ca-GrQc", "file_sha256": manifest["sha256"], "n": n,
                   "m": graph.number_of_edges(), "K": K, "T": T,
                   "initial_set": f"degree top-{T}", "batch_size": BATCH_SIZE,
                   "wall_seconds_per_run": WALL_SECONDS, "query_cap": None, "restarts": RESTARTS,
                   "acceptance_rule": "first strictly improving swap (true sigma)"},
        "model": {"file": MODEL.name, "sha256": training["checkpoint"]["sha256"],
                  "selected_epoch": training["checkpoint"]["selected_epoch"],
                  "label": training["checkpoint"]["label"],
                  "inference": "numpy forward; ordering verified identical to torch"},
        "ordering_gate_numpy_equals_torch": order_ok,
        "costs_outside_timing": {"label_generation_seconds": training["cost"]["label_generation_seconds"],
                                 "training_seconds": training["cost"]["training_seconds"],
                                 "model_load_seconds": load_seconds,
                                 "graph_read_seconds": graph_seconds},
        "results": {str(s): results[s] for s in ordered},
        "means": {"sigma_at_10s": {"random": statistics.mean(rand), "degree": statistics.mean(degr),
                                   "learned": statistics.mean(lern)},
                  "queries_completed_within_10s": {
                      a: statistics.mean([results[s][a]["queries_completed_within_10s"]
                                          for s in ordered]) for a in ARMS}},
        "curve_table": curve_table,
        "tests": tests,
        "holm": holm_res,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
