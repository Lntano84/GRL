"""Three-arm scan-order pilot: can cheap structural information improve the real search trajectory?

Frozen for all three arms: graph, K=4, T=8, the five pre-fixed search seeds, initialisation, restart
rule, acceptance rule (true sigma), caching, tie-break policy, and a 5,000 sigma-query budget that
counts initialisation and restarts.  The only thing that differs is the order candidates are examined
in: ``random`` (the original), ``degree``, and ``threshold`` (the one-round F1 proxy).

What is reported per checkpoint (100 / 300 / 1000 / 3000 / 5000):
  the current best set's true sigma/|V|, the query requests, the cache misses, and the cumulative wall
  time including candidate scoring and sorting -- per run and averaged over the five seeds.

What is deliberately NOT done: no candidate is evaluated to fill in top-k labels.  A scan that stops at
the first improvement never asked about the candidates behind it, so the log cannot say what fraction
of the candidate set would have improved.

Decision rule, fixed before the run:
  B or C reaches similar or better quality with FEWER queries AND less time
      -> cheap structure already helps; a future learner must beat the stronger of these, not just
         random scanning
  queries fall but time does not improve
      -> the ordering has real computational cost; judge the actual trade-off, do not announce a speed-up
  neither B nor C clearly improves
      -> no positive evidence for learning; first check whether the score fails to separate candidates,
         or whether the new order walks into worse local optima.  Do not reach for a model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.model import DEFAULT_K, DEFAULT_T, load_graph
from ccim.scan_order import METHODS, ordered_search

GRAPHS = ("football", "polbooks")
SEEDS = (0, 1, 2, 3, 4)
BUDGET = 5_000
CHECKPOINTS = (100, 300, 1_000, 3_000, 5_000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "scan_order_pilot.json")
    args = parser.parse_args()

    artifact = {
        "script": Path(__file__).name,
        "status": "zero-training scan-order pilot; no model is fitted anywhere in this run",
        "setting": {"graphs": list(args.graphs), "K": DEFAULT_K, "T": DEFAULT_T,
                    "budget": args.budget, "seeds": list(args.seeds),
                    "checkpoints": list(CHECKPOINTS)},
        "frozen": ["graph", "K", "T", "seeds", "initialisation", "restart rule",
                   "acceptance rule (true sigma)", "caching", "tie-break policy", "budget"],
        "manipulated": "the order in which candidate swaps are examined",
        "methods": {
            "random": "uniformly random scan order (the original search)",
            "degree": "degree(v) - degree(u) descending",
            "threshold": "q(u,v|S) = F1(S') - F1(S) descending, with "
                         "F1(S) = |S| + |{x not in S : |N(x) cap S| >= K}|, one round only",
        },
        "not_done": ["no candidate is evaluated just to fill in top-k labels; a scan that stops at the "
                     "first improvement never asked about the candidates behind it",
                     "old logs were not re-sorted and presented as a new result",
                     "the 36,369 tier is not run"],
        "decision_rule": {
            "cheaper_queries_and_time_similar_or_better":
                "cheap structure already helps; a future learner must beat the stronger of B/C",
            "queries_fall_time_does_not":
                "ordering has computational cost; judge the trade-off, do not announce a speed-up",
            "neither_improves":
                "no positive evidence for learning; first check whether the score cannot separate "
                "candidates or whether the order walks into worse local optima",
        },
        "graphs": {},
    }

    for name in args.graphs:
        g = load_graph(name)
        n = g.number_of_nodes()
        print(f"\n===== {name} n={n} K={DEFAULT_K} T={DEFAULT_T} budget={args.budget} =====",
              flush=True)
        entry = {"n": n, "m": g.number_of_edges(), "arms": {}}
        for method in args.methods:
            runs = []
            for seed in args.seeds:
                t0 = time.perf_counter()
                res = ordered_search(g, method=method, budget=args.budget, seed=seed,
                                     checkpoints=CHECKPOINTS)
                wall = time.perf_counter() - t0
                runs.append({
                    "method": method, "seed": seed,
                    "final_sigma": res.sigma, "final_normalized": res.normalized,
                    "budget_used": res.cost["decision_evaluations"],
                    "distinct_queries": res.cost["distinct_queries"],
                    "cache_hits": res.cost["cached_repeats"],
                    "wall_seconds": wall,
                    "checkpoint_curve": {str(c["checkpoint"]): {
                        "best_normalized": c["best_normalized"],
                        "queries": c["queries"],
                        "distinct_queries": c["distinct_queries"],
                        "cumulative_seconds": c["cumulative_seconds"],
                        "reached": c["reached"]} for c in res.extra["checkpoint_curve"]},
                    "scan_summary": res.extra["scan_summary"],
                    "scan_log": res.extra["scan_log"],
                })
                tail = "  ".join(
                    f"@{c['checkpoint']}="
                    + (f"{c['best_normalized']:.4f}" if c["best_normalized"] is not None else "n/a")
                    for c in res.extra["checkpoint_curve"])
                print(f"  {method:<9} seed {seed}: {tail}  final {res.normalized:.4f}  "
                      f"wall {wall:.2f}s  scans {res.extra['scan_summary']['scans']} "
                      f"(exhausted {res.extra['scan_summary']['scans_exhausted']})", flush=True)
            entry["arms"][method] = {"runs": runs}
            print(f"  -> {method}: final mean "
                  f"{statistics.fmean(r['final_normalized'] for r in runs):.4f}", flush=True)
        artifact["graphs"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ per-run table
    print("\n" + "=" * 116)
    print("  PER-RUN RESULTS")
    print("=" * 116)
    for name, entry in artifact["graphs"].items():
        print(f"\n  {name}")
        print(f"    {'method':<10}{'seed':>5}" + "".join(f"{'@' + str(c):>11}" for c in CHECKPOINTS)
              + f"{'queries':>9}{'misses':>8}{'wall_s':>9}")
        for method, arm in entry["arms"].items():
            for r in arm["runs"]:
                cells = []
                for c in CHECKPOINTS:
                    v = r["checkpoint_curve"][str(c)]["best_normalized"]
                    cells.append(f"{v:>11.4f}" if v is not None else f"{'n/a':>11}")
                print(f"    {method:<10}{r['seed']:>5}" + "".join(cells)
                      + f"{r['budget_used']:>9}{r['distinct_queries']:>8}"
                      + f"{r['wall_seconds']:>9.2f}")

    # ------------------------------------------------------------------ means
    print("\n" + "=" * 116)
    print("  FIVE-SEED MEANS")
    print("=" * 116)
    print(f"  {'graph':<10}{'method':<10}" + "".join(f"{'@' + str(c):>11}" for c in CHECKPOINTS)
          + f"{'final':>10}{'misses':>9}{'wall_s':>9}")
    summary = {}
    for name, entry in artifact["graphs"].items():
        summary[name] = {}
        for method, arm in entry["arms"].items():
            row = {}
            for c in CHECKPOINTS:
                vals = [r["checkpoint_curve"][str(c)]["best_normalized"] for r in arm["runs"]
                        if r["checkpoint_curve"][str(c)]["best_normalized"] is not None]
                row[f"@{c}"] = statistics.fmean(vals) if vals else None
            row["final"] = statistics.fmean(r["final_normalized"] for r in arm["runs"])
            row["distinct"] = statistics.fmean(r["distinct_queries"] for r in arm["runs"])
            row["wall"] = statistics.fmean(r["wall_seconds"] for r in arm["runs"])
            row["scoring_seconds"] = statistics.fmean(
                r["scan_summary"]["scoring_seconds_total"] for r in arm["runs"])
            row["scans"] = statistics.fmean(r["scan_summary"]["scans"] for r in arm["runs"])
            row["scans_exhausted"] = statistics.fmean(
                r["scan_summary"]["scans_exhausted"] for r in arm["runs"])
            summary[name][method] = row
            cells = []
            for c in CHECKPOINTS:
                cells.append(f"{row['@' + str(c)]:>11.4f}" if row[f"@{c}"] is not None
                             else f"{'n/a':>11}")
            print(f"  {name:<10}{method:<10}" + "".join(cells)
                  + f"{row['final']:>10.4f}{row['distinct']:>9.0f}{row['wall']:>9.2f}")
    artifact["summary"] = summary

    # ------------------------------------------------------------------ scan behaviour
    print("\n" + "=" * 116)
    print("  SCAN BEHAVIOUR (five-seed means)")
    print("=" * 116)
    print(f"  {'graph':<10}{'method':<10}{'scans':>8}{'exhausted':>11}{'score_s':>10}"
          f"{'wall_s':>9}{'scoring share':>15}")
    for name, entry in artifact["graphs"].items():
        for method, row in summary[name].items():
            share = row["scoring_seconds"] / row["wall"] if row["wall"] else 0
            print(f"  {name:<10}{method:<10}{row['scans']:>8.1f}{row['scans_exhausted']:>11.1f}"
                  f"{row['scoring_seconds']:>10.3f}{row['wall']:>9.2f}{share:>14.1%}")
    print("\n  first-improvement positions within a scan are recorded per scan in the artifact;")
    print("  candidates behind the first improvement were never evaluated and are not inferred.")

    # ------------------------------------------------------------------ verdict check
    print("\n" + "=" * 116)
    print("  AGAINST THE PRE-FIXED DECISION RULE")
    print("=" * 116)
    artifact["verdict"] = {}
    for name in artifact["graphs"]:
        base = summary[name]["random"]
        for method in ("degree", "threshold"):
            arm = summary[name][method]
            dq = arm["distinct"] - base["distinct"]
            dt = arm["wall"] - base["wall"]
            dquality = arm["final"] - base["final"]
            artifact["verdict"].setdefault(name, {})[method] = {
                "final_minus_random": dquality, "distinct_queries_minus_random": dq,
                "wall_seconds_minus_random": dt}
            print(f"    {name:<10}{method:<10} quality {dquality:+.4f}  "
                  f"distinct queries {dq:+.0f}  wall {dt:+.2f}s")
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
