"""Collect the search traces the previous logs were missing, and answer A / B / C.

Runs the SAME non-learning local search, on the same two graphs, with the same five search seeds, at
the same query budget.  Nothing is trained.  Three things come out of it:

  A. the quality-vs-query curve at 100 / 300 / 1000 / 3000 / 5000 queries, for all five seeds -- this
     time genuinely recorded as the best set found *within* the first C queries, not inferred;
  B. the query-purpose decomposition per trajectory;
  C. the per-move table a swap ranker would be trained on.

Two regression assertions guard the claim that recording changed nothing: each traced run must
reproduce the value previously recorded at the full budget, and its best-within-5000 must equal the
value previously recorded for a 5000-query run with the same seed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.model import DEFAULT_K, DEFAULT_T, load_graph
from ccim.search import local_search

GRAPHS = ("football", "polbooks")
SEARCH_SEEDS = (0, 1, 2, 3, 4)
CHECKPOINTS = (100, 300, 1_000, 3_000, 5_000)
BUDGET = 36_369

COLUMNS = ["cumulative_queries", "phase", "restart", "S", "u", "v", "value_before",
           "value_after", "delta", "accepted", "duplicate_query", "best_so_far"]
PHASE_CODES = {"init": 0, "1swap": 1, "2swap": 2}
PHASE_NAMES = {v: k for k, v in PHASE_CODES.items()}


def decompose(trace: list[dict]) -> dict:
    """B: what the queries were spent on, and what the proposals looked like."""
    by_phase: dict[str, dict] = {}
    for row in trace:
        phase = row["phase"]
        d = by_phase.setdefault(phase, {"queries": 0, "accepted": 0, "duplicate_queries": 0,
                                        "delta_positive": 0, "delta_zero": 0, "delta_negative": 0})
        d["queries"] += 1
        d["accepted"] += int(bool(row["accepted"]))
        d["duplicate_queries"] += int(bool(row["duplicate_query"]))
        if row["delta"] is None:
            continue
        if row["delta"] > 0:
            d["delta_positive"] += 1
        elif row["delta"] == 0:
            d["delta_zero"] += 1
        else:
            d["delta_negative"] += 1
    proposal = {"queries": 0, "accepted": 0, "delta_positive": 0, "delta_zero": 0,
                "delta_negative": 0, "duplicate_queries": 0}
    for phase in ("1swap", "2swap"):
        d = by_phase.get(phase)
        if not d:
            continue
        for k in proposal:
            proposal[k] += d[k]
    proposal["acceptance_rate"] = (proposal["accepted"] / proposal["queries"]
                                   if proposal["queries"] else None)
    proposal["strictly_positive_rate"] = (proposal["delta_positive"] / proposal["queries"]
                                          if proposal["queries"] else None)
    proposal["non_positive_queries"] = proposal["delta_zero"] + proposal["delta_negative"]
    return {"by_phase": by_phase, "proposal_queries_only": proposal,
            "total_queries": len(trace),
            "note": "zero- and negative-gain proposals are NOT counted as waste: they may be what the "
                    "search needs to find a good solution.  This only sizes the task a ranker faces."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEARCH_SEEDS))
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "search_trace.json")
    parser.add_argument("--trace-output", type=Path,
                        default=ROOT / "results" / "search_trace_rows.json")
    args = parser.parse_args()

    previous = json.loads((ROOT / "results" / "gate2_shaping.json").read_text(encoding="utf-8"))
    artifact = {"script": Path(__file__).name,
                "note": "non-learning search only; no model is trained in this run",
                "budget": args.budget, "checkpoints": list(CHECKPOINTS),
                "columns": COLUMNS, "phase_codes": PHASE_CODES,
                "grouping_constraint":
                    "when splitting this table into train/test, group by (graph, search_seed) or by "
                    "trajectory; rows sharing a current set S are highly dependent and a random row "
                    "split leaks across the boundary",
                "graphs": {}}
    rows_out: dict[str, list] = {}

    for name in args.graphs:
        g = load_graph(name)
        n = g.number_of_nodes()
        print(f"\n===== {name} n={n} budget={args.budget} =====", flush=True)
        per_seed, traces = [], {}
        for seed in args.seeds:
            trace: list[dict] = []
            res = local_search(g, budget=args.budget, seed=seed, trace=trace,
                               checkpoints=CHECKPOINTS)
            # regression: the traced run must reproduce the untraced results on record
            prior = None
            for row in previous["graphs"][name]["search"]:
                for per in row["per_seed"]:
                    if per["search_seed"] == seed and row["budget"] == args.budget:
                        prior = per["normalized"]
            at5 = None
            for row in previous["graphs"][name]["search"]:
                if row["budget"] == 5_000:
                    for per in row["per_seed"]:
                        if per["search_seed"] == seed:
                            at5 = per["normalized"]
            curve = {c["checkpoint"]: c for c in res.extra["checkpoint_curve"]}
            match_full = prior is not None and abs(res.normalized - prior) < 1e-12
            match_5k = at5 is not None and curve[5_000]["best_within_checkpoint"] is not None \
                and abs(curve[5_000]["best_within_checkpoint"] / n - at5) < 1e-12
            per_seed.append({
                "search_seed": seed,
                "final_normalized": res.normalized,
                "final_sigma": res.sigma,
                "queries_used": res.cost["decision_evaluations"],
                "queries_distinct": res.cost["distinct_queries"],
                "cache_hits": res.cost["cached_repeats"],
                "budget_exhausted": res.extra["budget_exhausted"],
                "checkpoint_curve": {str(c["checkpoint"]):
                                     (None if c["best_within_checkpoint"] is None
                                      else c["best_within_checkpoint"] / n)
                                     for c in res.extra["checkpoint_curve"]},
                "checkpoint_reached": {str(c["checkpoint"]): c["reached"]
                                       for c in res.extra["checkpoint_curve"]},
                "matches_previous_full_budget": match_full,
                "matches_previous_5000_run": match_5k,
                "decomposition": decompose(trace),
            })
            traces[seed] = trace
            print(f"  seed {seed}: final {res.normalized:.4f}  used {res.cost['decision_evaluations']}"
                  f"  curve {[('%.4f' % (curve[c]['best_within_checkpoint'] / n))
                              if curve[c]['best_within_checkpoint'] is not None else 'MISSING'
                              for c in CHECKPOINTS]}"
                  f"  regression full={match_full} @5000={match_5k}", flush=True)

        artifact["graphs"][name] = {
            "n": n,
            "per_seed": per_seed,
            "checkpoint_table": {str(c): [r["checkpoint_curve"][str(c)] for r in per_seed]
                                 for c in CHECKPOINTS},
            "all_regressions_pass": all(r["matches_previous_full_budget"]
                                        and r["matches_previous_5000_run"] for r in per_seed),
        }
        rows_out[name] = {
            str(seed): [[r["q"], PHASE_CODES[r["phase"]], r["restart"], r["S"],
                         r["u"] if not isinstance(r["u"], tuple) else list(r["u"]),
                         r["v"] if not isinstance(r["v"], tuple) else list(r["v"]),
                         r["before"], r["after"], r["delta"], int(bool(r["accepted"])),
                         int(bool(r["duplicate_query"])), r["best_so_far"]]
                        for r in traces[seed]]
            for seed in args.seeds
        }

    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    args.trace_output.write_text(json.dumps(rows_out), encoding="utf-8")

    # ------------------------------------------------------------------ A
    print("\n" + "=" * 108)
    print("  A. QUALITY-VS-QUERY CURVE (best set found within the first C queries; genuinely recorded)")
    print("=" * 108)
    for name, entry in artifact["graphs"].items():
        print(f"\n  {name}   (regressions all pass: {entry['all_regressions_pass']})")
        print(f"    {'search_seed':>11} |" + "".join(f"{c:>11}" for c in CHECKPOINTS)
              + f"{'final':>11}")
        for r in entry["per_seed"]:
            cells = []
            for c in CHECKPOINTS:
                v = r["checkpoint_curve"][str(c)]
                cells.append(f"{v:>11.4f}" if v is not None else f"{'MISSING':>11}")
            print(f"    {r['search_seed']:>11} |" + "".join(cells) + f"{r['final_normalized']:>11.4f}")
    print("\n  across the five seeds:")
    print(f"    {'graph':<10}" + "".join(f"{'@' + str(c):>12}" for c in CHECKPOINTS)
          + f"{'final':>12}")
    for name, entry in artifact["graphs"].items():
        cells = []
        for c in CHECKPOINTS:
            vals = [v for v in entry["checkpoint_table"][str(c)] if v is not None]
            cells.append(f"{statistics.fmean(vals):>12.4f}" if vals else f"{'MISSING':>12}")
        finals = [r["final_normalized"] for r in entry["per_seed"]]
        print(f"    {name:<10}" + "".join(cells) + f"{statistics.fmean(finals):>12.4f}")

    # ------------------------------------------------------------------ B
    print("\n" + "=" * 108)
    print("  B. QUERY-PURPOSE DECOMPOSITION")
    print("=" * 108)
    for name, entry in artifact["graphs"].items():
        agg = {}
        for r in entry["per_seed"]:
            for phase, d in r["decomposition"]["by_phase"].items():
                a = agg.setdefault(phase, {k: 0 for k in d})
                for k, v in d.items():
                    a[k] += v
        print(f"\n  {name} (summed over 5 seeds)")
        print(f"    {'phase':<8}{'queries':>9}{'accepted':>10}{'dup queries':>13}"
              f"{'delta>0':>9}{'delta=0':>9}{'delta<0':>9}")
        for phase in ("init", "1swap", "2swap"):
            d = agg.get(phase)
            if not d:
                continue
            print(f"    {phase:<8}{d['queries']:>9}{d['accepted']:>10}{d['duplicate_queries']:>13}"
                  f"{d['delta_positive']:>9}{d['delta_zero']:>9}{d['delta_negative']:>9}")
        prop = {"queries": 0, "accepted": 0, "delta_positive": 0, "delta_zero": 0,
                "delta_negative": 0, "duplicate_queries": 0}
        for phase in ("1swap", "2swap"):
            d = agg.get(phase)
            if d:
                for k in prop:
                    prop[k] += d[k]
        print(f"    proposals only: {prop['queries']} queries, {prop['accepted']} accepted "
              f"({prop['accepted'] / prop['queries']:.4f}), "
              f"delta>0 {prop['delta_positive']} ({prop['delta_positive'] / prop['queries']:.4f}), "
              f"delta=0 {prop['delta_zero']}, delta<0 {prop['delta_negative']}, "
              f"duplicates {prop['duplicate_queries']}")

    # ------------------------------------------------------------------ C
    total_rows = sum(len(v) for graph in rows_out.values() for v in graph.values())
    print("\n" + "=" * 108)
    print("  C. PER-MOVE TABLE")
    print("=" * 108)
    print(f"    rows: {total_rows}")
    print(f"    columns: {COLUMNS}")
    print(f"    written to {args.trace_output}")
    print(f"    grouping constraint recorded in the artifact: split by (graph, search_seed)")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
