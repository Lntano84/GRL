#!/usr/bin/env python3
"""Summarize runtime rating-entry and coarsening pass audit events."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from gp01_common import EVAL_SEEDS, GRAPHS, K_VALUES, METHODS


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "work/gp01/outputs"
    runs = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    rows = []
    for graph in GRAPHS:
        for k in K_VALUES:
            for method in METHODS:
                group = [r for r in runs if r["graph"] == graph and r["k"] == k and r["method"] == method]
                pass_counts = [len(r["coarsening_pass_events"]) for r in group]
                context_counts = [len({e["context"] for e in r["coarsening_pass_events"]}) for r in group]
                max_pass = [max((int(e["pass"]) for e in r["coarsening_pass_events"]), default=-1) for r in group]
                rows.append({"graph": graph, "k": k, "method": method, "runs": len(group),
                             "rating_entry_calls": sum(len(r["rating_entry_events"]) for r in group),
                             "runs_with_rating_entry": sum(bool(r["rating_entry_events"]) for r in group),
                             "pass_events_total": sum(pass_counts),
                             "mean_pass_events_per_run": f"{sum(pass_counts)/len(pass_counts):.3f}",
                             "mean_contexts_per_run": f"{sum(context_counts)/len(context_counts):.3f}",
                             "max_pass_index": max(max_pass)})
    with (out / "scoring_coarsening_audit.csv").open("wt", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    interface = json.loads((root / "work/gp01/runs/interface-audit/interface_audit_results.json").read_text(encoding="utf-8"))
    interface_rows = []
    for pair in interface["pairs"]:
        interface_rows.append({"graph": pair["graph"], "k": pair["k"], "seed": pair["seed"],
                               "calls": 2,
                               "model_scoring_calls": len(pair["model"]["rating_entry_events"]),
                               "replay_scoring_calls": len(pair["replay"]["rating_entry_events"]),
                               "model_pass_events": len(pair["model"]["coarsening_pass_events"]),
                               "replay_pass_events": len(pair["replay"]["coarsening_pass_events"]),
                               "same_partition_hash": pair["partition_hash_equal"],
                               "same_cut": pair["cut_equal"]})
    with (out / "interface_audit_summary.csv").open("wt", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(interface_rows[0]))
        writer.writeheader()
        writer.writerows(interface_rows)
    model_calls = sum(r["rating_entry_calls"] for r in rows)
    base_calls = sum(r["runs"] for r in rows if r["method"] == "BASE")
    print(json.dumps({"evaluation_rating_entry_calls": model_calls,
                      "base_runs_without_guided_rating": base_calls,
                      "evaluation_runs": sum(r["runs"] for r in rows),
                      "interface_pairs": len(interface_rows)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
