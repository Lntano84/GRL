#!/usr/bin/env python3
"""Produce the frozen GP01 cell, comparison, and timing summaries."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from gp01_common import EVAL_SEEDS, GRAPHS, K_VALUES, METHODS


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("wt", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "work/gp01/outputs"
    eval_status = json.loads((out / "evaluation_status.json").read_text(encoding="utf-8"))
    teacher_status = json.loads((out / "teacher_status.json").read_text(encoding="utf-8"))
    if not eval_status.get("all_valid") or not teacher_status.get("all_valid"):
        raise SystemExit("Cannot summarize an incomplete or invalid GP01 matrix")
    runs = json.loads((out / "evaluation_results.json").read_text(encoding="utf-8"))
    key = {(r["graph"], r["k"], r["seed"], r["method"]): r for r in runs}
    expected = {(g, k, s, m) for g in GRAPHS for k in K_VALUES for s in EVAL_SEEDS for m in METHODS}
    if len(key) != 144 or set(key) != expected:
        raise SystemExit(f"Expected 144 unique cells, found {len(key)}")

    cell_values: dict[tuple[str, int, str], dict] = {}
    cell_rows = []
    for graph in GRAPHS:
        for k in K_VALUES:
            row = {"graph": graph, "k": k}
            for method in METHODS:
                vals = [key[(graph, k, seed, method)] for seed in EVAL_SEEDS]
                mean_cut = sum(x["cut"] for x in vals) / len(vals)
                mean_wall = sum(x["wall_seconds"] for x in vals) / len(vals)
                cell_values[(graph, k, method)] = {"mean_cut": mean_cut, "runs": vals,
                                                   "mean_wall_seconds": mean_wall}
                row[method] = f"{mean_cut:.3f}"
                row[f"{method}_seed_cuts"] = ";".join(f"{s}:{key[(graph,k,s,method)]['cut']}" for s in EVAL_SEEDS)
                row[f"{method}_mean_wall_s"] = f"{mean_wall:.4f}"
            cell_rows.append(row)
    write_csv(out / "evaluation_cell_summary.csv", cell_rows)

    comparison_rows = []
    comparison_summary = {}
    for control in ("BASE", "AUTHOR", "CAL-TIE", "CAL-EXACT"):
        improvements = {}
        for graph in GRAPHS:
            for k in K_VALUES:
                c = cell_values[(graph, k, control)]["mean_cut"]
                ck = cell_values[(graph, k, "K")]["mean_cut"]
                improvements[(graph, k)] = (c - ck) / c if c else 0.0
        per_k = {k: sum(improvements[(g, k)] for g in GRAPHS) / len(GRAPHS) for k in K_VALUES}
        per_graph = {g: sum(improvements[(g, k)] for k in K_VALUES) / len(K_VALUES) for g in GRAPHS}
        per_seed = {}
        for seed in EVAL_SEEDS:
            values = []
            for graph in GRAPHS:
                for k in K_VALUES:
                    c = key[(graph, k, seed, control)]["cut"]
                    ck = key[(graph, k, seed, "K")]["cut"]
                    values.append((c - ck) / c if c else 0.0)
            per_seed[seed] = sum(values) / len(values)
        item = {"control": control,
                "overall_equal_weight_gain": sum(improvements.values()) / len(improvements),
                "by_k": per_k, "by_graph": per_graph, "by_seed_overall_direction": per_seed,
                "positive_cells": sum(v > 0 for v in improvements.values()),
                "positive_graphs": sum(v > 0 for v in per_graph.values()),
                "positive_seeds": sum(v > 0 for v in per_seed.values())}
        item["meets_frozen_gate"] = (item["overall_equal_weight_gain"] >= 0.005 and
                                      all(per_k[k] > 0 for k in K_VALUES) and
                                      item["positive_graphs"] >= 3 and item["positive_seeds"] >= 2)
        comparison_summary[control] = item
        comparison_rows.append({"control": control,
                                "overall_gain_pct": f"{100*item['overall_equal_weight_gain']:.6f}",
                                "k4_gain_pct": f"{100*per_k[4]:.6f}",
                                "k32_gain_pct": f"{100*per_k[32]:.6f}",
                                "positive_cells": item["positive_cells"],
                                "positive_graphs": item["positive_graphs"],
                                "seed100_gain_pct": f"{100*per_seed[100]:.6f}",
                                "seed101_gain_pct": f"{100*per_seed[101]:.6f}",
                                "seed102_gain_pct": f"{100*per_seed[102]:.6f}",
                                "positive_seeds": item["positive_seeds"],
                                "meets_frozen_gate": item["meets_frozen_gate"]})
    write_csv(out / "comparison_summary.csv", comparison_rows)

    time_rows = []
    for method in METHODS:
        method_runs = [r for r in runs if r["method"] == method]
        times = [r["wall_seconds"] for r in method_runs]
        time_rows.append({"method": method, "runs": len(method_runs),
                          "total_solver_process_wall_s": f"{sum(times):.4f}",
                          "mean_solver_process_wall_s": f"{sum(times)/len(times):.4f}",
                          "median_solver_process_wall_s": f"{median(times):.6f}",
                          "max_solver_process_wall_s": f"{max(times):.4f}"})
    write_csv(out / "evaluation_time_summary.csv", time_rows)

    teachers = json.loads((out / "teacher_results.json").read_text(encoding="utf-8"))
    teacher_times = [r["wall_seconds"] for r in teachers]
    teacher_grouped = {}
    for graph in GRAPHS:
        for k in K_VALUES:
            group = [r for r in teachers if r["graph"] == graph and r["k"] == k]
            teacher_grouped[f"{graph}_k{k}"] = {"runs": len(group),
                                                 "total_wall_seconds": sum(r["wall_seconds"] for r in group),
                                                 "mean_wall_seconds": sum(r["wall_seconds"] for r in group)/len(group)}
    clock = json.loads((out / "gp01_experiment_clock.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(clock["deadline_utc"])
    # On an offline refresh, preserve the completion-time snapshot instead of
    # counting report maintenance as part of the experiment batch.
    summary_path = out / "gp01_summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    previous_clock = previous.get("wall_clock", {})
    elapsed = previous_clock.get(
        "teacher_evaluation_batch_elapsed_seconds",
        previous_clock.get(
            "elapsed_seconds_at_summary",
            (datetime.now(timezone.utc) - datetime.fromisoformat(clock["started_utc"])).total_seconds(),
        ),
    )
    summary = {
        "protocol": "GP01 frozen; 4 graphs, k=4/32, eps=0.03, c-t=160, one thread",
        "evaluation_runs": {"expected": 144, "valid": len(runs), "all_valid": eval_status["all_valid"]},
        "teacher_runs": {"expected": 80, "valid": len(teachers), "all_valid": teacher_status["all_valid"],
                         "total_solver_wall_seconds": sum(teacher_times), "by_graph_k": teacher_grouped},
        "wall_clock": {"started_utc": clock["started_utc"], "deadline_utc": clock["deadline_utc"],
                       "scope": "teacher generation through formal evaluation summary; excludes data acquisition, graph conversion, build, and the initial interface audit",
                       "teacher_evaluation_batch_elapsed_seconds": elapsed,
                       "teacher_evaluation_batch_within_two_hour_budget": elapsed <= 7200},
        "comparisons": comparison_summary,
        "evaluation_time_by_method": {r["method"]: r for r in time_rows},
        "evaluation_results_sha256": __import__("hashlib").sha256(
            (out / "evaluation_results.json").read_bytes()).hexdigest().upper(),
        "teacher_results_sha256": __import__("hashlib").sha256(
            (out / "teacher_results.json").read_bytes()).hexdigest().upper(),
        "rating_manifest_sha256": __import__("hashlib").sha256(
            (out / "rating_manifest.json").read_bytes()).hexdigest().upper(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison_summary": comparison_rows,
                      "teacher_solver_seconds": round(sum(teacher_times), 3),
                      "evaluation_solver_seconds": {r["method"]: round(float(r["total_solver_process_wall_s"]), 3)
                                                     for r in time_rows},
                      "elapsed_seconds": round(elapsed, 3)}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
