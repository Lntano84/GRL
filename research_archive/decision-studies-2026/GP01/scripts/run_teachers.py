#!/usr/bin/env python3
"""Run the frozen 80-run GP01 non-learning teacher matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gp01_common import GRAPHS, K_VALUES, load_clock, run_solver, seconds_left


def save(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "work/gp01/outputs"
    run_results = out / "teacher_results.json"
    status_path = out / "teacher_status.json"
    clock = load_clock(root)
    existing = json.loads(run_results.read_text(encoding="utf-8")) if run_results.exists() else []
    done = {(r["graph"], r["k"], r["seed"]): r for r in existing}
    for graph in GRAPHS:
        for k in K_VALUES:
            for seed in range(10):
                key = (graph, k, seed)
                if key in done:
                    continue
                if seconds_left(clock) <= 60:
                    break
                rec = run_solver(root, graph, k, seed, "BASE")
                rec["role"] = "teacher"
                done[key] = rec
                save(run_results, list(done.values()))
                print(json.dumps({"stage": "teacher", "graph": graph, "k": k, "seed": seed,
                                  "status": rec["status"], "wall_seconds": round(rec["wall_seconds"], 3),
                                  "cut": rec.get("cut")}, separators=(",", ":")), flush=True)
            if seconds_left(clock) <= 60:
                break
        if seconds_left(clock) <= 60:
            break

    expected = {(g, k, s) for g in GRAPHS for k in K_VALUES for s in range(10)}
    completed = {key for key, rec in done.items() if rec.get("status") == "ok" and rec.get("balanced")}
    state = {"expected_runs": len(expected), "recorded_runs": len(done),
             "valid_runs": len(completed), "missing_or_invalid": sorted(expected - completed),
             "all_valid": completed == expected, "elapsed_budget_seconds": 7200 - max(seconds_left(clock), 0)}
    save(status_path, state)
    print(json.dumps({"stage": "teacher_complete", **state}, separators=(",", ":")), flush=True)
    if not state["all_valid"]:
        raise SystemExit("Teacher matrix incomplete or invalid; do not construct q ratings or evaluate")


if __name__ == "__main__":
    main()
