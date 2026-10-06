#!/usr/bin/env python3
"""Run the frozen 144-call GP01 six-arm evaluation matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gp01_common import (EVAL_SEEDS, GRAPHS, K_VALUES, METHODS, load_clock,
                        run_solver, seconds_left)


def save(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "work/gp01/outputs"
    teacher_state = json.loads((out / "teacher_status.json").read_text(encoding="utf-8"))
    if not teacher_state.get("all_valid"):
        raise SystemExit("Teacher matrix incomplete; evaluation is not permitted")
    if not (out / "rating_manifest.json").exists():
        raise SystemExit("Frozen ratings are missing; run build_ratings.py first")
    rating_manifest = json.loads((out / "rating_manifest.json").read_text(encoding="utf-8"))
    rating_paths = {(g["name"], f["target_k"], f["method"]): Path(f["path"])
                    for g in rating_manifest["graphs"] for f in g["files"]}
    run_results = out / "evaluation_results.json"
    state_path = out / "evaluation_status.json"
    clock = load_clock(root)
    existing = json.loads(run_results.read_text(encoding="utf-8")) if run_results.exists() else []
    done = {(r["graph"], r["k"], r["seed"], r["method"]): r for r in existing}

    for graph in GRAPHS:
        for k in K_VALUES:
            for seed in EVAL_SEEDS:
                for method in METHODS:
                    key = (graph, k, seed, method)
                    if key in done:
                        continue
                    if seconds_left(clock) <= 60:
                        break
                    score_file = None if method in {"BASE", "AUTHOR"} else rating_paths[(graph, k, method)]
                    rec = run_solver(root, graph, k, seed, method, score_file)
                    done[key] = rec
                    save(run_results, list(done.values()))
                    print(json.dumps({"stage": "evaluation", "graph": graph, "k": k,
                                      "seed": seed, "method": method, "status": rec["status"],
                                      "wall_seconds": round(rec["wall_seconds"], 3),
                                      "cut": rec.get("cut")}, separators=(",", ":")), flush=True)
                if seconds_left(clock) <= 60:
                    break
            if seconds_left(clock) <= 60:
                break
        if seconds_left(clock) <= 60:
            break

    expected = {(g, k, s, method) for g in GRAPHS for k in K_VALUES
                for s in EVAL_SEEDS for method in METHODS}
    complete = {key for key, rec in done.items() if rec.get("status") == "ok" and rec.get("balanced")}
    state = {"expected_runs": len(expected), "recorded_runs": len(done), "valid_runs": len(complete),
             "missing_or_invalid": [list(key) for key in sorted(expected - complete)],
             "all_valid": complete == expected,
             "elapsed_budget_seconds": 7200 - max(seconds_left(clock), 0)}
    save(state_path, state)
    print(json.dumps({"stage": "evaluation_complete", **state}, separators=(",", ":")), flush=True)
    if not state["all_valid"]:
        raise SystemExit("Evaluation matrix incomplete or invalid; report missing cells as missing")


if __name__ == "__main__":
    main()
