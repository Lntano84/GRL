"""Frozen no-class ablation for the attribute VOI probe.

Uses exactly the candidate pools saved by the main probe.  Removes the class split
from the Beta-Bernoulli model, then independently plans and chooses final seeds.
The final true-graph IC stream is identical to the main probe's confirmation stream.
No hyperparameters, candidate pools, or baselines are reselected here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attribute_voi_probe as P  # noqa: E402

OUT = ROOT / "results" / "attribute_voi"


def main():
    graph, ids, classes, i, j, same, full, pair, info = P.load_graph()
    no_class = np.zeros_like(same, dtype=bool)
    plan_hash = P.digest(P.PLAN)
    rows = []
    for index in range(P.N_CONFIRM):
        saved = json.loads((OUT / "confirmation" / f"state_{index:03d}.json").read_text(encoding="utf-8"))
        assert saved["plan_sha256"] == plan_hash
        original = saved["result"]
        number = 1000 + index
        assert original["state"] == number and original["split"] == "confirmation"
        state = P.build_state("confirmation", number, len(ids), i, j, no_class, full)
        assert state["query_order"] == original["query_order"]
        pool = original["pool"]
        t0 = time.process_time()
        base = P.solve(len(ids), i, j, state["q"], P.seed("base_solver", "confirmation", number))
        node, plan_scores = P.plan_choice(state, pool, base, len(ids), i, j, no_class, full)
        planning_cpu = time.process_time() - t0
        final_q = P.after_query(state, node, full, i, j, no_class)
        final_seeds = P.solve(len(ids), i, j, final_q,
                              P.seed("final_solver", "confirmation", number))
        evaluator = P.truth_objective(len(ids), i, j, full, P.N_CONFIRM_MC,
                                      "true_confirmation", "confirmation", number)
        value = float(evaluator.value(final_seeds))
        rows.append({"state": number, "pool": pool, "no_class_choice": node,
                     "no_class_seeds": final_seeds, "no_class_value": value,
                     "with_class_choice": original["choices"]["voi"],
                     "with_class_value": original["values"]["voi"],
                     "difference": original["values"]["voi"] - value,
                     "planning_cpu_seconds": planning_cpu,
                     "total_cpu_seconds": time.process_time() - t0})
        P.atomic_json(OUT / "no_class" / f"state_{index:03d}.json", rows[-1])
        print(f"no-class {index + 1}/{P.N_CONFIRM} CPU={rows[-1]['total_cpu_seconds']:.1f}s", flush=True)
    diffs = [r["difference"] for r in rows]
    result = {"plan_sha256": plan_hash, "n": len(rows),
              "comparison": "attribute VOI minus no-class VOI; same finite candidate pool and true IC stream",
              "paired": P.paired(diffs),
              "same_choice": sum(r["with_class_choice"] == r["no_class_choice"] for r in rows),
              "no_class_mean": float(np.mean([r["no_class_value"] for r in rows])),
              "with_class_mean": float(np.mean([r["with_class_value"] for r in rows])),
              "mean_no_class_planning_cpu_seconds": float(np.mean([r["planning_cpu_seconds"] for r in rows]))}
    P.atomic_json(OUT / "no_class_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
