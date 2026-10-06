"""Supplementary finite-pool opportunity check for the high-school survey probe.

The privileged arm selects one of the already frozen <=8 candidates using 300
true-graph IC samples.  Its final value is measured on the independent 3000-sample
batch used by every deployable arm.  It is an opportunity diagnostic, not a policy
or a universal upper bound.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attribute_voi_probe as P  # noqa: E402

OUT = ROOT / "results" / "attribute_voi"
SELECT_MC = 300


def main():
    graph, ids, classes, i, j, same, full, pair, info = P.load_graph()
    plan_hash = P.digest(P.PLAN)
    rows = []
    for index in range(P.N_CONFIRM):
        saved = json.loads((OUT / "confirmation" / f"state_{index:03d}.json").read_text(encoding="utf-8"))
        assert saved["plan_sha256"] == plan_hash
        original = saved["result"]
        number = 1000 + index
        state = P.build_state("confirmation", number, len(ids), i, j, same, full)
        assert state["query_order"] == original["query_order"]
        t0 = time.process_time()
        selected_on = P.truth_objective(len(ids), i, j, full, SELECT_MC,
                                        "privileged_selection", "confirmation", number)
        confirmed_on = P.truth_objective(len(ids), i, j, full, P.N_CONFIRM_MC,
                                         "true_confirmation", "confirmation", number)
        scores = {}
        for node in original["pool"]:
            q_after = P.after_query(state, node, full, i, j, same)
            seeds = P.solve(len(ids), i, j, q_after,
                            P.seed("final_solver", "confirmation", number))
            scores[node] = {"choice_batch": float(selected_on.value(seeds)),
                            "confirm_batch": float(confirmed_on.value(seeds)),
                            "seeds": seeds}
        picked = max(original["pool"], key=lambda node: scores[node]["choice_batch"])
        # Exact agreement means this arm used the same terminal solver and confirmation stream.
        for arm, node in original["choices"].items():
            assert abs(scores[node]["confirm_batch"] - original["values"][arm]) < 1e-9
        row = {"state": number, "selected": picked,
               "value": scores[picked]["confirm_batch"],
               "observed_degree_value": original["values"]["observed_degree"],
               "voi_value": original["values"]["voi"],
               "selected_equals_observed_degree": picked == original["choices"]["observed_degree"],
               "selected_equals_voi": picked == original["choices"]["voi"],
               "cpu_seconds": time.process_time() - t0}
        P.atomic_json(OUT / "privileged" / f"state_{index:03d}.json", row)
        rows.append(row)
        print(f"privileged {index+1}/{P.N_CONFIRM} CPU={row['cpu_seconds']:.1f}s", flush=True)
    result = {"plan_sha256": plan_hash, "n": len(rows),
              "selection_samples": SELECT_MC, "independent_confirmation_samples": P.N_CONFIRM_MC,
              "mean": float(np.mean([r["value"] for r in rows])),
              "vs_observed_degree": P.paired([r["value"]-r["observed_degree_value"] for r in rows]),
              "vs_voi": P.paired([r["value"]-r["voi_value"] for r in rows]),
              "same_observed_degree": sum(r["selected_equals_observed_degree"] for r in rows),
              "same_voi": sum(r["selected_equals_voi"] for r in rows),
              "total_cpu_seconds": sum(r["cpu_seconds"] for r in rows)}
    P.atomic_json(OUT / "privileged_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
