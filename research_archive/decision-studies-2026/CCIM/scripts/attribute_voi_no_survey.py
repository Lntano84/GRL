"""Frozen protocol's final-seed value before spending the last survey."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attribute_voi_probe as P  # noqa: E402

OUT = ROOT / "results" / "attribute_voi"


def main():
    graph, ids, classes, i, j, same, full, pair, info = P.load_graph()
    plan_hash = P.digest(P.PLAN)
    rows = []
    for index in range(P.N_CONFIRM):
        original = json.loads((OUT / "confirmation" / f"state_{index:03d}.json").read_text(encoding="utf-8"))
        assert original["plan_sha256"] == plan_hash
        original = original["result"]
        number = 1000 + index
        state = P.build_state("confirmation", number, len(ids), i, j, same, full)
        base = P.solve(len(ids), i, j, state["q"], P.seed("base_solver", "confirmation", number))
        evaluator = P.truth_objective(len(ids), i, j, full, P.N_CONFIRM_MC,
                                      "true_confirmation", "confirmation", number)
        value = float(evaluator.value(base))
        rows.append({"state": number, "no_survey_value": value,
                     "voi_value": original["values"]["voi"],
                     "observed_degree_value": original["values"]["observed_degree"]})
    result = {"plan_sha256": plan_hash, "n": len(rows),
              "mean_no_survey": float(np.mean([r["no_survey_value"] for r in rows])),
              "voi_minus_no_survey": P.paired([r["voi_value"] - r["no_survey_value"] for r in rows]),
              "observed_degree_minus_no_survey": P.paired(
                  [r["observed_degree_value"] - r["no_survey_value"] for r in rows])}
    P.atomic_json(OUT / "no_survey_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
