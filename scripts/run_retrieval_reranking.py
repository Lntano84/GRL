from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.diagnostics.retrieval_reranking import run_retrieval_reranking_evaluation
from grl.utils import build_run_metadata, load_yaml_config, set_random_seed


def _resolve_output_root(value: str) -> Path:
    configured = Path(value)
    if configured.is_absolute() or not configured.parts or configured.parts[0] != "outputs":
        raise ValueError("experiment.output_dir must be a relative path under outputs/")
    return PROJECT_ROOT / configured


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for field, value in output.items():
                if isinstance(value, list):
                    output[field] = json.dumps(value, ensure_ascii=False)
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval plus seed-conditioned marginal reranking with "
            "separate restricted-oracle diagnostics and end-to-end trajectories."
        )
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    results = run_retrieval_reranking_evaluation(graph_data, config, PROJECT_ROOT)
    metadata = build_run_metadata(
        args.config, config, PROJECT_ROOT, "retrieval_reranking_evaluation_v1"
    )
    results["diagnostic"]["metadata"] = metadata
    results["end_to_end"]["metadata"] = metadata

    output_dir = (
        _resolve_output_root(str(config["experiment"]["output_dir"]))
        / datetime.now().strftime("%Y%m%d_%H%M%S")
        / "retrieval_reranking"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with (output_dir / "oracle_trajectory_diagnostic.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results["diagnostic"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "end_to_end_sequential.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(results["end_to_end"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(results["validation"], handle, ensure_ascii=False, indent=2)

    diagnostic_fields = [
        "step",
        "retriever",
        "ranker",
        "M",
        "repeat",
        "restricted_oracle_node",
        "restricted_oracle_gain",
        "pool_best_node",
        "pool_best_gain",
        "selected_node",
        "selected_gain",
        "candidate_loss",
        "ranking_loss",
        "total_regret",
        "decomposition_error",
        "retrieval_runtime_seconds",
        "reranking_runtime_seconds",
        "gain_evaluation_runtime_seconds",
    ]
    _write_csv(
        output_dir / "oracle_trajectory_diagnostic.csv",
        results["diagnostic"]["records"],
        diagnostic_fields,
    )
    _write_csv(
        output_dir / "end_to_end_sequential.csv",
        results["end_to_end"]["records"],
        ["method_id", *diagnostic_fields],
    )
    end_summary_fields = [
        "method_id",
        "retriever",
        "ranker",
        "M",
        "repeat",
        "selected_seeds",
        "final_spread_mean",
        "final_spread_std",
        "selection_runtime_seconds",
        "gain_evaluation_runtime_seconds",
        "spread_evaluation_runtime_seconds",
    ]
    _write_csv(
        output_dir / "end_to_end_summary.csv",
        results["end_to_end"]["summary"],
        end_summary_fields,
    )

    print(f"Retrieval-reranking evaluation finished: {output_dir}")
    print(
        f"oracle_scope={results['diagnostic']['oracle_scope']} "
        "(restricted; not full-graph)"
    )
    print(
        "FeatureDQN status: "
        + json.dumps(results["diagnostic"]["feature_dqn"], ensure_ascii=False)
    )
    print(
        "MarginalGainPredictor status: "
        + json.dumps(
            results["diagnostic"]["marginal_gain_predictor"], ensure_ascii=False
        )
    )
    print(
        "Loss decomposition holds: "
        + str(results["validation"]["loss_decomposition_holds"])
    )
    print("End-to-end methods:")
    for row in results["end_to_end"]["summary"]:
        repeat = "" if row["repeat"] is None else f" repeat={row['repeat']}"
        print(
            f"{row['retriever']}/{row['ranker']} M={row['M']}{repeat}: "
            f"spread={row['final_spread_mean']:.3f}, "
            f"selection_runtime={row['selection_runtime_seconds']:.6f}s"
        )


if __name__ == "__main__":
    main()
