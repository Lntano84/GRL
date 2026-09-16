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
from grl.diagnostics import run_candidate_benchmark
from grl.utils import build_run_metadata, load_yaml_config, set_random_seed


def _resolve_output_root(value: str) -> Path:
    configured = Path(value)
    if configured.is_absolute() or not configured.parts or configured.parts[0] != "outputs":
        raise ValueError("experiment.output_dir must be a relative path under outputs/")
    return PROJECT_ROOT / configured


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate retrievers against a restricted top-N-by-degree oracle."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    graph_data = load_graph_from_config(config)
    results = run_candidate_benchmark(graph_data, config, PROJECT_ROOT)
    results["metadata"] = build_run_metadata(
        args.config, config, PROJECT_ROOT, "candidate_retrieval_benchmark_v1"
    )

    output_dir = (
        _resolve_output_root(str(config["experiment"]["output_dir"]))
        / datetime.now().strftime("%Y%m%d_%H%M%S")
        / "candidate_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with (output_dir / "candidate_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "candidate_benchmark.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results["summary"][0]))
        writer.writeheader()
        writer.writerows(results["summary"])

    print(f"Candidate benchmark finished: {output_dir}")
    print(f"oracle_scope={results['oracle_scope']} (restricted; not full-graph)")
    print(f"FeatureDQN status: {json.dumps(results['feature_dqn'], ensure_ascii=False)}")
    print("Retriever          M   Recall@M   CandidateLoss   Runtime(s)")
    for row in results["summary"]:
        print(
            f"{row['retriever']:<18} {row['M']:>3} "
            f"{row['recall_at_m']:>10.3f} {row['mean_candidate_loss']:>15.4f} "
            f"{row['mean_runtime_seconds']:>12.6f}"
        )


if __name__ == "__main__":
    main()
