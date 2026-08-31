from __future__ import annotations

import argparse
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
from grl.experiments.overlap import run_overlap_experiment
from grl.utils import load_yaml_config, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Direct-MLP with Direct-MLP+Overlap.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    set_random_seed(int(config.get("experiment", {}).get("random_seed", 42)))
    graph_data = load_graph_from_config(config)
    base_output = Path(config["experiment"]["output_dir"])
    if base_output.is_absolute() or not base_output.parts or base_output.parts[0] != "outputs":
        raise ValueError("experiment.output_dir must be relative and under outputs/")
    output_dir = PROJECT_ROOT / base_output / datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_overlap_experiment(graph_data, config, output_dir)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"Overlap experiment finished: {output_dir}")
    print(json.dumps({"models": result["models"], "seeds": result["seeds"], "aggregate_metrics_csv": result["aggregate_metrics_csv"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
