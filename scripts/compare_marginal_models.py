import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from grl.data import load_graph_from_config
from grl.training.comparison import MarginalComparisonExperiment
from grl.utils import build_run_metadata, load_yaml_config, set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    set_random_seed(int(config["experiment"]["random_seed"]))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_data = load_graph_from_config(config)
    experiment = MarginalComparisonExperiment(graph_data, config, output_dir)
    result = experiment.run()
    result["metadata"] = build_run_metadata(
        args.config,
        config,
        PROJECT_ROOT,
        "marginal_model_comparison_v1",
    )
    (output_dir / "comparison_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print(f"Comparison artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
