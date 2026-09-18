"""Smoke-check that the script-level normalise wrapper routes through grl.data.weights."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import evaluate_overexposure_paper_graphs as m  # noqa: E402
from grl.data import (  # noqa: E402
    AS_GIVEN,
    CLIP_TO_ONE,
    SUM_TO_ONE,
    describe_normalisation,
)


def main() -> int:
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.01)
    graph.add_edge("b", "t", weight=0.01)
    graph.add_edge("t", "z", weight=0.01)

    for strategy in (SUM_TO_ONE, AS_GIVEN, CLIP_TO_ONE):
        normalised = m.normalise_in_weights(graph.copy(), strategy)
        described = describe_normalisation(normalised)
        print(
            f"  {strategy:<14} strategy={described['strategy']:<14} "
            f"max_total={described['max_in_weight_total']:.4f} "
            f"nodes_at_one={described['nodes_reaching_one']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
