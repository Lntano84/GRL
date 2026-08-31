from pathlib import Path

import networkx as nx

from grl.baselines import rank_degree_discount_candidates
from grl.data import GraphData
from grl.diagnostics.candidate_benchmark import (
    build_candidate_record,
    load_feature_dqn_ranker,
)


def test_degree_discount_ranking_changes_with_selected_seeds():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
        ]
    )

    empty_context = rank_degree_discount_candidates(graph, [], [1, 2], 0.1)
    updated_context = rank_degree_discount_candidates(graph, [0], [1, 2], 0.1)

    assert empty_context == [1, 2]
    assert updated_context == [2, 1]


def test_candidate_loss_is_non_negative():
    record = build_candidate_record(
        step=1,
        retriever="Degree",
        pool_size=1,
        pool=[1],
        oracle_node=0,
        oracle_gain=1.0,
        gains={0: 1.0, 1: 1.0 + 1e-12},
        runtime_seconds=0.1,
        oracle_gain_runtime_seconds=1.0,
    )
    assert record["candidate_loss"] == 0.0


def test_recalled_oracle_has_zero_candidate_loss():
    record = build_candidate_record(
        step=1,
        retriever="Degree",
        pool_size=2,
        pool=[1, 0],
        oracle_node=0,
        oracle_gain=4.0,
        gains={0: 4.0, 1: 2.0},
        runtime_seconds=0.1,
        oracle_gain_runtime_seconds=1.0,
    )
    assert record["recalled"] == 1.0
    assert record["candidate_loss"] == 0.0


def test_missing_feature_dqn_dependencies_are_explicit(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph_data = GraphData(
        "toy", Path("toy.txt"), True, graph, 2, 1, 0, 0, 1.0, 1, 1, 2
    )
    ranker, status = load_feature_dqn_ranker(
        graph_data,
        {
            "feature_dqn_path": "missing/model.pth",
            "embedding_paths": ["missing/embedding.pth"],
        },
        [0, 1],
        tmp_path,
    )

    assert ranker is None
    assert status["status"] == "skipped"
    assert "checkpoint not found" in status["reason"]
    assert "embedding not found" in status["reason"]
