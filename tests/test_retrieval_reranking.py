from pathlib import Path

import networkx as nx

from grl.data import GraphData
from grl.diagnostics.retrieval_reranking import (
    build_reranking_record,
    load_marginal_gain_ranker,
)


def _record(pool, oracle_node, selected_node, gains):
    return build_reranking_record(
        step=1,
        retriever="Degree",
        ranker="MarginalGainPredictor",
        pool_size=len(pool),
        pool=pool,
        oracle_node=oracle_node,
        oracle_gain=gains[oracle_node],
        selected_node=selected_node,
        gains=gains,
        retrieval_runtime_seconds=0.1,
        reranking_runtime_seconds=0.2,
        gain_runtime_seconds=0.3,
    )


def test_loss_decomposition_holds():
    record = _record(
        pool=[1, 2],
        oracle_node=0,
        selected_node=2,
        gains={0: 10.0, 1: 7.0, 2: 4.0},
    )
    assert record["candidate_loss"] == 3.0
    assert record["ranking_loss"] == 3.0
    assert record["total_regret"] == 6.0
    assert abs(
        record["candidate_loss"]
        + record["ranking_loss"]
        - record["total_regret"]
    ) < 1e-12


def test_pool_containing_restricted_oracle_has_zero_candidate_loss():
    record = _record(
        pool=[1, 0],
        oracle_node=0,
        selected_node=1,
        gains={0: 10.0, 1: 7.0},
    )
    assert record["candidate_loss"] == 0.0


def test_selecting_pool_best_has_zero_ranking_loss():
    record = _record(
        pool=[1, 2],
        oracle_node=0,
        selected_node=1,
        gains={0: 10.0, 1: 7.0, 2: 4.0},
    )
    assert record["ranking_loss"] == 0.0


def test_missing_marginal_checkpoint_is_explicit(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph_data = GraphData(
        "toy", Path("toy.txt"), True, graph, 2, 1, 0, 0, 1.0, 1, 1, 2
    )
    scorer, status = load_marginal_gain_ranker(
        graph_data,
        {
            "marginal_model_path": "missing/model.pth",
            "marginal_embedding_path": "missing/embedding.pth",
        },
        tmp_path,
    )
    assert scorer is None
    assert status["status"] == "skipped"
    assert "checkpoint not found" in status["reason"]
    assert "embedding not found" in status["reason"]
