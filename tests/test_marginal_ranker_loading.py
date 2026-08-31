from pathlib import Path

import networkx as nx
import torch

from grl.data import GraphData
from grl.diagnostics.retrieval_reranking import load_marginal_gain_ranker
from grl.models import MarginalGainPredictor


def test_marginal_ranker_loads_checkpoint_and_scores_candidate_pool(tmp_path):
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2)])
    graph_data = GraphData(
        "toy", Path("toy.txt"), True, graph, 3, 2, 0, 0, 4 / 3, 2, 1, 3
    )
    model_path = tmp_path / "model.pth"
    embedding_path = tmp_path / "embedding.pth"
    torch.save(MarginalGainPredictor(4, hidden_dim=8).state_dict(), model_path)
    torch.save(torch.randn(3, 4), embedding_path)

    scorer, status = load_marginal_gain_ranker(
        graph_data,
        {
            "marginal_model_path": str(model_path),
            "marginal_embedding_path": str(embedding_path),
        },
        tmp_path,
    )

    assert status["status"] == "loaded"
    assert scorer is not None
    scores = scorer([0], [1, 2])
    assert set(scores) == {1, 2}
    assert all(isinstance(value, float) for value in scores.values())
