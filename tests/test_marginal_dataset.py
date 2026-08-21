import networkx as nx

from grl.data import GraphData
from grl.training import build_marginal_dataset


def test_marginal_dataset_contains_multiple_seed_set_sizes():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    data = GraphData("toy", "toy.txt", True, graph, 3, 2, 0, 0, 1.33, 1, 1, 3)
    config = {
        "seed": {"budget": 3},
        "gnn": {"contexts": 12, "candidates_per_context": 2, "mc_runs_train": 2},
        "diffusion": {"probability": 1.0},
        "experiment": {"random_seed": 1},
    }
    splits = build_marginal_dataset(data, config)
    samples = [sample for values in splits.values() for sample in values]
    assert samples
    assert {len(sample.seed_set) for sample in samples} >= {0, 1}
    assert all(sample.candidate not in sample.seed_set for sample in samples)

    context_splits = {
        name: {sample.context_id for sample in split}
        for name, split in splits.items()
    }
    assert context_splits["train"].isdisjoint(context_splits["validation"])
    assert context_splits["train"].isdisjoint(context_splits["test"])
    assert context_splits["validation"].isdisjoint(context_splits["test"])
    for sample in samples:
        assert sample.seed_set_size == len(sample.seed_set)
        assert sample.extended_spread - sample.base_spread == sample.marginal_gain

    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.context_id, []).append(sample)
    assert all(len(context_samples) >= 2 for context_samples in grouped.values())
