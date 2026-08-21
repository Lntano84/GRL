import networkx as nx

from grl.diffusion import estimate_marginal_gains, estimate_spread


def test_diffusion_is_reproducible_with_same_seed():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)

    first = estimate_spread(graph, [0], mc_runs=5, random_seed=123)
    second = estimate_spread(graph, [0], mc_runs=5, random_seed=123)

    assert first == second


def test_grouped_marginal_labels_are_paired_and_consistent():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)

    result = estimate_marginal_gains(graph, [0], [1, 3], mc_runs=3, random_seed=5)

    assert result[1]["mean"] == 0.0
    assert result[3]["mean"] == 1.0
    for values in result.values():
        assert values["extended_spread"] - values["base_spread"] == values["mean"]
