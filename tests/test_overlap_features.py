import networkx as nx
import numpy as np
import pytest

from grl.features import OVERLAP_FEATURE_NAMES, OverlapFeatureExtractor


def test_overlap_features_have_expected_values_on_chain():
    graph = nx.path_graph(4)
    extractor = OverlapFeatureExtractor(graph, distance_cap=6)
    values = extractor.transform_one([0], 2)

    assert len(values) == len(OVERLAP_FEATURE_NAMES) == 9
    assert values[0] == pytest.approx(2 / 6)
    assert values[1] == 0.0
    assert values[2] == 0.0
    assert values[3] == pytest.approx(1 / 3)
    assert values[4] == pytest.approx(3 / 4)
    assert values[5] == pytest.approx(3 / 4)
    assert values[6] == 1.0
    assert values[7] == pytest.approx(3 / 4)
    assert values[8] == 0.0
    assert np.isfinite(values).all()


def test_shortest_path_is_hop_count_not_edge_weight():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.01)
    values = OverlapFeatureExtractor(graph, distance_cap=2).transform_one([0], 1)
    assert values[0] == pytest.approx(0.5)


def test_empty_seed_set_is_explicit_and_finite():
    values = OverlapFeatureExtractor(nx.path_graph(3)).transform_one([], 1)
    assert values.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_unreachable_candidate_is_explicit():
    graph = nx.Graph([(0, 1), (2, 3)])
    values = OverlapFeatureExtractor(graph).transform_one([0], 3)
    assert values[0] == 1.0
    assert values[1] == 1.0
    assert np.isfinite(values).all()


def test_duplicate_seed_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        OverlapFeatureExtractor(nx.path_graph(3)).transform_one([0, 0], 2)
