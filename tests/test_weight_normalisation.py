"""Tests for in-edge weight normalisation (audit item P1-3.8).

The confound: the model requires ``sum_u omega_uv <= 1`` per node, but every script scaled in-weights
to sum to *exactly* 1 and said nothing about it.  That is a different model, and it sets the exposure
scale, which decides whether any node is over-exposed at all.  The tests below pin that the choice is
named, recorded, verified against the allowance, and actually different between strategies.
"""

from __future__ import annotations

import networkx as nx
import pytest

from grl.data.weights import (
    AS_GIVEN,
    CLIP_TO_ONE,
    METADATA_KEY,
    NORMALISATIONS,
    SUM_TO_ONE,
    UNIFORM_SHARE,
    WeightAllowanceError,
    describe_normalisation,
    in_weight_totals,
    max_in_weight_total,
    normalise_in_weights,
    verify_in_weight_allowance,
)


def sparse_graph() -> nx.DiGraph:
    """In-weights summing to well under 1, as a uniformly weighted sparse file would give."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.01)
    graph.add_edge("b", "t", weight=0.01)
    graph.add_edge("t", "z", weight=0.01)
    return graph


def heavy_graph() -> nx.DiGraph:
    """A node whose in-weights exceed the allowance, which the model forbids."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.9)
    graph.add_edge("b", "t", weight=0.6)
    return graph


# ---------------------------------------------------------------------------------------------
# the allowance is real and enforced
# ---------------------------------------------------------------------------------------------
def test_a_graph_over_the_allowance_is_refused():
    with pytest.raises(WeightAllowanceError, match="more than the allowed total in-weight of 1"):
        verify_in_weight_allowance(heavy_graph())


def test_as_given_refuses_a_graph_the_model_forbids():
    with pytest.raises(WeightAllowanceError):
        normalise_in_weights(heavy_graph(), AS_GIVEN)


def test_as_given_leaves_weights_alone():
    graph = sparse_graph()
    before = sorted(d["weight"] for _, _, d in graph.edges(data=True))
    normalise_in_weights(graph, AS_GIVEN)
    after = sorted(d["weight"] for _, _, d in graph.edges(data=True))
    assert before == after
    assert max_in_weight_total(graph) < 1.0, "a sparse graph stays sparse under as_given"


def test_every_normalisation_satisfies_the_allowance():
    for strategy in NORMALISATIONS:
        for build in (sparse_graph, heavy_graph):
            if strategy == AS_GIVEN and build is heavy_graph:
                # as_given must refuse rather than silently repair a graph the model forbids
                with pytest.raises(WeightAllowanceError):
                    normalise_in_weights(build(), strategy, copy=True)
                continue
            graph = normalise_in_weights(build(), strategy, copy=True)
            verify_in_weight_allowance(graph)
            assert max_in_weight_total(graph) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------------------------
# the strategies differ, which is why the choice has to be recorded
# ---------------------------------------------------------------------------------------------
def test_sum_to_one_and_as_given_differ_on_a_sparse_graph():
    """This is the confound in one assertion: the same file, two different exposure scales."""
    sparse = normalise_in_weights(sparse_graph(), SUM_TO_ONE, copy=True)
    given = normalise_in_weights(sparse_graph(), AS_GIVEN, copy=True)
    assert abs(in_weight_totals(sparse)["t"] - 1.0) < 1e-9
    assert abs(in_weight_totals(given)["t"] - 0.02) < 1e-9
    assert in_weight_totals(sparse)["t"] > 10 * in_weight_totals(given)["t"], (
        "sum_to_one invents weight the file did not provide; on a sparse graph that is a 50x "
        "change in the exposure scale, which is exactly why the choice must be reported"
    )


def test_clip_to_one_only_scales_down():
    """clip_to_one never invents weight, so a sparse node keeps a small delta."""
    sparse = normalise_in_weights(sparse_graph(), CLIP_TO_ONE, copy=True)
    assert abs(in_weight_totals(sparse)["t"] - 0.02) < 1e-9

    heavy = normalise_in_weights(heavy_graph(), CLIP_TO_ONE, copy=True)
    assert abs(in_weight_totals(heavy)["t"] - 1.0) < 1e-9
    # the ratio between the two contributors is preserved
    weights = {u: d["weight"] for u, _, d in heavy.edges(data=True)}
    assert abs(weights["a"] / weights["b"] - 0.9 / 0.6) < 1e-9


def test_uniform_share_ignores_the_file_magnitudes():
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.9)
    graph.add_edge("b", "t", weight=0.1)
    graph.add_edge("c", "t", weight=0.05)
    uniform = normalise_in_weights(graph, UNIFORM_SHARE, copy=True)
    weights = sorted(d["weight"] for _, _, d in uniform.edges(data=True))
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown normalisation"):
        normalise_in_weights(sparse_graph(), "whatever")


# ---------------------------------------------------------------------------------------------
# the choice must be visible in the output
# ---------------------------------------------------------------------------------------------
def test_the_strategy_is_recorded_on_the_graph():
    for strategy in NORMALISATIONS:
        graph = normalise_in_weights(sparse_graph(), strategy, copy=True)
        assert graph.graph[METADATA_KEY] == strategy


def test_copy_leaves_the_input_untouched():
    graph = sparse_graph()
    normalise_in_weights(graph, SUM_TO_ONE, copy=True)
    assert graph.graph.get(METADATA_KEY) is None, "the input must not be annotated"
    assert abs(in_weight_totals(graph)["t"] - 0.02) < 1e-9


def test_describe_normalisation_reports_the_exposure_scale():
    graph = normalise_in_weights(sparse_graph(), SUM_TO_ONE, copy=True)
    described = describe_normalisation(graph)
    assert described["strategy"] == SUM_TO_ONE
    assert abs(described["max_in_weight_total"] - 1.0) < 1e-9
    # only nodes WITH in-edges can reach a total of 1; 'a' and 'b' receive nothing
    assert described["nodes_reaching_one"] == 2  # t and z
    assert described["n"] == 4

    sparse = normalise_in_weights(sparse_graph(), AS_GIVEN, copy=True)
    assert describe_normalisation(sparse)["nodes_reaching_one"] == 0
    assert max(describe_normalisation(sparse)[k] for k in ("max_in_weight_total",)) < 0.1
