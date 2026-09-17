"""Edge-list format coverage for :func:`grl.data.graph_loader._parse_graph_file`.

Motivated by a real failure: the Inf. Sci. 2026 evaluation graphs include
``congress.edgelist``, which is a NetworkX edge list -- attributes are serialised as a
Python dict literal on the line::

    0 4 {'weight': 0.002105263157894737}

The loader split on whitespace and cast field 3 to ``float``, so it raised
``ValueError: could not convert string to float: "{'weight':"``.  Plain ``u v`` and
``u v w`` files worked, which is why the gap went unnoticed.

These tests pin down all three formats plus the fallback behaviour.
"""

from __future__ import annotations

import networkx as nx
import pytest

from grl.data import graph_loader as gl


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # plain numeric third column
        (["0", "1", "0.25"], 0.25),
        (["0", "1", "0.25", "extra"], 0.25),
        # no third column -> default
        (["0", "1"], 0.01),
        # NetworkX dict literal, already word-split by the caller
        (["0", "4", "{'weight':", "0.002105263157894737}"], 0.002105263157894737),
        (["0", "4", "{'weight':", "1.0}"], 1.0),
        # scientific notation inside the dict
        (["0", "4", "{'weight':", "1e-05}"], 1e-05),
        (["0", "4", "{'weight':", "+2.5e-3}"], 2.5e-3),
        # dict literal without a weight key -> default
        (["0", "4", "{'color':", "'red'}"], 0.01),
        # int-valued weight
        (["0", "4", "{'weight':", "1}"], 1.0),
    ],
)
def test_parse_weight_formats(line, expected):
    assert gl._parse_weight(line, 0.01) == pytest.approx(expected)


def test_parse_weight_rejects_unparseable_line():
    with pytest.raises(gl.GraphValidationError):
        gl._parse_weight(["0", "1", "not-a-number"], 0.01)


def test_networkx_edgelist_round_trip(tmp_path):
    """A graph written by ``nx.write_edgelist`` must load with its weights intact."""
    source = nx.DiGraph()
    source.add_edge(0, 1, weight=0.5)
    source.add_edge(1, 2, weight=0.125)
    source.add_edge(2, 0, weight=1.0)
    path = tmp_path / "graph.edgelist"
    nx.write_edgelist(source, path, data=True)

    loaded, duplicates = gl._parse_graph_file(path, True, 0.01)

    assert duplicates == 0
    assert loaded.number_of_edges() == 3
    for u, v, data in source.edges(data=True):
        assert loaded[u][v]["weight"] == pytest.approx(data["weight"])


def test_plain_two_column_file_uses_default_probability(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("30 1412\n30 3352\n30 5254\n", encoding="utf-8")

    loaded, _ = gl._parse_graph_file(path, True, 0.037)

    assert loaded.number_of_edges() == 3
    assert all(
        data["weight"] == pytest.approx(0.037) for _, _, data in loaded.edges(data=True)
    )


def test_two_column_first_line_that_looks_like_a_header(tmp_path):
    """A leading edge line must not be eaten as a header, even for small node ids.

    ``0 1`` satisfies ``m >= n`` and would be swallowed by a naive two-integer check.  The
    header rule therefore also requires the first field to be at least a tenth of the
    largest node id in the file, which no genuine edge satisfies in these datasets.
    """
    path = tmp_path / "small_ids.txt"
    path.write_text("0 1\n1 2\n2 3\n", encoding="utf-8")

    loaded, _ = gl._parse_graph_file(path, True, 0.01)

    assert loaded.number_of_edges() == 3
    assert loaded.has_edge(0, 1)


def test_real_header_is_still_skipped(tmp_path):
    """A realistic ``n m`` header must still be recognised and dropped."""
    path = tmp_path / "with_header.txt"
    rows = ["100 400"] + [f"{i} {i + 1}" for i in range(99)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    loaded, _ = gl._parse_graph_file(path, True, 0.01)

    assert loaded.number_of_edges() == 99
    assert not loaded.has_edge(100, 400)


def test_header_requires_edge_count_at_least_node_count(tmp_path):
    """``100 2`` cannot be a header for a simple graph, so it is kept as an edge."""
    path = tmp_path / "not_a_header.txt"
    path.write_text("100 2\n1 2\n2 3\n100 1\n", encoding="utf-8")

    loaded, _ = gl._parse_graph_file(path, True, 0.01)

    assert loaded.has_edge(100, 2)


def test_weighted_file_ignores_default_probability(tmp_path):
    """The default must NOT override weights that the file already provides.

    This is the behaviour that made ``configs/nethept_uniform.yaml`` silently a no-op:
    NetHEPT.txt carries its own weight column, so setting ``diffusion.probability`` had
    no effect until the file was replaced with a two-column copy.
    """
    path = tmp_path / "weighted.txt"
    path.write_text("0 1 0.9\n1 2 0.4\n", encoding="utf-8")

    loaded, _ = gl._parse_graph_file(path, True, 0.01)

    assert loaded[0][1]["weight"] == pytest.approx(0.9)
    assert loaded[1][2]["weight"] == pytest.approx(0.4)
