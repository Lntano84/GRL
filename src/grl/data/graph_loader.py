from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


class GraphValidationError(ValueError):
    pass


# ``congress.edgelist`` (and anything written by ``nx.write_edgelist``) stores attributes
# as a Python dict literal, e.g. ``0 4 {'weight': 0.002105263157894737}``.  Splitting on
# whitespace yields ``['0', '4', "{'weight':", "0.002105263157894737}"]``, so casting field
# 3 to float raises ValueError.  This pattern pulls the weight out instead.
_NETWORKX_WEIGHT_RE = re.compile(r"['\"]weight['\"]\s*:\s*([0-9eE.+-]+)")


def _parse_weight(fields: list[str], default_probability: float) -> float:
    """Extract an edge probability from a tokenised line.

    Handles plain numeric third columns and NetworkX dict literals.  A line with no
    weight information falls back to ``default_probability``.
    """
    if len(fields) < 3:
        return float(default_probability)

    rest = " ".join(fields[2:])
    try:
        return float(fields[2])
    except ValueError:
        pass

    match = _NETWORKX_WEIGHT_RE.search(rest)
    if match:
        return float(match.group(1))

    # A dict literal without a weight key carries no probability information.
    if rest.lstrip().startswith("{"):
        return float(default_probability)

    raise GraphValidationError(f"cannot parse edge weight from: {' '.join(fields)}")


@dataclass
class GraphData:
    name: str
    graph_path: Path
    directed: bool
    graph: nx.Graph | nx.DiGraph
    num_nodes: int
    num_edges: int
    self_loops: int
    duplicate_edges: int
    average_degree: float
    max_degree: int
    connected_components: int
    largest_component_size: int
    node_to_idx: dict[int, int] = field(default_factory=dict)
    idx_to_node: list[int] = field(default_factory=list)

    def to_index(self, node: int) -> int:
        """Convert an original node id to the contiguous internal index."""
        if self.node_to_idx:
            return self.node_to_idx[node]
        return int(node)

    def to_original(self, index: int) -> int:
        """Convert an internal index back to the original node id."""
        if self.idx_to_node:
            return self.idx_to_node[int(index)]
        return int(index)

    def restore_nodes(self, indices: list[int]) -> list[int]:
        return [self.to_original(index) for index in indices]


def _build_graph(directed: bool) -> nx.Graph | nx.DiGraph:
    return nx.DiGraph() if directed else nx.Graph()


def _looks_like_edge(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True


def _find_header_end(lines: list[str]) -> int:
    """Return the index of the first edge line, skipping an ``n m`` header if present.

    A leading line of two non-negative integers is ambiguous -- it can be a graph header
    or an ordinary edge of a two-column file.  Three conditions must all hold before the
    line is treated as a header, so that a plain edge list essentially never loses data:

    1. it has exactly two non-negative integer fields;
    2. ``m >= n``, since a simple graph on ``n`` nodes needs at least ``n - 1`` edges
       (an edge line carries no such constraint);
    3. ``n`` is at least one tenth of the largest node id appearing in the rest of the
       file -- a header's first field *is* the node count, whereas an edge's first field
       is just one endpoint id.

    Condition 3 is what separates ``15233 32235`` (a real header: the largest id is
    15232) from an edge such as ``30 1412`` (a real edge in a file whose largest id is
    far above 30).
    """
    if not lines:
        return 0
    first = lines[0].split()
    if len(first) != 2 or not all(part.isdigit() for part in first):
        return 0

    node_count, edge_count = int(first[0]), int(first[1])
    if edge_count < node_count:
        return 0

    others = [line for line in lines[1:] if _looks_like_edge(line)]
    if not others:
        return 0
    max_id_seen = max(max(int(part) for part in line.split()[:2]) for line in others)

    return 1 if node_count * 10 >= max_id_seen else 0


def _parse_graph_file(graph_path: Path, directed: bool, default_probability: float) -> tuple[nx.Graph | nx.DiGraph, int]:
    if not graph_path.exists():
        raise GraphValidationError(f"Graph file does not exist: {graph_path}")

    graph = _build_graph(directed)
    seen_edges: set[tuple[int, int]] = set()
    duplicate_edges = 0

    with graph_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in (raw.strip() for raw in handle) if line]
        start = _find_header_end(lines)

        for line in lines[start:]:
            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            probability = _parse_weight(parts, default_probability)

            edge_key = (u, v) if directed else tuple(sorted((u, v)))
            if edge_key in seen_edges:
                duplicate_edges += 1
            seen_edges.add(edge_key)
            graph.add_edge(u, v, weight=probability)

    return graph, duplicate_edges


def _component_stats(graph: nx.Graph | nx.DiGraph, directed: bool) -> tuple[int, int]:
    if graph.number_of_nodes() == 0:
        return 0, 0

    if directed:
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    return len(components), largest


def _validate_expected_counts(graph: nx.Graph | nx.DiGraph, expected_num_nodes: Any, expected_num_edges: Any) -> None:
    if expected_num_nodes is not None and graph.number_of_nodes() != int(expected_num_nodes):
        raise GraphValidationError(
            f"Expected {expected_num_nodes} nodes, got {graph.number_of_nodes()}"
        )
    if expected_num_edges is not None and graph.number_of_edges() != int(expected_num_edges):
        raise GraphValidationError(
            f"Expected {expected_num_edges} edges, got {graph.number_of_edges()}"
        )


def load_graph_from_config(config: dict[str, Any]) -> GraphData:
    dataset = config["dataset"]
    diffusion = config.get("diffusion", {})
    name = str(dataset["name"])
    graph_path = Path(str(dataset["graph_path"]))
    directed = bool(dataset.get("directed", True))
    default_probability = float(diffusion.get("probability", 0.01))

    raw_graph, duplicate_edges = _parse_graph_file(graph_path, directed, default_probability)
    idx_to_node = list(raw_graph.nodes())
    node_to_idx = {node: index for index, node in enumerate(idx_to_node)}
    graph = nx.relabel_nodes(
        raw_graph,
        node_to_idx,
        copy=True,
    )
    nx.set_node_attributes(graph, {index: node for index, node in enumerate(idx_to_node)}, "original_id")
    _validate_expected_counts(
        graph,
        dataset.get("expected_num_nodes"),
        dataset.get("expected_num_edges"),
    )

    degrees = [degree for _, degree in graph.degree()]
    average_degree = float(sum(degrees) / len(degrees)) if degrees else 0.0
    max_degree = max(degrees, default=0)
    components, largest_component = _component_stats(graph, directed)

    return GraphData(
        name=name,
        graph_path=graph_path,
        directed=directed,
        graph=graph,
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        self_loops=nx.number_of_selfloops(graph),
        duplicate_edges=duplicate_edges,
        average_degree=average_degree,
        max_degree=max_degree,
        connected_components=components,
        largest_component_size=largest_component,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
    )
