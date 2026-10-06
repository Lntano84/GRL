"""Deterministic K-complex-contagion influence maximization, as defined in Chen et al., IJCAI 2023.

The paper's model
-----------------
Cascade probability depends on the number of activated neighbours ``k``:

    p(k) = p0  if k >= K
    p(k) = p1  if k <  K                                    (Eq. 1)

The experimental section sets ``p1 = 0, p0 = 1`` so that **propagation is deterministic**: a node
becomes active exactly when at least ``K`` of its neighbours are active.  ``K = 4`` for the medium
networks (Football, Polbooks) and the seed budget is ``T = 8``.

``sigma(G, S)`` counts the **whole active set including the seeds**, which is what makes the paper's
reward ``r = sigma(G, S u {a}) - sigma(G, S)`` start at 1 for the first seed (their Section 5.1
motivating example).  Table 1 reports the influence **normalized by |V|**.

Why this setting matters for us
-------------------------------
With ``p0 = 1, p1 = 0`` there is no Monte-Carlo noise anywhere: ``sigma`` is an exact deterministic
function of the seed set, so every comparison in this experiment is exact and a difference in the
numbers is a difference in the decision, never a difference in sampling luck.  The independent
cascade is the *least* fixed point of the threshold process, and because activation is monotone the
fixed point is unique.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

DATA = Path(__file__).resolve().parents[1] / "_data"

#: The paper's basic setting: p1 = 0, p0 = 1, K = 4 for medium networks, T = 8.
P0 = 1
P1 = 0
DEFAULT_K = 4
DEFAULT_T = 8

#: Table 1 of the paper, normalized influence (sigma / |V|).  Football and Polbooks have no OPT
#: column: the MIP optimum was only computed for the five small networks.
PAPER_TABLE1 = {
    "football": {"Greedy": 0.3217, "DPIM": 0.3913, "Random+": 0.0926, "RL4IM": 0.5130,
                 "RL4CCIM": 1.0000, "OPT": None},
    "polbooks": {"Greedy": 0.4571, "DPIM": 0.6762, "Random+": 0.1486, "RL4IM": 0.7429,
                 "RL4CCIM": 0.8190, "OPT": None},
}

#: Reference sizes, used to confirm the downloaded graphs are the canonical ones.
EXPECTED = {"football": (115, 613), "polbooks": (105, 441)}


def load_graph(name: str) -> nx.Graph:
    """Load one of the paper's two public graphs with contiguous integer labels.

    Labels are re-assigned by ascending numeric value of the original GML id so that node order is
    reproducible; the graph is made simple (no self-loops or parallel edges).
    """
    if name not in EXPECTED:
        raise KeyError(f"unknown graph {name!r}; expected one of {sorted(EXPECTED)}")
    path = DATA / f"{name}.gml"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the canonical GML must be present")
    graph = nx.Graph(nx.read_gml(path))
    graph.remove_edges_from(nx.selfloop_edges(graph))

    def sort_key(label):
        text = str(label)
        return (0, int(text), "") if text.lstrip("-").isdigit() else (1, 0, text)

    mapping = {old: new for new, old in enumerate(sorted(graph.nodes(), key=sort_key))}
    graph = nx.relabel_nodes(graph, mapping)
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")

    n, m = graph.number_of_nodes(), graph.number_of_edges()
    if (n, m) != EXPECTED[name]:
        raise ValueError(f"{name}: got n={n}, m={m}, expected {EXPECTED[name]}; "
                         f"this is not the graph the paper used")
    return graph


def cascade(graph: nx.Graph, seeds, K: int = DEFAULT_K, stats: dict | None = None) -> set:
    """The deterministic threshold-``K`` activation set, seeds included.

    A node activates exactly when it has at least ``K`` active neighbours.  Activation is monotone, so
    the least fixed point is unique and the result does not depend on the order of processing.

    **What this rule requires, and what it does not.**  It requires ``K`` *active neighbours*.  It does
    **not** require the node to lie in the graph's ``K``-core, and a large ``K``-core does not imply
    that anything spreads.  Two counterexamples, both kept as regression tests:

    * a four-leaf star with ``K = 4`` and all four leaves seeded: the 4-core is **empty**, yet the
      centre activates and the whole graph ends up active;
    * ``K5`` with ``K = 4`` and three seeds: the 4-core is the **entire** graph, yet nothing beyond the
      three seeds activates.

    ``K``-core size is therefore a structural description only, never a propagation gate.

    If ``stats`` is supplied it is filled with work counters for cost measurement.  Passing it changes
    no result.
    """
    active: set[int] = set()
    active_count = {v: 0 for v in graph.nodes()}
    queue: list[int] = []

    if stats is not None:
        stats.clear()
        stats.update({"nodes_initialised": graph.number_of_nodes(), "edges_scanned": 0,
                      "nodes_dequeued": 0, "waves": 0, "activating_rounds": 0, "seed_count": 0})

    def activate(v: int) -> None:
        if v in active:
            return
        active.add(v)
        queue.append(v)
        for u in graph.neighbors(v):
            active_count[u] += 1
            if stats is not None:
                stats["edges_scanned"] += 1

    for s in seeds:
        activate(int(s))
    if stats is not None:
        stats["seed_count"] = len(active)

    head = 0
    while head < len(queue):
        wave_end = len(queue)
        before = len(active)
        while head < wave_end:
            v = queue[head]
            head += 1
            if stats is not None:
                stats["nodes_dequeued"] += 1
            for u in graph.neighbors(v):
                if u not in active and active_count[u] >= K:
                    activate(u)
        if stats is not None:
            stats["waves"] += 1
            if len(active) > before:
                stats["activating_rounds"] += 1
    return active


def sigma(graph: nx.Graph, seeds, K: int = DEFAULT_K) -> int:
    """``sigma(G, S)`` -- the number of active nodes at the end, seeds included."""
    return len(cascade(graph, seeds, K))


def normalized(graph: nx.Graph, seeds, K: int = DEFAULT_K) -> float:
    """``sigma(G, S) / |V|``, the quantity Table 1 reports."""
    return sigma(graph, seeds, K) / graph.number_of_nodes()


def effective_candidates(graph: nx.Graph, K: int = DEFAULT_K) -> list[int]:
    """The paper's solution-filtering set ``A`` (Algorithm 1, lines 4-5).

    ``A = union of N(v) over all v with |N(v)| >= K``.  Returned for reference and ablation; the
    DQN in this experiment deliberately uses the FULL action space with a legality mask instead, so
    that the paper's filter is not silently doing the work for it.
    """
    out: set[int] = set()
    for v in graph.nodes():
        if graph.degree(v) >= K:
            out.update(graph.neighbors(v))
    return sorted(out)


def degree_ranking(graph: nx.Graph) -> list[int]:
    """Nodes by descending degree, ties by ascending id."""
    return sorted(graph.nodes(), key=lambda v: (-graph.degree(v), v))
