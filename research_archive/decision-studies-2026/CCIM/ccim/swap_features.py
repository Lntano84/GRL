"""Features for ranking a single swap ``(u, v)`` given the current set ``S``.

Every feature here is computable from ``S`` and the graph **without running the diffuser**.  Nothing in
the vector is derived from the outcome of the query being predicted: no query position, no acceptance
flag, and no post-swap spread.  That is the leakage rule for this pilot, and
``tests/test_swap_features.py`` checks it by confirming the features are unchanged when the labels are
permuted.

The one-round threshold quantities are the ones the CCIM rule actually depends on: a node activates
when it has at least ``K`` active neighbours, so what matters about ``v`` is how many of its neighbours
are already in ``S``, and how many nodes it would push across the threshold on its own.

Two omissions that are deliberate
---------------------------------
* ``active - K`` is not carried alongside ``active``: with ``K`` fixed the two differ by a constant and
  are exactly collinear, so the second adds no information.
* the state's own ``sigma(S)`` is **not** a feature.  Within one batch every candidate shares the same
  ``S``, so a state-constant term shifts every score equally and cannot change the within-batch order.
  Carrying it would only add a quantity whose training values were not recorded, i.e. a silent
  distribution shift between training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .model import DEFAULT_K

FEATURE_NAMES = (
    "deg_v", "deg_u", "deg_diff",
    "active_neighbours_v", "active_neighbours_u",
    "one_round_gain_v", "one_round_loss_u",
    "f1_delta",
)


@dataclass
class StateContext:
    """Per-state quantities, built once and reused for every candidate of that state."""

    graph: nx.Graph
    S: frozenset
    K: int
    degree: dict
    neighbours: dict
    active_count: dict          # node -> number of neighbours currently in S

    def features(self, u: int, v: int) -> list[float]:
        """O(deg u + deg v).  Only ``S``-derived quantities; the label never enters."""
        K, S = self.K, self.S
        ac = self.active_count
        Nu, Nv = self.neighbours[u], self.neighbours[v]
        ac_v, ac_u = ac[v], ac[u]

        # what v's addition alone would push over the threshold in one round (u kept)
        gain_v = sum(1 for x in Nv if x not in S and x != v and ac[x] + 1 >= K)
        # nodes whose activation currently rests exactly on u: one step below the threshold
        loss_u = sum(1 for x in Nu if x not in S and x != u and ac[x] == K)

        # exact one-round F1 delta q(u,v|S) = F1(S') - F1(S); only N(u)|N(v)|{u,v} can flip
        Sp = (S - {u}) | {v}
        delta = 0
        for x in Nu | Nv | {u, v}:
            before = (x not in S) and ac[x] >= K
            after = (x not in Sp) and (ac[x] - (x in Nu) + (x in Nv)) >= K
            delta += after - before

        return [
            float(self.degree[v]), float(self.degree[u]),
            float(self.degree[v] - self.degree[u]),
            float(ac_v), float(ac_u),
            float(gain_v), float(loss_u),
            float(delta),
        ]


def build_context(graph: nx.Graph, S, K: int = DEFAULT_K) -> StateContext:
    """Compute the per-state tables once: O(n + m), independent of the candidate pool size."""
    Sset = frozenset(int(s) for s in S)
    degree = {v: graph.degree(v) for v in graph.nodes()}
    neighbours = {v: set(graph.neighbors(v)) for v in graph.nodes()}
    active_count = {v: 0 for v in graph.nodes()}
    for s in Sset:
        for x in neighbours[s]:
            active_count[x] += 1
    return StateContext(graph=graph, S=Sset, K=K,
                        degree=degree, neighbours=neighbours, active_count=active_count)
