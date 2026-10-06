"""Directed 2-hop reachability, the N01 hand graph, and the summed-local-proxy arithmetic check.

Why this file exists separately from ``n01_common``
---------------------------------------------------
The N01 condition is defined on the **message-passing direction**, so the 2-hop region ``R2(x)`` is a
*forward* reachability set from ``x``.  On ca-GrQc each undirected edge becomes two propagation arcs,
so forward reachability in the directed version coincides with hop distance in the undirected one.
The hand graph is genuinely directed, and there the two notions differ sharply -- which is the whole
point of the construction -- so the directed reading is implemented explicitly and used everywhere.
"""

from __future__ import annotations

from collections import deque


class Digraph:
    """Forward adjacency, built from an undirected edge list interpreted as two arcs per edge."""

    def __init__(self, n: int, undirected_edges):
        self.n = n
        self.out: list[list[int]] = [[] for _ in range(n)]
        for u, v in undirected_edges:
            self.out[u].append(v)
            self.out[v].append(u)
        for i in range(n):
            self.out[i] = sorted(self.out[i])
        self.m = len(undirected_edges)

    def degree(self, v: int) -> int:
        return len(self.out[v])


def forward_within_hops(g: Digraph, source: int, hops: int) -> set[int]:
    """Nodes reachable from ``source`` in <= ``hops`` forward steps; ``source`` itself excluded.

    Directed reading: only ``source -> neighbour`` direction is traversed.  A node that can only reach
    ``source`` (a predecessor) is NOT in ``R2(source)``.
    """
    dist = {source: 0}
    q = deque([source])
    out: set[int] = set()
    while q:
        v = q.popleft()
        d = dist[v]
        if d == hops:
            continue
        for u in g.out[v]:
            if u not in dist:
                dist[u] = d + 1
                out.add(u)
                q.append(u)
    return out


def disjoint_condition(r2: dict[int, set[int]], a: int, b: int, c: int, d: int) -> dict:
    left = r2[a] | r2[b]
    right = r2[c] | r2[d]
    inter = left & right
    return {
        "holds": not inter,
        "intersection_size": len(inter),
        "intersection": sorted(inter)[:8],
        "r2": {"a": sorted(r2[a]), "b": sorted(r2[b]), "c": sorted(r2[c]), "d": sorted(r2[d])},
        "r2_sizes": {"a": len(r2[a]), "b": len(r2[b]), "c": len(r2[c]), "d": len(r2[d])},
    }


# --------------------------------------------------------------------------- the hand graph

HAND_CANDIDATES = ("a", "b", "c", "d")
HAND_LEAVES = {
    "P": ["P0", "P1", "P2", "P3", "P4"],
    "Q": ["Q0", "Q1", "Q2", "Q3", "Q4"],
}


def build_hand_graph():
    """The frozen 24-node / 22-edge construction.

    Four directed paths of length three, ``x -> x1 -> x2 -> hub``, for ``x`` in {a,c} into ``P`` and
    ``x`` in {b,d} into ``Q``; each hub then points at five private leaves.  No reverse edges and no
    other edges.  Every edge is used in exactly one direction, so this *is* the undirected edge list
    with a single orientation imposed -- ``Digraph`` will add the reverse arcs, which is why the path
    structure is what decides the 2-hop regions.
    """
    nodes = []
    nodes += ["a", "a1", "a2", "c", "c1", "c2", "b", "b1", "b2", "d", "d1", "d2", "P", "Q"]
    for hub in ("P", "Q"):
        nodes += HAND_LEAVES[hub]
    idx = {name: i for i, name in enumerate(nodes)}

    forward_edges = []
    for x, hub in (("a", "P"), ("c", "P"), ("b", "Q"), ("d", "Q")):
        forward_edges.append((x, f"{x}1"))
        forward_edges.append((f"{x}1", f"{x}2"))
        forward_edges.append((f"{x}2", hub))
    for hub, leaves in HAND_LEAVES.items():
        for leaf in leaves:
            forward_edges.append((hub, leaf))
    assert len(forward_edges) == 22, len(forward_edges)

    # A directed edge list used to build forward adjacency only: strip the automatic reverse arcs
    # that Digraph would add, by wiring the arcs directly.
    g = Digraph(len(nodes), [])
    for u, v in forward_edges:
        g.out[idx[u]].append(idx[v])
    for i in range(g.n):
        g.out[i] = sorted(g.out[i])
    g.m = len(forward_edges)

    labels = {i: name for name, i in idx.items()}
    return g, idx, labels, forward_edges


def hand_expected():
    """The four activation counts the construction predicts under p = 1, U = empty, k = 2."""
    return {("a", "c"): 12, ("b", "c"): 18, ("a", "d"): 18, ("b", "d"): 12}


# --------------------------------------------------------------------------- proxy arithmetic


def summed_local_proxy_differences(phi: dict, a, b, c, d, u_seeds):
    """The two proxy differences for ANY fixed-parameter two-layer summed local proxy.

    Such a proxy is a function ``Sigma_hat(S) = sum_{v in S} phi(v) + const`` where the const does not
    depend on S (it is whatever the shared background contributes, and the two-layer receptive field
    cannot see across the a/b-side and c/d-side split by the structural condition).  Therefore

        delta_hat_c = Sigma_hat(S_ac) - Sigma_hat(S_bc) = phi(a) - phi(b)
        delta_hat_d = Sigma_hat(S_ad) - Sigma_hat(S_bd) = phi(a) - phi(b)

    so ``delta_hat_c == delta_hat_d`` **identically**, for every parameter setting and every random
    initialisation.  This computes that arithmetic for a given score vector ``phi``.  It is an
    arithmetic identity check on the DEFINITION of the proxy class -- it is NOT a run of the official
    SIMBA network, and the report says so explicitly.
    """
    sets = {
        "S_ac": (*u_seeds, a, c), "S_bc": (*u_seeds, b, c),
        "S_ad": (*u_seeds, a, d), "S_bd": (*u_seeds, b, d),
    }
    pred = {k: sum(phi[v] for v in v_) for k, v_ in sets.items()}
    if phi.get("__scale__") is not None:
        for k in pred:
            pred[k] *= phi["__scale__"]
    d_c = pred["S_ac"] - pred["S_bc"]
    d_d = pred["S_ad"] - pred["S_bd"]
    return {
        "S_hat": {k: round(v, 12) for k, v in pred.items()},
        "delta_hat_c": round(d_c, 12),
        "delta_hat_d": round(d_d, 12),
        "difference": round(d_c - d_d, 12),
    }
