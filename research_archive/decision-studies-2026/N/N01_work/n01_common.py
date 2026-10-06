"""N01 shared machinery: graph loading, 2-hop reachability, and live-edge independent cascade.

No third-party dependencies.  The environment has no ``networkx``, ``numpy`` or ``scipy`` and no
network access, so everything here is pure standard library.  One consequence is recorded in the
report as a limitation: the diffuser is not cross-checked against ``networkx``.

Design decisions that matter for the N01 claim
----------------------------------------------
* **Graph reading.**  ``ca-GrQc.txt`` stores each undirected pair once (the header says "each
  unordered pair of nodes is saved once").  The CCIM probe symmetrised it, and so do we, which is why
  the undirected edge count is half the data rows.  The raw sha256 is frozen in the cases file.
* **Propagation directions.**  Each undirected edge becomes two directed propagation arcs.  Every arc
  ``(u, v)`` is live with probability ``1 / deg(v)`` where ``deg`` is the **undirected** degree, i.e.
  the number of distinct neighbours.  The two directions of a pair are sampled independently.
* **Common random numbers.**  ONE live/dead state is drawn per directed arc per world, from a
  ``random.Random(world_seed)`` stream.  All four seed sets of a comparison are then evaluated by
  deterministic reachability in that SAME state, so the four activations share every edge outcome
  exactly.  This is the live-edge view of IC (the set reachable from the seeds through live arcs is
  exactly the set that ends up active), not a re-seeded round-by-round simulation.  Re-seeding per
  seed set is explicitly NOT used: different propagation paths consume random draws in different
  orders, which would silently decouple the four sets.
* **Horizon.**  BFS depth is capped at 100, matching the stated "at most 100 rounds" limit.  The
  live-edge reachability equivalence is only exact for the unbounded horizon, so the achieved maximum
  depth is recorded per world; when it stays below 100 the cap provably never bound.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from pathlib import Path

# --------------------------------------------------------------------------- frozen constants

SOURCE_FILE = Path(r"C:\Users\windows\Desktop\_grl_merge\merged\data\paper\ca-GrQc.txt")

#: Verified against the existing CCIM probe manifest (results/ca_grqc_probe.json).
EXPECTED_SHA256 = "F8CE6E931E068B878044B783DA99EF603F566C87BCBCE7991CD53720879F1660"
EXPECTED_N = 5241
EXPECTED_M = 14484

MAX_ROUNDS = 100
K_SEEDS = 10
N_DEV_WORLDS = 256
N_CONFIRM_WORLDS = 2048
DEV_WORLD_IDS = tuple(range(0, N_DEV_WORLDS))
CONFIRM_WORLD_IDS = tuple(range(1_000_000, 1_000_000 + N_CONFIRM_WORLDS))


# --------------------------------------------------------------------------- graph


def parse_ca_grqc(path: Path = SOURCE_FILE):
    """Return ``(arcs, undirected_edges, node_ids, manifest)`` with the raw file hashed."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    rows: list[tuple[int, int]] = []
    comment_lines = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            comment_lines += 1
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        rows.append((int(parts[0]), int(parts[1])))

    self_loops = [r for r in rows if r[0] == r[1]]
    non_loop = [r for r in rows if r[0] != r[1]]
    undirected = sorted({tuple(sorted(r)) for r in non_loop})
    node_ids = sorted({x for r in non_loop for x in r})

    # Directed propagation arcs: both orientations of every undirected edge.
    arcs: list[tuple[int, int]] = []
    for u, v in undirected:
        arcs.append((u, v))
        arcs.append((v, u))

    manifest = {
        "file": str(path),
        "file_name": path.name,
        "sha256": digest,
        "comment_lines": comment_lines,
        "data_rows": len(rows),
        "self_loop_rows": len(self_loops),
        "self_loop_nodes": sorted({r[0] for r in self_loops}),
        "distinct_node_ids_in_file": len({x for r in rows for x in r}),
        "nodes_in_simple_undirected_graph": len(node_ids),
        "undirected_edges": len(undirected),
        "directed_propagation_arcs": len(arcs),
        "expected": {"sha256": EXPECTED_SHA256, "n": EXPECTED_N, "m": EXPECTED_M},
        "matches_expected": (digest == EXPECTED_SHA256 and len(node_ids) == EXPECTED_N
                             and len(undirected) == EXPECTED_M),
    }
    return arcs, undirected, node_ids, manifest


class Graph:
    """Adjacency + degree, with contiguous integer labels over the simple undirected graph."""

    def __init__(self, arcs, undirected, node_ids):
        # Node mapping: ascending original id -> contiguous 0..n-1.  Frozen in the cases file.
        self.original_ids = list(node_ids)
        self.node_to_idx = {orig: i for i, orig in enumerate(self.original_ids)}
        self.n = len(node_ids)
        self.arcs_original = [tuple(a) for a in arcs]
        self.undirected_original = [tuple(e) for e in undirected]

        self.adj: list[list[int]] = [[] for _ in range(self.n)]
        self.degree = [0] * self.n
        self.arc_index: list[tuple[int, int, int]] = []   # (u_idx, v_idx, arc_id)
        for arc_id, (u, v) in enumerate(arcs):
            ui = self.node_to_idx[u]
            vi = self.node_to_idx[v]
            self.arc_index.append((ui, vi, arc_id))
            self.adj[ui].append(vi)
        for i in range(self.n):
            self.adj[i] = sorted(self.adj[i])
            self.degree[i] = len(self.adj[i])
        self.m = len(undirected)
        self.n_arcs = len(arcs)

    def degree_ranking(self) -> list[int]:
        """Descending degree, ties broken by ascending contiguous index (the frozen mapping)."""
        return sorted(range(self.n), key=lambda v: (-self.degree[v], v))

    def components(self) -> list[list[int]]:
        seen = [False] * self.n
        comps: list[list[int]] = []
        for s in range(self.n):
            if seen[s]:
                continue
            comp: list[int] = []
            q = deque([s])
            seen[s] = True
            while q:
                v = q.popleft()
                comp.append(v)
                for u in self.adj[v]:
                    if not seen[u]:
                        seen[u] = True
                        q.append(u)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        return comps


# --------------------------------------------------------------------------- structure


def within_hops(graph: Graph, source: int, hops: int, include_source: bool = False) -> set[int]:
    """Nodes reachable from ``source`` within ``hops`` steps.

    ``include_source=False`` (default) returns the proper successors only, excluding ``source``.
    ``include_source=True`` returns the **0-to-``hops`` receptive field**, which is what a ``hops``-layer
    message-passing stack actually sees: layer 0 is the node itself, so the source is always inside its
    own receptive field.

    The frozen N01 condition is stated on ``R2``, the set a node can *influence*, which excludes the
    node itself.  ``audit_claims.py`` re-checks all 40 comparisons under the inclusive reading as well;
    they hold either way, so the frozen comparisons do not depend on which convention is meant.
    """
    dist = {source: 0}
    q = deque([source])
    out: set[int] = set()
    if include_source:
        out.add(source)
    while q:
        v = q.popleft()
        d = dist[v]
        if d == hops:
            continue
        for u in graph.adj[v]:
            if u not in dist:
                dist[u] = d + 1
                out.add(u)
                q.append(u)
    return out


def check_2hop_disjoint(r2: dict[int, set[int]], a: int, b: int, c: int, d: int) -> dict:
    """The frozen structural condition ``(R2(a) u R2(b)) n (R2(c) u R2(d)) = empty``."""
    left_ab = r2[a] | r2[b]
    left_cd = r2[c] | r2[d]
    inter_ab_cd = left_ab & left_cd
    return {
        "holds": not inter_ab_cd,
        "intersection_size": len(inter_ab_cd),
        "intersection_sample": sorted(inter_ab_cd)[:5],
        "r2_a_size": len(r2[a]), "r2_b_size": len(r2[b]),
        "r2_c_size": len(r2[c]), "r2_d_size": len(r2[d]),
        "union_ab_size": len(left_ab), "union_cd_size": len(left_cd),
        "also_pairwise_disjoint_candidates":
            not (r2[a] | r2[b]) & (r2[c] | r2[d]),
    }


# --------------------------------------------------------------------------- diffusion


def draw_live_arcs(graph: Graph, world_seed: int):
    """One live/dead draw per directed arc, in a fixed arc order, from a seeded stream.

    ``random.Random(world_seed)`` is consumed exactly once per arc in arc-id order, so the state for a
    given world is an exact function of ``world_seed`` alone -- independent of which seed set is being
    evaluated.  That is what makes the four sets share a world.
    """
    rng = random.Random(world_seed)
    thresh = [1.0 / graph.degree[v] for _, v, _ in graph.arc_index]
    return [rng.random() < thresh[i] for i in range(graph.n_arcs)]


def build_live_adjacency(graph: Graph, live) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(graph.n)]
    for (u, v, arc_id) in graph.arc_index:
        if live[arc_id]:
            adj[u].append(v)
    return adj


def ic_spread(live_adj, seeds, max_rounds: int = MAX_ROUNDS):
    """Activated set (seeds included) and the achieved BFS depth, in the live-edge world.

    The set reachable from the seeds through live arcs is exactly the final IC active set; the BFS
    depth is the number of propagation rounds actually used.
    """
    seen = bytearray(len(live_adj))
    depth = [0] * len(live_adj)
    q = deque()
    for s in seeds:
        if not seen[s]:
            seen[s] = 1
            q.append(s)
    count = len(q)
    max_depth = 0
    while q:
        v = q.popleft()
        d = depth[v]
        if d >= max_rounds:
            continue
        nd = d + 1
        for u in live_adj[v]:
            if not seen[u]:
                seen[u] = 1
                depth[u] = nd
                if nd > max_depth:
                    max_depth = nd
                count += 1
                q.append(u)
    return count, max_depth
