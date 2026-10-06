"""N02 shared machinery: exact t quantiles (computed, never transcribed) and reachable-set helpers.

The quantile is computed from the closed-form t CDF by bisection at call time.  No constant is copied
from a table, which is the failure mode that produced the N01 statistics error.
"""
from __future__ import annotations

import math
from collections import deque


# ------------------------------------------------------------------ t distribution

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        if abs(d * c - 1.0) < 1e-16:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    fr = math.exp(lb + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return fr * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_cdf(df: int, t: float) -> float:
    z = df / (df + t * t)
    if t <= 0:
        return 0.5 * _betai(df / 2.0, 0.5, z)
    return 1.0 - 0.5 * _betai(df / 2.0, 0.5, z)


def t_quantile(df: int, level: float) -> float:
    a, b = 0.0, 200.0
    for _ in range(300):
        m = 0.5 * (a + b)
        if t_cdf(df, m) > level:
            b = m
        else:
            a = m
        if b - a < 1e-14:
            break
    return 0.5 * (a + b)


def simultaneous_t(n_worlds: int, n_comparisons: int) -> float:
    """Two-sided t for a family of ``2 * n_comparisons`` intervals at simultaneous 95%.

    Each interval gets alpha = 0.05/(2*n_comparisons), so the two-sided quantile level is
    ``1 - 0.05/(4*n_comparisons)``.
    """
    level = 1.0 - 0.05 / (4 * n_comparisons)
    return t_quantile(n_worlds - 1, level)


# ------------------------------------------------------------------ reachability


def reach_within(graph, source: int, hops: int, include_source: bool = True) -> set[int]:
    """Nodes reachable from ``source`` within ``hops`` forward steps.

    ``include_source=True`` is the N02 convention: the 0-to-``hops`` receptive field contains the
    source itself.
    """
    dist = {source: 0}
    q = deque([source])
    out: set[int] = {source} if include_source else set()
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


def draw_live(graph, world_seed: int, rng_factory):
    """One live/dead draw per directed arc, in a fixed arc order, from a seeded stream."""
    rng = rng_factory(world_seed)
    thresh = [1.0 / graph.degree[v] for _, v, _ in graph.arc_index]
    return [rng.random() < thresh[i] for i in range(graph.n_arcs)]


def live_adjacency(graph, live):
    adj: list[list[int]] = [[] for _ in range(graph.n)]
    for (u, v, arc_id) in graph.arc_index:
        if live[arc_id]:
            adj[u].append(v)
    return adj


def spread(adj, seeds, max_rounds: int = 100):
    """Activated set size (seeds included) and the BFS depth reached, in one live-edge world."""
    seen = bytearray(len(adj))
    depth = [0] * len(adj)
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
        for u in adj[v]:
            if not seen[u]:
                seen[u] = 1
                depth[u] = nd
                if nd > max_depth:
                    max_depth = nd
                count += 1
                q.append(u)
    return count, max_depth


def stats(xs, t_ord: float, t_adj: float):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return {
        "n": n, "mean": m, "sd": sd, "se": se,
        "t_ordinary": t_ord, "t_adjusted": t_adj,
        "ci_low": m - t_ord * se, "ci_high": m + t_ord * se,
        "adj_low": m - t_adj * se, "adj_high": m + t_adj * se,
    }


def signed_reversal_magnitude(dc: float, dd: float) -> float:
    """N02's ``r``: positive when the two point estimates already have opposite signs.

    ``r = max{ min(-dc, dd), min(dc, -dd) }``.  For (dc, dd) = (-6, +6) it is +6; for (+1, +2) and
    (-1, -2) it is -1.
    """
    return max(min(-dc, dd), min(dc, -dd))
