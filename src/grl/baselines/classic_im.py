"""Three classic IM baselines, implemented so the comparison matches the source model's.

The source model (Inf. Sci. 744 (2026) 123375, Sec. 7.3) compares its methods against
``Random``, ``Max_Degree``, ``IMRank`` and ``PageRank``, and also cites ``CELF``.  Our draft so
far compares only against out-degree (its ``Max_Degree``), exact Monte-Carlo greedy and our own
closed form, so three of its five baselines were missing.  This module supplies them.

A caveat that belongs in the paper, not just here
-------------------------------------------------
CELF (Leskovec et al., KDD 2007) is a *lazy-evaluation* speedup for greedy maximisation of a
**monotone submodular** function.  Its guarantee comes from submodularity: if a candidate's
marginal gain has dropped below the best value still in the priority queue, it can be skipped
without loss.  The overexposure objective is **not** submodular --- the source model's own
Theorem 3 states this --- so the lazy bound is not valid and CELF applied here is a heuristic,
not an exact accelerator.  We implement it and report it as such.  (The source model's Lemma 2
establishes submodularity only for ``sigma^kappa`` and ``sigma^tau`` individually, not for the
objective, so citing CELF for the objective is a gap in its own baseline set.)

IMRank
------
We implement the influence-ranking formulation as described in the IM literature: each node
carries a rank; ranks are propagated along out-edges weighted by edge influence; nodes are then
selected in order of their accumulated rank.  **The exact update rule varies between the papers
that introduce IMRank**, and we could not verify the source model's cited variant from the
material on hand, so this is labelled as our reading, exactly as the upper-bound arms are.
It is a heuristic ranking method; no optimisation is performed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from grl.diffusion.overexposure import (  # noqa: E402
    estimate_overexposure_spread_over_configs,
)


def random_seeds(graph: nx.DiGraph, pool: list[int], budget: int,
                 rng: random.Random) -> list[int]:
    """The source model's ``Random`` baseline."""
    return rng.sample(pool, min(budget, len(pool)))


def max_degree_seeds(graph: nx.DiGraph, pool: list[int], budget: int) -> list[int]:
    """The source model's ``Max_Degree`` baseline (out-degree, our reading of 'degree')."""
    degree = dict(graph.out_degree())
    return sorted(pool, key=lambda v: (-degree[v], v))[:budget]


def pagerank_seeds(
    graph: nx.DiGraph, pool: list[int], budget: int, alpha: float = 0.85,
    iterations: int = 100, tol: float = 1e-10,
) -> list[int]:
    """The source model's ``PageRank`` baseline.

    Edge weights are used as transition probabilities, which is the natural reading for a
    weighted graph.  Implemented directly rather than via ``networkx`` so the weighting is
    explicit and danging nodes are handled deterministically.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return []
    out_weight = {v: 0.0 for v in nodes}
    for u, v, data in graph.edges(data=True):
        out_weight[u] += float(data.get("weight", 0.0))

    rank = {v: 1.0 / n for v in nodes}
    for _ in range(iterations):
        nxt = {v: (1.0 - alpha) / n for v in nodes}
        dangling = 0.0
        for v in nodes:
            if out_weight[v] <= 0.0:
                dangling += rank[v]
        share = alpha * dangling / n
        for v in nodes:
            nxt[v] += share
        for u, v, data in graph.edges(data=True):
            if out_weight[u] > 0.0:
                nxt[v] += alpha * rank[u] * float(data.get("weight", 0.0)) / out_weight[u]
        delta = sum(abs(nxt[v] - rank[v]) for v in nodes)
        rank = nxt
        if delta < tol:
            break

    return sorted(pool, key=lambda v: (-rank[v], v))[:budget]


def imrank_seeds(
    graph: nx.DiGraph, pool: list[int], budget: int, rounds: int = 20,
) -> list[int]:
    """Our reading of the IMRank heuristic.

    Iteratively: rank nodes by accumulated influence, then propagate each node's rank to its
    out-neighbours along edge weights.  The top-``budget`` nodes by final rank are returned.
    No spread evaluation is performed, so this costs zero cascades.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return []
    rank = {v: 1.0 / n for v in nodes}
    for _ in range(rounds):
        nxt = {v: 0.0 for v in nodes}
        for u, v, data in graph.edges(data=True):
            nxt[v] += rank[u] * float(data.get("weight", 0.0))
        total = sum(nxt.values())
        if total <= 0.0:
            break
        rank = {v: nxt[v] / total for v in nodes}
    return sorted(pool, key=lambda v: (-rank[v], v))[:budget]


def celf_seeds(
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    mc: int,
    base_seed: int,
    allow_negative_gain: bool = False,
) -> tuple[list[int], int]:
    """CELF, implemented as an *exact* greedy because the lazy bound is invalid here.

    Returns ``(seeds, cascades)``; ``cascades`` counts candidate spread evaluations times ``mc``.

    Why there is no lazy evaluation
    -------------------------------
    CELF (Leskovec et al., KDD 2007) accelerates greedy maximisation of a **monotone submodular**
    function by keeping a stale upper bound per candidate: if the recomputed gain of the heap
    head still dominates the runner-up's stale gain, the head is the true argmax and the
    remaining candidates can be skipped.  That argument depends entirely on submodularity and
    monotonicity.  The overexposure objective has neither -- the source model's Theorem 3 states
    it is neither submodular nor supermodular, and its Theorem 2 states it is non-monotone -- so
    the skip test is unsound: a candidate whose gain dropped can rise again later, and the bound
    gives no guarantee.

    We therefore implement CELF as the exact greedy it is meant to accelerate.  It produces the
    same seed set as plain greedy at the same cost, which is the honest baseline; reporting a
    lazy version would give an artificially *worse* method whose shortfall came from a violated
    assumption rather than from the algorithm.  The methodological point -- that CELF is cited in
    this literature but its precondition does not hold for this objective -- is made in the
    paper's related work rather than baked into a fragile baseline.

    A greedy step may also encounter a situation where the best available candidate has a
    negative marginal gain.  Since the objective is non-monotone that is a real possibility, and
    adding the seed would reduce the spread; the search stops unless ``allow_negative_gain``.
    """
    seeds: list[int] = []
    remaining = list(pool)
    cascades = 0

    for step in range(budget):
        if not remaining:
            break
        configurations = [list(seeds)] + [[*seeds, c] for c in remaining]
        estates = estimate_overexposure_spread_over_configs(
            graph, configurations, mc, base_seed + 17 * step
        )
        cascades += len(remaining) * mc
        base = estates[0]["mean"]
        gains = [estates[i + 1]["mean"] - base for i in range(len(remaining))]
        best = max(range(len(gains)), key=lambda i: gains[i])
        if gains[best] <= 0.0 and not allow_negative_gain:
            break
        seeds.append(remaining.pop(best))
    return seeds, cascades
