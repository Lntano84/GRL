"""Regression tests for the CCIM diffuser: the two K-core counterexamples, boundaries, and truth.

The two counterexamples exist because an earlier report in this project claimed that a node can only
be activated if it lies in the graph's ``K``-core, and used the ``K``-core fraction to predict whether
a threshold-``K`` cascade would spread.  Both halves of that claim are false, and these tests pin the
falsification down so the mistake cannot come back.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.model import DEFAULT_K, DEFAULT_T, cascade, degree_ranking, effective_candidates, \
    load_graph, sigma  # noqa: E402


# ------------------------------------------------------------------ K-core counterexamples
def test_star_empty_core_still_propagates():
    """A four-leaf star at K=4 has an EMPTY 4-core, yet seeding the leaves activates everything."""
    star = nx.star_graph(4)                       # node 0 is the centre, 1..4 are leaves
    assert nx.k_core(star, k=4).number_of_nodes() == 0, "the 4-core must be empty for this to bite"
    active = cascade(star, [1, 2, 3, 4], K=4)
    assert active == {0, 1, 2, 3, 4}


def test_complete_graph_full_core_no_propagation():
    """K5 at K=4 has the 4-core equal to the whole graph, yet three seeds activate nothing extra."""
    k5 = nx.complete_graph(5)
    assert nx.k_core(k5, k=4).number_of_nodes() == 5, "the 4-core must be the whole graph"
    assert cascade(k5, [0, 1, 2], K=4) == {0, 1, 2}


def test_core_size_is_not_a_gate():
    """The two graphs above show core size and propagation are independent in BOTH directions."""
    star, k5 = nx.star_graph(4), nx.complete_graph(5)
    assert (nx.k_core(star, k=4).number_of_nodes(), len(cascade(star, [1, 2, 3, 4], 4))) == (0, 5)
    assert (nx.k_core(k5, k=4).number_of_nodes(), len(cascade(k5, [0, 1, 2], 4))) == (5, 3)


# ------------------------------------------------------------------ boundary behaviour
def test_empty_seed_set():
    g = nx.path_graph(6)
    assert cascade(g, [], K=2) == set()


def test_k_equal_to_one_is_the_connected_component():
    """K=1 means one active neighbour suffices, so the closure is the whole component."""
    g = nx.Graph([(0, 1), (1, 2), (3, 4)])       # component {0,1,2} plus an isolated edge {3,4}
    assert cascade(g, [0], K=1) == {0, 1, 2}
    assert cascade(g, [0, 3], K=1) == {0, 1, 2, 3, 4}


def test_k_above_every_degree_gives_no_propagation():
    g = nx.path_graph(6)                          # max degree 2
    for seeds in ([0], [0, 1], [0, 1, 2]):
        assert cascade(g, seeds, K=3) == set(seeds)


def test_all_nodes_seeded_activates_everything():
    g = nx.gnp_random_graph(20, 0.2, seed=7)
    assert cascade(g, list(g.nodes()), K=DEFAULT_K) == set(g.nodes())


def test_seeds_are_always_in_the_result():
    g = nx.gnp_random_graph(30, 0.15, seed=11)
    seeds = [0, 5, 9]
    assert set(seeds) <= cascade(g, seeds, K=3)


def test_monotone_in_the_seed_set():
    """Adding a seed can never shrink the final active set in this model."""
    g = nx.gnp_random_graph(25, 0.2, seed=13)
    base = [1, 2]
    for extra in (3, 7, 11, 19):
        assert cascade(g, base, K=2) <= cascade(g, base + [extra], K=2)


def test_order_independence():
    g = nx.gnp_random_graph(40, 0.12, seed=17)
    seeds = [3, 8, 15, 22]
    results = {frozenset(cascade(g, random.Random(s).sample(seeds, len(seeds)), K=3))
               for s in range(10)}
    assert len(results) == 1, f"the fixed point must not depend on processing order: {results}"


# ------------------------------------------------------------------ statistical counters
def test_stats_do_not_change_the_result():
    g = nx.gnp_random_graph(30, 0.2, seed=19)
    seeds = [0, 4, 9, 14]
    plain = cascade(g, seeds, K=2)
    stats: dict = {}
    counted = cascade(g, seeds, K=2, stats=stats)
    assert plain == counted
    assert stats["nodes_initialised"] == g.number_of_nodes()
    assert stats["seed_count"] == len(set(seeds))
    assert stats["edges_scanned"] > 0 and stats["waves"] >= 1


# ------------------------------------------------------------------ agreement with brute force
def brute_force(graph, seeds, K):
    """Deliberately naive reference: rescan every node until a full pass changes nothing."""
    active = set(int(s) for s in seeds)
    changed = True
    while changed:
        changed = False
        for v in graph.nodes():
            if v in active:
                continue
            if sum(1 for u in graph.neighbors(v) if u in active) >= K:
                active.add(v)
                changed = True
    return active


@pytest.mark.parametrize("n,p", [(6, 0.3), (10, 0.2), (14, 0.15), (9, 0.5)])
def test_matches_brute_force_on_small_random_graphs(n, p):
    rng = random.Random(n * 100 + int(p * 100))
    g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10 ** 6))
    for K in (1, 2, 3, 4):
        for size in (1, 2, 3):
            if size > n:
                continue
            for seeds in itertools.combinations(range(n), size):
                assert cascade(g, seeds, K) == brute_force(g, seeds, K), \
                    f"n={n} p={p} K={K} seeds={seeds}"


@pytest.mark.parametrize("name", ["football", "polbooks"])
def test_matches_brute_force_on_the_paper_graphs(name):
    """Set equality, not just equal counts: the two implementations must agree node for node."""
    g = load_graph(name)
    nodes = list(g.nodes())
    rng = random.Random(hash(name) % 10_000)
    for _ in range(60):
        seeds = rng.sample(nodes, rng.randint(1, 12))
        for K in (2, 3, 4, 5):
            fast = cascade(g, seeds, K)
            slow = brute_force(g, seeds, K)
            assert fast == slow, f"{name} K={K} seeds={seeds}: symmetric difference " \
                                 f"{sorted(fast ^ slow)[:8]}"


# ------------------------------------------------------------------ loaded graphs are the papers'
@pytest.mark.parametrize("name,expected", [("football", (115, 613)), ("polbooks", (105, 441))])
def test_graphs_are_canonical(name, expected):
    g = load_graph(name)
    assert (g.number_of_nodes(), g.number_of_edges()) == expected
    assert not g.is_directed()
    assert nx.number_of_selfloops(g) == 0


def test_paper_reference_values_reproduce():
    """The two published Greedy values, which is what established the model in the first place."""
    football = load_graph("football")
    polbooks = load_graph("polbooks")
    by_degree = lambda g: sorted(g.nodes(), key=lambda v: (-g.degree(v), v))

    seeds_f, current = [], 0
    for _ in range(DEFAULT_T):
        best, best_s = None, current
        for v in football.nodes():
            if v in seeds_f:
                continue
            s = sigma(football, seeds_f + [v], DEFAULT_K)
            if s > best_s:
                best, best_s = v, s
        seeds_f.append(best)
        current = best_s
    assert sigma(football, seeds_f, DEFAULT_K) == 37
    assert abs(sigma(football, seeds_f, DEFAULT_K) / 115 - 0.3217) < 5e-5

    seeds_p, current = [], 0
    for _ in range(DEFAULT_T):
        best, best_s = None, current
        for v in polbooks.nodes():
            if v in seeds_p:
                continue
            s = sigma(polbooks, seeds_p + [v], DEFAULT_K)
            if s > best_s:
                best, best_s = v, s
        seeds_p.append(best)
        current = best_s
    # the earlier run reproduced 48 -> 0.4571 exactly; allow the empty-pool fallback to differ only
    # if it ever fires, which it must not on this graph
    assert sigma(polbooks, seeds_p, DEFAULT_K) == 48
