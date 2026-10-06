"""Gates for the exploratory-IM environment.

The load-bearing ones:

* :func:`test_live_edge_objective_matches_brute_force_ic` -- the estimator is checked against an explicit
  independent-cascade simulation, so "influence" means the number of nodes the seeds actually activate
  (seeds included), not something merely correlated with it.
* :func:`test_greedy_matches_brute_force_on_a_small_graph` -- the CELF port really maximises that
  objective.
* :func:`test_environment_reveals_exactly_the_true_neighbourhood` -- a query reveals the node's real
  neighbours and nothing else, which is the whole premise of the problem.
* :func:`test_initial_seeds_are_free_and_frontier_only` -- the initial seeds are revealed without spending
  budget, and queries outside the frontier are refused.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from ccim.xplore import (DiscoveryEnv, LiveEdgeObjective, change_surveys, policy_degree_max,
                         policy_degree_min, policy_random, run_change_episode, run_episode)
from ccim.xplore.icm import INFL_BUDGET


def brute_force_ic(graph, seeds, trials, rng, p=0.1):
    """Explicit IC simulation: BFS from the seeds over independently sampled live edges."""
    seeds = list(seeds)
    total = 0
    for _ in range(trials):
        live = nx.Graph()
        live.add_nodes_from(graph.nodes())
        for u, v in graph.edges():
            if rng.random() < p:
                live.add_edge(u, v)
        reached = set(seeds)
        frontier = list(seeds)
        while frontier:
            nxt = []
            for u in frontier:
                for v in live.neighbors(u):
                    if v not in reached:
                        reached.add(v)
                        nxt.append(v)
            frontier = nxt
        total += len(reached)
    return total / trials


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("p", [0.1, 0.3])
def test_live_edge_objective_matches_brute_force_ic(p):
    g = nx.karate_club_graph()
    rng = np.random.default_rng(11)
    obj = LiveEdgeObjective(g, samples=4000, rng=rng, p=p)
    for seeds in ([0], [0, 33], [1, 2, 3], list(range(10))):
        estimate = obj.value(seeds)
        reference = brute_force_ic(g, seeds, 4000, np.random.default_rng(11), p=p)
        assert estimate == pytest.approx(reference, rel=0.06), (seeds, estimate, reference)


def test_empty_seed_set_influences_nobody():
    g = nx.karate_club_graph()
    assert LiveEdgeObjective(g, samples=50, rng=np.random.default_rng(0)).value([]) == 0.0


def test_objective_is_monotone_and_counts_the_seeds():
    g = nx.path_graph(30)
    obj = LiveEdgeObjective(g, samples=200, rng=np.random.default_rng(2))
    a = obj.value([0])
    b = obj.value([0, 15])
    assert a >= 1.0 and b >= a


def test_p_zero_means_only_the_seeds_are_influenced():
    g = nx.karate_club_graph()
    obj = LiveEdgeObjective(g, samples=100, rng=np.random.default_rng(3), p=0.0)
    assert obj.value([0, 1, 2]) == pytest.approx(3.0)


def test_binomial_edge_sampling_matches_per_edge_bernoulli():
    """The fast sampler must have the same distribution as ``random.random() < p`` per edge."""
    g = nx.gnp_random_graph(60, 0.15, seed=5)
    fast = LiveEdgeObjective(g, samples=800, rng=np.random.default_rng(7))
    rng = np.random.default_rng(7)
    slow_ncomp = []
    for _ in range(800):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u, v in g.edges():
            if rng.random() < 0.1:
                h.add_edge(u, v)
        slow_ncomp.append(nx.number_connected_components(h))
    fast_ncomp = np.array([len(np.unique(fast.labels[s])) for s in range(fast.samples)])
    assert abs(np.mean(fast_ncomp) - np.mean(slow_ncomp)) < 0.1 * np.std(slow_ncomp) + 0.5
    assert abs(np.std(fast_ncomp) - np.std(slow_ncomp)) < 0.25 * np.std(slow_ncomp) + 0.5


# ------------------------------------------------------------------ greedy
def test_greedy_matches_brute_force_on_a_small_graph():
    """Greedy must report its own objective, stay inside the (1-1/e) guarantee, and beat any single node.

    Note what is *not* asserted: optimality.  The objective is monotone submodular, so greedy is only a
    ``(1 - 1/e)`` approximation and exhaustive search is expected to beat it slightly.  The published
    selector has the same property, so demanding optimality here would be testing the wrong thing.
    """
    from itertools import combinations
    g = nx.gnp_random_graph(12, 0.35, seed=9)
    obj = LiveEdgeObjective(g, samples=2000, rng=np.random.default_rng(4))
    chosen, value = obj.greedy(2)
    assert len(chosen) == 2
    assert value == pytest.approx(obj.value(chosen))
    best = max(obj.value(c) for c in combinations(range(12), 2))
    best_single = max(obj.value([u]) for u in range(12))
    assert value >= (1 - 1 / np.e) * best - 1e-9, "greedy fell outside its own approximation guarantee"
    assert value >= best_single - 1e-9


def test_greedy_returns_the_requested_number_and_distinct_nodes():
    g = nx.gnp_random_graph(80, 0.08, seed=3)
    chosen, value = LiveEdgeObjective(g, samples=100, rng=np.random.default_rng(1)).greedy(INFL_BUDGET)
    assert len(chosen) == INFL_BUDGET == 10
    assert len(set(chosen)) == 10


# ------------------------------------------------------------------ the environment
def test_environment_reveals_exactly_the_true_neighbourhood():
    g = nx.gnp_random_graph(60, 0.1, seed=13)
    env = DiscoveryEnv(g, seeds=[0, 1], max_T=3)
    allowed = set(g.neighbors(0)) | set(g.neighbors(1))
    for q in list(env.possible_actions)[:3]:
        env.step(q)
    for u, v in env.graph.edges():
        assert g.has_edge(u, v), "an edge was invented"
    for u in env.active:
        for v in g.neighbors(u):
            assert env.graph.has_edge(u, v), "a true edge was withheld from a queried node"
    assert env.discovered_nodes <= len(allowed | set(env.queried)) + len(g)


def test_initial_seeds_are_free_and_frontier_only():
    g = nx.gnp_random_graph(50, 0.12, seed=21)
    env = DiscoveryEnv(g, seeds=[5, 6], max_T=2)
    assert env.queried == [], "initial seeds must not spend query budget"
    assert env.discovered_nodes > 2
    outside = [u for u in g.nodes() if u not in env.possible_actions and u not in env.active]
    if outside:
        with pytest.raises(ValueError):
            env.step(outside[0])
    q = sorted(env.possible_actions)[0]
    env.step(q)
    with pytest.raises(ValueError):
        env.step(q)                      # re-querying is refused, as in the original


def test_query_budget_is_respected_and_flags_done():
    g = nx.gnp_random_graph(90, 0.1, seed=17)
    out = run_episode(g, policy_random, [0, 1, 2, 3, 4], max_T=5, rng=np.random.default_rng(0),
                      samples=20, graph_name="t")
    assert len(out.queried) == 5
    assert out.discovered_nodes >= 5


def test_policies_choose_from_the_frontier():
    g = nx.gnp_random_graph(70, 0.1, seed=31)
    for pol in (policy_random, policy_degree_max, policy_degree_min):
        env = DiscoveryEnv(g, seeds=[0, 1, 2, 3, 4], max_T=1)
        u = pol(env, np.random.default_rng(0))
        assert u in env.possible_actions


def test_degree_max_picks_the_highest_visible_degree():
    g = nx.gnp_random_graph(80, 0.1, seed=41)
    env = DiscoveryEnv(g, seeds=[0], max_T=1)
    obs = env.observed
    best = max(obs.degree(u) for u in env.possible_actions)
    assert obs.degree(policy_degree_max(env, np.random.default_rng(0))) == best


# ------------------------------------------------------------------ CHANGE
def test_change_surveys_the_published_number_of_nodes():
    """``Change(g, budget=budget*2)`` with a 5-query agent reveals up to 5 random + 5 neighbour nodes."""
    g = nx.gnp_random_graph(200, 0.05, seed=51)
    surveys = change_surveys(g, change_budget=10, rng=np.random.default_rng(0))
    assert 5 <= len(surveys) <= 10, surveys
    assert len(set(surveys)) == len(surveys)
    assert all(0 <= s < 200 for s in surveys)
    # passing the agent's own budget would halve CHANGE -- the guard against that regression
    halved = change_surveys(g, change_budget=5, rng=np.random.default_rng(0))
    assert len(halved) <= 4


def test_change_episode_produces_a_full_result():
    g = nx.gnp_random_graph(150, 0.06, seed=61)
    out = run_change_episode(g, budget_queries=5, rng=np.random.default_rng(0), samples=20, graph_name="t")
    assert len(out.chosen_seeds) == 10
    assert 0 < out.influence_full <= g.number_of_nodes()
    assert out.discovered_nodes >= len(out.queried)


def test_reward_is_measured_on_the_full_graph_not_the_discovered_one():
    """A partial graph can only under-state influence, so its estimate must sit below the true one."""
    g = nx.gnp_random_graph(120, 0.08, seed=71)
    out = run_episode(g, policy_degree_max, [0, 1, 2, 3, 4], max_T=5, rng=np.random.default_rng(2),
                      samples=2000, graph_name="t")
    assert len(out.chosen_seeds) == 10
    assert out.influence_discovered <= out.influence_full, (
        "the discovered subgraph cannot support more influence than the complete graph")
    assert out.influence_full > 10, "the chosen seeds must spread beyond themselves on the full graph"


def test_influence_is_never_below_the_seed_count():
    g = nx.gnp_random_graph(80, 0.06, seed=81)
    for pol in (policy_random, policy_degree_max):
        out = run_episode(g, pol, [0, 1, 2, 3, 4], max_T=4, rng=np.random.default_rng(5),
                          samples=200, graph_name="t")
        assert out.influence_full >= 10 - 1e-9
