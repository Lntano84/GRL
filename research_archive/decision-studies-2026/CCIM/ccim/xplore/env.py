"""The network-discovery environment, and the published exploration policies.

Environment, matching ``expts/net_env.py`` from the official repository
-----------------------------------------------------------------------
The agent is given ``extra_seeds`` random nodes whose neighbourhoods are revealed for free.  It then has
``max_T`` queries.  A query on ``u`` reveals every true neighbour of ``u``.  Queries may only be spent on
**discovered but unqueried** nodes (the frontier); picking an already-queried node is an error, as in the
original.  When the budget is spent, a seed set is chosen by greedy IM **on the discovered graph** and its
influence is measured **on the complete graph**.

Two access models, and they are not the same
--------------------------------------------
``random`` / ``degree_*`` walk the frontier: they can only ask about someone they have already heard of.
``CHANGE`` asks **globally** -- the published implementation samples uniformly from all ``n`` nodes
(``random.sample(range(len(self.graph)), budget/2)``) and then one random neighbour of each.  That global
access is part of the published protocol and is preserved here rather than quietly removed, but it is a
structural advantage and the report says so.

Defaults are the official ones (``train.py`` argparse): ``extra_seeds = 5``, ``budget = 5``, ``p = 0.1``,
``infl_budget = 10``, ``samples = 100``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .icm import INFL_BUDGET, PROP_PROBAB, SAMPLES, LiveEdgeObjective

EXTRA_SEEDS = 5        # train.py --extra-seeds default
QUERY_BUDGET = 5       # train.py --sample-budget default


@dataclass
class StepRecord:
    query: int
    node: int
    discovered_before: int
    discovered_after: int
    frontier_after: int


@dataclass
class EpisodeResult:
    policy: str
    graph: str
    budget: int
    seeds: list
    queried: list
    discovered_nodes: int
    frontier_nodes: int
    chosen_seeds: list
    influence_full: float
    influence_discovered: float
    steps: list = field(default_factory=list)


class DiscoveryEnv:
    """Faithful port of ``expts/net_env.py`` (single-graph, no reward shaping)."""

    def __init__(self, full_graph: nx.Graph, seeds, max_T: int = QUERY_BUDGET,
                 p: float = PROP_PROBAB, infl_budget: int = INFL_BUDGET):
        self.full = full_graph
        self.seeds = list(seeds)
        self.max_T = int(max_T)
        self.p = p
        self.infl_budget = int(infl_budget)
        self.reset()

    def reset(self, seeds=None) -> None:
        if seeds is not None:
            self.seeds = list(seeds)
        # the original adds every node of the full graph, so undiscovered nodes are isolated vertices
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.full.nodes())
        self.active: set = set()
        self.possible_actions: set = set()
        self.queried: list = []
        self.steps: list = []
        for v in self.seeds:
            self._enlarge(v)

    # ------------------------------------------------------------------ mechanics
    def _enlarge(self, u) -> None:
        if u not in self.graph:
            raise KeyError(f"no such node: {u}")
        self.active.add(u)
        for v in self.full.neighbors(u):
            self.graph.add_edge(u, v)
            if v not in self.active:
                self.possible_actions.add(v)
        self.possible_actions.discard(u)

    def step(self, u) -> None:
        """Query ``u``.  Only frontier nodes are legal, exactly as in the original."""
        if u in self.possible_actions:
            before = self.discovered_nodes
            self._enlarge(u)
            self.queried.append(u)
            self.steps.append(StepRecord(len(self.queried), u, before, self.discovered_nodes,
                                         len(self.possible_actions)))
            return
        if u in self.active:
            raise ValueError(f"node {u} has already been queried")
        raise ValueError(f"node {u} is not on the frontier")

    @property
    def discovered_nodes(self) -> int:
        return len(self.active | self.possible_actions)

    @property
    def observed(self) -> nx.Graph:
        """The subgraph the agent can see: queried nodes plus the frontier."""
        return nx.subgraph(self.graph, self.active | self.possible_actions).copy()

    def done(self) -> bool:
        return len(self.queried) >= self.max_T or not self.possible_actions

    # ------------------------------------------------------------------ reward
    def evaluate(self, rng, samples: int = SAMPLES):
        """Greedy IM on the discovered graph, then measure that seed set on the complete graph."""
        disc = LiveEdgeObjective(self.graph, samples=samples, rng=rng, p=self.p)
        seeds, value_disc = disc.greedy(self.infl_budget)
        full = LiveEdgeObjective(self.full, samples=samples, rng=rng, p=self.p)
        return seeds, full.value(seeds), value_disc


# ---------------------------------------------------------------------------- policies
def frontier_list(env: DiscoveryEnv) -> list:
    return sorted(env.possible_actions)


def policy_random(env: DiscoveryEnv, rng: np.random.Generator):
    frontier = frontier_list(env)
    if not frontier:
        return None
    return int(frontier[int(rng.integers(len(frontier)))])


def _degree_choice(env: DiscoveryEnv, rng: np.random.Generator, highest: bool):
    frontier = frontier_list(env)
    if not frontier:
        return None
    obs = env.observed
    deg = {u: obs.degree(u) for u in frontier}
    target = max(deg.values()) if highest else min(deg.values())
    tied = [u for u in frontier if deg[u] == target]
    return int(tied[int(rng.integers(len(tied)))])


def policy_degree_max(env: DiscoveryEnv, rng: np.random.Generator):
    """The visible-degree strategy: ask the frontier node with the largest observed degree."""
    return _degree_choice(env, rng, True)


def policy_degree_min(env: DiscoveryEnv, rng: np.random.Generator):
    """The paper's H1/H2 heuristic family: the learned policy was observed to prefer minimum degree."""
    return _degree_choice(env, rng, False)


def change_surveys(full_graph: nx.Graph, change_budget: int, rng: np.random.Generator) -> list:
    """``expts/change_baseline.py``: half the surveys uniform over ALL nodes, half one neighbour of each.

    ``change_budget`` is CHANGE's **own** budget parameter, which the published driver sets to twice the
    agent's query budget (``Change(g, budget=budget*2)`` with ``budget = 5``).  Inside, ``sample_graph1``
    takes ``int(change_budget/2)`` random nodes and ``sample_graph2`` takes one random neighbour of each,
    so a query budget of 5 reveals up to 10 nodes -- the same number the agent sees once its 5 free
    initial seeds are counted.  Passing the agent's budget here instead would silently halve CHANGE.
    """
    nodes = list(full_graph.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    half = max(1, int(change_budget / 2))
    if change_budget <= 0:
        return []                      # T=0 means no surveys at all, for every policy
    v1 = [nodes[i] for i in rng.choice(len(nodes), size=min(half, len(nodes)), replace=False)]
    surveyed = list(v1)
    for u in v1:
        nb = sorted(set(full_graph.neighbors(u)) - set(v1))
        if nb:
            surveyed.append(nb[int(rng.integers(len(nb)))])
    out, seen = [], set()
    for u in surveyed:
        if u not in seen:
            seen.add(u)
            out.append(int(index[u]))
    return out


def run_change_episode(full_graph: nx.Graph, budget_queries: int, rng: np.random.Generator,
                       samples: int = SAMPLES, p: float = PROP_PROBAB, infl_budget: int = INFL_BUDGET,
                       graph_name: str = "") -> EpisodeResult:
    """CHANGE as the published baseline: global surveys, then the same greedy IM and the same reward."""
    env = DiscoveryEnv(full_graph, [], max_T=0, p=p, infl_budget=infl_budget)
    queried = []
    for u in change_surveys(full_graph, 2 * budget_queries, rng):
        env._enlarge(u)
        queried.append(u)
        env.steps.append(StepRecord(len(queried), u, 0, env.discovered_nodes, len(env.possible_actions)))
    env.queried = queried
    seeds, influence_full, influence_disc = env.evaluate(rng, samples=samples)
    return EpisodeResult(policy="change", graph=graph_name, budget=budget_queries,
                         seeds=list(env.seeds), queried=queried,
                         discovered_nodes=env.discovered_nodes,
                         frontier_nodes=len(env.possible_actions), chosen_seeds=list(seeds),
                         influence_full=float(influence_full),
                         influence_discovered=float(influence_disc), steps=list(env.steps))


def run_episode(full_graph: nx.Graph, policy, seeds, max_T: int, rng: np.random.Generator,
                samples: int = SAMPLES, p: float = PROP_PROBAB,
                infl_budget: int = INFL_BUDGET, graph_name: str = "") -> EpisodeResult:
    """One discovery episode on the frontier.  ``policy`` is ``(env, rng) -> node | None``."""
    env = DiscoveryEnv(full_graph, seeds, max_T=max_T, p=p, infl_budget=infl_budget)
    while not env.done():
        u = policy(env, rng)
        if u is None:
            break
        env.step(u)
    chosen_seeds, influence_full, influence_disc = env.evaluate(rng, samples=samples)
    return EpisodeResult(
        policy=getattr(policy, "name", getattr(policy, "__name__", str(policy))),
        graph=graph_name, budget=max_T, seeds=list(env.seeds), queried=list(env.queried),
        discovered_nodes=env.discovered_nodes, frontier_nodes=len(env.possible_actions),
        chosen_seeds=list(chosen_seeds), influence_full=float(influence_full),
        influence_discovered=float(influence_disc), steps=list(env.steps))
