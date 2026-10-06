"""Gates for the ported learner.

The load-bearing one is :func:`test_observation_cannot_see_the_hidden_network`.  The whole problem is that
the policy must decide under partial information; if the encoder could read a true degree, a centrality or
an embedding precomputed on the complete graph, the comparison against the heuristics would be meaningless.
The test builds two complete graphs that agree exactly on everything the agent has observed and differ
everywhere else, and requires the encodings to be bit-identical.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from ccim.xplore import DiscoveryEnv, policy_degree_max
from ccim.xplore.learner import (GCNEncoder, GeoDQN, NODE_FEATURES, QNet, encode_observation,
                                 normalise_adjacency)


def _env_with_hidden_tail(hidden_extra: bool):
    """A graph whose observed part is fixed; the rest is either bare or densely rewired."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
    if hidden_extra:
        g.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (5, 9), (6, 9)])
    else:
        g.add_nodes_from(range(5, 10))
    env = DiscoveryEnv(g, seeds=[0], max_T=2)
    return g, env


def test_observation_cannot_see_the_hidden_network():
    g1, env1 = _env_with_hidden_tail(False)
    g2, env2 = _env_with_hidden_tail(True)
    # the two graphs agree on everything the agent can reach from seed 0
    assert sorted(env1.active | env1.possible_actions) == sorted(env2.active | env2.possible_actions)
    o1, o2 = encode_observation(env1, 0, 5), encode_observation(env2, 0, 5)
    assert torch.equal(o1["x"], o2["x"]), "node features differ although the observation is identical"
    assert torch.equal(o1["adj"], o2["adj"]), "adjacency differs although the observation is identical"
    assert o1["frontier"] == o2["frontier"]

    agent = GeoDQN(seed=0)
    q1 = agent.q(o1["x"], o1["adj"], torch.arange(len(o1["nodes"])), o1["extra"])
    q2 = agent.q(o2["x"], o2["adj"], torch.arange(len(o2["nodes"])), o2["extra"])
    assert torch.allclose(q1, q2), "the Q-values depend on structure the agent never observed"


def test_features_contain_no_true_degree():
    """Doubling every hidden degree must not move a single feature."""
    g1, env1 = _env_with_hidden_tail(False)
    g2 = g1.copy()
    for u in range(5, 10):
        for v in range(5, 10):
            if u != v:
                g2.add_edge(u, v)
    env2 = DiscoveryEnv(g2, seeds=[0], max_T=2)
    o1, o2 = encode_observation(env1, 0, 5), encode_observation(env2, 0, 5)
    assert torch.equal(o1["x"], o2["x"])


def test_normalise_adjacency_is_symmetric_with_unit_diagonal():
    a = normalise_adjacency(4, [(0, 1), (1, 2)])
    assert torch.allclose(a, a.T, atol=1e-6)
    assert torch.all(a.diag() > 0)


def test_encoder_and_qnet_shapes():
    x = torch.randn(6, NODE_FEATURES)
    adj = normalise_adjacency(6, [(0, 1), (2, 3)])
    emb = GCNEncoder()(x, adj)
    assert emb.shape == (6, 60), "the published action dimension is 60"
    net = QNet()
    q = net(x, adj, torch.tensor([0, 2, 4]), torch.zeros(4))
    assert q.shape == (3,)


def test_actions_are_always_frontier_nodes():
    g = nx.gnp_random_graph(60, 0.08, seed=3)
    env = DiscoveryEnv(g, seeds=[0, 1, 2, 3, 4], max_T=5)
    agent = GeoDQN(seed=0)
    t = 0
    while t < 5 and not env.done():
        obs = encode_observation(env, t, 5)
        a = agent.act(obs, explore=True)
        assert a in env.possible_actions, "the policy proposed an illegal survey"
        env.step(a)
        t += 1


def test_greedy_policy_is_deterministic_and_exploration_is_not():
    g = nx.gnp_random_graph(70, 0.08, seed=5)
    env = DiscoveryEnv(g, seeds=[0, 1, 2, 3, 4], max_T=3)
    agent = GeoDQN(seed=0)
    obs = encode_observation(env, 0, 5)
    assert len({agent.act(obs, explore=False) for _ in range(5)}) == 1
    assert len({agent.act(obs, explore=True) for _ in range(40)}) > 1


def test_buffer_is_capped_and_target_sync_copies_weights():
    g = nx.gnp_random_graph(40, 0.1, seed=7)
    env = DiscoveryEnv(g, seeds=[0, 1], max_T=2)
    agent = GeoDQN(seed=0, buffer_size=10)
    obs = encode_observation(env, 0, 5)
    a = agent.act(obs, explore=True)
    env.step(a)
    nxt = encode_observation(env, 1, 5)
    for _ in range(25):
        agent.observe(obs, a, 1.0, nxt, False)
    assert len(agent.buffer) == 10, "the replay buffer grew past its cap"
    with torch.no_grad():
        for p in agent.q.parameters():
            p.add_(1.0)
    agent.sync_target()
    for a_, b_ in zip(agent.q.parameters(), agent.target.parameters()):
        assert torch.allclose(a_, b_)


def test_update_returns_none_until_the_buffer_has_enough():
    agent = GeoDQN(seed=0, batch_size=100)
    assert agent.update() is None


def test_training_step_reduces_the_loss_on_a_fixed_batch():
    """A sanity check that gradients flow: repeated updates on the same batch must reduce its loss."""
    g = nx.gnp_random_graph(50, 0.1, seed=11)
    env = DiscoveryEnv(g, seeds=[0, 1, 2], max_T=3)
    agent = GeoDQN(seed=0, lr=1e-3, batch_size=20)
    obs = encode_observation(env, 0, 5)
    a = agent.act(obs, explore=True)
    env.step(a)
    nxt = encode_observation(env, 1, 5)
    for _ in range(250):
        agent.observe(obs, a, 5.0, nxt, False)
    first = agent.update()
    for _ in range(30):
        last = agent.update()
    assert first is not None and last is not None
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"
