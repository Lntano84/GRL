"""The original double-DQN for CCIM, with terminal handling and legal action masking fixed.

What is "original" here
-----------------------
The base RL method of Chen et al. (IJCAI 2023) is a double-DQN over the MDP whose steps are seed
selections and whose reward is the true marginal influence, ``r = sigma(S u {a}) - sigma(S)``
(their Eq. 4).  This module implements exactly that base, **without** the three components RL4CCIM
adds on top: no solution filtering, no prioritised experience replay, no reward shaping.  So it is
the "plain DQN" arm, trained directly on the true diffusion reward.

Deviations, stated plainly
--------------------------
* The paper's graph representation is a graph attention network; this uses a two-layer GCN.  The
  encoder is simpler on purpose: the question here is whether the *decision process* has value, not
  whether a particular encoder wins.
* Discount ``gamma = 1``.  With a fixed horizon and rewards that telescope into
  ``sigma(S_T) - sigma(S_t)``, ``gamma = 1`` makes the Q-value the actual remaining influence, so the
  terminal-handling fix has a precise meaning rather than being a convention.

The two fixes
-------------
``legal masking``
    an already-selected node is never a legal action.  Letting the agent pick it again both wastes the
    step and turns the "seed set" into a multiset, so ``sigma`` no longer describes what the agent
    believes it selected.  The mask is applied in **three** places: exploration sampling, greedy
    action selection, and the bootstrap ``max`` over the next state.
``terminal handling``
    the episode ends exactly at ``t = T - 1`` and the target there is ``r`` with **no** bootstrap.
    Bootstrapping past the horizon makes the last seed look better than it is.

Both are toggles (``mask_actions``, ``bootstrap_at_terminal``) so the size of each defect can be
measured rather than asserted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import torch
import torch.nn as nn

from .baselines import Ledger
from .model import DEFAULT_K, DEFAULT_T

NEG_INF = -1e9


# ------------------------------------------------------------------------------------ environment
@dataclass
class Step:
    state: tuple
    action: int
    reward: float
    next_state: tuple
    done: bool


class CCIMEnv:
    """The paper's MDP: ``T`` sequential seed selections, reward = true marginal influence."""

    def __init__(self, graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K,
                 ledger: Ledger | None = None):
        self.graph = graph
        self.T = T
        self.K = K
        self.n = graph.number_of_nodes()
        self.ledger = ledger if ledger is not None else Ledger()
        self.seeds: list[int] = []
        self._value = 0

    def reset(self) -> tuple:
        self.seeds = []
        self._value = self.ledger.sigma(self.graph, [], self.K)
        return self.state()

    def state(self) -> tuple:
        s = torch.zeros(self.n, dtype=torch.float32)
        for v in self.seeds:
            s[v] = 1.0
        return (s, len(self.seeds))

    def legal_actions(self) -> list[int]:
        chosen = set(self.seeds)
        return [v for v in range(self.n) if v not in chosen]

    def step(self, action: int) -> Step:
        before_state = self.state()
        before_value = self._value
        self.seeds.append(int(action))
        self._value = self.ledger.sigma(self.graph, self.seeds, self.K)
        reward = float(self._value - before_value)
        done = len(self.seeds) >= self.T
        return Step(before_state, int(action), reward, self.state(), done)


# ------------------------------------------------------------------------------------ Q-network
class QNet(nn.Module):
    """Two-layer GCN node embeddings; the Q head is batched over states.

    ``embed`` depends only on the graph, so it is computed once per update; ``head`` maps a batch of
    ``(selected mask, step)`` states to a full ``(batch, n)`` Q-table.  That keeps the inner loop
    vectorised instead of looping over replay samples.
    """

    def __init__(self, in_features: int, hidden: int = 64, embed: int = 64):
        super().__init__()
        self.gcn1 = nn.Linear(in_features, embed)
        self.gcn2 = nn.Linear(embed, embed)
        self.head = nn.Sequential(
            nn.Linear(embed + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def embed(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gcn1(adj @ features))
        return torch.relu(self.gcn2(adj @ h))

    def head_q(self, h: torch.Tensor, degree: torch.Tensor, selected: torch.Tensor,
               step_fraction: torch.Tensor) -> torch.Tensor:
        """``selected``: (B, n); ``step_fraction``: (B,) -> returns (B, n)."""
        batch, n, e = selected.shape[0], selected.shape[1], h.shape[1]
        hh = h.unsqueeze(0).expand(batch, n, e)
        deg = degree.unsqueeze(0).expand(batch, n, 1)
        st = step_fraction.view(batch, 1, 1).expand(batch, n, 1)
        z = torch.cat([hh, selected.unsqueeze(2), deg, st], dim=2)
        return self.head(z).squeeze(2)


def normalized_adjacency(graph: nx.Graph) -> torch.Tensor:
    """``D^-1/2 (A + I) D^-1/2`` as a dense tensor; the graphs here are tiny."""
    n = graph.number_of_nodes()
    a = torch.zeros(n, n)
    for u, v in graph.edges():
        a[u, v] = 1.0
        a[v, u] = 1.0
    a = a + torch.eye(n)
    deg = a.sum(dim=1).clamp(min=1e-9)
    d_inv_sqrt = torch.diag(deg.pow(-0.5))
    return d_inv_sqrt @ a @ d_inv_sqrt


def node_features(graph: nx.Graph) -> torch.Tensor:
    """Degree (max-normalized) and a constant column; the selected flag is added per state."""
    n = graph.number_of_nodes()
    deg = torch.tensor([graph.degree(v) for v in range(n)], dtype=torch.float32)
    return torch.stack([deg / deg.max().clamp(min=1.0), torch.ones(n)], dim=1)


# ------------------------------------------------------------------------------------ training
def _mask(q: torch.Tensor, selected: torch.Tensor, mask: bool) -> torch.Tensor:
    return q.masked_fill(selected.bool(), NEG_INF) if mask else q


def train_dqn(graph: nx.Graph, T: int = DEFAULT_T, K: int = DEFAULT_K,
              episodes: int = 1000, seed: int = 0, gamma: float = 1.0,
              lr: float = 1e-3, batch_size: int = 64, buffer_size: int = 20_000,
              target_sync: int = 200, eps_start: float = 1.0, eps_end: float = 0.05,
              eps_decay: float = 500, validate_every: int = 50,
              mask_actions: bool = True, bootstrap_at_terminal: bool = False,
              ledger: Ledger | None = None) -> dict:
    """Train the plain double-DQN.  Returns the validation curve and the selected checkpoint."""
    torch.manual_seed(seed)
    random.seed(seed)

    ledger = ledger if ledger is not None else Ledger()
    adj = normalized_adjacency(graph)
    feats = node_features(graph)
    degree = feats[:, :1]
    n = graph.number_of_nodes()
    env = CCIMEnv(graph, T=T, K=K, ledger=ledger)

    online = QNet(feats.shape[1])
    target = QNet(feats.shape[1])
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=lr)
    buffer: list[tuple] = []

    validation: list[dict] = []
    best = {"normalized": -1.0, "episode": -1, "state_dict": None, "seeds": None}

    for episode in range(1, episodes + 1):
        state = env.reset()
        eps = eps_end + (eps_start - eps_end) * math.exp(-episode / eps_decay)
        while True:
            legal = env.legal_actions()
            if not legal:
                break
            if random.random() < eps:
                action = random.choice(legal) if mask_actions else random.randrange(n)
            else:
                with torch.no_grad():
                    h = online.embed(adj, feats)
                    q = online.head_q(h, degree, state[0].unsqueeze(0),
                                      torch.tensor([state[1] / max(1, state[1] + 1)]))
                action = int(torch.argmax(_mask(q, state[0].unsqueeze(0), mask_actions)).item())
            step = env.step(action)
            buffer.append((step.state[0], step.state[1], step.action, step.reward,
                           step.next_state[0], step.next_state[1], step.done))
            if len(buffer) > buffer_size:
                buffer.pop(0)
            state = step.next_state

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s0 = torch.stack([b[0] for b in batch])
                t0 = torch.tensor([b[1] for b in batch], dtype=torch.float32)
                acts = torch.tensor([b[2] for b in batch], dtype=torch.long)
                rew = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                s1 = torch.stack([b[4] for b in batch])
                t1 = torch.tensor([b[5] for b in batch], dtype=torch.float32)
                done = torch.tensor([b[6] for b in batch], dtype=torch.float32)
                frac0 = t0 / (t0 + 1.0)
                frac1 = t1 / (t1 + 1.0)

                h_on = online.embed(adj, feats)
                q_pred = _mask(online.head_q(h_on, degree, s0, frac0), s0, mask_actions)
                q_pred = q_pred.gather(1, acts.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next_online = _mask(online.head_q(h_on, degree, s1, frac1), s1, mask_actions)
                    best_next = q_next_online.argmax(dim=1, keepdim=True)
                    h_tg = target.embed(adj, feats)
                    q_next_target = _mask(target.head_q(h_tg, degree, s1, frac1),
                                          s1, mask_actions).gather(1, best_next).squeeze(1)
                    if not bootstrap_at_terminal:
                        q_next_target = q_next_target * (1.0 - done)
                    target_values = rew + gamma * q_next_target
                loss = nn.functional.mse_loss(q_pred, target_values)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimiser.step()
            if step.done:
                break
        if episode % target_sync == 0:
            target.load_state_dict(online.state_dict())

        if episode % validate_every == 0:
            value, seeds = greedy_rollout(graph, online, adj, feats, T, K, ledger, mask_actions)
            validation.append({"episode": episode, "sigma": value,
                               "normalized": value / n, "seeds": seeds})
            if value / n > best["normalized"]:
                best = {"normalized": value / n, "episode": episode,
                        "state_dict": {k: v.clone() for k, v in online.state_dict().items()},
                        "seeds": seeds}

    final_value, final_seeds = greedy_rollout(graph, online, adj, feats, T, K, ledger, mask_actions)
    return {"validation": validation, "best": best, "final_sigma": final_value,
            "final_seeds": final_seeds, "seed": seed, "episodes": episodes,
            "mask_actions": mask_actions, "bootstrap_at_terminal": bootstrap_at_terminal}


def greedy_rollout(graph: nx.Graph, net: QNet, adj, feats, T: int, K: int,
                   ledger: Ledger, mask: bool = True) -> tuple[int, list[int]]:
    """One deterministic, exploration-free episode.  This is the policy that gets evaluated."""
    degree = feats[:, :1]
    env = CCIMEnv(graph, T=T, K=K, ledger=ledger)
    state = env.reset()
    while True:
        legal = env.legal_actions()
        if not legal:
            break
        with torch.no_grad():
            h = net.embed(adj, feats)
            q = net.head_q(h, degree, state[0].unsqueeze(0),
                           torch.tensor([state[1] / max(1, state[1] + 1)]))
        action = int(torch.argmax(_mask(q, state[0].unsqueeze(0), mask)).item())
        if action not in legal:  # masking disabled: refuse to select a node twice
            action = legal[0]
        step = env.step(action)
        state = step.next_state
        if step.done:
            break
    seeds = env.seeds
    assert len(set(seeds)) == len(seeds), "the policy selected a node twice; masking failed"
    assert len(seeds) == T, f"rollout produced {len(seeds)} seeds, expected {T}"
    value = ledger.sigma(graph, seeds, K, decision=False)
    return value, sorted(seeds)
