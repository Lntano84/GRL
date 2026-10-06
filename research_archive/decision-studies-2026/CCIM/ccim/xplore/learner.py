"""Geometric-DQN, ported: a DQN that chooses which frontier node to survey next.

What is ported, and what is substituted
---------------------------------------
The published system (Kamarthi et al., AAMAS 2020, official code ``kage08/graph_sample_rl``) is a DQN
whose state is the observed sub-graph, whose actions are the frontier nodes, and whose reward is the
influence the finally chosen seed set achieves on the complete network.  This module reproduces that
formulation.  Two components are **substituted**, and the substitution is recorded rather than hidden:

===========================  ==========================================  =================================
published                    here                                        why
===========================  ==========================================  =================================
DeepWalk node embeddings     an end-to-end GCN encoder over the          ``gensim`` is not installed, and the
recomputed each step         observed sub-graph, trained with the DQN    published loop retrains DeepWalk at
                                                                         every one of 10,000 x 5 steps
DiffPool hierarchical         mean + max pooling over observed nodes      DiffPool needs the authors' own
graph pooling                                                            ``diffpool/`` module and a
                                                                         different tensor pipeline
===========================  ==========================================  =================================

The paper itself reports that structure-based encoders trained for the actions were unstable, which is a
reason to expect this port to be *weaker* than the published model, not stronger.  It is reported that way.

What is deliberately **not** substituted: the reward.  It is the official one -- the influence of the
greedily selected seed set measured on the complete graph.  ``train.py`` runs with ``opt_reward = 0`` and
``norm_reward = 0``, so the training signal is the *raw* influence, not the paper's Eq. (1) ratio.  That
code-versus-paper difference is preserved and recorded.

Legality of the observation
---------------------------
The encoder sees only the observed sub-graph: which nodes are queried, which are on the frontier, and the
degrees *within the observed sub-graph*.  No true degree, centrality or precomputed embedding of the
complete graph ever enters the policy.  ``tests/test_xplore_learner.py`` checks that two episodes with
identical observations but different hidden structure produce identical encodings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ACTION_DIM = 60          # train.py --actiondim default
GCN_LAYERS = 2           # train.py --gcn_layers default
HIDDEN = 128
NODE_FEATURES = 5


class GCNEncoder(nn.Module):
    """Two-layer GCN with symmetric normalisation, as a drop-in for the published node encoder."""

    def __init__(self, in_dim: int = NODE_FEATURES, hidden: int = ACTION_DIM,
                 layers: int = GCN_LAYERS):
        super().__init__()
        self.lins = nn.ModuleList()
        for i in range(layers):
            self.lins.append(nn.Linear(in_dim if i == 0 else hidden, hidden))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for lin in self.lins:
            h = torch.relu(adj @ lin(h))
        return h


class QNet(nn.Module):
    """``Q(s, a)`` for a frontier node ``a``: its embedding, the graph summary, and the budget left."""

    def __init__(self, action_dim: int = ACTION_DIM, extra: int = 4, hidden: int = HIDDEN):
        super().__init__()
        self.encoder = GCNEncoder(hidden=action_dim)
        self.head = nn.Sequential(
            nn.Linear(action_dim * 3 + extra, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, adj, node_sel, extra):
        emb = self.encoder(x, adj)
        mean_pool = emb.mean(dim=0)
        max_pool = emb.max(dim=0).values
        chosen = emb[node_sel]
        g = mean_pool.unsqueeze(0).expand(chosen.shape[0], -1)
        m = max_pool.unsqueeze(0).expand(chosen.shape[0], -1)
        e = extra.unsqueeze(0).expand(chosen.shape[0], -1)
        return self.head(torch.cat([chosen, g, m, e], dim=1)).squeeze(-1)


# ------------------------------------------------------------------ observation
def normalise_adjacency(n: int, edges) -> torch.Tensor:
    a = torch.zeros((n, n), dtype=torch.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    a = a + torch.eye(n)
    deg = a.sum(dim=1).clamp(min=1e-9).pow(-0.5)
    return deg.unsqueeze(1) * a * deg.unsqueeze(0)


def encode_observation(env, t: int, t_max: int) -> dict:
    """Everything the policy is allowed to know: the observed sub-graph and the budget, nothing else."""
    observed = list(env.active | env.possible_actions)
    index = {u: i for i, u in enumerate(observed)}
    sub = env.observed
    deg = np.array([sub.degree(u) for u in observed], dtype=np.float32)
    deg_max = float(deg.max()) if len(deg) else 1.0
    order = np.array([0.0 if u in env.active else 1.0 for u in observed], dtype=np.float32)
    feats = np.stack([
        np.array([1.0 if u in env.active else 0.0 for u in observed], dtype=np.float32),
        order,
        deg / max(deg_max, 1.0),
        np.full(len(observed), len(observed) / max(env.full.number_of_nodes(), 1), dtype=np.float32),
        np.ones(len(observed), dtype=np.float32),
    ], axis=1)
    edges = [(index[u], index[v]) for u, v in sub.edges()]
    frontier = sorted(env.possible_actions)
    return {"nodes": observed, "index": index, "x": torch.tensor(feats),
            "adj": normalise_adjacency(len(observed), edges), "frontier": frontier,
            "extra": torch.tensor([len(observed) / max(env.full.number_of_nodes(), 1),
                                   len(frontier) / max(env.full.number_of_nodes(), 1),
                                   (t_max - t) / t_max, t / t_max], dtype=torch.float32)}


# ------------------------------------------------------------------ agent
class GeoDQN:
    """DQN over frontier actions with a target network and a replay buffer."""

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99, batch_size: int = 100,
                 buffer_size: int = 4000, epsilon: float = 0.1, eps_min: float = 0.01,
                 eps_decay: float = 0.999, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.q = QNet().to(device)
        self.target = QNet().to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma, self.batch_size = gamma, batch_size
        self.buffer: list = []
        self.buffer_size = buffer_size
        self.epsilon, self.eps_min, self.eps_decay = epsilon, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.updates = 0

    # -------------------------------------------------------------- acting
    def act(self, obs: dict, explore: bool = True) -> int:
        frontier = obs["frontier"]
        if not frontier:
            return -1
        if explore and self.rng.random() < self.epsilon:
            return int(frontier[int(self.rng.integers(len(frontier)))])
        with torch.no_grad():
            sel = torch.tensor([obs["index"][u] for u in frontier], dtype=torch.long)
            q = self.q(obs["x"], obs["adj"], sel, obs["extra"])
        best = torch.argmax(q).item()
        return int(frontier[best])

    def q_values(self, obs: dict) -> np.ndarray:
        frontier = obs["frontier"]
        if not frontier:
            return np.zeros(0)
        with torch.no_grad():
            sel = torch.tensor([obs["index"][u] for u in frontier], dtype=torch.long)
            return self.q(obs["x"], obs["adj"], sel, obs["extra"]).numpy()

    # -------------------------------------------------------------- learning
    def observe(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, int(action), float(reward), next_obs, bool(done)))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def update(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, 200):
            return None
        idx = self.rng.integers(len(self.buffer), size=self.batch_size)
        loss_total = 0.0
        self.opt.zero_grad()
        for i in idx:
            obs, action, reward, next_obs, done = self.buffer[int(i)]
            if action not in obs["index"]:
                continue
            sel = torch.tensor([obs["index"][action]], dtype=torch.long)
            q = self.q(obs["x"], obs["adj"], sel, obs["extra"])
            with torch.no_grad():
                if done or not next_obs["frontier"]:
                    target = torch.tensor([reward], dtype=torch.float32)
                else:
                    nsel = torch.tensor([next_obs["index"][u] for u in next_obs["frontier"]],
                                        dtype=torch.long)
                    nq = self.q(next_obs["x"], next_obs["adj"], nsel, next_obs["extra"])
                    amax = int(torch.argmax(nq).item())
                    tq = self.target(next_obs["x"], next_obs["adj"], nsel[amax:amax + 1],
                                     next_obs["extra"])
                    target = reward + self.gamma * tq.detach()
            loss_total = loss_total + self.loss_fn(q, target)
        if loss_total == 0.0:
            return None
        loss = loss_total / self.batch_size
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        return float(loss.detach())

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
