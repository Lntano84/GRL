"""Reward reshaping from Chen et al. (IJCAI 2023) §5.5, Eqs. (6)-(7), under a unified pipeline.

The reshaping, exactly as specified
-----------------------------------
For an episode whose final seed set is ``S_T`` (with ``S_t`` the first ``t`` seeds):

    r_t  = sigma(S_t) - sigma(S_{t-1})                          (raw, Eq. 4)
    c_t  = sigma(S_T) - sigma(S_T \\ {a_t})                      (leave-one-out contribution)
    r'_t = r_t + omega * c_t                                    (Eq. 7)

``c_t`` is what the final set loses if this particular seed is removed -- it is **not** the final
return spread evenly over the steps.  ``omega = 0`` recovers the raw reward exactly, so the two arms
differ only in the added term.

Two things that must not be confused
------------------------------------
* The shaping term does **not** guarantee that the optimal policy of the original objective is
  preserved.  Final quality is therefore judged **only** by the raw ``sigma(S)`` of the resulting seed
  set; a shaped return is never evidence of progress.
* ``omega = 1`` puts ``r_t`` (typically ~1, the seed itself) and ``c_t`` (potentially tens of nodes) on
  the same numeric scale, so the shaped reward is much larger in magnitude.  That is a property of the
  rule as specified, not a tunable here; ``mean_abs_reward`` is recorded for both arms so the
  difference is visible instead of implicit, and the diagnostics report whether ``c_t`` actually
  discriminates between seeds or is nearly constant (in which case plain amplification would explain
  any gain).

Unified pipeline
----------------
Both arms buffer a whole episode, insert at episode end, and run the same number of gradient updates
(``T`` per episode, matching one update per step).  A transition is inserted **once**, with the reward
its own arm uses -- never twice in two versions.  ``omega = 0`` is re-run under this pipeline rather
than reusing the earlier numbers, because the insertion timing itself changed.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

import networkx as nx
import torch
import torch.nn as nn

from .baselines import Ledger
from .dqn import CCIMEnv, QNet, _mask, node_features, normalized_adjacency
from .model import DEFAULT_K, DEFAULT_T


@dataclass
class Diagnostics:
    """Step-level reward structure, recorded per episode and summarised incrementally."""

    episodes: int = 0
    steps: int = 0
    steps_with_extra: int = 0                     # r_t > 1
    first_extra_step_hist: dict = field(default_factory=dict)
    raw_rewards: list = field(default_factory=list)          # sample of per-step raw rewards
    loo_samples: list = field(default_factory=list)          # sample of per-episode c_t vectors
    step_reward_mean: list = field(default_factory=list)     # mean r_t by step index
    step_loo_mean: list = field(default_factory=list)        # mean c_t by step index
    early_raw_vs_loo: list = field(default_factory=list)     # (r_t, c_t) pairs, for the correlation
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def record(self, raw: list[float], loo: list[float] | None, sample: bool = True) -> None:
        self.episodes += 1
        self.steps += len(raw)
        self.steps_with_extra += sum(1 for r in raw if r > 1)
        first = next((i for i, r in enumerate(raw) if r > 1), None)
        self.first_extra_step_hist[first] = self.first_extra_step_hist.get(first, 0) + 1
        while len(self.step_reward_mean) < len(raw):
            self.step_reward_mean.append(0.0)
        for i, r in enumerate(raw):
            self.step_reward_mean[i] += r
        if loo is not None:
            while len(self.step_loo_mean) < len(loo):
                self.step_loo_mean.append(0.0)
            for i, c in enumerate(loo):
                self.step_loo_mean[i] += c
            self.early_raw_vs_loo.extend(zip(raw, loo))
        if sample and self._rng.random() < 0.10:
            self.raw_rewards.append(list(raw))
            if loo is not None:
                self.loo_samples.append(list(loo))

    def summary(self) -> dict:
        out = {
            "episodes": self.episodes,
            "steps": self.steps,
            "fraction_of_steps_with_r_gt_1": self.steps_with_extra / self.steps if self.steps else None,
            "first_extra_step_histogram":
                {("none" if k is None else str(k)): v
                 for k, v in sorted(self.first_extra_step_hist.items(),
                                    key=lambda kv: (kv[0] is None, kv[0]))},
            "mean_raw_reward_by_step":
                [v / self.episodes for v in self.step_reward_mean] if self.episodes else [],
        }
        if self.step_loo_mean and self.loo_samples:
            out["mean_loo_by_step"] = [v / self.episodes for v in self.step_loo_mean]
            distinct = [len(set(c)) for c in self.loo_samples]
            spreads = [max(c) - min(c) for c in self.loo_samples]
            out["loo_discrimination"] = {
                "sampled_episodes": len(self.loo_samples),
                "mean_distinct_values_per_episode": statistics.fmean(distinct),
                "fraction_of_episodes_with_all_loo_equal":
                    sum(1 for d in distinct if d == 1) / len(distinct),
                "mean_within_episode_spread": statistics.fmean(spreads),
                "max_within_episode_spread": max(spreads),
            }
            pairs = self.early_raw_vs_loo
            if pairs:
                small = [c for r, c in pairs if r <= 1]
                large = [c for r, c in pairs if r > 1]
                out["early_small_reward_vs_later"] = {
                    "mean_loo_when_raw_le_1": statistics.fmean(small) if small else None,
                    "n_raw_le_1": len(small),
                    "mean_loo_when_raw_gt_1": statistics.fmean(large) if large else None,
                    "n_raw_gt_1": len(large),
                }
        return out


def _episode_cost(T: int, omega: float) -> int:
    """Queries one episode issues: reset + T steps + T leave-one-out sets when shaping is on."""
    return 1 + T + (T if omega else 0)


def train_shaped(graph: nx.Graph, omega: float, T: int = DEFAULT_T, K: int = DEFAULT_K,
                 query_cap: int = 36_369, seed: int = 0, gamma: float = 1.0, lr: float = 1e-3,
                 batch_size: int = 64, buffer_size: int = 20_000, target_sync: int = 200,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 200,
                 validate_every: int = 100, common_episodes: int = 2000,
                 ledger: Ledger | None = None, diagnostics: Diagnostics | None = None) -> dict:
    """Train one arm under the unified pipeline until the shared query cap is reached."""
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed + 9973)

    ledger = ledger if ledger is not None else Ledger()
    diag = diagnostics if diagnostics is not None else Diagnostics()
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

    per_episode = _episode_cost(T, omega)
    validation_history: list[dict] = []
    common_snapshot: dict | None = None
    episode = 0
    reward_abs_sum = 0.0
    reward_abs_count = 0

    while ledger.decision_evaluations + per_episode <= query_cap:
        episode += 1
        eps = eps_end + (eps_start - eps_end) * math.exp(-episode / eps_decay)
        state = env.reset()
        episode_records = []
        while True:
            legal = env.legal_actions()
            if not legal:
                break
            if rng.random() < eps:
                action = rng.choice(legal)
            else:
                with torch.no_grad():
                    h = online.embed(adj, feats)
                    q = online.head_q(h, degree, state[0].unsqueeze(0),
                                      torch.tensor([state[1] / max(1, state[1] + 1)]))
                action = int(torch.argmax(q).item())
                if action not in legal:
                    action = legal[0]
            pre_selected, pre_step = state[0], state[1]
            step = env.step(action)
            episode_records.append([pre_selected, pre_step, action, step.reward])
            state = step.next_state
            if step.done:
                break

        raw = [rec[3] for rec in episode_records]
        seeds = list(env.seeds)
        loo = None
        if omega:
            final_value = ledger.sigma(graph, seeds, K)
            loo = []
            for a in seeds:
                without = [v for v in seeds if v != a]
                loo.append(float(final_value - ledger.sigma(graph, without, K)))
        diag.record(raw, loo)

        # ---- insert once, at episode end, with this arm's reward ----
        for i, (pre_selected, pre_step, action, raw_r) in enumerate(episode_records):
            reward = raw_r + (omega * loo[i] if loo is not None else 0.0)
            reward_abs_sum += abs(reward)
            reward_abs_count += 1
            nxt = episode_records[i + 1][0] if i + 1 < len(episode_records) else \
                torch.zeros(n, dtype=torch.float32)
            nxt_step = episode_records[i + 1][1] if i + 1 < len(episode_records) else T
            buffer.append((pre_selected, pre_step, action, reward, nxt, nxt_step,
                           i == len(episode_records) - 1))
        while len(buffer) > buffer_size:
            buffer.pop(0)

        # ---- T gradient updates per episode, identical for both arms ----
        if len(buffer) >= batch_size:
            for _ in range(len(episode_records)):
                batch = rng.sample(buffer, batch_size)
                s0 = torch.stack([b[0] for b in batch])
                t0 = torch.tensor([b[1] for b in batch], dtype=torch.float32)
                acts = torch.tensor([b[2] for b in batch], dtype=torch.long)
                rew = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                s1 = torch.stack([b[4] for b in batch])
                t1 = torch.tensor([b[5] for b in batch], dtype=torch.float32)
                done = torch.tensor([float(b[6]) for b in batch])
                h_on = online.embed(adj, feats)
                q_pred = _mask(online.head_q(h_on, degree, s0, t0 / (t0 + 1.0)), s0, True)
                q_pred = q_pred.gather(1, acts.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = _mask(online.head_q(h_on, degree, s1, t1 / (t1 + 1.0)), s1, True)
                    best_next = q_next.argmax(dim=1, keepdim=True)
                    h_tg = target.embed(adj, feats)
                    q_tg = _mask(target.head_q(h_tg, degree, s1, t1 / (t1 + 1.0)), s1, True)
                    q_tg = q_tg.gather(1, best_next).squeeze(1) * (1.0 - done)
                    y = rew + gamma * q_tg
                loss = nn.functional.mse_loss(q_pred, y)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimiser.step()
        if episode % target_sync == 0:
            target.load_state_dict(online.state_dict())

        if episode % validate_every == 0:
            value, chosen = greedy_rollout(graph, online, adj, feats, T, K, ledger)
            validation_history.append({"episode": episode, "sigma": value,
                                       "normalized": value / n, "seeds": chosen})
        if episode == common_episodes and common_snapshot is None:
            value, chosen = greedy_rollout(graph, online, adj, feats, T, K, ledger)
            common_snapshot = {"episode": episode, "sigma": value, "normalized": value / n,
                               "seeds": chosen}

    final_value, final_seeds = greedy_rollout(graph, online, adj, feats, T, K, ledger)
    return {
        "omega": omega, "seed": seed, "episodes": episode,
        "query_cap": query_cap, "queries_issued": ledger.decision_evaluations,
        "queries_distinct": ledger.as_dict()["distinct_queries"],
        "cache_hits": ledger.cache_hits,
        "final_sigma": final_value, "final_normalized": final_value / n, "final_seeds": final_seeds,
        "common_episode_snapshot": common_snapshot,
        "validation_history": validation_history,
        "mean_abs_reward": reward_abs_sum / reward_abs_count if reward_abs_count else None,
        "diagnostics": diag.summary(),
    }


def greedy_rollout(graph: nx.Graph, net: QNet, adj, feats, T: int, K: int,
                   ledger: Ledger) -> tuple[int, list[int]]:
    """Deterministic, exploration-free episode.  Always masked: the fix is not on trial here."""
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
        action = int(torch.argmax(_mask(q, state[0].unsqueeze(0), True)).item())
        if action not in legal:
            action = legal[0]
        step = env.step(action)
        state = step.next_state
        if step.done:
            break
    seeds = env.seeds
    assert len(set(seeds)) == len(seeds), "policy selected a node twice"
    assert len(seeds) == T, f"rollout produced {len(seeds)} seeds, expected {T}"
    value = ledger.sigma(graph, seeds, K, decision=False)
    return value, sorted(seeds)
