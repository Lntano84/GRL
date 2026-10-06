"""Is the gradient update bound by threads, by per-sample calls, or by FLOPs?

Run one thread count per process (``--threads N``), because ``torch.set_num_threads`` cannot be trusted to
re-tune an already-warm process.

The replay is filled with synthetic transitions of **realistic shape** rather than by running episodes: the
observed subgraph runs 16..63 nodes over an episode, so each entry gets a random size in that range.  Using
synthetic entries is what makes the measurement affordable -- filling the buffer from real episodes would
cost ~110 x 21 s before the first update could be timed, and the update does not care where the tensors came
from.  Every update timed here is a **real** ``gradient_update_sarsa`` call on the authors' unmodified code.

Segmentation replicates the body of ``gradient_update_sarsa`` (dqn.py:631-663) statement for statement,
with timers between the phases, so the split is of the same work rather than of a lookalike.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402
from rl_alg import utils as utls  # noqa: E402
from torch.autograd import Variable  # noqa: E402

BATCH = 100
MIN_NODES, MAX_NODES = 16, 63          # observed-subgraph sizes measured during a real episode
WARMUP_UPDATES = 3
TIMED_UPDATES = 25


def make_entry(rng, n_nodes: int):
    m = int(rng.integers(MIN_NODES, MAX_NODES + 1))
    ones = np.ones((m, OFF.INPUT_DIM), dtype=np.float32)
    adj = np.zeros((m, m), dtype=np.float32)
    for i in range(m - 1):
        if rng.random() < 0.4:
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    emb = rng.normal(size=OFF.EMB_DIM).astype(np.float32)
    emb1 = rng.normal(size=OFF.EMB_DIM).astype(np.float32)
    return [ones, adj], emb, float(rng.normal()), [ones, adj], emb1, int(rng.integers(m))


def fill(replay, rng, count=600):
    for _ in range(count):
        s, a, r, s1, a1, ano = make_entry(rng, 0)
        replay.add(s, a, r, s1, a1, ano, td=float(abs(rng.normal())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)
    acmodel = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                         replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                         gcn_num_layers=2, num_pooling=1)

    print(f"threads={args.threads}  replay.size={replay.size}  batch={BATCH}  "
          f"torch.get_num_threads()={torch.get_num_threads()}")

    # ---- whole-update timing on the real method
    for _ in range(WARMUP_UPDATES):
        acmodel.gradient_update_sarsa(batch_size=BATCH)
    whole = []
    for _ in range(TIMED_UPDATES):
        t = time.perf_counter()
        acmodel.gradient_update_sarsa(batch_size=BATCH)
        whole.append(time.perf_counter() - t)

    # ---- segmented replay of the same body, to see where the time sits
    seg = {"replay_and_assembly": [], "forward": [], "backward": [], "optimizer_step": []}
    for i in range(TIMED_UPDATES):
        t0 = time.perf_counter()
        s, a, r, s1, a1, ano = replay.sample_(BATCH)
        sa = [Variable(torch.from_numpy(np.array(x[0], dtype=np.float32)).reshape((1,) + np.array(x[0], dtype=np.float32).shape)) for x in s]
        sb = [Variable(torch.from_numpy(np.array(x[1], dtype=np.float32)).reshape((1,) + np.array(x[1], dtype=np.float32).shape)) for x in s]
        a1t = [Variable(torch.from_numpy(np.array(x, dtype=np.float32)).reshape((1,) + np.array(x, dtype=np.float32).shape)) for x in a1]
        at = Variable(torch.from_numpy(np.array(a, dtype=np.float32)))
        rt = Variable(torch.from_numpy(np.array(r, dtype=np.float32)))
        s1a = [Variable(torch.from_numpy(np.array(x[0], dtype=np.float32)).reshape((1,) + np.array(x[0], dtype=np.float32).shape)) for x in s1]
        s1b = [Variable(torch.from_numpy(np.array(x[1], dtype=np.float32)).reshape((1,) + np.array(x[1], dtype=np.float32).shape)) for x in s1]
        t1 = time.perf_counter()
        q_next1 = torch.squeeze(torch.cat([acmodel.target_actor_critic.critic_forward1(x[0], x[1], x[2], adjust_dim=False, convert_torch=False, node_labels=False).detach() for x in zip(s1a, s1b, a1t)], dim=0))
        q_next2 = torch.squeeze(torch.cat([acmodel.target_actor_critic.critic_forward2(x[0], x[1], x[2], adjust_dim=False, convert_torch=False, node_labels=False).detach() for x in zip(s1a, s1b, a1t)], dim=0))
        q_next = torch.min(q_next1, q_next2).detach()
        q_expected = rt + acmodel.gamma * q_next
        q_pred1 = torch.squeeze(torch.cat([acmodel.actor_critic.forward(x[0], x[1], x[2].view((1,) + x[2].shape), adjust_dim=False, convert_torch=False, node_labels=False)[1] for x in zip(sa, sb, at)], dim=0))
        q_pred2 = torch.squeeze(torch.cat([acmodel.actor_critic.forward(x[0], x[1], x[2].view((1,) + x[2].shape), adjust_dim=False, convert_torch=False, node_labels=False)[2] for x in zip(sa, sb, at)], dim=0))
        loss = torch.nn.functional.mse_loss(q_pred1, q_expected) + torch.nn.functional.mse_loss(q_pred2, q_expected)
        t2 = time.perf_counter()
        acmodel.critic_opt.zero_grad()
        loss.backward()
        t3 = time.perf_counter()
        acmodel.critic_opt.step()
        utls.copy_parameters(acmodel.target_actor_critic, acmodel.actor_critic, acmodel.eta)
        t4 = time.perf_counter()
        seg["replay_and_assembly"].append(t1 - t0)
        seg["forward"].append(t2 - t1)
        seg["backward"].append(t3 - t2)
        seg["optimizer_step"].append(t4 - t3)

    mean_whole = statistics.mean(whole)
    print(f"  whole gradient_update_sarsa : {mean_whole * 1000:8.1f} ms   "
          f"(median {statistics.median(whole) * 1000:.1f}, min {min(whole) * 1000:.1f})")
    print(f"    {'segment':<22}{'mean ms':>10}{'share':>8}")
    seg_mean = {}
    for k, v in seg.items():
        mu = statistics.mean(v)
        seg_mean[k] = mu
        print(f"    {k:<22}{mu * 1000:>10.1f}{100 * mu / mean_whole:>7.0f}%")
    print(f"    {'sum of segments':<22}{sum(seg_mean.values()) * 1000:>10.1f}")

    result = {"threads": args.threads, "torch_threads": torch.get_num_threads(),
              "replay_size": replay.size, "batch": BATCH,
              "whole_update_ms": mean_whole * 1000,
              "whole_update_median_ms": statistics.median(whole) * 1000,
              "whole_update_min_ms": min(whole) * 1000,
              "segments_ms": {k: v * 1000 for k, v in seg_mean.items()},
              "timed_updates": TIMED_UPDATES,
              "per_episode_updates": 10}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
