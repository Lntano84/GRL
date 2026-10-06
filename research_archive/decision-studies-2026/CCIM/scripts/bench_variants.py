"""Timing comparison of the three update variants, plus the full episode, at threads=1.

Variants, from the equivalence check (``scripts/equiv_check.py``):

* ``published``            -- ``DQNTrainer.gradient_update_sarsa``, authors' code untouched
* ``compute_loss_off_only`` -- published call pattern, auxiliary losses not computed.
  **Bitwise identical** on every checked quantity, so this timing difference is a free win.
* ``fast``                 -- embeddings shared (6 -> 2 embedder forwards per sample) *and* auxiliary
  losses off.  Agrees with the published update to float32 precision (relative error ~4e-7 on parameters)
  but is **not bitwise identical**, so it is a numerically-equivalent reordering rather than a
  provably-identical one.

Only after the per-update numbers come out is the full episode measured, with the same environment and
reward pipeline as the frozen run.
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
from bench_update import BATCH, fill  # noqa: E402
from ccim.xplore.dqn_opt import gradient_update_sarsa_fast  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

OUTPUT = ROOT / "results" / "bench_variants.json"


def time_updates(threads: int, updates: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)
    out = {}
    for kind in ("published", "compute_loss_off_only", "fast"):
        torch.manual_seed(0)
        model = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                           replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                           gcn_num_layers=2, num_pooling=1)
        step = {
            "published": lambda: model.gradient_update_sarsa(batch_size=BATCH),
            "compute_loss_off_only": lambda: gradient_update_sarsa_fast(
                model, batch_size=BATCH, compute_loss=False, share_embeddings=False),
            "fast": lambda: gradient_update_sarsa_fast(
                model, batch_size=BATCH, compute_loss=False, share_embeddings=True),
        }[kind]
        for _ in range(2):
            step()
        times = []
        for _ in range(updates):
            t = time.perf_counter()
            step()
            times.append(time.perf_counter() - t)
        out[kind] = {"mean_ms": statistics.mean(times) * 1000,
                     "median_ms": statistics.median(times) * 1000,
                     "min_ms": min(times) * 1000}
    return out


def time_episode(threads: int, episodes: int, variant: str) -> dict:
    """Full episode with the real environment and reward, using one update variant."""
    torch.set_num_threads(threads)
    import xplore_official as OFF2
    from expts.influence import influence
    graphs = {n: __import__("pickle").loads((OFF2.SIMP.DATA / f"{n}.pkl").read_bytes())
              for n in OFF2.TRAIN_GRAPHS}
    for g in graphs.values():
        for u, v in g.edges():
            g[u][v]["p"] = 0.1
    g = graphs[OFF2.TRAIN_GRAPHS[0]]
    opt = OFF2.make_opt(lambda a, b: OFF2.influence_equivalent(a, b), g)
    torch.manual_seed(0)
    model = DQNTrainer(input_dim=OFF2.INPUT_DIM, state_dim=OFF2.EMB_DIM, action_dim=OFF2.EMB_DIM,
                       replayBuff=PriortizedReplay(OFF2.BUFF, 10, beta=0.6), lr=OFF2.LR,
                       use_cuda=False, gamma=OFF2.GAMMA, gcn_num_layers=2, num_pooling=1)
    if variant == "fast":
        model.gradient_update_sarsa = lambda batch_size=BATCH: gradient_update_sarsa_fast(
            model, batch_size=batch_size, compute_loss=False, share_embeddings=True)
    elif variant == "compute_loss_off_only":
        model.gradient_update_sarsa = lambda batch_size=BATCH: gradient_update_sarsa_fast(
            model, batch_size=batch_size, compute_loss=False, share_embeddings=False)
    rng = np.random.default_rng(0)
    srng = np.random.default_rng(1)
    # the update only fires once replay.size > batch_size (dqn.py:230), and an episode adds only
    # T_MAX = 5 transitions, so ~21 episodes must run before the first update.  Warming up on a single
    # episode -- as an earlier version of this script did -- silently measures an update-free episode.
    warm = 0
    while model.replay.size <= BATCH and warm < 60:
        OFF2.episode(g, model, model.replay, opt, None, rng, srng, OFF2.EPS0)
        warm += 1
    if model.replay.size <= BATCH:
        raise RuntimeError(f"replay never passed batch_size after {warm} episodes")
    times = []
    for _ in range(episodes):
        t = time.perf_counter()
        OFF2.episode(g, model, model.replay, opt, None, rng, srng, OFF2.EPS0)
        times.append(time.perf_counter() - t)
    return {"variant": variant, "episodes": episodes, "mean_s": statistics.mean(times),
            "median_s": statistics.median(times), "replay_size": model.replay.size,
            "warmup_episodes": warm, "updates_active": model.replay.size > BATCH}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--updates", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    print("=" * 100)
    print(f"  VARIANT TIMING -- threads={args.threads}")
    print("=" * 100)
    up = time_updates(args.threads, args.updates)
    print(f"    {'variant':<26}{'mean ms':>10}{'median ms':>11}{'min ms':>9}{'vs published':>14}")
    base = up["published"]["mean_ms"]
    for k, v in up.items():
        print(f"    {k:<26}{v['mean_ms']:>10.1f}{v['median_ms']:>11.1f}{v['min_ms']:>9.1f}"
              f"{base / v['mean_ms']:>13.2f}x")

    eps = []
    for variant in ("published", "compute_loss_off_only", "fast"):
        r = time_episode(args.threads, args.episodes, variant)
        eps.append(r)
        print(f"    full episode [{variant:<23}] {r['mean_s']:6.2f} s  "
              f"(median {r['median_s']:.2f})  replay={r['replay_size']}")

    frozen_target = 10_000 * 2.88
    print(f"\n    frozen budget is 10,000 episodes; 8 h would need 2.88 s/episode")
    for r in eps:
        print(f"      {r['variant']:<24} -> {r['mean_s'] * 10000 / 3600:6.1f} h for the frozen budget")

    args.output.write_text(json.dumps({"threads": args.threads, "updates": args.updates,
                                       "updates_ms": up, "episodes": eps}, ensure_ascii=False,
                                      indent=2), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
