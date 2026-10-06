"""Per-update drift of the batched path, and its speed: 25 updates plus full episodes.

Drift matters more than the single-step error.  If the parameter gap is at float32 level after one update
but grows by roughly a constant factor each update, then over 20,000 updates (2000 episodes x 10) the two
implementations decorrelate completely -- which makes the batched path a **new implementation** for long
training even though every individual step agrees to float32 precision.
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from ccim.xplore.dqn_batched import gradient_update_sarsa_batched, swap_in_batched_encoder  # noqa: E402
from ccim.xplore.dqn_opt import gradient_update_sarsa_fast  # noqa: E402
from equiv_check import maxdiff, params  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

OUTPUT = ROOT / "results" / "bench_batched.json"


def build(seed, replay, batched=False):
    torch.manual_seed(seed)
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    if batched:
        swap_in_batched_encoder(m)
    return m


def drift(replay, updates: int):
    ref = build(0, replay)
    bat = build(0, replay, batched=True)
    curve = []
    for it in range(updates):
        replay.rg = np.random.RandomState(1000 + it)
        gradient_update_sarsa_fast(ref, batch_size=BATCH, compute_loss=False, share_embeddings=True)
        replay.rg = np.random.RandomState(1000 + it)
        gradient_update_sarsa_batched(bat, batch_size=BATCH)
        _, rel = maxdiff(params(ref), params(bat))
        curve.append(rel)
    return curve


def time_updates(replay, updates: int, threads: int):
    torch.set_num_threads(threads)
    out = {}
    steps = {
        "published": lambda m: m.gradient_update_sarsa(batch_size=BATCH),
        "fast": lambda m: gradient_update_sarsa_fast(m, batch_size=BATCH, compute_loss=False,
                                                     share_embeddings=True),
        "batched": lambda m: gradient_update_sarsa_batched(m, batch_size=BATCH),
    }
    for kind, fn in steps.items():
        m = build(0, replay, batched=(kind == "batched"))
        for _ in range(2):
            fn(m)
        ts = []
        for _ in range(updates):
            t = time.perf_counter()
            fn(m)
            ts.append(time.perf_counter() - t)
        out[kind] = {"mean_ms": statistics.mean(ts) * 1000, "min_ms": min(ts) * 1000}
    return out


def time_episode(threads: int, episodes: int, kind: str):
    torch.set_num_threads(threads)
    graphs = {n: pickle.loads((OFF.SIMP.DATA / f"{n}.pkl").read_bytes()) for n in OFF.TRAIN_GRAPHS}
    for g in graphs.values():
        for u, v in g.edges():
            g[u][v]["p"] = 0.1
    g = graphs[OFF.TRAIN_GRAPHS[0]]
    opt = OFF.make_opt(lambda a, b: OFF.influence_equivalent(a, b), g)
    m = build(0, PriortizedReplay(OFF.BUFF, 10, beta=0.6), batched=(kind == "batched"))
    if kind == "fast":
        m.gradient_update_sarsa = lambda batch_size=BATCH: gradient_update_sarsa_fast(
            m, batch_size=batch_size, compute_loss=False, share_embeddings=True)
    elif kind == "batched":
        m.gradient_update_sarsa = lambda batch_size=BATCH: gradient_update_sarsa_batched(
            m, batch_size=batch_size)
    rng = np.random.default_rng(0)
    srng = np.random.default_rng(1)
    warm = 0
    while m.replay.size <= BATCH and warm < 60:      # updates only fire past batch_size
        OFF.episode(g, m, m.replay, opt, None, rng, srng, OFF.EPS0)
        warm += 1
    assert m.replay.size > BATCH, "replay never passed batch_size"
    ts = []
    for _ in range(episodes):
        t = time.perf_counter()
        OFF.episode(g, m, m.replay, opt, None, rng, srng, OFF.EPS0)
        ts.append(time.perf_counter() - t)
    return {"variant": kind, "mean_s": statistics.mean(ts), "median_s": statistics.median(ts),
            "replay_size": m.replay.size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--updates", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)

    print("=" * 100)
    print(f"  BATCHED: DRIFT AND SPEED -- threads={args.threads}")
    print("=" * 100)
    curve = drift(replay, 5)
    print("    parameter drift vs the shared-embedding reference, after each update:")
    for i, v in enumerate(curve, 1):
        print(f"      after update {i}: relative {v:.3e}")
    growth = (curve[-1] / curve[0]) ** (1 / max(1, len(curve) - 1)) if curve[0] > 0 else float("nan")
    print(f"    mean per-update growth factor: {growth:.2f}x")

    up = time_updates(replay, args.updates, args.threads)
    print(f"\n    {'variant':<14}{'mean ms/update':>17}{'min ms':>10}{'vs published':>14}")
    b = up["published"]["mean_ms"]
    for k, v in up.items():
        print(f"    {k:<14}{v['mean_ms']:>17.1f}{v['min_ms']:>10.1f}{b / v['mean_ms']:>13.2f}x")

    eps = [time_episode(args.threads, args.episodes, k) for k in ("published", "fast", "batched")]
    print()
    for r in eps:
        print(f"    full episode [{r['variant']:<10}] {r['mean_s']:6.2f} s "
              f"(median {r['median_s']:.2f}, replay {r['replay_size']}) -> "
              f"{r['mean_s'] * 10000 / 3600:5.1f} h for 10,000 episodes")

    args.output.write_text(json.dumps({"threads": args.threads, "drift_curve": curve,
                                       "drift_growth_per_update": growth,
                                       "updates_ms": up, "episodes": eps},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
