"""Aggregate the short-horizon cells from disk, and measure batched peak memory properly.

The training run finished all six cells and saved each one; only its *final* memory probe crashed (a fresh
model was built with an **empty** replay, so ``pad_batch`` hit ``max()`` on an empty list).  So the table is
produced here from the saved ``*_result.json`` files rather than by re-running ~3 hours of episodes, and the
memory probe is redone with a filled replay.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import xplore_official as OFF  # noqa: E402
from bench_update import fill  # noqa: E402
from ccim.xplore.dqn_batched import gradient_update_sarsa_batched, swap_in_batched_encoder  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

CKPT = ROOT / "results" / "shorthorizon_ckpt"
OUTPUT = ROOT / "results" / "shorthorizon_table.json"
SEEDS = (0, 1, 2)
VARIANTS = ("fast", "batched")
CHECKPOINTS = (0, 100, 200)


def load(variant, seed):
    p = CKPT / f"{variant}_seed{seed}_result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    torch.set_num_threads(1)

    cells = {(v, s): load(v, s) for v in VARIANTS for s in SEEDS}
    missing = [k for k, v in cells.items() if v is None]
    print("=" * 104)
    print("  SHORT-HORIZON TABLE -- 200 episodes, seeds 0/1/2, threads=1")
    print("=" * 104)
    if missing:
        print(f"  MISSING CELLS: {missing}")

    print("\n  (1) VALIDATION SPREAD")
    print(f"    {'seed':>5}{'variant':>10}{'ep0':>10}{'ep100':>10}{'ep200':>10}"
          f"{'batched-fast@100':>18}{'batched-fast@200':>18}")
    for s in SEEDS:
        a, b = cells[("fast", s)], cells[("batched", s)]
        for v, c in (("fast", a), ("batched", b)):
            if c is None:
                continue
            cur = {x["episode"]: x["val_mean_spread"] for x in c["val_curve"]}
            print(f"    {s:>5}{v:>10}{cur.get(0, float('nan')):>10.2f}{cur.get(100, float('nan')):>10.2f}"
                  f"{cur.get(200, float('nan')):>10.2f}" +
                  (f"{cur.get(100, float('nan')) - {x['episode']: x['val_mean_spread'] for x in a['val_curve']}.get(100, float('nan')):>18.2f}"
                   if False else ""))
        if a and b:
            ca = {x["episode"]: x["val_mean_spread"] for x in a["val_curve"]}
            cb = {x["episode"]: x["val_mean_spread"] for x in b["val_curve"]}
            d1 = cb.get(100, float("nan")) - ca.get(100, float("nan"))
            d2 = cb.get(200, float("nan")) - ca.get(200, float("nan"))
            print(f"    {'':>15}{'':>10}{'':>10}{'':>10}{d1:>+18.2f}{d2:>+18.2f}")
    ds = []
    for s in SEEDS:
        a, b = cells[("fast", s)], cells[("batched", s)]
        if a and b:
            ca = {x["episode"]: x["val_mean_spread"] for x in a["val_curve"]}
            cb = {x["episode"]: x["val_mean_spread"] for x in b["val_curve"]}
            ds.append((cb.get(200, float("nan")) - ca.get(200, float("nan")), s))
    if ds:
        vals = [d for d, _ in ds if d == d]
        print(f"\n    batched - fast at ep200, per seed: "
              + ", ".join(f"s{s}={d:+.2f}" for d, s in ds))
        print(f"    mean {statistics.mean(vals):+.2f}   min {min(vals):+.2f}   max {max(vals):+.2f}")

    print("\n  (2) DECISIONS ON FIXED VISIBLE STATES")
    agree = total = 0
    per_seed = {}
    for s in SEEDS:
        a, b = cells[("fast", s)], cells[("batched", s)]
        if not (a and b):
            continue
        ag = tt = 0
        for ep in CHECKPOINTS:
            qa = (a["q_ordering"] or {}).get(str(ep)) or (a["q_ordering"] or {}).get(ep)
            qb = (b["q_ordering"] or {}).get(str(ep)) or (b["q_ordering"] or {}).get(ep)
            if not qa or not qb:
                continue
            for x, y in zip(qa, qb):
                tt += 1
                ag += int(x["preferred_node"] == y["preferred_node"])
        per_seed[s] = [ag, tt]
        agree += ag
        total += tt
        print(f"    seed {s}: preferred survey action identical on {ag}/{tt} state-checkpoint pairs")
    if total:
        print(f"    overall: {agree}/{total} = {100 * agree / total:.1f}%")

    print("\n  (3) SPEED")
    for v in VARIANTS:
        ss = [cells[(v, s)]["episode_seconds_mean"] for s in SEEDS if cells[(v, s)]]
        if ss:
            print(f"    {v:<8} {statistics.mean(ss):5.2f} s/episode  "
                  f"(per seed: {', '.join(f'{x:.2f}' for x in ss)})  "
                  f"-> 10,000 episodes {statistics.mean(ss) * 10000 / 3600:5.1f} h")

    # ---- peak memory, now with a filled replay
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, np.random.default_rng(0))
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    swap_in_batched_encoder(m)
    for _ in range(2):
        gradient_update_sarsa_batched(m, batch_size=100)
    tracemalloc.start()
    for _ in range(5):
        gradient_update_sarsa_batched(m, batch_size=100)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n    batched update peak python-heap allocation: {peak / 1e6:.1f} MB "
          f"(current {cur / 1e6:.1f} MB)")

    args.output.write_text(json.dumps({
        "cells": {f"{v}_s{s}": cells[(v, s)] for v in VARIANTS for s in SEEDS},
        "preferred_action_agreement": {"per_seed": per_seed, "overall": [agree, total]},
        "batched_peak_mb": peak / 1e6}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
