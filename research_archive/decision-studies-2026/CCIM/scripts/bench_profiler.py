"""Where inside forward/backward does the time go -- many tiny ops, or a few big ones?

Runs at the best thread count found by ``scripts/bench_update.py`` (1), profiles a few real
``gradient_update_sarsa`` calls with ``torch.profiler``, and prints the top operators both by CPU time and
by call count.  The call count is the decisive column: a handful of large ops would mean the update is
compute-bound, whereas thousands of near-identical small ops mean it is launch/Python-bound -- and only the
second reading supports batched calls as the fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from bench_update import BATCH, fill  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "bench_profiler.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)
    acmodel = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                         replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                         gcn_num_layers=2, num_pooling=1)
    for _ in range(2):
        acmodel.gradient_update_sarsa(batch_size=BATCH)

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for _ in range(args.updates):
            acmodel.gradient_update_sarsa(batch_size=BATCH)

    stats = defaultdict(lambda: [0.0, 0])          # name -> [cpu_us_total, calls]
    for ev in prof.key_averages():
        if ev.count and ev.self_cpu_time_total > 0:
            stats[ev.key][0] += ev.self_cpu_time_total
            stats[ev.key][1] += ev.count

    by_time = sorted(stats.items(), key=lambda kv: -kv[1][0])
    total_us = sum(v[0] for v in stats.values())
    total_calls = sum(v[1] for v in stats.values())

    print("=" * 96)
    print(f"  PROFILER -- {args.updates} real updates, threads={args.threads}")
    print("=" * 96)
    print(f"    distinct operators: {len(stats):,}   total op calls: {total_calls:,} "
          f"({total_calls / args.updates:,.0f} per update)")
    print(f"    total CPU time: {total_us / 1e6:.2f} s over {args.updates} updates "
          f"({total_us / 1e3 / args.updates:.0f} ms per update)")
    print(f"\n    {'operator':<44}{'cpu ms/upd':>12}{'calls/upd':>12}{'us/call':>9}{'share':>8}")
    for name, (us, calls) in by_time[:22]:
        print(f"    {name[:43]:<44}{us / 1e3 / args.updates:>12.2f}{calls / args.updates:>12.0f}"
              f"{us / max(calls, 1):>9.1f}{100 * us / total_us:>7.1f}%")

    per_update_calls = total_calls / args.updates
    verdict = ("many small repeated ops -> launch/Python bound, batching is the lever"
               if per_update_calls > 500 else
               "few large ops -> compute bound, threads/GPU are the lever")
    print(f"\n    verdict: {verdict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "threads": args.threads, "updates": args.updates,
        "total_op_calls": total_calls, "calls_per_update": per_update_calls,
        "cpu_ms_per_update": total_us / 1e3 / args.updates,
        "top_operators": [{"name": n, "cpu_ms_per_update": us / 1e3 / args.updates,
                           "calls_per_update": c / args.updates}
                          for n, (us, c) in by_time[:30]],
        "verdict": verdict}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
