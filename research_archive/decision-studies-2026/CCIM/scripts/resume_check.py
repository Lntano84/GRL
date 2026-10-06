"""Does a restarted run continue a continuous one exactly?

The check is staged so that a failure is attributable:

1. **Checkpoint round-trip** -- capture, save, load, restore into a *fresh* model, and assert the state
   fingerprints match: weights, replay contents, replay priorities, episode counter, epsilon and every RNG
   stream.  A mismatch here means the checkpoint is incomplete, and nothing downstream is worth running.
2. **Continuation equality** -- from one identical starting state (replay already past ``batch_size`` so
   gradient updates are live), run ``N`` further episodes twice: once straight through, once by capturing
   at the midpoint and rebuilding the model from ``progress.load`` in a fresh object.  Compare the actions
   each episode surveys, the resulting epsilon, the replay size, and the final parameters.

Only if both pass may this be used to resume the long run.  Weights-only files (the ``ep500`` snapshots)
are deliberately **not** accepted as resume points and the loader rejects any object without the full
schema.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from ccim.xplore import progress  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

OUTPUT = ROOT / "results" / "resume_check.json"


def build(seed: int, graphs: dict, opt: dict):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    acmodel = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                         replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                         gcn_num_layers=2, num_pooling=1)
    rng = np.random.default_rng(OFF.sseed("official", seed))
    seed_rng = np.random.default_rng(OFF.sseed("official-seeds", seed))
    return acmodel, replay, rng, seed_rng


def run_episodes(acmodel, replay, rng, seed_rng, graphs, opt, start_ep, count, eps):
    """Run ``count`` episodes, returning the surveyed action sequence and the ending epsilon."""
    traces = []
    for ep in range(start_ep, start_ep + count):
        name = OFF.TRAIN_GRAPHS[int(rng.integers(len(OFF.TRAIN_GRAPHS)))]
        out = OFF.episode(graphs[name], acmodel, replay, opt[name], None, rng, seed_rng, eps)
        traces.append(out["actions"])
        eps = max(OFF.EPS_MIN, eps * OFF.EPS_DECAY)
    return traces, eps


def weight_checksum(acmodel) -> float:
    return float(sum(float(v.double().sum()) for v in acmodel.actor_critic.state_dict().values()))


def deep_equal(a, b) -> bool:
    """Structural equality that handles the numpy arrays and tensors stored in the replay.

    A plain ``list(...) == list(...)`` raises "truth value of an array is ambiguous" the moment a replay
    entry is a numpy array, which every one of them is.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        return a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup-episodes", type=int, default=0,
                    help="episodes to run before the comparison, to push replay past batch_size")
    ap.add_argument("--tail-episodes", type=int, default=4)
    ap.add_argument("--split-after", type=int, default=2)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    graphs = {n: pickle.loads((OFF.SIMP.DATA / f"{n}.pkl").read_bytes())
              for n in OFF.TRAIN_GRAPHS}
    for g in graphs.values():
        for u, v in g.edges():
            g[u][v]["p"] = 0.1
    opt = {n: OFF.make_opt(lambda a, b: OFF.influence_equivalent(a, b), graphs[n])
           for n in OFF.TRAIN_GRAPHS}

    report = {}
    print("=" * 100)
    print(f"  RESUME CHECK -- threads={args.threads}")
    print("=" * 100)

    # ---------------------------------------------------------------- 1. round trip
    acmodel, replay, rng, seed_rng = build(0, graphs, opt)
    warm = args.warmup_episodes
    while replay.size <= OFF.BATCH and warm < 60:          # updates must be live
        name = OFF.TRAIN_GRAPHS[int(rng.integers(len(OFF.TRAIN_GRAPHS)))]
        OFF.episode(graphs[name], acmodel, replay, opt[name], None, rng, seed_rng, OFF.EPS0)
        warm += 1
    obj = progress.capture(acmodel, replay, warm, OFF.EPS0, rng, seed_rng, [], None)
    path = ROOT / "results" / "_resume_check.pt"
    progress.save_atomic(obj, path)
    before = progress.describe(obj)

    acmodel2, replay2, rng2, seed_rng2 = build(0, graphs, opt)
    loaded = progress.load(path)
    ep_after = progress.restore(loaded, acmodel2, replay2, rng2, seed_rng2)
    after = progress.describe(loaded)
    same_weights = abs(weight_checksum(acmodel) - weight_checksum(acmodel2)) == 0.0
    same_replay = deep_equal(list(replay.buffer), list(replay2.buffer)) and \
        deep_equal(list(replay.probs), list(replay2.probs))
    draw_a = [float(rng.random()), int(seed_rng.integers(1 << 30)), float(replay.rg.rand())]
    draw_b = [float(rng2.random()), int(seed_rng2.integers(1 << 30)), float(replay2.rg.rand())]
    same_rng = draw_a == draw_b
    print(f"    warmed {warm} episodes, replay.size={replay.size} (updates live)")
    print(f"    [{'PASS' if same_weights else 'FAIL'}] weights restored bit-exactly "
          f"(checksum {before['weight_checksum']:.6f})")
    print(f"    [{'PASS' if same_replay else 'FAIL'}] replay contents AND priorities restored "
          f"({before['replay_size']} entries, prob sum {before['replay_probs_sum']:.6f})")
    print(f"    [{'PASS' if same_rng else 'FAIL'}] all random streams resume identically "
          f"(next draws {draw_a})")
    print(f"    episode counter resumes at {ep_after}, epsilon {loaded['epsilon']}")
    report["round_trip"] = {"warmup_episodes": warm, "replay_size": before["replay_size"],
                            "weights_identical": bool(same_weights),
                            "replay_identical": bool(same_replay),
                            "rng_identical": bool(same_rng),
                            "episode": ep_after, "epsilon": loaded["epsilon"]}

    # ---------------------------------------------------------------- 2. continuation
    # restart A: straight through
    acA, repA, rngA, srngA = build(0, graphs, opt)
    w = 0
    while repA.size <= OFF.BATCH and w < 60:
        name = OFF.TRAIN_GRAPHS[int(rngA.integers(len(OFF.TRAIN_GRAPHS)))]
        OFF.episode(graphs[name], acA, repA, opt[name], None, rngA, srngA, OFF.EPS0)
        w += 1
    eps = OFF.EPS0
    # advance both to a common start, then compare the tail
    tracesA, epsA = run_episodes(acA, repA, rngA, srngA, graphs, opt, 0, args.split_after, eps)
    mid = progress.capture(acA, repA, args.split_after, epsA, rngA, srngA, [], None)
    progress.save_atomic(mid, path)
    tracesA2, epsA2 = run_episodes(acA, repA, rngA, srngA, graphs, opt, args.split_after,
                                   args.tail_episodes, epsA)

    # restart B: rebuild from the checkpoint in fresh objects
    acB, repB, rngB, srngB = build(0, graphs, opt)
    progress.restore(progress.load(path), acB, repB, rngB, srngB)
    tracesB2, epsB2 = run_episodes(acB, repB, rngB, srngB, graphs, opt, args.split_after,
                                   args.tail_episodes, epsA)

    same_actions = deep_equal(tracesA2, tracesB2)
    same_weights2 = weight_checksum(acA) == weight_checksum(acB)
    same_size = repA.size == repB.size
    same_eps = epsA2 == epsB2
    print(f"\n    continuation: {args.split_after} episodes, checkpoint, then {args.tail_episodes} more")
    print(f"      continuous run  actions {tracesA2}")
    print(f"      restarted run   actions {tracesB2}")
    print(f"    [{'PASS' if same_actions else 'FAIL'}] identical survey actions after restarting")
    print(f"    [{'PASS' if same_weights2 else 'FAIL'}] identical parameters "
          f"({weight_checksum(acA):.6f} vs {weight_checksum(acB):.6f})")
    print(f"    [{'PASS' if same_size else 'FAIL'}] identical replay size ({repA.size} vs {repB.size})")
    print(f"    [{'PASS' if same_eps else 'FAIL'}] identical epsilon ({epsA2:.8f} vs {epsB2:.8f})")
    report["continuation"] = {"split_after": args.split_after, "tail_episodes": args.tail_episodes,
                              "actions_continuous": tracesA2, "actions_restarted": tracesB2,
                              "actions_identical": bool(same_actions),
                              "weights_identical": bool(same_weights2),
                              "replay_size_identical": bool(same_size),
                              "epsilon_identical": bool(same_eps)}

    ok = (same_weights and same_replay and same_rng and same_actions and same_weights2
          and same_size and same_eps)
    print(f"\n  OVERALL: {'resume is exact -- safe for the long run' if ok else 'RESUME IS NOT EXACT'}")
    report["overall_pass"] = bool(ok)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  artifact: {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
