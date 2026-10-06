"""Short-horizon check: does the batched update visibly damage early learning or decisions?

Frozen settings: 3 pre-fixed seeds (0, 1, 2), 200 episodes each, `fast` and `batched` both at 1 thread,
the same training graphs, the same initial observations and the same validation graphs.  Checkpoints are
taken at ep0 / ep100 / ep200.

Deliverables
------------
* per-seed validation-spread curves,
* Q ordering and the preferred survey action on a fixed set of visible network states,
* measured wall clock for a full episode in each version, and peak memory for the batched version.

200 episodes only asks whether early learning and decision-making are obviously broken.  It cannot show
that the two versions have the same long-run quality, and it is not reported as such.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from ccim.xplore.dqn_batched import gradient_update_sarsa_batched, swap_in_batched_encoder  # noqa: E402
from ccim.xplore.dqn_opt import gradient_update_sarsa_fast  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

OUTPUT = ROOT / "results" / "shorthorizon.json"
CKPT = ROOT / "results" / "shorthorizon_ckpt"
SEEDS = (0, 1, 2)          # pre-fixed, before any run
EPISODES = 200
CHECKPOINTS = (0, 100, 200)
VARIANTS = ("fast", "batched")


def build(seed, replay, batched=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    if batched:
        swap_in_batched_encoder(m)
    return m


def install(model, variant):
    if variant == "fast":
        model.gradient_update_sarsa = lambda batch_size=100: gradient_update_sarsa_fast(
            model, batch_size=batch_size, compute_loss=False, share_embeddings=True)
    else:
        model.gradient_update_sarsa = lambda batch_size=100: gradient_update_sarsa_batched(
            model, batch_size=batch_size)


def _state_arrays(env, g):
    """Build the (features, adjacency) pair directly instead of via ``observed_state``.

    ``observed_state`` is fine for the training loop but its return value goes through
    ``np.array([...])`` inside ``get_values2``, which chokes on anything ragged.  Constructing the two
    arrays here makes the shapes explicit and independent of how the environment stores its subgraph.
    """
    _ = env.state                                   # the authors' property sets env.sub
    node_list = sorted(env.active.union(env.possible_actions))
    index = {u: i for i, u in enumerate(node_list)}
    attrs = np.ones((len(node_list), OFF.INPUT_DIM), dtype=np.float32)
    adj = np.zeros((len(node_list), len(node_list)), dtype=np.float32)
    for u, v in env.sub.edges():
        if u in index and v in index:
            adj[index[u], index[v]] = adj[index[v], index[u]] = 1.0
    return attrs, adj, node_list


def q_ordering(model, states):
    """Preferred action (argmax Q of critic 1) and the full Q ordering on fixed visible states."""
    out = []
    with torch.no_grad():
        for entry in states:
            attrs, adj, _ = _state_arrays(entry["env"], entry["g"])
            state = [attrs, adj]
            grid = [OFF.greedy_action(model, state, entry["embs"], entry["frontier"], 0)]
            qs = []
            for v in entry["frontier"]:
                act = np.asarray(entry["embs"][v], dtype=np.float32).reshape(-1)
                qs.append(float(model.get_values2(attrs, adj, act)[0]))
            order = sorted(range(len(entry["frontier"])), key=lambda i: -qs[i])
            out.append({"preferred_node": int(grid[0][0]),
                        "q_order_nodes": [int(entry["frontier"][i]) for i in order[:5]],
                        "q_top5": [round(qs[i], 6) for i in order[:5]]})
    return out


def make_fixed_states(graphs, n=6):
    """A fixed set of visible network states, built once and reused by both versions."""
    states = []
    for i in range(n):
        g = graphs[OFF.TRAIN_GRAPHS[i % len(OFF.TRAIN_GRAPHS)]]
        rng = np.random.default_rng(5000 + i)
        seeds = [int(x) for x in rng.choice(g.number_of_nodes(), OFF.EXTRA_SEEDS, replace=False)]
        env = OFF.NetworkEnv(g, seeds=seeds, max_T=OFF.T_MAX, influence_algo=None, opt_reward=0,
                             nop_r=0, times_mean=1, bad_reward=0, normalize=False)
        env.reset()
        for _ in range(i % 3):
            f = sorted(env.possible_actions)
            if f:
                env.step(int(f[0]))
        obs = OFF.observed_state(env, g)
        frontier = sorted(env.possible_actions)
        embs = OFF.get_embeds(env.sub, num_walks=OFF.NUM_WALKS, walk_length=OFF.WALK_LEN,
                              window=OFF.WINDOW, iterations=OFF.EMB_ITERS, embed_size=OFF.EMB_DIM,
                              seed=9000 + i)
        states.append({"env": env, "g": g, "embs": embs, "frontier": frontier})
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    CKPT.mkdir(parents=True, exist_ok=True)

    graphs = {n: pickle.loads((OFF.SIMP.DATA / f"{n}.pkl").read_bytes()) for n in OFF.TRAIN_GRAPHS}
    for g in graphs.values():
        for u, v in g.edges():
            g[u][v]["p"] = 0.1
    val_graphs = {n: pickle.loads((OFF.SIMP.DATA / f"{n}.pkl").read_bytes()) for n in OFF.VAL_GRAPHS}
    plan = json.loads(OFF.PLAN.read_text(encoding="utf-8"))
    opt = {n: OFF.make_opt(lambda a, b: OFF.influence_equivalent(a, b), graphs[n])
           for n in OFF.TRAIN_GRAPHS}
    fixed_states = make_fixed_states(graphs)

    print("=" * 104)
    print(f"  SHORT-HORIZON CHECK -- {args.episodes} episodes, seeds {SEEDS}, threads={args.threads}")
    print("=" * 104)
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for seed in SEEDS:
            replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
            model = build(seed, replay, batched=(variant == "batched"))
            install(model, variant)
            rng = np.random.default_rng(OFF.sseed("short", seed))
            srng = np.random.default_rng(OFF.sseed("short-seeds", seed))
            eps = OFF.EPS0
            curve, snaps, ep_times = [], {}, []
            snaps[0] = {"ac": copy.deepcopy(model.actor_critic.state_dict()), "ep": 0}
            for ep in range(1, args.episodes + 1):
                name = OFF.TRAIN_GRAPHS[int(rng.integers(len(OFF.TRAIN_GRAPHS)))]
                t0 = time.perf_counter()
                OFF.episode(graphs[name], model, replay, opt[name], None, rng, srng, eps)
                ep_times.append(time.perf_counter() - t0)
                eps = max(OFF.EPS_MIN, eps * OFF.EPS_DECAY)
                if ep in CHECKPOINTS:
                    vals = []
                    for gname, tag, seeds in plan["validation_observations"]:
                        ev = OFF.LiveEdgeObjective(
                            val_graphs[gname], samples=1000,
                            rng=np.random.default_rng(OFF.sseed("val", gname, tag)),
                            p=OFF.PROP_PROBAB)
                        out = OFF.evaluate(val_graphs[gname], model, [seeds], ev, gname, f"val_{tag}")
                        vals.append(out["spread"][0])
                    spread = statistics.mean(vals)
                    curve.append({"episode": ep, "val_mean_spread": spread})
                    snaps[ep] = {"ac": copy.deepcopy(model.actor_critic.state_dict()), "ep": ep}
                    print(f"    {variant:<8} seed {seed}  ep{ep:<4} val spread {spread:7.2f}"
                          f"  replay {replay.size}", flush=True)
                    # flush at EVERY checkpoint, not only when the cell finishes: this machine is
                    # switched off often enough that a kill at ep150 must not discard ep0 and ep100
                    torch.save(snaps, CKPT / f"{variant}_seed{seed}.pt")
                    (CKPT / f"{variant}_seed{seed}_curve.json").write_text(
                        json.dumps({"variant": variant, "seed": seed, "val_curve": curve,
                                    "episodes_done": ep,
                                    "episode_seconds_mean": statistics.mean(ep_times),
                                    "replay_size": replay.size},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
            # save BEFORE the decision probe: a crash in q_ordering must not discard 200 episodes
            torch.save(snaps, CKPT / f"{variant}_seed{seed}.pt")
            qs = {ep: q_ordering(model, fixed_states) for ep in CHECKPOINTS}
            results[variant][seed] = {
                "val_curve": curve, "q_ordering": qs,
                "episode_seconds_mean": statistics.mean(ep_times[60:]) if len(ep_times) > 60
                else statistics.mean(ep_times),
                "replay_size": replay.size}
            (CKPT / f"{variant}_seed{seed}_result.json").write_text(
                json.dumps({"val_curve": curve, "q_ordering": qs,
                            "episode_seconds_mean": results[variant][seed]["episode_seconds_mean"],
                            "replay_size": replay.size}, ensure_ascii=False, indent=2),
                encoding="utf-8")
    # ---- peak memory of the batched forward
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    m = build(0, PriortizedReplay(OFF.BUFF, 10, beta=0.6), batched=True)
    for _ in range(2):
        gradient_update_sarsa_batched(m, batch_size=100)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 104)
    print("  VALIDATION CURVES")
    print("=" * 104)
    for seed in SEEDS:
        for variant in VARIANTS:
            c = results[variant][seed]["val_curve"]
            print(f"    seed {seed} {variant:<8} " + "  ".join(
                f"ep{x['episode']}={x['val_mean_spread']:.2f}" for x in c))
        a = [x["val_mean_spread"] for x in results["fast"][seed]["val_curve"]]
        b = [x["val_mean_spread"] for x in results["batched"][seed]["val_curve"]]
        print(f"      -> batched - fast: " + "  ".join(f"{y - x:+.2f}" for x, y in zip(a, b)))

    print("\n" + "=" * 104)
    print("  DECISIONS ON FIXED VISIBLE STATES")
    print("=" * 104)
    agree_pref = total = 0
    for seed in SEEDS:
        for ep in CHECKPOINTS:
            qa = results["fast"][seed]["q_ordering"][ep]
            qb = results["batched"][seed]["q_ordering"][ep]
            for s, (x, y) in enumerate(zip(qa, qb)):
                total += 1
                agree_pref += int(x["preferred_node"] == y["preferred_node"])
    print(f"    preferred survey action identical: {agree_pref}/{total} state-checkpoint pairs")

    print("\n" + "=" * 104)
    print("  TIMING AND MEMORY")
    print("=" * 104)
    for variant in VARIANTS:
        s = [results[variant][k]["episode_seconds_mean"] for k in SEEDS]
        print(f"    {variant:<8} mean episode {statistics.mean(s):6.2f} s   "
              f"-> 10,000 episodes {statistics.mean(s) * 10000 / 3600:5.1f} h")
    print(f"    batched update peak python-heap allocation: {peak / 1e6:.1f} MB "
          f"(current {cur / 1e6:.1f} MB)")

    args.output.write_text(json.dumps({"seeds": list(SEEDS), "episodes": args.episodes,
                                       "results": results,
                                       "preferred_action_agreement": [agree_pref, total],
                                       "batched_peak_mb": peak / 1e6},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
