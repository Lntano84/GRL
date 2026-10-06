"""One question: under equal frontier access and the same budget, does Geometric-DQN beat random and degree?

This round changes exactly one thing
------------------------------------
The environment, the budget, the seed-selection protocol and the reward are the ones already verified in
``scripts/xplore_baseline.py``.  Nothing about them is touched here.  The only new element is a learned
survey policy, because changing environment and model together would make the result uninterpretable.

What is answered
----------------
*Can an already-existing learning method work in our environment at all* -- before asking whether our own
gain-prediction module adds anything.  If it cannot, the second question is moot.

Frozen before training, by ``--plan-only``
------------------------------------------
train / validation / development-evaluation graph lists; five training seeds, each reported; the author's
hyperparameters; the episode budget recorded as an explicit deviation; the checkpoint rule (**highest mean
final spread on the independent validation graphs, ties to the earliest**); and the 20 development
observations.  The checkpoint is never chosen by development-graph spread, discovered-node counts, or the
training reward.

Statistics, stated honestly
---------------------------
Five model seeds x 20 observations is **not** 100 independent replicates: the 20 observations are shared,
so the five models are correlated and are reported separately.  Their spread is reported as a spread
across seeds, not converted into a test the design does not support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402
from ccim.xplore.learner import GeoDQN, encode_observation  # noqa: E402

DATA = ROOT / "_data" / "xplore"
PLAN = ROOT / "results" / "xplore_geodqn_plan.json"
OUTPUT = ROOT / "results" / "xplore_geodqn.json"

TRAIN_GRAPHS = ("interact__ia-crime-moreno", "interact__ia-infect-dublin",
                "interact__ia-infect-hyper")
VAL_GRAPHS = ("interact__ia-enron-only", "interact__rt-twitter-copen")
DEV_GRAPH = "interact__soc-wiki-Vote"      # never trained on, never used to choose a checkpoint
TRAIN_SEEDS = (0, 1, 2, 3, 4)
BUDGET = T_MAX = 5
EXTRA_SEEDS = 5
SELECT_SAMPLES = 100
EVAL_SAMPLES = 1000
DEV_OBSERVATIONS = 20
VAL_OBSERVATIONS_PER_GRAPH = 5
EPISODES = 2000                            # a recorded deviation from train.py's 10000
VAL_EVAL_POINTS = (500, 1000, 2000)        # the checkpoint rule is applied at these fixed points
HYPER = {"lr": 1e-4, "gamma": 0.99, "batch_size": 100, "buffer_size": 4000,
         "epsilon": 0.1, "eps_min": 0.01, "eps_decay": 0.999, "target_sync_every": 200}
PLAN_RNG_SEED = 20240120


def sseed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63)


def greedy_seeds(env, graph, tag) -> list:
    sel = LiveEdgeObjective(env.graph, samples=SELECT_SAMPLES,
                            rng=np.random.default_rng(sseed("select", graph, tag)), p=PROP_PROBAB)
    seeds, _ = sel.greedy(INFL_BUDGET)
    return list(seeds)


def rollout(g, kind, seeds, agent, graph, tag) -> dict:
    """One episode to T_MAX under a fixed policy, then the official selection pipeline."""
    env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
    t = 0
    while t < T_MAX and not env.done():
        obs = encode_observation(env, t, T_MAX)
        if not obs["frontier"]:
            break
        if kind == "geodqn":
            a = agent.act(obs, explore=False)
        else:
            rng = np.random.default_rng(sseed("pick", graph, tag, t, kind))
            if kind == "random":
                a = int(obs["frontier"][int(rng.integers(len(obs["frontier"])))])
            else:
                deg = {u: env.observed.degree(u) for u in obs["frontier"]}
                target = max(deg.values()) if kind == "degree_max" else min(deg.values())
                tied = [u for u in obs["frontier"] if deg[u] == target]
                a = int(tied[int(rng.integers(len(tied)))])
        env.step(a)
        t += 1
    chosen = greedy_seeds(env, graph, tag)
    discovered = env.active | env.possible_actions
    return {"chosen": chosen, "surveys": len(env.queried), "discovered": env.discovered_nodes,
            "undiscovered_seeds": int(sum(1 for s in chosen if s not in discovered))}


def score(g, chosen, graph, tag, samples=SELECT_SAMPLES) -> float:
    obj = LiveEdgeObjective(g, samples=samples,
                            rng=np.random.default_rng(sseed("reward", graph, tag)), p=PROP_PROBAB)
    return obj.value(chosen)


# ------------------------------------------------------------------ training
def validate(agent, graphs, val_plan) -> float:
    """Mean final spread over the independent validation observations -- the checkpoint criterion."""
    vals = []
    for graph, tag, obs_seeds in val_plan:
        out = rollout(graphs[graph], "geodqn", obs_seeds, agent, graph, f"val_{tag}")
        vals.append(score(graphs[graph], out["chosen"], graph, f"val_{tag}"))
    return statistics.mean(vals)


def train_one(seed: int, graphs: dict, val_plan: list, episodes: int, log,
              snapshot_at=(), snapshot_out: dict | None = None) -> dict:
    """Train one model seed.  ``snapshot_at`` records weight copies at those episodes.

    Snapshotting only copies tensors -- it consumes no randomness -- so a run with snapshots must
    reproduce a run without them **exactly**.  That is what makes it legitimate to recover the models of
    the previous round by re-executing it, given that the weights were not archived at the time.
    """
    import copy
    agent = GeoDQN(seed=seed, **{k: v for k, v in HYPER.items() if k != "target_sync_every"})
    rng = np.random.default_rng(sseed("train", seed))
    interactions, t0, losses = 0, time.perf_counter(), []
    curve, best = [], {"spread": -1.0, "episode": None, "state": None}
    eval_points = [p for p in VAL_EVAL_POINTS if p <= episodes]
    if episodes not in eval_points:
        eval_points.append(episodes)
    snapshot_at = set(snapshot_at)
    if snapshot_out is not None and 0 in snapshot_at:
        snapshot_out[0] = copy.deepcopy(agent.q.state_dict())   # the untrained initialisation
    for ep in range(1, episodes + 1):
        graph = TRAIN_GRAPHS[int(rng.integers(len(TRAIN_GRAPHS)))]
        g = graphs[graph]
        seeds = [int(x) for x in rng.choice(g.number_of_nodes(), EXTRA_SEEDS, replace=False)]
        env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
        hist, t = [], 0
        while t < T_MAX and not env.done():
            obs = encode_observation(env, t, T_MAX)
            if not obs["frontier"]:
                break
            a = agent.act(obs, explore=True)
            env.step(a)
            interactions += 1
            hist.append((obs, a))
            t += 1
        tag = f"{seed}:{ep}"
        chosen = greedy_seeds(env, graph, tag)
        reward = score(g, chosen, graph, tag)      # official terminal reward, fresh samples
        for i, (obs, a) in enumerate(hist):
            done = (i == len(hist) - 1)
            agent.observe(obs, a, reward if done else 0.0, obs if done else hist[i + 1][0], done)
        loss = agent.update()
        if loss is not None:
            losses.append(loss)
        if ep % HYPER["target_sync_every"] == 0:
            agent.sync_target()
        agent.decay_epsilon()
        if snapshot_out is not None and ep in snapshot_at:
            snapshot_out[ep] = copy.deepcopy(agent.q.state_dict())
        if ep in eval_points:
            spread = validate(agent, graphs, val_plan)
            curve.append({"episode": ep, "val_mean_spread": spread,
                          "epsilon": agent.epsilon, "updates": agent.updates,
                          "seconds": time.perf_counter() - t0})
            # highest validation spread wins; ties go to the earliest evaluation
            if spread > best["spread"]:
                best = {"spread": spread, "episode": ep,
                        "state": copy.deepcopy(agent.q.state_dict())}
            log(f"      seed {seed}: episode {ep}/{episodes}  eps={agent.epsilon:.4f}"
                f"  updates={agent.updates}  val spread={spread:.2f}"
                f"  best={best['spread']:.2f}@{best['episode']}"
                f"  {time.perf_counter() - t0:.0f}s")
    agent.q.load_state_dict(best["state"])
    return {"agent": agent, "interactions": interactions,
            "train_seconds": time.perf_counter() - t0,
            "val_mean_spread": best["spread"], "selected_episode": best["episode"],
            "val_curve": curve, "final_epsilon": agent.epsilon, "updates": agent.updates}


def build_plan() -> dict:
    sizes = {}
    for name in TRAIN_GRAPHS + VAL_GRAPHS + (DEV_GRAPH,):
        g = pickle.loads((DATA / f"{name}.pkl").read_bytes())
        sizes[name] = {"n": g.number_of_nodes(), "m": g.number_of_edges()}
    val_obs = []
    for graph in VAL_GRAPHS:
        n = sizes[graph]["n"]
        for i in range(VAL_OBSERVATIONS_PER_GRAPH):
            val_obs.append([graph, f"{graph}_{i}",
                            sorted(int(x) for x in np.random.default_rng(sseed(graph, i, "valinit"))
                                   .choice(n, EXTRA_SEEDS, replace=False))])
    dev = []
    n = sizes[DEV_GRAPH]["n"]
    for i in range(DEV_OBSERVATIONS):
        dev.append({"i": i, "initial_nodes": sorted(
            int(x) for x in np.random.default_rng(sseed(DEV_GRAPH, i, "devinit"))
            .choice(n, EXTRA_SEEDS, replace=False))})
    return {"plan_rng_seed": PLAN_RNG_SEED,
            "graphs": {"train": list(TRAIN_GRAPHS), "validation": list(VAL_GRAPHS),
                       "development_evaluation": DEV_GRAPH},
            "graph_role_note": (
                "the development-evaluation graph is never trained on and never used for checkpoint "
                "selection; it took part in the direction choice in the previous round, so it is a "
                "development graph and NOT a final independent test set"),
            "graph_sizes": sizes, "train_seeds": list(TRAIN_SEEDS), "episodes": EPISODES,
            "budget": BUDGET, "t_max": T_MAX, "extra_seeds": EXTRA_SEEDS,
            "select_samples": SELECT_SAMPLES, "eval_samples": EVAL_SAMPLES,
            "hyperparameters": HYPER,
            "deviations_from_paper": [
                f"episodes: {EPISODES}/seed here vs 10000 in train.py; the published count is not "
                f"reachable at this scale and the per-episode cost is measured and reported",
                "node embeddings: DeepWalk recomputed per step -> end-to-end GCN encoder "
                "(gensim is not installed and the published loop retrains DeepWalk at every step)",
                "graph pooling: DiffPool -> mean+max pooling (the authors' diffpool module does not "
                "port to this tensor pipeline)",
                "reward: raw final influence with 1x100 samples, matching train.py opt_reward=0 / "
                "norm_reward=0, which already differs from the paper's Eq. (1); the official code also "
                "averages times_mean_env=5 repeats per step, which is not done here",
            ],
            "checkpoint_rule": ("highest mean final spread over the validation graphs, evaluated at the "
                                "fixed points "
                                f"{list(VAL_EVAL_POINTS)}, ties to the earliest; never the development "
                                "graph, never discovered-node counts, never the training reward.  "
                                "Evaluating at fixed points also shows whether training was still "
                                "improving when the episode budget ran out."),
            "validation_observations": val_obs, "dev_observations": dev,
            "statistics_note": ("5 model seeds x 20 observations is not 100 independent replicates; the "
                                "observations are shared, so models are correlated and are reported "
                                "separately")}


def paired(a: dict, b: dict, keys) -> dict:
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    if len(d) < 2:
        return {"n": len(d), "mean": float(d.mean()) if len(d) else None, "lo": None, "hi": None,
                "p": None, "wins": None, "losses": None}
    mean = float(d.mean())
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(d), "mean": mean, "lo": mean - half, "hi": mean + half,
            "p": float(stats.ttest_1samp(d, 0.0).pvalue),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.plan_only:
        plan = build_plan()
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=" * 100)
        print("  Geometric-DQN PILOT PLAN FROZEN -- written before any training")
        print("=" * 100)
        print(f"    train      : {list(TRAIN_GRAPHS)}")
        print(f"    validation : {list(VAL_GRAPHS)}   ({VAL_OBSERVATIONS_PER_GRAPH} observations each)")
        print(f"    dev eval   : {DEV_GRAPH}  ({DEV_OBSERVATIONS} observations)")
        for name, s in plan["graph_sizes"].items():
            print(f"        {name:<28} n={s['n']:<5} m={s['m']}")
        print(f"    train seeds: {list(TRAIN_SEEDS)}   episodes/seed: {EPISODES}   budget T={BUDGET}")
        print(f"    hyper      : {HYPER}")
        print(f"    checkpoint : {plan['checkpoint_rule']}")
        print("    recorded deviations from the paper/code:")
        for d in plan["deviations_from_paper"]:
            print(f"      - {d}")
        print(f"\n    written to {args.plan}")
        return 0

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    episodes = int(plan["episodes"])
    graphs = {name: pickle.loads((DATA / f"{name}.pkl").read_bytes())
              for name in plan["graphs"]["train"] + plan["graphs"]["validation"]
              + [plan["graphs"]["development_evaluation"]]}

    print("=" * 100)
    print("  Geometric-DQN PILOT -- equal frontier access, T=5, official reward, official seeding")
    print("=" * 100)
    print(f"    train {list(TRAIN_GRAPHS)} | val {list(VAL_GRAPHS)} | dev-eval {DEV_GRAPH}")
    print(f"    {episodes} episodes/seed x {len(TRAIN_SEEDS)} seeds", flush=True)

    models = {}
    for seed in TRAIN_SEEDS:
        print(f"\n    training model seed {seed}", flush=True)
        models[seed] = train_one(seed, graphs, plan["validation_observations"], episodes, print)
        m = models[seed]
        print(f"      seed {seed}: val spread {m['val_mean_spread']:.2f}  "
              f"interactions {m['interactions']:,}  {m['train_seconds']:.0f}s", flush=True)

    print(f"\n    development evaluation on {DEV_GRAPH}, {DEV_OBSERVATIONS} fixed observations",
          flush=True)
    g = graphs[DEV_GRAPH]
    evals = {d["i"]: LiveEdgeObjective(
        g, samples=EVAL_SAMPLES, rng=np.random.default_rng(sseed(DEV_GRAPH, d["i"], "deveval")),
        p=PROP_PROBAB) for d in plan["dev_observations"]}
    arms = {}
    for kind in ("random", "degree_max", "degree_min"):
        arms[kind] = {}
        for d in plan["dev_observations"]:
            out = rollout(g, kind, d["initial_nodes"], None, DEV_GRAPH, f"dev{d['i']}")
            out["influence"] = evals[d["i"]].value(out["chosen"])
            out["inference_seconds"] = 0.0
            arms[kind][d["i"]] = out
    for seed in TRAIN_SEEDS:
        name = f"geodqn_s{seed}"
        arms[name] = {}
        for d in plan["dev_observations"]:
            t0 = time.perf_counter()
            out = rollout(g, "geodqn", d["initial_nodes"], models[seed]["agent"], DEV_GRAPH,
                          f"dev{d['i']}")
            out["inference_seconds"] = time.perf_counter() - t0
            out["influence"] = evals[d["i"]].value(out["chosen"])
            arms[name][d["i"]] = out

    keys = [d["i"] for d in plan["dev_observations"]]
    base = {k: arms["random"][k]["influence"] for k in keys}
    dmax = {k: arms["degree_max"][k]["influence"] for k in keys}
    rows = {}
    print("\n" + "=" * 100)
    print(f"  RESULTS -- {DEV_GRAPH} (n={g.number_of_nodes()}), T={BUDGET}, "
          f"{DEV_OBSERVATIONS} paired observations")
    print("=" * 100)
    print(f"    {'method / model seed':<22}{'spread':>9}{'vs random':>23}{'vs degree_max':>23}"
          f"{'discovered':>11}{'infer s':>9}")
    for name in arms:
        sp = {k: arms[name][k]["influence"] for k in keys}
        r1, r2 = paired(sp, base, keys), paired(sp, dmax, keys)
        rows[name] = {"spread_mean": statistics.mean(sp.values()), "vs_random": r1,
                      "vs_degree_max": r2,
                      "discovered_mean": statistics.mean(arms[name][k]["discovered"] for k in keys),
                      "inference_seconds_mean": statistics.mean(
                          arms[name][k]["inference_seconds"] for k in keys),
                      "per_observation": sp}
        f1 = f"{r1['mean']:+.2f} [{r1['lo']:+.2f},{r1['hi']:+.2f}]" if r1["lo"] is not None else "-"
        f2 = f"{r2['mean']:+.2f} [{r2['lo']:+.2f},{r2['hi']:+.2f}]" if r2["lo"] is not None else "-"
        print(f"    {name:<22}{statistics.mean(sp.values()):>9.2f}{f1:>23}{f2:>23}"
              f"{rows[name]['discovered_mean']:>11.1f}{rows[name]['inference_seconds_mean']:>9.4f}")

    tr_sec = sum(m["train_seconds"] for m in models.values())
    tr_int = sum(m["interactions"] for m in models.values())
    geod = [rows[f"geodqn_s{s}"]["spread_mean"] for s in TRAIN_SEEDS]
    print(f"\n    total training time       : {tr_sec:.0f} s ({tr_sec / 60:.1f} min) over "
          f"{len(TRAIN_SEEDS)} seeds")
    print(f"    total environment steps   : {tr_int:,}  ({tr_int // len(TRAIN_SEEDS):,}/seed)")
    print(f"    per-seed validation spread: "
          + ", ".join(f"s{s}={models[s]['val_mean_spread']:.2f}" for s in TRAIN_SEEDS))
    print(f"\n    Geometric-DQN across the 5 model seeds: min {min(geod):.2f}  max {max(geod):.2f}"
          f"  mean {statistics.mean(geod):.2f}")
    print(f"    random {rows['random']['spread_mean']:.2f}   "
          f"degree_max {rows['degree_max']['spread_mean']:.2f}   "
          f"degree_min {rows['degree_min']['spread_mean']:.2f}")

    artifact = {"script": Path(__file__).name, "plan": plan,
                "protocol": {"budget": BUDGET, "extra_seeds": EXTRA_SEEDS, "p": PROP_PROBAB,
                             "infl_budget": INFL_BUDGET, "select_samples": SELECT_SAMPLES,
                             "eval_samples": EVAL_SAMPLES,
                             "reward": "official: final influence of the greedy seed set on the complete "
                                       "graph; train.py opt_reward=0 / norm_reward=0",
                             "access": "frontier only, all four arms",
                             "reference_impls": "kage08/graph_sample_rl (icm.py, expts/net_env.py, "
                                                "expts/influence.py, utils.greedy, train.py)"},
                "arms": arms, "rows": rows,
                "training": {"per_seed": {str(s): {"val_mean_spread": models[s]["val_mean_spread"],
                                                   "selected_episode": models[s]["selected_episode"],
                                                   "val_curve": models[s]["val_curve"],
                                                   "interactions": models[s]["interactions"],
                                                   "train_seconds": models[s]["train_seconds"],
                                                   "updates": models[s]["updates"],
                                                   "final_epsilon": models[s]["final_epsilon"]}
                                          for s in TRAIN_SEEDS},
                             "total_seconds": tr_sec, "total_interactions": tr_int,
                             "episodes_per_seed": episodes},
                "geodqn_spread_across_seeds": {"min": min(geod), "max": max(geod),
                                               "mean": statistics.mean(geod),
                                               "per_seed": {str(s): rows[f"geodqn_s{s}"]["spread_mean"]
                                                            for s in TRAIN_SEEDS}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
