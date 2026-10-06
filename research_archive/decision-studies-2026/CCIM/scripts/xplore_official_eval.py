"""The two agreed tables for the frozen adapted version.  No new training, no configuration change.

Terminology, fixed by instruction
---------------------------------
The arm is called the **official-architecture low-budget adapted version** ("adapted version" below).
The two primary comparisons are **adapted − random** and **adapted − degree_min**.  ``degree_min`` was
picked as the strong heuristic reference from an earlier round's results, so it is a result-informed
choice and is labelled as such.

What the reward substitution does and does not establish
--------------------------------------------------------
``influence_equivalent()`` is an **alternative implementation of the same target**, not a proven
equivalent, and the function name is not evidence.  The recorded 1.85% cross-check (max observed relative
difference at 4000 samples) supports compatibility of the *propagation estimate on a fixed seed set*.  It
does **not** establish that the whole stochastic reward pipeline -- greedy selection on 100 samples and
then an independent evaluation -- agrees, because finite-sample error can change which seeds greedy picks.
Both the evidence and this limitation are reported; no further validation is attempted this round.

Reward repeats
--------------
Five repeats are actually executed.  The published ``parallel_influence`` would run **four**, because its
index arithmetic drops one worker for ``times=5``.  With i.i.d. repeats four and five have the same
expectation but different variance and cost, so executing five is an explicit **behavioural correction**,
not a compatibility patch.  The actual repeat count is logged in the artifact.  ``r / opt`` is reported as
normalisation against a **full-knowledge greedy reference**; ``opt`` is not a proven global optimum.
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

import xplore_learn as SIMP  # noqa: E402
import xplore_official as OFF  # noqa: E402
# these two live inside xplore_official.main(), so they are not module attributes of OFF
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402
from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402

CKPT = ROOT / "results" / "official_checkpoints"
PREV = ROOT / "results" / "xplore_geodqn.json"
OUTPUT = ROOT / "results" / "xplore_official_tables.json"
DEV_GRAPH = OFF.DEV_GRAPH
EVAL_SAMPLES = 1000
OBS = 20


def sseed(*parts) -> int:
    return SIMP.sseed(*parts)


def dev_observations(g) -> list:
    n = g.number_of_nodes()
    return [sorted(int(x) for x in np.random.default_rng(sseed(DEV_GRAPH, i, "devinit"))
                   .choice(n, 5, replace=False)) for i in range(OBS)]


def heuristic_rollout(g, kind, seeds, tag) -> dict:
    """The three frontier heuristics, with the same streams as the earlier rounds so numbers reconcile."""
    env = DiscoveryEnv(g, seeds, max_T=5, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
    taken, t = [], 0
    while t < 5 and not env.done():
        frontier = sorted(env.possible_actions)
        if not frontier:
            break
        rng = np.random.default_rng(sseed("pick", DEV_GRAPH, tag, t, kind))
        if kind == "random":
            a = int(frontier[int(rng.integers(len(frontier)))])
        else:
            deg = {u: env.observed.degree(u) for u in frontier}
            target = max(deg.values()) if kind == "degree_max" else min(deg.values())
            tied = [u for u in frontier if deg[u] == target]
            a = int(tied[int(rng.integers(len(tied)))])
        env.step(a)
        taken.append(a)
        t += 1
    chosen = SIMP.greedy_seeds(env, DEV_GRAPH, tag)
    return {"actions": taken, "chosen": chosen, "discovered": env.discovered_nodes,
            "frontier": len(env.possible_actions)}


def official_rollout(g, acmodel, seeds, tag) -> dict:
    """One deterministic rollout of the adapted version, using the authors' environment."""
    from expts.net_env import NetworkEnv
    env = NetworkEnv(g, seeds=list(seeds), max_T=5, influence_algo=None, opt_reward=0, nop_r=0,
                     times_mean=1, bad_reward=0, normalize=False)
    env.reset()
    s, _ = OFF.observed_state(env, g)
    embs = OFF.get_embeds(env.sub, num_walks=OFF.NUM_WALKS, walk_length=OFF.WALK_LEN,
                          window=OFF.WINDOW, iterations=OFF.EMB_ITERS, embed_size=OFF.EMB_DIM,
                          seed=sseed("official", DEV_GRAPH, tag, 0))
    taken = []
    for step in range(5):
        frontier = sorted(env.possible_actions)
        if not frontier:
            break
        a, _ = OFF.greedy_action(acmodel, s, embs, frontier, 0)
        env.step(a)
        taken.append(int(a))
        s, _ = OFF.observed_state(env, g)
        embs = OFF.get_embeds(env.sub, num_walks=OFF.NUM_WALKS, walk_length=OFF.WALK_LEN,
                              window=OFF.WINDOW, iterations=OFF.EMB_ITERS, embed_size=OFF.EMB_DIM,
                              seed=sseed("official", DEV_GRAPH, tag, step + 1))
    chosen = SIMP.greedy_seeds(env, DEV_GRAPH, tag)
    return {"actions": taken, "chosen": chosen, "discovered": len(env.active | env.possible_actions),
            "frontier": len(env.possible_actions)}


def paired(a: dict, b: dict, keys) -> dict:
    from scipy import stats
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    mean = float(d.mean())
    if len(d) < 2:
        return {"n": len(d), "mean": mean, "lo": None, "hi": None, "p": None}
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(d), "mean": mean, "lo": mean - half, "hi": mean + half,
            "p": float(stats.ttest_1samp(d, 0.0).pvalue),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    plan = json.loads(OFF.PLAN.read_text(encoding="utf-8"))
    graphs = {n: pickle.loads((SIMP.DATA / f"{n}.pkl").read_bytes())
              for n in list(OFF.TRAIN_GRAPHS) + list(OFF.VAL_GRAPHS) + [DEV_GRAPH]}
    dev_g = graphs[DEV_GRAPH]
    obs = dev_observations(dev_g)
    evals = {i: LiveEdgeObjective(dev_g, samples=EVAL_SAMPLES,
                                  rng=np.random.default_rng(sseed(DEV_GRAPH, i, "deveval")),
                                  p=PROP_PROBAB) for i in range(OBS)}

    print("=" * 104)
    print("  OFFICIAL-ARCHITECTURE LOW-BUDGET ADAPTED VERSION -- agreed tables")
    print("=" * 104)
    print(f"    dev graph {DEV_GRAPH} (n={dev_g.number_of_nodes()})   {OBS} fixed paired observations")
    print(f"    reward repeats executed: {OFF.TIMES_MEAN_ENV} (the published helper would run 4)")

    # ---------------------------------------------------------------- table 2 arms
    arms = {}
    for kind in ("random", "degree_max", "degree_min"):
        arms[kind] = {}
        for i in range(OBS):
            out = heuristic_rollout(dev_g, kind, obs[i], f"dev{i}")
            out["spread"] = evals[i].value(out["chosen"])
            out["inference_seconds"] = 0.0
            arms[kind][i] = out
    for seed in OFF.TRAIN_SEEDS:
        blob = torch.load(CKPT / f"official_seed{seed}.pt", weights_only=False)
        name = f"adapted_s{seed}"
        arms[name] = {}
        for ck in ("epoch0", "selected", "final"):
            ep = 0 if ck == "epoch0" else (blob["selected"] if ck == "selected" else max(blob["snapshots"]))
            acmodel = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM,
                                     action_dim=OFF.EMB_DIM,
                                     replayBuff=PriortizedReplay(OFF.BUFF, 10, beta=0.6),
                                     lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                                     gcn_num_layers=2, num_pooling=1)
            OFF.restore(acmodel, blob["snapshots"][ep])
            acmodel.actor_critic.eval()
            for i in range(OBS):
                t0 = time.perf_counter()
                out = official_rollout(dev_g, acmodel, obs[i], f"dev{i}")
                out["inference_seconds"] = time.perf_counter() - t0
                out["spread"] = evals[i].value(out["chosen"])
                arms.setdefault(f"{name}_{ck}", {})[i] = out
        print(f"    seed {seed}: selected ep{blob['selected']}  "
              f"val curve {[round(c['val_mean_spread'], 2) for c in blob['val_curve']]}", flush=True)

    non_check = [k for k in arms if k in ("random", "degree_max", "degree_min")]
    table2 = [k for k in arms if k not in non_check and k.endswith("selected")]
    keys = list(range(OBS))
    base = {i: arms["random"][i]["spread"] for i in keys}
    dmin = {i: arms["degree_min"][i]["spread"] for i in keys}

    print("\n" + "=" * 104)
    print("  TABLE 2 -- DEVELOPMENT EVALUATION (all five model seeds)")
    print("=" * 104)
    print(f"    {'arm':<22}{'spread':>9}{'adapted-random':>26}{'adapted-degree_min':>26}"
          f"{'disc':>7}{'infer s':>9}")
    rows2 = {}
    for name in ["random", "degree_max", "degree_min"] + sorted(table2):
        sp = {i: arms[name][i]["spread"] for i in keys}
        r1, r2 = paired(sp, base, keys), paired(sp, dmin, keys)
        disc = statistics.mean(arms[name][i]["discovered"] for i in keys)
        inf = statistics.mean(arms[name][i]["inference_seconds"] for i in keys)
        rows2[name] = {"spread_mean": statistics.mean(sp.values()), "vs_random": r1,
                       "vs_degree_min": r2, "discovered_mean": disc,
                       "inference_seconds_mean": inf, "per_observation": sp}
        f1 = f"{r1['mean']:+.2f} [{r1['lo']:+.2f},{r1['hi']:+.2f}]" if r1["lo"] is not None else "-"
        f2 = f"{r2['mean']:+.2f} [{r2['lo']:+.2f},{r2['hi']:+.2f}]" if r2["lo"] is not None else "-"
        print(f"    {name:<22}{statistics.mean(sp.values()):>9.2f}{f1:>26}{f2:>26}"
              f"{disc:>7.0f}{inf:>9.4f}")

    # gate: the heuristics must reproduce the earlier round exactly
    try:
        prev = json.loads(PREV.read_text(encoding="utf-8"))
        ok = True
        for kind in ("random", "degree_max", "degree_min"):
            before = statistics.mean(prev["arms"][kind][str(i)]["influence"] for i in keys)
            after = rows2[kind]["spread_mean"]
            same = abs(before - after) < 1e-6
            ok &= same
            print(f"    [{'PASS' if same else 'FAIL'}] {kind} reproduces the earlier round "
                  f"({before:.4f} vs {after:.4f})")
    except Exception as exc:                                    # noqa: BLE001
        ok = None
        print(f"    heuristic reconciliation unavailable: {exc}")

    artifact = {"script": Path(__file__).name,
                "arm_name": "official-architecture low-budget adapted version",
                "primary_comparisons": ["adapted - random", "adapted - degree_min"],
                "degree_min_note": ("chosen as the strong heuristic reference from an earlier round's "
                                    "results; a result-informed choice"),
                "reward_repeats_executed": OFF.TIMES_MEAN_ENV,
                "reward_repeats_note": ("the published parallel_influence runs 4 for times=5 because its "
                                        "index arithmetic drops a worker; executing 5 is an explicit "
                                        "behavioural correction, not a compatibility patch"),
                "reward_implementation_note": (
                    "influence_equivalent() is an alternative implementation of the same target, not a "
                    "proven equivalent; the 1.85% cross-check supports compatibility of the propagation "
                    "estimate on a fixed seed set only, and does not establish that the full stochastic "
                    "reward pipeline (greedy on 100 samples then an independent evaluation) agrees, "
                    "because finite-sample error can change the greedy selection"),
                "opt_note": ("r / opt is normalisation against a full-knowledge greedy reference; opt is "
                             "not a proven global optimum"),
                "dev_graph_note": ("a repeatedly used development graph, not new independent evidence; "
                                   "no cross-graph generalisation is claimed"),
                "heuristics_reconcile": bool(ok),
                "rows": rows2}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
