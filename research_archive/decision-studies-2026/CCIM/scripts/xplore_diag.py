"""Did this implementation gain anything from training?  A before/after diagnostic, no new training.

Why the models had to be rebuilt
---------------------------------
The previous round never wrote its weights to disk -- ``results/xplore_geodqn.json`` holds numbers only --
so the "earliest saved checkpoint" and the "selected checkpoint" did not exist as files.  The models are
therefore recovered by re-executing the **identical** deterministic training (same seeds, same episodes,
same hyperparameters, same RNG flow) with snapshotting enabled.

That is only legitimate if the rebuild is the *same* model, so the script gates on it: **every** validation
spread recorded in the previous round must be reproduced exactly.  A single mismatch voids the diagnostic.
``torch.manual_seed(seed)`` in ``GeoDQN.__init__`` makes epoch 0 exactly reproducible too, so the untrained
network is the original initialisation and not a fresh random draw.

What is measured
----------------
Four checkpoints per model seed -- **epoch 0** (untrained), **ep500** (the earliest saved), **ep1000**,
**ep2000** (the last) -- evaluated on fixed observations of the **training graphs and the validation graphs
separately**, never pooled.  The reward is the final spread on the complete graph, computed with its own
independent random stream; the training reward is never substituted for it.

For every decision the survey action sequence is recorded, so the script can report how many of the five
decisions actually *changed* after training.  A policy that never changes an action cannot have gained
anything from training, whatever its score does.

This is an explanatory diagnostic.  It does not select a new checkpoint, and its numbers are not fed back
into the previous round's results.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import xplore_learn as L  # noqa: E402
from ccim.xplore import DiscoveryEnv  # noqa: E402
from ccim.xplore.icm import INFL_BUDGET, PROP_PROBAB, LiveEdgeObjective  # noqa: E402
from ccim.xplore.learner import GeoDQN, encode_observation  # noqa: E402

PLAN = ROOT / "results" / "xplore_geodqn_plan.json"
PREVIOUS = ROOT / "results" / "xplore_geodqn.json"
SNAPSHOT_DIR = ROOT / "results" / "geodqn_snapshots"
OUTPUT = ROOT / "results" / "xplore_geodqn_diagnostic.json"

CHECKPOINTS = (0, 500, 1000, 2000)
OBSERVATIONS_PER_GRAPH = 20
EVAL_SAMPLES = 1000
T_MAX = 5


def sseed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63)


def rollout_actions(g, agent, seeds, graph, tag) -> dict:
    """Greedy rollout: record the survey sequence and the greedily chosen seed set."""
    env = DiscoveryEnv(g, seeds, max_T=T_MAX, p=PROP_PROBAB, infl_budget=INFL_BUDGET)
    actions, t = [], 0
    while t < T_MAX and not env.done():
        obs = encode_observation(env, t, T_MAX)
        if not obs["frontier"]:
            break
        a = agent.act(obs, explore=False)
        actions.append(int(a))
        env.step(a)
        t += 1
    sel = LiveEdgeObjective(env.graph, samples=L.SELECT_SAMPLES,
                            rng=np.random.default_rng(sseed("diagselect", graph, tag)),
                            p=PROP_PROBAB)
    chosen, _ = sel.greedy(INFL_BUDGET)
    return {"actions": actions, "chosen": list(chosen), "discovered": env.discovered_nodes,
            "surveys": len(env.queried)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(L.TRAIN_SEEDS))
    parser.add_argument("--skip-training", action="store_true",
                        help="reuse snapshots already written to results/geodqn_snapshots")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    graph_names = list(L.TRAIN_GRAPHS) + list(L.VAL_GRAPHS)
    graphs = {n: pickle.loads((L.DATA / f"{n}.pkl").read_bytes()) for n in graph_names}

    print("=" * 104)
    print("  TRAINING-CONTRIBUTION DIAGNOSTIC -- epoch 0 / earliest / mid / last, no new training")
    print("=" * 104)
    print(f"    train graphs     : {list(L.TRAIN_GRAPHS)}")
    print(f"    validation graphs: {list(L.VAL_GRAPHS)}   (reported separately, never pooled)")
    print(f"    checkpoints      : {list(CHECKPOINTS)}   observations per graph: "
          f"{OBSERVATIONS_PER_GRAPH}")

    # ---------------------------------------------------------------- rebuild + gate
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots, gate = {}, {}
    for seed in args.seeds:
        path = SNAPSHOT_DIR / f"seed{seed}.pt"
        if args.skip_training and path.exists():
            snapshots[seed] = torch.load(path, weights_only=False)
            gate[seed] = {"rebuilt": False, "source": str(path.name)}
            print(f"\n    seed {seed}: loaded snapshots from {path.name}")
            continue
        print(f"\n    seed {seed}: re-executing the identical training to recover the weights")
        snap = {}
        t0 = time.perf_counter()
        res = L.train_one(seed, graphs, plan["validation_observations"], int(plan["episodes"]),
                          lambda s: None, snapshot_at=CHECKPOINTS, snapshot_out=snap)
        torch.save(snap, path)
        recorded = [c["val_mean_spread"] for c in
                    previous["training"]["per_seed"][str(seed)]["val_curve"]]
        rebuilt = [c["val_mean_spread"] for c in res["val_curve"]]
        exact = recorded == rebuilt
        gate[seed] = {"rebuilt": True, "recorded": recorded, "rebuilt_curve": rebuilt,
                      "exact_match": exact, "selected_episode": res["selected_episode"],
                      "seconds": time.perf_counter() - t0, "snapshots": sorted(snap)}
        snapshots[seed] = snap
        print(f"      val curve recorded {recorded}")
        print(f"      val curve rebuilt  {rebuilt}")
        print(f"      [{'PASS' if exact else 'FAIL'}] exact reproduction of the previous round")

    all_exact = all(v.get("exact_match", True) for v in gate.values())
    print(f"\n  REBUILD GATE: {'ALL PASS' if all_exact else 'FAILED'}")
    if not all_exact:
        print("  a mismatch means these are not the previous round's models; diagnostic aborted")
        return 1

    # ---------------------------------------------------------------- fixed observations
    obs = {}
    for name in graph_names:
        n = graphs[name].number_of_nodes()
        obs[name] = [sorted(int(x) for x in np.random.default_rng(sseed("diag", name, i, "init"))
                            .choice(n, L.EXTRA_SEEDS, replace=False))
                     for i in range(OBSERVATIONS_PER_GRAPH)]
    evals = {(name, i): LiveEdgeObjective(
        graphs[name], samples=EVAL_SAMPLES,
        rng=np.random.default_rng(sseed("diag", name, i, "eval")), p=PROP_PROBAB)
        for name in graph_names for i in range(OBSERVATIONS_PER_GRAPH)}

    # ---------------------------------------------------------------- evaluate
    results = {}
    for seed in args.seeds:
        results[seed] = {}
        for ck in CHECKPOINTS:
            agent = GeoDQN(seed=seed, **{k: v for k, v in L.HYPER.items()
                                         if k != "target_sync_every"})
            agent.q.load_state_dict(snapshots[seed][ck])
            agent.q.eval()
            per_graph = {}
            for name in graph_names:
                spreads, actions, disc = [], {}, []
                for i in range(OBSERVATIONS_PER_GRAPH):
                    out = rollout_actions(graphs[name], agent, obs[name][i], name, f"{seed}:{ck}:{i}")
                    spreads.append(evals[(name, i)].value(out["chosen"]))
                    actions[str(i)] = out["actions"]
                    disc.append(out["discovered"])
                per_graph[name] = {"spread": spreads, "actions": actions,
                                   "discovered": disc,
                                   "split": "train" if name in L.TRAIN_GRAPHS else "validation"}
            results[seed][ck] = per_graph
            tr = statistics.mean(statistics.mean(per_graph[n]["spread"]) for n in L.TRAIN_GRAPHS)
            va = statistics.mean(statistics.mean(per_graph[n]["spread"]) for n in L.VAL_GRAPHS)
            print(f"    seed {seed} ep{ck:<5}: train spread {tr:7.2f}   val spread {va:7.2f}",
                  flush=True)

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 104)
    print("  (1) FINAL SPREAD BY CHECKPOINT -- mean over 20 paired observations per graph")
    print("=" * 104)
    for split, names in (("TRAIN graphs", L.TRAIN_GRAPHS), ("VALIDATION graphs", L.VAL_GRAPHS)):
        print(f"\n    {split}")
        print(f"      {'seed':>5}" + "".join(f"{'ep' + str(c):>12}" for c in CHECKPOINTS)
              + f"{'ep2000-ep0':>13}{'ep2000-ep500':>15}")
        for seed in args.seeds:
            vals = [statistics.mean(statistics.mean(results[seed][c][n]["spread"]) for n in names)
                    for c in CHECKPOINTS]
            print(f"      {seed:>5}" + "".join(f"{v:>12.2f}" for v in vals)
                  + f"{vals[-1] - vals[0]:>+13.2f}{vals[-1] - vals[1]:>+15.2f}")
        mean_by_ck = [statistics.mean(
            statistics.mean(statistics.mean(results[s][c][n]["spread"]) for n in names)
            for s in args.seeds) for c in CHECKPOINTS]
        print(f"      {'mean':>5}" + "".join(f"{v:>12.2f}" for v in mean_by_ck)
              + f"{mean_by_ck[-1] - mean_by_ck[0]:>+13.2f}"
              f"{mean_by_ck[-1] - mean_by_ck[1]:>+15.2f}")

    print("\n" + "=" * 104)
    print("  (2) DID THE DECISIONS ACTUALLY CHANGE?  survey actions vs the untrained network")
    print("=" * 104)
    print(f"    {'seed':>5}{'checkpoint':>12}{'decisions changed':>19}{'of total':>10}"
          f"{'train spread':>14}{'val spread':>13}")
    changed = {}
    for seed in args.seeds:
        base = results[seed][0]
        for ck in CHECKPOINTS:
            same = total = 0
            for name in graph_names:
                for i in range(OBSERVATIONS_PER_GRAPH):
                    a0, a1 = base[name]["actions"][str(i)], results[seed][ck][name]["actions"][str(i)]
                    for x, y in zip(a0, a1):
                        total += 1
                        same += int(x == y)
            tr = statistics.mean(statistics.mean(results[seed][ck][n]["spread"])
                                 for n in L.TRAIN_GRAPHS)
            va = statistics.mean(statistics.mean(results[seed][ck][n]["spread"])
                                 for n in L.VAL_GRAPHS)
            changed[(seed, ck)] = {"identical": same, "total": total}
            print(f"    {seed:>5}{'ep' + str(ck):>12}{total - same:>19}{total:>10}"
                  f"{tr:>14.2f}{va:>13.2f}")
    pool_id = sum(v["identical"] for (s, c), v in changed.items() if c > 0)
    pool_tot = sum(v["total"] for (s, c), v in changed.items() if c > 0)
    print(f"\n    across all five seeds and all trained checkpoints: "
          f"{pool_tot - pool_id:,}/{pool_tot:,} decisions differ from the untrained network "
          f"({100 * (pool_tot - pool_id) / pool_tot:.1f}%)")

    print("\n" + "=" * 104)
    print("  (3) RECOVERY GATE AND RAW CURVE")
    print("=" * 104)
    for seed in args.seeds:
        g = gate[seed]
        print(f"    seed {seed}: rebuilt={g['rebuilt']}  exact={g.get('exact_match', 'n/a')}  "
              f"selected ep{g.get('selected_episode', 'n/a')}")
        if g["rebuilt"]:
            print(f"      previous round's curve {g['recorded']}")
            print(f"      this rebuild's curve  {g['rebuilt_curve']}")

    artifact = {"script": Path(__file__).name, "checkpoints": list(CHECKPOINTS),
                "observations_per_graph": OBSERVATIONS_PER_GRAPH, "eval_samples": EVAL_SAMPLES,
                "rebuild_gate": {str(k): v for k, v in gate.items()}, "all_exact": all_exact,
                "train_graphs": list(L.TRAIN_GRAPHS), "validation_graphs": list(L.VAL_GRAPHS),
                "note": ("explanatory only; no checkpoint is re-selected and no number here is fed back "
                         "into the previous round's reported results"),
                "results": {str(s): {str(c): results[s][c] for c in CHECKPOINTS}
                            for s in args.seeds},
                "decisions_changed": {f"{s}:{c}": v for (s, c), v in changed.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    print(f"\n    snapshots: {SNAPSHOT_DIR}")
    print(f"    artifact : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
