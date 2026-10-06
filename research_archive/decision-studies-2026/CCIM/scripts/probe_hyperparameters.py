"""Pick ONE DQN configuration before the main run, by mean validation over training seeds.

A DQN that fails because of an unlucky learning rate would make the main experiment a statement about
my defaults rather than about multi-step decision-making.  So the configuration is chosen here, on
Football only, using the **mean over two training seeds** -- not the best seed -- and then frozen for
both graphs.  No further tuning happens after the main results are seen.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.baselines import Ledger
from ccim.dqn import train_dqn
from ccim.model import load_graph

CANDIDATES = [
    {"name": "default", "lr": 1e-3, "eps_decay": 500},
    {"name": "slow_eps", "lr": 1e-3, "eps_decay": 1500},
    {"name": "small_lr", "lr": 3e-4, "eps_decay": 1000},
    {"name": "fast_eps", "lr": 1e-3, "eps_decay": 200},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="football")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()

    g = load_graph(args.graph)
    n = g.number_of_nodes()
    print(f"hyperparameter probe on {args.graph} n={n}, {args.episodes} episodes, "
          f"seeds {args.seeds}, chosen by MEAN over seeds", flush=True)
    results = []
    for cand in CANDIDATES:
        vals, finals = [], []
        for seed in args.seeds:
            ledger = Ledger()
            t0 = time.time()
            out = train_dqn(g, episodes=args.episodes, seed=seed, validate_every=args.validate_every,
                            lr=cand["lr"], eps_decay=cand["eps_decay"], ledger=ledger)
            vals.append(out["best"]["normalized"])
            finals.append(out["final_sigma"] / n)
            print(f"    {cand['name']:<10} seed={seed} best {out['best']['normalized']:.4f} "
                  f"final {out['final_sigma'] / n:.4f}  cost {ledger.decision_evaluations} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results.append({"config": cand, "mean_best_validation": statistics.fmean(vals),
                        "per_seed_best": vals, "mean_final": statistics.fmean(finals)})
    results.sort(key=lambda r: -r["mean_best_validation"])
    print("\nranking by MEAN best-validation over seeds:", flush=True)
    for r in results:
        print(f"  {r['config']['name']:<10} mean {r['mean_best_validation']:.4f}  "
              f"per-seed {[round(v, 4) for v in r['per_seed_best']]}  "
              f"mean-final {r['mean_final']:.4f}", flush=True)
    print(f"\nSELECTED: {results[0]['config']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
