"""How much training does the plain DQN need before its result means anything?

A DQN that has not converged cannot support the conclusion "the DQN gains nothing" -- that would be a
statement about the training budget, not about multi-step decision-making.  This probe trains the
same configuration at several episode counts and reports the validation curve, so the main run can use
a budget where the curve has flattened rather than one chosen to make the DQN look bad.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.baselines import Ledger
from ccim.dqn import train_dqn
from ccim.model import DEFAULT_K, DEFAULT_T, load_graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="football")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", nargs="+", type=int, default=[500, 1500, 4000])
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()

    g = load_graph(args.graph)
    n = g.number_of_nodes()
    print(f"{args.graph} n={n}  K={DEFAULT_K}  T={DEFAULT_T}", flush=True)
    for episodes in args.episodes:
        for seed in args.seeds:
            ledger = Ledger()
            t0 = time.time()
            out = train_dqn(g, episodes=episodes, seed=seed,
                            validate_every=args.validate_every, ledger=ledger)
            curve = out["validation"]
            tail = [c["normalized"] for c in curve[-3:]] if curve else []
            print(f"  ep={episodes:<6} seed={seed}  best {out['best']['normalized']:.4f} "
                  f"(ep {out['best']['episode']})  final {out['final_sigma'] / n:.4f}  "
                  f"last3 {[round(x, 4) for x in tail]}  "
                  f"cost {ledger.decision_evaluations}  {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
