"""Does multi-step decision-making have value under the true CCIM reward?

The question, and the three-way reading fixed in advance
-------------------------------------------------------
Two public graphs from Chen et al. (IJCAI 2023), the deterministic setting they use for their basic
comparison (``p0 = 1, p1 = 0, K = 4``, seed budget ``T = 8``), and three decision procedures:

1. ``degree`` and true-reward ``greedy``      -- cheap, non-learning, no lookahead
2. ``local_search``                            -- non-learning, budget-controlled lookahead
3. ``dqn``                                     -- the original double-DQN, terminal handling and
                                                  legal action masking fixed, trained on the true
                                                  diffusion reward

Reading, decided BEFORE seeing any result:

* **both the search and the DQN gain** -> combinatorial decision has room; only then is it worth
  asking whether a learned policy can cut cost;
* **the search gains, the DQN does not** -> fix the RL representation and training first, and do not
  paper over it with a surrogate;
* **neither gains** -> do not scale up; these two settings give no basis for further investment.

Cost is reported in one unit for every method: **the number of ``sigma`` evaluations** (cascade
queries).  Degree spends zero, greedy spends ``sum_{t<T}(n-t)``, the search spends its cap, and the
DQN spends one per step of every training episode plus one per step of every validation rollout.
Wall time is not the metric; oracle calls are.

Every training seed is reported.  The paper's protocol selects the best of 15 runs by validation and
reports that number; reporting only the best would make the DQN look better than it is, so this script
prints all seeds and the mean, and shows what best-of-N would have claimed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.baselines import Ledger, degree_baseline, greedy_baseline, random_plus_baseline
from ccim.dqn import train_dqn
from ccim.model import DEFAULT_K, DEFAULT_T, PAPER_TABLE1, load_graph
from ccim.search import local_search

GRAPHS = ("football", "polbooks")
TRAIN_SEEDS = (0, 1, 2, 3, 4)
EPISODES = 4000
VALIDATE_EVERY = 100
#: Chosen by the pre-declared probe in ``probe_hyperparameters.py``: best MEAN over training seeds on
#: Football, then frozen for both graphs.  No tuning after the main results are seen.
DQN_CONFIG = {"name": "fast_eps", "lr": 1e-3, "eps_decay": 200}
SEARCH_BUDGETS = (1_000, 5_000, 20_000, 40_000)


def dqn_budget(episodes: int, T: int, validate_every: int) -> int:
    """sigma evaluations a DQN run spends: one per training step plus one per validation step."""
    return episodes * T + (episodes // validate_every) * T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--validate-every", type=int, default=VALIDATE_EVERY)
    parser.add_argument("--search-budgets", nargs="+", type=int, default=list(SEARCH_BUDGETS))
    parser.add_argument("--defect-seeds", nargs="+", type=int, default=[0],
                        help="training seeds used for the two defect toggles")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gate1_multistep_value.json")
    args = parser.parse_args()

    per_run = dqn_budget(args.episodes, DEFAULT_T, args.validate_every)
    matched_budget = per_run
    print(f"DQN budget per run: {args.episodes} episodes x {DEFAULT_T} steps + "
          f"{args.episodes // args.validate_every} validations x {DEFAULT_T} = {per_run} "
          f"sigma-evaluations (excluding the per-episode reset query)", flush=True)
    budgets = sorted(set([*args.search_budgets, matched_budget]))
    print(f"search budget ladder: {budgets}", flush=True)

    artifact = {
        "script": Path(__file__).name,
        "status": "exploratory probe of whether multi-step decision has value; not a benchmark "
                  "reproduction and not a claim about any published method",
        "setting": {"p0": 1, "p1": 0, "propagation": "deterministic threshold-K bootstrap",
                    "K": DEFAULT_K, "T": DEFAULT_T,
                    "source": "Chen et al., IJCAI 2023, basic deterministic comparison"},
        "cost_unit": "sigma evaluations (cascade queries); degree spends 0 to decide",
        "decision_rule": {
            "search_gains_and_dqn_gains": "combinatorial decision has room; next ask whether a "
                                          "learned policy can cut cost",
            "search_gains_dqn_does_not": "fix RL representation and training first; do not add a "
                                         "surrogate to mask the cause",
            "neither_gains": "do not scale up; these settings give no basis for further investment",
        },
        "dqn_protocol": {"episodes": args.episodes, "validate_every": args.validate_every,
                         "train_seeds": args.seeds, "budget_per_run": per_run,
                         "selection": "best validation checkpoint, as in the paper's protocol",
                         "reporting": "ALL seeds reported, plus mean/median, plus what best-of-N "
                                      "would have claimed"},
        "graphs": {},
    }

    for name in args.graphs:
        g = load_graph(name)
        n = g.number_of_nodes()
        print(f"\n===== {name}  n={n}  K={DEFAULT_K}  T={DEFAULT_T} =====", flush=True)
        entry = {"n": n, "m": g.number_of_edges(), "paper_table1": PAPER_TABLE1[name]}

        # ---- cheap baselines ----
        t0 = time.time()
        deg = degree_baseline(g)
        grd = greedy_baseline(g)
        rnd = random_plus_baseline(g, runs=50, seed=0)
        entry["baselines"] = {"degree": deg.as_dict(), "greedy": grd.as_dict(),
                              "random_plus": rnd.as_dict()}
        print(f"  degree      sigma {deg.sigma:>4} ({deg.normalized:.4f})  "
              f"cost {deg.cost['decision_evaluations']}", flush=True)
        print(f"  greedy      sigma {grd.sigma:>4} ({grd.normalized:.4f})  "
              f"cost {grd.cost['decision_evaluations']}   "
              f"paper Greedy {PAPER_TABLE1[name]['Greedy']}", flush=True)
        print(f"  random+     best {rnd.sigma:>4} ({rnd.normalized:.4f})  "
              f"mean {rnd.extra['mean_sigma']:.1f} ({rnd.extra['mean_normalized']:.4f})  "
              f"cost {rnd.cost['decision_evaluations']}", flush=True)

        # ---- budget-controlled non-learning search ----
        ladder = []
        for b in budgets:
            res = local_search(g, budget=b, seed=0)
            ladder.append({"budget": b, "sigma": res.sigma, "normalized": res.normalized,
                           "used": res.cost["decision_evaluations"],
                           "exhausted": res.extra["budget_exhausted"], "seeds": res.seeds})
            print(f"  search b={b:>6}  sigma {res.sigma:>4} ({res.normalized:.4f})  "
                  f"used {res.cost['decision_evaluations']}", flush=True)
        entry["search_ladder"] = ladder
        matched = next(r for r in ladder if r["budget"] == matched_budget)
        entry["search_at_matched_budget"] = matched

        # ---- the DQN, every training seed ----
        runs = []
        for seed in args.seeds:
            ledger = Ledger()
            t1 = time.time()
            out = train_dqn(g, episodes=args.episodes, seed=seed,
                            validate_every=args.validate_every, ledger=ledger, **{
                                k: v for k, v in DQN_CONFIG.items() if k != "name"})
            runs.append({
                "train_seed": seed,
                "best_validation_sigma": int(round(out["best"]["normalized"] * n)),
                "best_validation_normalized": out["best"]["normalized"],
                "best_epoch": out["best"]["episode"],
                "best_seeds": out["best"]["seeds"],
                "final_epoch_sigma": out["final_sigma"],
                "final_epoch_normalized": out["final_sigma"] / n,
                "final_epoch_seeds": out["final_seeds"],
                "cost": ledger.as_dict(),
                "validation_curve": out["validation"],
                "seconds": time.time() - t1,
            })
            print(f"  dqn seed {seed}: best-val {out['best']['normalized']:.4f} "
                  f"(ep {out['best']['episode']})   final-epoch {out['final_sigma'] / n:.4f}   "
                  f"issued {ledger.decision_evaluations}   {time.time() - t1:.0f}s", flush=True)
        values = [r["best_validation_normalized"] for r in runs]
        finals = [r["final_epoch_normalized"] for r in runs]
        entry["dqn_runs"] = runs
        entry["dqn_summary"] = {
            "n_seeds": len(runs),
            "best_validation_all_seeds": values,
            "best_validation_mean": statistics.fmean(values),
            "best_validation_median": statistics.median(values),
            "best_validation_min": min(values),
            "best_validation_max": max(values),
            "best_of_n_as_the_paper_reports": max(values),
            "final_epoch_mean": statistics.fmean(finals),
        }
        print(f"  DQN over {len(runs)} seeds: mean {statistics.fmean(values):.4f}  "
              f"median {statistics.median(values):.4f}  min {min(values):.4f}  max {max(values):.4f}"
              f"   (best-of-N would report {max(values):.4f})", flush=True)

        # ---- how much do the two fixes matter?  measured, not asserted ----
        defects = {}
        for label, kwargs in (("no_action_masking", {"mask_actions": False}),
                              ("bootstrap_at_terminal", {"bootstrap_at_terminal": True})):
            vals = []
            for seed in args.defect_seeds:
                ledger = Ledger()
                try:
                    out = train_dqn(g, episodes=args.episodes, seed=seed,
                                    validate_every=args.validate_every, ledger=ledger, **kwargs)
                    vals.append(out["best"]["normalized"])
                    note = "ok"
                except AssertionError as exc:
                    vals.append(None)
                    note = f"rollout violated the seed-set contract: {exc}"
                print(f"  defect {label:<22} seed={seed} -> "
                      f"{'n/a' if vals[-1] is None else f'{vals[-1]:.4f}'}   {note}", flush=True)
            defects[label] = {"per_seed": vals,
                              "note": "best validation normalized influence with the fix reverted"}
        entry["defect_toggles"] = defects
        print(f"  [{time.time() - t0:.0f}s for {name}]", flush=True)
        artifact["graphs"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ verdict table
    print("\n" + "=" * 100)
    print("  SUMMARY -- normalized influence (sigma / |V|)")
    print("=" * 100)
    print(f"  {'graph':<10}{'degree':>9}{'greedy':>9}{'search':>9}{'DQN mean':>10}{'DQN best':>10}"
          f"{'random+':>10}   cost: search vs DQN")
    for name, entry in artifact["graphs"].items():
        s = entry["search_at_matched_budget"]
        d = entry["dqn_summary"]
        dqn_cost = entry["dqn_runs"][0]["cost"]["decision_evaluations"]
        print(f"  {name:<10}{entry['baselines']['degree']['normalized']:>9.4f}"
              f"{entry['baselines']['greedy']['normalized']:>9.4f}"
              f"{s['normalized']:>9.4f}{d['best_validation_mean']:>10.4f}"
              f"{d['best_of_n_as_the_paper_reports']:>10.4f}"
              f"{entry['baselines']['random_plus']['mean_normalized']:>10.4f}"
              f"   {s['used']} vs {dqn_cost}")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
