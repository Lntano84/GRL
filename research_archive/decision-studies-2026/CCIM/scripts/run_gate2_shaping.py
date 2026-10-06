"""Gate 2: does the paper's reward reshaping help, and is it worth its label cost?

Both arms run the SAME pipeline (episode buffering, insertion at episode end, ``T`` gradient updates
per episode, same network, optimiser and exploration schedule).  ``omega = 0`` is re-run here rather
than reused, because the insertion timing itself changed.

Two cost views, both reported
-----------------------------
* **same query cap** (the primary comparison): each arm runs until it has spent 36,369 ``sigma``
  queries, counting training steps, the leave-one-out sets the reshaping needs, and every validation
  rollout.  ``omega = 1`` pays 8 extra queries per episode, so it completes fewer episodes.
* **same episode count** (auxiliary): the policy each arm had at the common checkpoint, so the effect
  of the reshaping on learning can be read without the label cost folded in.

The reported policy is the one at the pre-determined budget, **not** the best validation checkpoint:
selecting on training-time spread would make the reshaping look better than it is.

Three diagnostics are recorded alongside, because a larger reward is not the same as better credit
assignment: the step-wise raw reward distribution, whether the leave-one-out contributions actually
differ between seeds within an episode, and whether early low-reward actions receive larger ones.
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
from ccim.model import DEFAULT_K, DEFAULT_T, load_graph
from ccim.search import local_search
from ccim.shaping import Diagnostics, train_shaped

GRAPHS = ("football", "polbooks")
OMEGAS = (0.0, 1.0)
TRAIN_SEEDS = (0, 1, 2, 3, 4)
SEARCH_SEEDS = (0, 1, 2, 3, 4)
QUERY_CAP = 36_369
COMMON_EPISODES = 2000
SEARCH_POINTS = (5_000, QUERY_CAP)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS))
    parser.add_argument("--omega", nargs="+", type=float, default=list(OMEGAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--query-cap", type=int, default=QUERY_CAP)
    parser.add_argument("--common-episodes", type=int, default=COMMON_EPISODES)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gate2_shaping.json")
    args = parser.parse_args()

    artifact = {
        "script": Path(__file__).name,
        "setting": {"p0": 1, "p1": 0, "K": DEFAULT_K, "T": DEFAULT_T,
                    "graphs": list(args.graphs)},
        "shaping": {"r_t": "sigma(S_t) - sigma(S_{t-1})",
                    "c_t": "sigma(S_T) - sigma(S_T \\ {a_t})",
                    "r'_t": "r_t + omega * c_t",
                    "source": "Chen et al. IJCAI 2023, Section 5.5, Eqs. (6)-(7)",
                    "omega_control": 0.0, "omega_treatment": 1.0,
                    "omega_note": "1.0 is THIS round's pre-fixed setting, not a claimed paper "
                                  "default; no weight is swept",
                    "no_scaling": "no rescaling is applied to either arm; r_t and c_t are both node "
                                  "counts, so omega=1 makes the shaped reward numerically larger -- "
                                  "mean_abs_reward is recorded for both arms so this is visible",
                    "fairness_caveat": "the shaping term does not guarantee the original objective's "
                                       "optimal policy is preserved, so final quality is judged ONLY "
                                       "by raw sigma(S), never by shaped return"},
        "pipeline": {"buffering": "whole episode, inserted at episode end",
                     "updates_per_episode": DEFAULT_T,
                     "insertion": "each transition inserted once, with its own arm's reward",
                     "checkpoint_reported": "the policy at the pre-determined budget, NOT best "
                                            "validation",
                     "frozen": ["graph", "diffusion params", "budget", "network", "optimiser",
                                "exploration schedule", "training seeds"]},
        "cost_views": {
            "primary": f"same total query cap ({args.query_cap}); counts training, leave-one-out "
                       f"and validation queries",
            "auxiliary": f"same episode count ({args.common_episodes}), from the common checkpoint",
        },
        "graphs": {},
    }

    for name in args.graphs:
        g = load_graph(name)
        n = g.number_of_nodes()
        print(f"\n===== {name} n={n} K={DEFAULT_K} T={DEFAULT_T}  cap={args.query_cap} =====",
              flush=True)
        entry = {"n": n, "m": g.number_of_edges()}

        deg, grd, rnd = degree_baseline(g), greedy_baseline(g), random_plus_baseline(g, runs=50)
        entry["baselines"] = {"degree": deg.as_dict(), "greedy": grd.as_dict(),
                              "random_plus": rnd.as_dict()}
        print(f"  degree {deg.normalized:.4f} | greedy {grd.normalized:.4f} | "
              f"random+ mean {rnd.extra['mean_normalized']:.4f}", flush=True)

        # ---- non-learning search, now with all five pre-fixed seeds ----
        search_rows = []
        for budget in SEARCH_POINTS:
            vals = []
            for s in SEARCH_SEEDS:
                res = local_search(g, budget=budget, seed=s)
                vals.append({"search_seed": s, "sigma": res.sigma, "normalized": res.normalized,
                             "used": res.cost["decision_evaluations"], "seeds": res.seeds})
            best = max(v["normalized"] for v in vals)
            search_rows.append({"budget": budget, "per_seed": vals,
                                "mean": statistics.fmean(v["normalized"] for v in vals),
                                "median": statistics.median(v["normalized"] for v in vals),
                                "min": min(v["normalized"] for v in vals), "max": best})
            print(f"  search @{budget}: mean {search_rows[-1]['mean']:.4f} "
                  f"median {search_rows[-1]['median']:.4f} min {search_rows[-1]['min']:.4f} "
                  f"max {best:.4f}   (5 seeds)", flush=True)
        entry["search"] = search_rows

        # ---- the two reward arms ----
        arms = {}
        for omega in args.omega:
            runs = []
            pooled_diag = Diagnostics()
            for seed in args.seeds:
                ledger = Ledger()
                t0 = time.time()
                out = train_shaped(g, omega=omega, query_cap=args.query_cap, seed=seed,
                                   common_episodes=args.common_episodes, ledger=ledger)
                snap = out["common_episode_snapshot"]
                runs.append({
                    "omega": omega, "train_seed": seed,
                    "episodes": out["episodes"],
                    "queries_issued": out["queries_issued"],
                    "queries_distinct": out["queries_distinct"],
                    "cache_hits": out["cache_hits"],
                    "final_normalized": out["final_normalized"],
                    "final_sigma": out["final_sigma"],
                    "final_seeds": out["final_seeds"],
                    "common_episode_normalized": snap["normalized"] if snap else None,
                    "mean_abs_reward": out["mean_abs_reward"],
                    "seconds": time.time() - t0,
                    "diagnostics": out["diagnostics"],
                })
                print(f"  omega={omega} seed={seed}: final {out['final_normalized']:.4f} "
                      f"(common-ep {snap['normalized'] if snap else float('nan'):.4f})  "
                      f"episodes {out['episodes']}  queries {out['queries_issued']}  "
                      f"{time.time() - t0:.0f}s", flush=True)
            finals = [r["final_normalized"] for r in runs]
            commons = [r["common_episode_normalized"] for r in runs
                       if r["common_episode_normalized"] is not None]
            arms[f"omega_{omega}"] = {
                "omega": omega, "runs": runs,
                "final_mean": statistics.fmean(finals),
                "final_median": statistics.median(finals),
                "final_min": min(finals), "final_max": max(finals),
                "final_best_of_n": max(finals),
                "common_episode_mean": statistics.fmean(commons) if commons else None,
                "common_episode_median": statistics.median(commons) if commons else None,
                "mean_abs_reward": statistics.fmean(r["mean_abs_reward"] for r in runs),
                "mean_queries": statistics.fmean(r["queries_issued"] for r in runs),
                "mean_episodes": statistics.fmean(r["episodes"] for r in runs),
                "diagnostics": runs[0]["diagnostics"],
            }
            print(f"  omega={omega}: final mean {arms[f'omega_{omega}']['final_mean']:.4f} "
                  f"median {arms[f'omega_{omega}']['final_median']:.4f} "
                  f"min {arms[f'omega_{omega}']['final_min']:.4f} "
                  f"max {arms[f'omega_{omega}']['final_max']:.4f} | "
                  f"mean |reward| {arms[f'omega_{omega}']['mean_abs_reward']:.3f}", flush=True)
        entry["arms"] = arms
        artifact["graphs"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ per-run table
    print("\n" + "=" * 118)
    print("  PER-RUN TABLE -- normalized sigma(S) of the policy at the pre-determined budget")
    print("=" * 118)
    print(f"  {'graph':<10}{'omega':>6}{'seed':>5}{'episodes':>10}{'queries':>9}"
          f"{'final':>9}{'common-ep':>11}{'mean|r|':>9}{'seconds':>9}")
    for name, entry in artifact["graphs"].items():
        for key in ("omega_0.0", "omega_1.0"):
            for r in entry["arms"][key]["runs"]:
                ce = r["common_episode_normalized"]
                print(f"  {name:<10}{r['omega']:>6.0f}{r['train_seed']:>5}{r['episodes']:>10}"
                      f"{r['queries_issued']:>9}{r['final_normalized']:>9.4f}"
                      f"{(f'{ce:.4f}' if ce is not None else 'n/a'):>11}"
                      f"{r['mean_abs_reward']:>9.3f}{r['seconds']:>9.0f}")
    print()
    print(f"  {'graph':<10}{'view':<16}{'omega=0':>10}{'omega=1':>10}{'difference':>12}"
          f"   (mean over 5 seeds)")
    for name, entry in artifact["graphs"].items():
        a, b = entry["arms"]["omega_0.0"], entry["arms"]["omega_1.0"]
        print(f"  {name:<10}{'same query cap':<16}{a['final_mean']:>10.4f}{b['final_mean']:>10.4f}"
              f"{b['final_mean'] - a['final_mean']:>+12.4f}")
        print(f"  {name:<10}{'same episodes':<16}{a['common_episode_mean']:>10.4f}"
              f"{b['common_episode_mean']:>10.4f}"
              f"{b['common_episode_mean'] - a['common_episode_mean']:>+12.4f}")
        s5 = next(r for r in entry["search"] if r["budget"] == 5_000)
        sc = next(r for r in entry["search"] if r["budget"] == QUERY_CAP)
        print(f"  {name:<10}{'search @5000 (mean)':<16}{'':>10}{s5['mean']:>10.4f}")
        print(f"  {name:<10}{'search @cap (mean)':<16}{'':>10}{sc['mean']:>10.4f}")
    print("\n  DIAGNOSTICS")
    for name, entry in artifact["graphs"].items():
        for key in ("omega_0.0", "omega_1.0"):
            d = entry["arms"][key]["diagnostics"]
            print(f"    {name} {key}: r_t>1 in "
                  f"{d['fraction_of_steps_with_r_gt_1']:.4f} of steps; "
                  f"first-extra-step hist {d['first_extra_step_histogram']}")
            if "loo_discrimination" in d:
                ld = d["loo_discrimination"]
                print(f"        LOO: mean distinct {ld['mean_distinct_values_per_episode']:.2f}, "
                      f"all-equal in {ld['fraction_of_episodes_with_all_loo_equal']:.3f} of episodes, "
                      f"mean spread {ld['mean_within_episode_spread']:.3f}")
                es = d.get("early_small_reward_vs_later", {})
                print(f"        LOO when r_t<=1: {es.get('mean_loo_when_raw_le_1')} "
                      f"(n={es.get('n_raw_le_1')})   when r_t>1: "
                      f"{es.get('mean_loo_when_raw_gt_1')} (n={es.get('n_raw_gt_1')})")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
