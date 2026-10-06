"""Time the pieces before freezing the constants, and sanity-check that they mean something.

Nothing here selects a configuration by outcome.  The only thing measured is cost, because the
pre-registration has to state a budget that is actually affordable, and because the cost of the
look-ahead is itself one of the reported results.  The one quality check (seed-set value versus RR
budget) uses the *same* model for every budget so the comparison is of budgets, not of samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccim.eim import graphdata, rules                       # noqa: E402
from ccim.eim.gm import HomophilyModel                       # noqa: E402
from ccim.eim.ic import RRSolver, TrueInfluence, sample_rr_sets   # noqa: E402
from ccim.eim.states import (FIXED_RANDOM_SURVEYS, FREE_SURVEYS, PRIMARY_K,  # noqa: E402
                             PRIMARY_P, run_state)


def adjacency(edges, n):
    a = np.zeros((n, n), dtype=bool)
    a[edges[:, 0], edges[:, 1]] = True
    return a | a.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "eim_timing.json"))
    args = ap.parse_args()

    out: dict = {}
    nodes, edges, attr, meta = graphdata.load_network()
    out["data"] = meta
    n = len(nodes)
    E = np.asarray(edges, dtype=np.int64)
    print(f"graph: n={n} m={len(E)} day={meta['chosen_day']} groups={len(meta['classes'])} "
          f"within={meta['within_class_edges']}/{len(E)} ({meta['within_class_edges']/len(E):.3f}) "
          f"isolated={meta['n_isolated_on_day']} mean_deg={2*len(E)/n:.2f}")

    # ---------------------------------------------------------------- model fit
    adj = adjacency(E, n)
    rng = np.random.default_rng(7)
    model = HomophilyModel(n, attr)
    picked = [int(u) for u in rng.choice(n, size=FREE_SURVEYS + FIXED_RANDOM_SURVEYS, replace=False)]
    t0 = time.perf_counter()
    for u in picked:
        model.resolve(u, [int(w) for w in np.nonzero(adj[u])[0]])
    out["resolve_and_fit_seconds_each"] = (time.perf_counter() - t0) / len(picked)
    out["model_summary"] = model.summary()
    out["model_attribute_table"] = {k: {kk: round(vv["p"], 4) for kk, vv in v.items()}
                                    for k, v in __import__("ccim.eim.gm", fromlist=["x"])
                                    .attribute_pair_table(model).items()}
    print("model after 14 surveys:", json.dumps(model.summary(), indent=1)[:520])

    # ---------------------------------------------------------------- RR cost and quality
    true_scores = {}
    grader = TrueInfluence(n, E, PRIMARY_P, 3000, np.random.default_rng(13))
    fixed_sample = model.edges_for_solver(np.random.default_rng(12))
    for theta in (100, 200, 500, 1000, 2000):
        r = np.random.default_rng(11)
        t0 = time.perf_counter()
        sizes, sets = sample_rr_sets(n, E, np.full(len(E), PRIMARY_P), theta, r)
        t_rr = time.perf_counter() - t0
        solver = RRSolver(n, PRIMARY_K, theta, np.random.default_rng(11))
        t0 = time.perf_counter()
        idx, predicted = solver.solve(*fixed_sample)
        t_solve = time.perf_counter() - t0
        val = grader.value([int(i) for i in idx])
        true_scores[str(theta)] = {"rr_seconds": t_rr, "solve_seconds": t_solve,
                                   "mean_rr_size": float(sizes.mean()),
                                   "predicted": float(predicted), "sigma_true": float(val)}
        print(f"theta={theta:5d}: rr {t_rr:.3f}s solve {t_solve:.3f}s mean|Q|={sizes.mean():.1f} "
              f"predicted {predicted:.1f} -> sigma_true {val:.1f}")
    out["rr_and_solver"] = true_scores
    out["grader_reference"] = {
        "empty": grader.value([]),
        "random_10": float(np.mean([grader.value(list(np.random.default_rng(s).choice(n, 10, replace=False)))
                                    for s in range(5)])),
        "top_degree_10": grader.value(sorted(range(n), key=lambda v: (-adj[v].sum(), v))[:10]),
        "all_nodes": grader.value(list(range(n))),
    }
    print("grader reference:", json.dumps(out["grader_reference"], indent=0))

    # ---------------------------------------------------------------- full state
    cfg = {"seed": 1, "pool_size": 8, "rule_samples": 32, "rule_solver_samples": 1000,
           "own_solver_samples": 1000, "lookahead_samples": 8, "confirm_samples": 3000}
    t0 = time.perf_counter()
    rec = run_state(1, nodes, edges, attr, cfg)
    total = time.perf_counter() - t0
    out["full_state"] = {"cfg": cfg, "seconds": total,
                         "arm_seconds": {k: round(v["seconds"], 3) for k, v in rec["arms"].items()},
                         "sigma_true": {k: round(v["sigma_true"], 2) for k, v in rec["arms"].items()},
                         "chosen": {k: v["chosen_node"] for k, v in rec["arms"].items()},
                         "survey_yield": {k: v["survey_yield"] for k, v in rec["arms"].items()},
                         "pool": [nodes[i] for i in rec["candidate_pool"]],
                         "legal_count": rec["legal_count"]}
    print(f"full state: {total:.1f}s")
    print("  arm seconds:", json.dumps(out["full_state"]["arm_seconds"], indent=0))
    print("  sigma_true :", json.dumps(out["full_state"]["sigma_true"], indent=0))
    print("  chosen     :", json.dumps(out["full_state"]["chosen"], indent=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
