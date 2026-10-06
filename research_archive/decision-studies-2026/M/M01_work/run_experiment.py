"""M01 driver: freeze the 32 configurations on disk FIRST, then solve each in its own process.

Enforces the stated budgets: 60 s CPU per configuration (the worker raises ``Timeout`` on its own
deadline) with a 90 s process wall-clock cap as a hard backstop, and 1 hour total wall clock for the
main experiment.  A configuration that does not finish is left MISSING -- no graph is substituted, no
scale is shrunk, and no time is added automatically.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m01_worker import generate_configs  # noqa: E402

PER_CONFIG_PROCESS_CAP = 90.0
TOTAL_WALL_BUDGET = 3600.0


def main() -> int:
    configs = generate_configs()
    print("=" * 100)
    print("  M01 -- freezing configurations before any solving")
    print("=" * 100)
    print(f"  configurations: {len(configs)}")

    # duplicate disclosure: same (n,m), same edge set AND same probability model
    seen = {}
    duplicates = []
    for i, c in enumerate(configs):
        key = (c["n"], c["m"], c["model"], tuple(c["edges"]), tuple(c["probs"]))
        if key in seen:
            duplicates.append({"config_a": configs[seen[key]]["config_id"],
                               "config_b": c["config_id"]})
        else:
            seen[key] = i
    # base graphs shared between HOM and HET are by construction the same graph
    n_base_graphs = len({(c["n"], c["m"], tuple(c["edges"])) for c in configs})
    print(f"  distinct base graphs: {n_base_graphs}")
    print(f"  duplicate (graph, model) configurations: {len(duplicates)}")
    for d in duplicates:
        print(f"    {d}")

    frozen = {
        "task": "M01 -- value of a full finite-horizon policy versus one-step lookahead with greedy "
                "continuation, in three-round stochastic matching with recourse",
        "frozen_before_solving": True,
        "generation_rule": {
            "vertices": "0..n-1",
            "candidate_edges": "all (u,v) with u<v, lexicographic order",
            "base_index": "s = 0..3",
            "graph_seed": "10000*n + 100*m + s",
            "edge_draw": "random.Random(graph_seed).sample(candidates, m)",
            "final_edge_order": "re-sorted lexicographically",
            "no_connectivity_or_outcome_filtering": True,
            "duplicates_kept_and_disclosed": True,
            "HOM": "every q_e = 0.5",
            "HET": "cycle 0.2,0.5,0.8 truncated to m, shuffled by random.Random(20000*n+200*m+s), "
                   "assigned along the sorted edge list",
        },
        "protocol": {
            "horizon_rounds": 3,
            "h_convention": "h counts rounds remaining including the current one; V_0 = 0",
            "objective": "expected number of matched vertices; a successful edge contributes 2",
            "eps": "1e-10 * max(1, n)",
            "tie_break": "lexicographically smallest sorted edge tuple",
            "non_maximal_matchings_allowed": True,
            "monte_carlo_used": False,
            "budget_per_config_seconds": 60.0,
            "process_cap_seconds": PER_CONFIG_PROCESS_CAP,
            "total_wall_budget_seconds": TOTAL_WALL_BUDGET,
        },
        "duplicate_disclosure": duplicates,
        "configurations": configs,
    }
    path = HERE / "M01_configs_frozen.json"
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"  wrote {path}   <-- frozen before solving")

    (HERE / "results_tmp").mkdir(exist_ok=True)

    order = list(range(len(configs)))
    if len(sys.argv) > 1 and sys.argv[1] == "--pilot":
        order = [0, 12, 16, 28]      # one per (n,m)档 under HOM
        print(f"\n  PILOT mode: solving {order}")

    start = time.perf_counter()
    for pos, idx in enumerate(order):
        elapsed = time.perf_counter() - start
        if elapsed > TOTAL_WALL_BUDGET:
            print(f"\n  TOTAL WALL BUDGET ({TOTAL_WALL_BUDGET:.0f}s) reached after {pos} configs; "
                  f"stopping. Remaining configurations are left MISSING.")
            break
        cfg = configs[idx]
        out = HERE / f"results_tmp/config_{idx:02d}.json"
        if out.exists():
            print(f"  [{pos+1}/{len(order)}] {cfg['config_id']} already done, skipping")
            continue
        print(f"\n  [{pos+1}/{len(order)}] {cfg['config_id']}", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(HERE / "m01_worker.py"), str(idx),
                                   "--out", str(out)],
                                  timeout=PER_CONFIG_PROCESS_CAP, check=False,
                                  capture_output=True, text=True, cwd=str(HERE))
            if proc.returncode != 0:
                print(f"    WORKER FAILED (returncode {proc.returncode}); stderr tail:")
                for line in (proc.stderr or "").strip().splitlines()[-6:]:
                    print(f"      {line}")
                out.write_text(json.dumps(
                    {"config_id": cfg["config_id"], "n": cfg["n"], "m": cfg["m"], "s": cfg["s"],
                     "model": cfg["model"], "graph_seed": cfg["graph_seed"],
                     "prob_seed": cfg["prob_seed"], "edges": cfg["edges"], "probs": cfg["probs"],
                     "timeout": True, "worker_failed": True,
                     "timeout_reason": f"worker returncode {proc.returncode}",
                     "stderr_tail": (proc.stderr or "").strip().splitlines()[-12:]},
                    indent=2), encoding="utf-8")
            else:
                for line in (proc.stdout or "").strip().splitlines()[-4:]:
                    print(f"      {line}")
        except subprocess.TimeoutExpired:
            print(f"    PROCESS CAP {PER_CONFIG_PROCESS_CAP:.0f}s exceeded -> recorded as missing")
            out.write_text(json.dumps(
                {"config_id": cfg["config_id"], "n": cfg["n"], "m": cfg["m"], "s": cfg["s"],
                 "model": cfg["model"], "graph_seed": cfg["graph_seed"],
                 "prob_seed": cfg["prob_seed"], "edges": cfg["edges"], "probs": cfg["probs"],
                 "timeout": True, "timeout_reason": "process wall-clock cap exceeded",
                 "wall_seconds": time.perf_counter() - t0}, indent=2), encoding="utf-8")
        dt = time.perf_counter() - t0
        print(f"    process wall {dt:.1f}s   (elapsed total {time.perf_counter()-start:.1f}s)",
              flush=True)

    print(f"\n  driver finished in {time.perf_counter()-start:.1f}s")
    done = sorted(p.name for p in (HERE / "results_tmp").glob("config_*.json"))
    print(f"  result files present: {len(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
