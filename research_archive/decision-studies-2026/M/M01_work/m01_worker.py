"""M01 main experiment -- one configuration per process, so the CPU budget is enforced cleanly.

Configurations are generated and WRITTEN TO DISK BEFORE any solving.  The generation rule is fixed:
    vertices 0..n-1; candidate edges all (u,v), u<v, in lexicographic order; base index s = 0..3;
    graph seed 10000*n + 100*m + s; draw m edges with independent random.Random(seed).sample;
    re-sort the final edge list lexicographically.  No connectivity, degree or outcome filtering and
    no redrawing.  Duplicates are kept and disclosed.
    HOM: every q_e = 0.5.
    HET: cycle 0.2, 0.5, 0.8 of length m, shuffled by an independent RNG seeded 20000*n + 200*m + s,
         then assigned along the sorted edge list.

Per configuration this worker computes, with SEPARATE caches for each method and the cache cleared
before each timing:
    V_3^G, V_3^R, V_3^*,  d_G, d_R,
    A_R (every first action within eps of the lookahead maximum),
    d_robust = V_3^* - max_{M in A_R} Q_3^*(G, M),
    the full first-action table (greedy score, lookahead score, Q_3^*).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m01_solver import Problem, Solver, Timeout  # noqa: E402

BUDGET_SECONDS = 60.0


def generate_configs():
    """The frozen 32 configurations, as plain dictionaries."""
    configs = []
    for n, m, s in [(6, 6, s) for s in range(4)] + [(6, 10, s) for s in range(4)] + \
                  [(8, 8, s) for s in range(4)] + [(8, 12, s) for s in range(4)]:
        seed = 10000 * n + 100 * m + s
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
        edges = sorted(random.Random(seed).sample(candidates, m))
        base = {"n": n, "m": m, "s": s, "graph_seed": seed, "edges": edges,
                "distinct_edges": len(set(edges))}
        # HOM
        configs.append({**base, "model": "HOM", "probs": [0.5] * m,
                        "prob_seed": None,
                        "config_id": f"n{n}_m{m}_s{s}_HOM"})
        # HET
        pseed = 20000 * n + 200 * m + s
        cycle = [0.2, 0.5, 0.8] * ((m + 2) // 3)
        cycle = cycle[:m]
        shuffled = cycle[:]
        random.Random(pseed).shuffle(shuffled)
        configs.append({**base, "model": "HET", "probs": shuffled,
                        "prob_seed": pseed, "cycle": cycle,
                        "config_id": f"n{n}_m{m}_s{s}_HET"})
    return configs


def solve_one(cfg: dict) -> dict:
    p = Problem(cfg["n"], cfg["edges"], cfg["probs"], label=cfg["config_id"])
    root = p.full_mask
    deadline = time.perf_counter() + BUDGET_SECONDS
    out = {"config_id": cfg["config_id"], "n": cfg["n"], "m": cfg["m"], "s": cfg["s"],
           "model": cfg["model"], "graph_seed": cfg["graph_seed"],
           "prob_seed": cfg["prob_seed"], "edges": cfg["edges"], "probs": cfg["probs"],
           "eps": 1e-10 * max(1, cfg["n"])}
    timing = {}

    # ---- OPT (its own fresh cache) ----
    t0 = time.perf_counter()
    s_opt = Solver(p, deadline=deadline)
    v_opt = s_opt.V_OPT(root, 3)
    opt_action, opt_val_root = s_opt.OPT_action(root, 3)
    timing["OPT_seconds"] = time.perf_counter() - t0
    opt_cache = s_opt.cache_sizes()

    # ---- G (its own fresh cache) ----
    t0 = time.perf_counter()
    s_g = Solver(p, deadline=deadline)
    v_g = s_g.V_G(root, 3)
    g_action = s_g.greedy_action(root)
    timing["G_seconds"] = time.perf_counter() - t0
    g_cache = s_g.cache_sizes()

    # ---- R (its own fresh cache; never reads the OPT cache) ----
    t0 = time.perf_counter()
    s_r = Solver(p, deadline=deadline)
    v_r = s_r.V_R(root, 3)
    r_action, argmax_set, r_best = s_r.R_action(root, 3)
    timing["R_seconds"] = time.perf_counter() - t0
    r_cache = s_r.cache_sizes()

    # ---- first-action table (built from the OPT solver's caches) ----
    root_matchings = p.all_matchings(root)
    table = []
    for M in root_matchings:
        g_score = 2.0 * sum(p.probs[i] for i in _bits(M))
        look = s_g.Qtilde(root, M, 3)
        q3 = s_opt.Q3_star(root, M)
        table.append({
            "action": [list(e) for e in p.matching_tuple(M)],
            "greedy_score": g_score,
            "lookahead_score": look,
            "Q3_star": q3,
        })
    table.sort(key=lambda row: tuple(tuple(e) for e in row["action"]))

    argmax_tuples = {p.matching_tuple(M) for M in argmax_set}
    q3_of_ar = [row["Q3_star"] for row in table
                if tuple(tuple(e) for e in row["action"]) in argmax_tuples]
    worst_ar = max(q3_of_ar) if q3_of_ar else None
    d_robust = (v_opt - worst_ar) if worst_ar is not None else None

    out.update({
        "V3_G": v_g, "V3_R": v_r, "V3_OPT": v_opt,
        "d_G": v_opt - v_g, "d_R": v_opt - v_r, "d_robust": d_robust,
        "d_G_over_n": (v_opt - v_g) / cfg["n"],
        "d_R_over_n": (v_opt - v_r) / cfg["n"],
        "d_robust_over_n": (d_robust / cfg["n"]) if d_robust is not None else None,
        "G_first_action": [list(e) for e in p.matching_tuple(g_action)],
        "R_first_action": [list(e) for e in p.matching_tuple(r_action)],
        "OPT_first_action": [list(e) for e in p.matching_tuple(opt_action)],
        "A_R": [[list(e) for e in p.matching_tuple(M)] for M in argmax_set],
        "n_A_R": len(argmax_set),
        "R_best_lookahead": r_best,
        "max_Q3_star_over_A_R": worst_ar,
        "OPT_root_value_check": opt_val_root,
        "n_matchings_root": len(root_matchings),
        "first_action_table": table,
        "timing_seconds": timing,
        "cache_sizes": {"OPT": opt_cache, "G": g_cache, "R": r_cache},
        "timeout": False,
    })
    return out


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: m01_worker.py <config_index> [--out PATH]")
    idx = int(sys.argv[1])
    out_path = HERE / f"results_tmp/config_{idx:02d}.json"
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    configs = generate_configs()
    cfg = configs[idx]
    print(f"  solving {cfg['config_id']} ...", flush=True)
    t0 = time.perf_counter()
    try:
        res = solve_one(cfg)
        res["timeout"] = False
    except Timeout as exc:
        res = {"config_id": cfg["config_id"], "n": cfg["n"], "m": cfg["m"], "s": cfg["s"],
               "model": cfg["model"], "graph_seed": cfg["graph_seed"],
               "prob_seed": cfg["prob_seed"], "edges": cfg["edges"], "probs": cfg["probs"],
               "timeout": True, "timeout_reason": str(exc)}
        print(f"    TIMEOUT after {time.perf_counter()-t0:.1f}s", flush=True)
    res["wall_seconds"] = time.perf_counter() - t0
    if not res.get("timeout"):
        print(f"    V3_G={res['V3_G']:.6f} V3_R={res['V3_R']:.6f} "
              f"V3_OPT={res['V3_OPT']:.6f} d_robust={res['d_robust']:.6f} "
              f"({res['wall_seconds']:.1f}s)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"    wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
