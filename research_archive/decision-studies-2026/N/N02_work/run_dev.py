"""N02 step 3 -- development batch, then freeze the 8 confirmation comparisons.

Frozen protocol:
  * 256 NEW worlds, seeds 2,000,000 .. 2,000,255 (disjoint from every earlier range);
  * all 128 comparisons are evaluated in every world, with ONE live/dead edge state per world shared
    by all four seed sets of a comparison;
  * r_dev = max{ min(-d_c, d_d), min(d_c, -d_d) } from the development means.  Positive means the
    development point estimates already reverse; negative means they do not;
  * select by r_dev descending: 4 uniform_U and 4 degree_U, 8 in total, each quadruple at most once,
    ties broken by comparison id.  Even if no r_dev is positive, the top of each list is still
    selected -- no re-scoring, no re-drawing;
  * PEEKING DISCLOSURE: this selection uses real simulation output, so it is a privileged diagnostic.
    It is not a deployable rule, and the reversal rate after selection does not estimate a natural
    rate.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "N01_work"))
sys.path.insert(0, str(HERE))

from n01_common import Graph, parse_ca_grqc  # noqa: E402
from n02_common import (  # noqa: E402
    draw_live, live_adjacency, signed_reversal_magnitude, spread, stats, t_quantile,
)

DEV_WORLD_IDS = tuple(range(2_000_000, 2_000_256))
N_PER_BACKGROUND = 4


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    stage2 = json.loads((HERE / "N02_stage2_selection.json").read_text(encoding="utf-8"))
    comparisons = stage2["comparisons"]
    n_cmp = len(comparisons)
    print("=" * 100)
    print("  N02 STEP 3 -- development batch")
    print("=" * 100)
    print(f"  worlds {DEV_WORLD_IDS[0]}..{DEV_WORLD_IDS[-1]} ({len(DEV_WORLD_IDS)}), "
          f"{n_cmp} comparisons")

    keys = ("S_ac", "S_bc", "S_ad", "S_bd")
    spreads = {k: [[] for _ in range(n_cmp)] for k in keys}
    depths = []
    t0 = time.perf_counter()
    for wi, wid in enumerate(DEV_WORLD_IDS):
        live = draw_live(graph, wid, random.Random)
        adj = live_adjacency(graph, live)
        for ci, cs in enumerate(comparisons):
            dmax = 0
            for k in keys:
                c, d = spread(adj, cs[k])
                spreads[k][ci].append(c)
                if d > dmax:
                    dmax = d
            depths.append(dmax)
        if (wi + 1) % 32 == 0:
            print(f"    {wi+1}/{len(DEV_WORLD_IDS)} worlds "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  development batch done in {time.perf_counter()-t0:.1f}s; "
          f"max BFS depth {max(depths)}")

    # ---------------------------------------------------------------- development statistics
    t_ord = t_quantile(len(DEV_WORLD_IDS) - 1, 0.975)
    t_adj = t_quantile(len(DEV_WORLD_IDS) - 1, 1.0 - 0.05 / (4 * n_cmp))
    records = []
    for ci, cs in enumerate(comparisons):
        xc = [a - b for a, b in zip(spreads["S_ac"][ci], spreads["S_bc"][ci])]
        xd = [a - b for a, b in zip(spreads["S_ad"][ci], spreads["S_bd"][ci])]
        sc = stats(xc, t_ord, t_adj)
        sd = stats(xd, t_ord, t_adj)
        rec = {
            "comparison_id": ci,
            "quadruple": cs["quadruple"], "Q3": cs["Q3"],
            "background_kind": cs["background_kind"], "U": cs["U"],
            "seeds": {k: cs[k] for k in keys},
            "dev": {
                "delta_c_mean": sc["mean"], "delta_c_se": sc["se"],
                "delta_c_adj_low": sc["adj_low"], "delta_c_adj_high": sc["adj_high"],
                "delta_d_mean": sd["mean"], "delta_d_se": sd["se"],
                "delta_d_adj_low": sd["adj_low"], "delta_d_adj_high": sd["adj_high"],
                "r_dev": signed_reversal_magnitude(sc["mean"], sd["mean"]),
            },
            "dev_X_c": xc, "dev_X_d": xd,
        }
        records.append(rec)

    print(f"\n  development t: ordinary {t_ord:.6f}, adjusted {t_adj:.6f} "
          f"(family 2N={2*n_cmp})")
    r_vals = [r["dev"]["r_dev"] for r in records]
    pos = [r for r in records if r["dev"]["r_dev"] > 0]
    print(f"  r_dev: min {min(r_vals):.3f}, median "
          f"{sorted(r_vals)[len(r_vals)//2]:.3f}, max {max(r_vals):.3f}")
    print(f"  comparisons with r_dev > 0 (development point estimates already reverse): "
          f"{len(pos)}/{n_cmp}")

    # ---------------------------------------------------------------- freeze the 8
    # global ranking by r_dev descending, ties by comparison id
    ranked = sorted(records, key=lambda r: (-r["dev"]["r_dev"], r["comparison_id"]))
    chosen: list[dict] = []
    used_quads: set = set()
    per_kind = {"uniform_U": 0, "degree_U": 0}
    for r in ranked:
        kind = r["background_kind"]
        if per_kind[kind] >= N_PER_BACKGROUND:
            continue
        q = tuple(sorted(r["quadruple"]))
        if q in used_quads:
            continue
        used_quads.add(q)
        per_kind[kind] += 1
        chosen.append(r)
        if len(chosen) == 2 * N_PER_BACKGROUND:
            break
    chosen.sort(key=lambda r: r["comparison_id"])

    print(f"\n  FROZEN confirmation set: {len(chosen)} comparisons "
          f"({per_kind['uniform_U']} uniform_U, {per_kind['degree_U']} degree_U)")
    print(f"  {'cmp':>4} {'kind':<10} {'Q3':>4} {'r_dev':>9} {'dc':>9} {'dd':>9}  quadruple")
    for r in chosen:
        print(f"  {r['comparison_id']:>4} {r['background_kind']:<10} {r['Q3']:>4} "
              f"{r['dev']['r_dev']:>+9.4f} {r['dev']['delta_c_mean']:>+9.4f} "
              f"{r['dev']['delta_d_mean']:>+9.4f}  {r['quadruple']}")

    # ---------------------------------------------------------------- save BEFORE confirming
    payload = {
        "task": "N02 -- targeted reversal search retaining the local-indistinguishability condition",
        "step": "3: development batch complete, confirmation set frozen before confirmation ran",
        "graph": {"manifest": manifest, "n": graph.n, "m": graph.m},
        "dev_batch": {
            "n_worlds": len(DEV_WORLD_IDS),
            "world_seeds": [DEV_WORLD_IDS[0], DEV_WORLD_IDS[-1]],
            "n_comparisons": n_cmp,
            "t_ordinary": t_ord, "t_adjusted": t_adj,
            "family_size": 2 * n_cmp,
        },
        "r_dev_definition": "max{ min(-d_c, d_d), min(d_c, -d_d) }; positive = development point "
                            "estimates already reverse, negative = they do not",
        "selection_rule": "r_dev descending, 4 per background, each quadruple at most once, ties by "
                          "comparison id; applied even when no r_dev is positive",
        "peeking_disclosure": "the confirmation set was chosen using real diffusion output, so this is "
                              "a privileged diagnostic, not a deployable rule; the post-selection "
                              "reversal frequency does not estimate a natural rate",
        "development_ranking": [
            {"comparison_id": r["comparison_id"], "background_kind": r["background_kind"],
             "r_dev": r["dev"]["r_dev"], "delta_c_mean": r["dev"]["delta_c_mean"],
             "delta_d_mean": r["dev"]["delta_d_mean"], "Q3": r["Q3"],
             "quadruple": r["quadruple"], "U": r["U"]}
            for r in sorted(records, key=lambda r: (-r["dev"]["r_dev"], r["comparison_id"]))
        ],
        "frozen_confirmation_set": [
            {"comparison_id": r["comparison_id"], "background_kind": r["background_kind"],
             "quadruple": r["quadruple"], "Q3": r["Q3"], "U": r["U"], "seeds": r["seeds"],
             "r_dev": r["dev"]["r_dev"]}
            for r in chosen
        ],
        "dev_spreads": {k: spreads[k] for k in keys},
        "dev_records": records,
    }
    out = HERE / "N02_stage3_dev_and_freeze.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    print("  confirmation set and all development results are now frozen on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
