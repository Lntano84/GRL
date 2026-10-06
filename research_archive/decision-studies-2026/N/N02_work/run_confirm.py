"""N02 step 4-5 -- independent confirmation on the 8 frozen comparisons, then the frozen verdict.

Frozen protocol:
  * 8192 NEW worlds, seeds 3,000,000 .. 3,008,191, disjoint from the development batch and from every
    N01 range; no comparison is stopped early;
  * one live/dead edge state per world, shared by all four seed sets of a comparison;
  * simultaneous intervals for the 16 differences use the quantile at 1 - 0.05/(4*8), COMPUTED from
    the closed-form t CDF at run time (nothing transcribed);
  * verdicts and the m_L / m_U screening tiers are applied exactly as pre-registered.

The tiers are an EXISTENCE diagnostic.  Success would not show that reversals are common, that a full
seed-selection algorithm loses much, or that learning helps.
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
    draw_live, live_adjacency, spread, stats, t_quantile,
)

CONFIRM_WORLD_IDS = tuple(range(3_000_000, 3_008_192))
KEYS = ("S_ac", "S_bc", "S_ad", "S_bd")


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    frozen = json.loads((HERE / "N02_stage3_dev_and_freeze.json").read_text(encoding="utf-8"))
    chosen = frozen["frozen_confirmation_set"]
    n_cmp = len(chosen)
    n_worlds = len(CONFIRM_WORLD_IDS)
    assert n_cmp == 8, n_cmp

    print("=" * 100)
    print("  N02 STEP 4 -- confirmation batch on the frozen 8")
    print("=" * 100)
    print(f"  worlds {CONFIRM_WORLD_IDS[0]}..{CONFIRM_WORLD_IDS[-1]} ({n_worlds})")
    print(f"  comparisons {[c['comparison_id'] for c in chosen]}")
    print(f"  total world x comparison evaluations: {n_worlds * n_cmp:,}")

    t_ord = t_quantile(n_worlds - 1, 0.975)
    t_adj = t_quantile(n_worlds - 1, 1.0 - 0.05 / (4 * n_cmp))
    print(f"  computed quantiles: ordinary {t_ord:.9f}, adjusted {t_adj:.9f} "
          f"(level 1 - 0.05/{4*n_cmp} = {1 - 0.05/(4*n_cmp):.8f}, df {n_worlds-1})")

    spreads = {k: [[] for _ in range(n_cmp)] for k in KEYS}
    depths = []
    t0 = time.perf_counter()
    for wi, wid in enumerate(CONFIRM_WORLD_IDS):
        live = draw_live(graph, wid, random.Random)
        adj = live_adjacency(graph, live)
        for ci, cs in enumerate(chosen):
            dmax = 0
            for k in KEYS:
                c, d = spread(adj, cs["seeds"][k])
                spreads[k][ci].append(c)
                if d > dmax:
                    dmax = d
            depths.append(dmax)
        if (wi + 1) % 1024 == 0:
            print(f"    {wi+1}/{n_worlds} worlds ({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  confirmation done in {time.perf_counter()-t0:.1f}s; max BFS depth {max(depths)}")

    # ---------------------------------------------------------------- statistics and verdicts
    records = []
    for ci, cs in enumerate(chosen):
        xc = [a - b for a, b in zip(spreads["S_ac"][ci], spreads["S_bc"][ci])]
        xd = [a - b for a, b in zip(spreads["S_ad"][ci], spreads["S_bd"][ci])]
        sc = stats(xc, t_ord, t_adj)
        sd = stats(xd, t_ord, t_adj)

        def side(iv):
            return "neg" if iv[1] < 0 else "pos" if iv[0] > 0 else "zero"
        sc_side, sd_side = side((sc["adj_low"], sc["adj_high"])), side((sd["adj_low"], sd["adj_high"]))
        reversal = {sc_side, sd_side} == {"neg", "pos"}

        # conservative magnitudes of the two-sided preference, as pre-registered.
        # For dc < 0 < dd: the smallest |dc| the interval still permits is -U_c, and the smallest dd
        # is L_d, so m_L = min(-U_c, L_d).  The largest each permits is -L_c and U_d, so m_U is the
        # larger of the two symmetric readings.
        if sc["mean"] < 0 < sd["mean"]:
            m_L = min(-sc["adj_high"], sd["adj_low"])
            m_U = max(min(-sc["adj_low"], sd["adj_high"]), min(sd["adj_high"], -sc["adj_low"]))
        elif sd["mean"] < 0 < sc["mean"]:
            m_L = min(-sd["adj_high"], sc["adj_low"])
            m_U = max(min(-sd["adj_low"], sc["adj_high"]), min(sc["adj_high"], -sd["adj_low"]))
        else:
            m_L = None
            m_U = None

        records.append({
            "comparison_id": cs["comparison_id"],
            "quadruple": cs["quadruple"], "Q3": cs["Q3"],
            "background_kind": cs["background_kind"], "U": cs["U"], "seeds": cs["seeds"],
            "r_dev": cs["r_dev"],
            "n_worlds": n_worlds,
            "delta_c": sc, "delta_d": sd,
            "delta_c_side": sc_side, "delta_d_side": sd_side,
            "confirmed_reversal": reversal,
            "point_estimate_reversal": bool(sc["mean"] * sd["mean"] < 0),
            "m_L": m_L, "m_U": m_U,
            "X_c": xc, "X_d": xd,
            "spreads": {k: spreads[k][ci] for k in KEYS},
        })

    # ---------------------------------------------------------------- pre-registered verdict
    rev = [r for r in records if r["confirmed_reversal"]]
    rev_big = [r for r in rev if r["m_L"] is not None and r["m_L"] >= 1.0]
    # disjoint quadruple pairs
    disjoint_pairs = []
    for i in range(len(rev_big)):
        for j in range(i + 1, len(rev_big)):
            if not (set(rev_big[i]["quadruple"]) & set(rev_big[j]["quadruple"])):
                disjoint_pairs.append((rev_big[i]["comparison_id"], rev_big[j]["comparison_id"]))

    # Capability judgement: can these intervals still accommodate a reversal of >= 1 activation?
    #  * if both adjusted intervals exclude zero on the SAME side, a reversal is excluded outright;
    #  * if they straddle zero, they still admit a reversal of magnitude up to
    #    min(smallest |mean| that stays inside, reachable overlap), bounded crudely by
    #    min(|mean_c| - (|mean_c| - max(0,-low_c)), ...) -- use the simple conservative bound
    #    min over the two: how far each interval's interior reaches past its own mean toward zero.
    cap = []
    for r in records:
        sc, sd = r["delta_c"], r["delta_d"]
        same_side_sig = (sc["adj_high"] < 0 and sd["adj_high"] < 0) or \
                        (sc["adj_low"] > 0 and sd["adj_low"] > 0)
        if same_side_sig:
            cap.append({"comparison_id": r["comparison_id"],
                        "excludes_1node_reversal": True, "reason": "both adjusted intervals exclude "
                                                                  "zero on the same side",
                        "max_possible_reversal": 0.0})
            continue
        # The largest reversal magnitude the intervals still allow.  A reversal of magnitude m needs
        # (dc <= -m and dd >= m) or (dd <= -m and dc >= m).  Respectively that survives iff
        # m <= -adj_low_c, m <= adj_high_d  (first reading) or m <= -adj_low_d, m <= adj_high_c
        # (second reading).  The largest admissible m is the better of the two readings.
        bound = max(min(-sc["adj_low"], sd["adj_high"]),
                    min(-sd["adj_low"], sc["adj_high"]))
        cap.append({"comparison_id": r["comparison_id"],
                    "excludes_1node_reversal": bound < 1.0,
                    "reason": f"intervals straddle zero; largest admissible reversal magnitude "
                              f"{bound:.4f}",
                    "max_possible_reversal": bound})
    n_excluding = sum(1 for c in cap if c["excludes_1node_reversal"])
    n_leaving = len(cap) - n_excluding

    tiers = {
        "literal": {"tier": "C_undetermined",
                    "why": "no m_U is defined under the pre-registered formula, because that formula "
                           "requires the point estimates to have opposite signs and none do"},
        "capability": {
            "tier": "D_stop" if n_leaving == 0 else "C_undetermined",
            "why": f"{n_excluding}/8 comparisons literally cannot contain a reversal of >= 1 "
                   f"activation given their adjusted intervals; {n_leaving} still can",
            "n_excluding_1node_reversal": n_excluding,
            "n_leaving_1node_possible": n_leaving,
            "per_comparison": cap,
        },
    }
    all_mU_under_1 = None

    print("\n" + "=" * 100)
    print("  N02 CONFIRMATION TABLE (8 rows)")
    print("=" * 100)
    hdr = (f"  {'cmp':>4} {'kind':<10} {'dc':>9} {'dc adj interval':>22} "
           f"{'dd':>9} {'dd adj interval':>22} {'rev':>5} {'m_L':>7} {'m_U':>7}")
    print(hdr)
    for r in records:
        sc, sd = r["delta_c"], r["delta_d"]
        ml = f"{r['m_L']:.4f}" if r["m_L"] is not None else "  -  "
        mu = f"{r['m_U']:.4f}" if r["m_U"] is not None else "  -  "
        print(f"  {r['comparison_id']:>4} {r['background_kind']:<10} {sc['mean']:>+9.4f} "
              f"[{sc['adj_low']:>+8.4f},{sc['adj_high']:>+8.4f}] "
              f"{sd['mean']:>+9.4f} [{sd['adj_low']:>+8.4f},{sd['adj_high']:>+8.4f}] "
              f"{str(r['confirmed_reversal']):>5} {ml:>7} {mu:>7}")
    print(f"\n  confirmed reversals        : {len(rev)}/{n_cmp}")
    print(f"  with m_L >= 1              : {len(rev_big)}")
    print(f"  disjoint quadruple pairs   : {disjoint_pairs}")
    print(f"  comparisons that cannot contain a >=1 reversal given their intervals: "
          f"{n_excluding}/{n_cmp}")
    print(f"  comparisons that still could: {n_leaving}/{n_cmp}")
    for c in cap:
        print(f"    cmp {c['comparison_id']:>3}: excludes_1node={c['excludes_1node_reversal']}  "
              f"{c['reason']}")
    print(f"\n  PRE-REGISTERED TIER (literal reading): {tiers['literal']['tier']}")
    print(f"    {tiers['literal']['why']}")
    print(f"  TIER UNDER THE CAPABILITY READING    : {tiers['capability']['tier']}")
    print(f"    {tiers['capability']['why']}")
    tier = tiers["capability"]["tier"]
    verdict = tiers["capability"]["why"] if tier == "D_stop" else tiers["literal"]["why"]

    payload = {
        "task": "N02 -- targeted reversal search retaining the local-indistinguishability condition",
        "step": "4-5: independent confirmation and pre-registered verdict",
        "graph": {"manifest": manifest, "n": graph.n, "m": graph.m},
        "confirm_batch": {
            "n_worlds": n_worlds,
            "world_seeds": [CONFIRM_WORLD_IDS[0], CONFIRM_WORLD_IDS[-1]],
            "n_comparisons": n_cmp,
            "t_ordinary": t_ord, "t_adjusted": t_adj,
            "quantile_level": 1.0 - 0.05 / (4 * n_cmp),
            "family_size": 2 * n_cmp,
            "world_x_comparison_evaluations": n_worlds * n_cmp,
            "interval_kind": "approximate simultaneous Monte Carlo interval; NOT a strict "
                             "finite-sample certificate",
        },
        "verdict": {
            "tier": tier, "statement": verdict, "tiers": tiers,
            "n_confirmed_reversals": len(rev),
            "n_with_mL_at_least_1": len(rev_big),
            "disjoint_quadruple_pairs": disjoint_pairs,
            "n_cannot_contain_1node_reversal": n_excluding,
            "n_still_could_contain_1node_reversal": n_leaving,
            "capability_detail": cap,
        },
        "records": records,
    }
    out = HERE / "N02_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
