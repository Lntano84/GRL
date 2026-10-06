"""Checks 1 and 2 of N01: the deterministic hand graph, and the summed-proxy arithmetic identity.

Check 1 (hand graph).  Build the frozen 24-node / 22-edge construction, confirm the four seed sets give
12 / 18 / 18 / 12 under IC with p = 1 and U = empty, and confirm the four two-hop regions are pairwise
disjoint.

Check 2 (proxy identity).  Evaluate the *definition* of a fixed-parameter two-layer summed local proxy
on the same four sets, for several different random score vectors, and confirm delta_hat_c and
delta_hat_d agree to floating point.  This is an arithmetic identity check on the proxy class, not a
run of the official SIMBA network (see N01_report.md for why the official forward pass is recorded as
not completed).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_structure import (  # noqa: E402
    HAND_CANDIDATES, build_hand_graph, disjoint_condition, forward_within_hops,
    hand_expected, summed_local_proxy_differences,
)


def ic_p1_closure(g, seeds):
    """Exact IC closure with every arc live (p = 1): forward reachability, rounds capped at 100."""
    seen = set(seeds)
    frontier = list(seeds)
    depth = 0
    while frontier and depth < 100:
        nxt = []
        for v in frontier:
            for u in g.out[v]:
                if u not in seen:
                    seen.add(u)
                    nxt.append(u)
        frontier = nxt
        depth += 1
    return seen


def main() -> int:
    print("=" * 96)
    print("  CHECK 1 -- deterministic hand graph (24 nodes, 22 directed edges, p = 1, U = empty, k = 2)")
    print("=" * 96)
    g, idx, labels, edges = build_hand_graph()
    print(f"  nodes {g.n}, directed edges {g.m}")
    if (g.n, g.m) != (24, 22):
        print(f"  !! expected 24 nodes / 22 edges, got {g.n} / {g.m}")
        return 2

    a, b, c, d = (idx[x] for x in HAND_CANDIDATES)
    r2 = {x: forward_within_hops(g, x, 2) for x in (a, b, c, d)}
    struct = disjoint_condition(r2, a, b, c, d)
    print("\n  two-hop influence regions R2(x):")
    for key in ("a", "b", "c", "d"):
        names = [labels[v] for v in struct["r2"][key]]
        print(f"    R2({key}) = {names}   (size {len(names)})")
    print(f"  (R2(a) u R2(b)) n (R2(c) u R2(d)) = {struct['intersection']} "
          f"-> size {struct['intersection_size']}, disjoint = {struct['holds']}")

    print("\n  seed set        activated     expected")
    got, ok_all = {}, True
    for s1, s2 in (("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")):
        active = ic_p1_closure(g, [idx[s1], idx[s2]])
        got[(s1, s2)] = len(active)
        exp = hand_expected()[(s1, s2)]
        flag = "OK" if len(active) == exp else "MISMATCH"
        ok_all &= (len(active) == exp)
        print(f"    {{{s1},{s2}}}          {len(active):>3}          {exp:>3}   {flag}")

    delta_c = got[("a", "c")] - got[("b", "c")]
    delta_d = got[("a", "d")] - got[("b", "d")]
    print(f"\n  delta_c = {delta_c}   delta_d = {delta_d}   "
          f"(expected -6 and +6; signs strictly opposite = "
          f"{delta_c * delta_d < 0})")

    print("\n" + "=" * 96)
    print("  CHECK 2 -- summed-local-proxy arithmetic identity")
    print("=" * 96)
    print("  For Sigma_hat(S) = sum_{v in S} phi(v) + const, delta_hat_c and delta_hat_d both equal")
    print("  phi(a) - phi(b), so their difference must be 0 for every phi.  Five fixed random")
    print("  initialisations of phi (seeds 1..5), 64 leaves each:\n")
    proxy_rows = []
    for seed in range(1, 6):
        rng = random.Random(seed)
        phi = {v: rng.uniform(-1.0, 1.0) for v in range(g.n)}
        row = summed_local_proxy_differences(phi, a, b, c, d, u_seeds=())
        proxy_rows.append({"init_seed": seed, **row})
        print(f"    init {seed}: S_hat(S_ac)={row['S_hat']['S_ac']:+.12f}  "
              f"S_hat(S_bc)={row['S_hat']['S_bc']:+.12f}  "
              f"S_hat(S_ad)={row['S_hat']['S_ad']:+.12f}  "
              f"S_hat(S_bd)={row['S_hat']['S_bd']:+.12f}")
        print(f"             delta_hat_c={row['delta_hat_c']:+.12f}  "
              f"delta_hat_d={row['delta_hat_d']:+.12f}  "
              f"difference={row['difference']:.3e}")
    max_dev = max(abs(r["difference"]) for r in proxy_rows)
    print(f"\n  max |delta_hat_c - delta_hat_d| over the five inits: {max_dev:.3e}  "
          f"(pure floating-point noise)")

    artifact = {
        "hand_graph": {
            "n": g.n, "m": g.m,
            "edges": [f"{u}->{v}" for u, v in edges],
            "candidates": list(HAND_CANDIDATES),
            "label_to_index": idx,
            "index_to_label": {str(i): labels[i] for i in range(g.n)},
            "propagation": {"model": "IC", "p": 1.0, "U": [], "k": 2, "rounds_cap": 100},
            "two_hop_regions": struct["r2"],
            "two_hop_regions_named": {
                key: sorted(labels[v] for v in struct["r2"][key])
                for key in ("a", "b", "c", "d")},
            "r2_sizes": struct["r2_sizes"],
            "disjointness_holds": struct["holds"],
            "intersection_size": struct["intersection_size"],
            "activations": {f"{{{k[0]},{k[1]}}}": v for k, v in got.items()},
            "activations_expected": {f"{{{k[0]},{k[1]}}}": v for k, v in hand_expected().items()},
            "activations_all_match": bool(ok_all),
            "delta_c": delta_c, "delta_d": delta_d,
            "signs_strictly_opposite": bool(delta_c * delta_d < 0),
        },
        "summed_proxy_identity": {
            "class": "Sigma_hat(S) = sum_{v in S} phi(v) + const, const independent of S",
            "claim": "delta_hat_c == delta_hat_d identically",
            "inits": proxy_rows,
            "max_abs_deviation": max_dev,
            "is_official_simba_run": False,
            "note": "arithmetic identity on the DEFINITION of a summed local proxy; NOT a forward pass "
                    "of the official SIMBA SpreadPredictor, which could not be run (no torch, no "
                    "network to fetch yl489/rethink-IM).",
        },
    }
    (HERE / "hand_graph_check.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'hand_graph_check.json'}")

    if not (ok_all and struct["holds"] and max_dev < 1e-9):
        print("\n  !! CHECK FAILED")
        return 1
    print("\n  CHECK 1 and CHECK 2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
