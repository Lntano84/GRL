"""Pre-flight on ca-GrQc: verify the frozen graph, then measure how restrictive the N01 structural
condition actually is.  This runs BEFORE any comparison is frozen and before any diffusion result is
seen; it only reads graph structure.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import Graph, check_2hop_disjoint, parse_ca_grqc, within_hops  # noqa: E402


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    print("=" * 96)
    print("  GRAPH MANIFEST (frozen source)")
    print("=" * 96)
    for k, v in manifest.items():
        print(f"    {k:<38} {v}")
    if not manifest["matches_expected"]:
        print("\n  !! manifest does NOT match the expected CCIM probe values -- STOP")
        return 2

    graph = Graph(arcs, undirected, node_ids)
    print(f"\n  n = {graph.n}, m = {graph.m}, directed propagation arcs = {graph.n_arcs}")

    comps = graph.components()
    print(f"  connected components: {len(comps)}; sizes head = {[len(c) for c in comps[:6]]}")
    lcc = comps[0]
    print(f"  largest connected component: {len(lcc)} nodes ({len(lcc)/graph.n:.4%} of n)")

    deg = graph.degree
    print(f"  degree: min {min(deg)}, median {statistics.median(deg)}, max {max(deg)}, "
          f"mean {statistics.fmean(deg):.4f}")
    # propagate p = 1/deg must be a probability
    bad = [v for v in range(graph.n) if not (0 < 1.0 / deg[v] <= 1.0)]
    print(f"  nodes with invalid 1/deg(v) probability: {len(bad)}")

    # ------------------------------------------------------------ 2-hop region sizes on the LCC
    sample_rng = random.Random(20261231)
    picks = sample_rng.sample(lcc, 200)
    sizes = []
    for v in picks:
        sizes.append(len(within_hops(graph, v, 2)))
    print(f"\n  |R2(v)| over 200 sampled LCC nodes: min {min(sizes)}, "
          f"median {statistics.median(sizes)}, mean {statistics.fmean(sizes):.1f}, max {max(sizes)}")
    huge = sum(1 for s in sizes if s > 1000)
    print(f"    of those, {huge}/200 have |R2| > 1000 (large neighbourhoods make disjointness hard)")

    # ------------------------------------------------------------ rejection-rate estimate
    trials = 20000
    rng = random.Random(20261231)
    ok = 0
    r2_cache: dict[int, set[int]] = {}
    for _ in range(trials):
        a, b, c, d = rng.sample(lcc, 4)
        for x in (a, b, c, d):
            if x not in r2_cache:
                r2_cache[x] = within_hops(graph, x, 2)
        if check_2hop_disjoint(r2_cache, a, b, c, d)["holds"]:
            ok += 1
    print(f"\n  uniform random 4-tuples from the LCC: {ok}/{trials} satisfy the 2-hop disjointness "
          f"condition (rate {ok/trials:.6f})")
    if ok == 0:
        print("    -> uniform rejection sampling alone would need >>100k draws; a structural "
              "pre-filter is required to reach 40 comparisons within the 100k attempt budget.")

    # ------------------------------------------------------------ structural pre-filter feasibility
    r2_all = {v: within_hops(graph, v, 2) for v in lcc}
    for x in lcc:
        r2_all[x].add(x)  # include self so that "no member of the other side" also blocks equality
    adj_sets = {v: set(graph.adj[v]) | {v} for v in lcc}

    def ok_pair_group(group) -> bool:
        g = list(group)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if r2_all[g[i]] & r2_all[g[j]]:
                    return False
        return True

    # greedy maximal independent set in the "2-hop overlap" conflict graph
    order = sorted(lcc, key=lambda v: (len(r2_all[v]), v))
    indep: list[int] = []
    blocked: set[int] = set()
    for v in order:
        if v in blocked:
            continue
        indep.append(v)
        blocked |= r2_all[v]
    print(f"\n  greedily built 2-hop-disjoint independent set: {len(indep)} nodes "
          f"(out of {len(lcc)}) -- sizes on the LCC are large, so this set is small")
    print(f"    max |R2| in that set = {max(len(r2_all[v]) for v in indep)}")

    summary = {
        "manifest": manifest,
        "n": graph.n, "m": graph.m, "n_arcs": graph.n_arcs,
        "n_components": len(comps),
        "lcc_size": len(lcc),
        "degree": {"min": min(deg), "median": statistics.median(deg), "max": max(deg),
                   "mean": statistics.fmean(deg)},
        "invalid_probability_nodes": len(bad),
        "r2_sizes_200_lcc_sample": {
            "min": min(sizes), "median": statistics.median(sizes),
            "mean": statistics.fmean(sizes), "max": max(sizes),
            "count_over_1000": huge},
        "uniform_quadruple_rejection": {"trials": trials, "accepted": ok,
                                        "rate": ok / trials},
        "greedy_2hop_independent_set_size": len(indep),
        "greedy_2hop_independent_set_max_r2": max(len(r2_all[v]) for v in indep),
    }
    (HERE / "preflight_ca_grqc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {HERE / 'preflight_ca_grqc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
