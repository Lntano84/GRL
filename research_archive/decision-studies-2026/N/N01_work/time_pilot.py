"""Timing pilot: how long does one world cost, so the full 256 + 2048 world run can be budgeted?"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import (  # noqa: E402
    DEV_WORLD_IDS, Graph, build_live_adjacency, draw_live_arcs, ic_spread,
    parse_ca_grqc,
)


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    print(f"n={graph.n} m={graph.m} arcs={graph.n_arcs}")

    cases = [
        {"S_ac": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"S_ac": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
    ]
    n_worlds = 64
    t0 = time.perf_counter()
    depths = []
    for wid in DEV_WORLD_IDS[:n_worlds]:
        live = draw_live_arcs(graph, wid)
        adj = build_live_adjacency(graph, live)
        for cs in cases:
            for key in ("S_ac",):
                c, d = ic_spread(adj, cs[key])
                depths.append(d)
    dt = time.perf_counter() - t0
    per_world = dt / n_worlds
    per_set = dt / (n_worlds * len(cases))

    print(f"\n  {n_worlds} worlds x {len(cases)} sets = {n_worlds*len(cases)} spreads in {dt:.3f} s")
    print(f"  per world            : {per_world*1000:.2f} ms")
    print(f"  per set evaluation   : {per_set*1000:.3f} ms")
    print(f"  max BFS depth seen   : {max(depths)} (cap is 100)")

    total_sets = 40 * 4 * (256 + 2048)
    total_worlds = 40 * (256 + 2048)
    print(f"\n  full run: {total_worlds} world-draws, {total_sets} set evaluations")
    print(f"  projected wall clock : {total_worlds*per_world/60:.2f} min "
          f"(world draws) + {total_sets*per_set/60:.2f} min (spreads)")
    print(f"  projected total      : {(total_worlds*per_world + total_sets*per_set)/60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
