"""Case-3 diagnostic for the scan-order pilot: WHY did the structural orders not help?

The pre-fixed rule says that when neither B nor C clearly improves, the next question is whether the
score fails to separate candidates, or whether the new order walks into worse local optima.  The scan
logs answer this directly:

* ``first_improvement_position`` -- where in the scan the first improving swap was found.  Smaller is
  better: it means the ordering put a good candidate near the front.  If B/C do not move this number
  down, the score is not separating candidates.
* how many distinct scores exist for the candidates of one state -- if the score is nearly constant,
  it cannot reorder anything.
* how many scans ended exhausted (no improving swap anywhere) -- these states offer nothing, and no
  ordering can help there.
* wall-clock spread across the five seeds -- to judge whether the timing differences are above noise.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccim.model import DEFAULT_K, load_graph
from ccim.scan_order import swap_scores

ARTIFACT = json.loads((ROOT / "results" / "scan_order_pilot.json").read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 100)
    print("  DOES THE ORDER PUT IMPROVING SWAPS EARLIER?")
    print("=" * 100)
    print(f"  {'graph':<10}{'method':<10}{'scans':>7}{'improved':>10}{'exhausted':>11}"
          f"{'median pos':>12}{'mean pos':>10}{'cands avail':>13}")
    for name, entry in ARTIFACT["graphs"].items():
        for method, arm in entry["arms"].items():
            pos, avail, scans, exhausted = [], [], 0, 0
            for run in arm["runs"]:
                for s in run["scan_log"]:
                    scans += 1
                    avail.append(s["candidates_available"])
                    if s["exhausted"]:
                        exhausted += 1
                    else:
                        pos.append(s["first_improvement_position"])
            print(f"  {name:<10}{method:<10}{scans:>7}{len(pos):>10}{exhausted:>11}"
                  f"{(statistics.median(pos) if pos else float('nan')):>12.0f}"
                  f"{(statistics.fmean(pos) if pos else float('nan')):>10.0f}"
                  f"{statistics.fmean(avail):>13.0f}")

    print("\n" + "=" * 100)
    print("  DOES THE SCORE SEPARATE CANDIDATES AT ALL?")
    print("=" * 100)
    print("  (recomputed on the initial degree top-8 of each graph and on a few states)")
    for name in ARTIFACT["graphs"]:
        g = load_graph(name)
        nodes = list(g.nodes())
        by_degree = sorted(nodes, key=lambda v: (-g.degree(v), v))
        states = [sorted(by_degree[:8])]
        # a couple of later states taken from the recorded trajectories
        for run in ARTIFACT["graphs"][name]["arms"]["random"]["runs"][:2]:
            for s in run["scan_log"][:3]:
                if s["accepted"]:
                    u, v = s["accepted"]
                    cur = sorted([x for x in states[0] if x != u] + [v])
                    if cur not in states:
                        states.append(cur)
        for i, S in enumerate(states):
            sc = swap_scores(g, S, DEFAULT_K)
            vals = list(sc.values())
            deg_keys = [g.degree(v) - g.degree(u) for (u, v) in sc]
            print(f"  {name} state {i} (S={S}): {len(vals)} candidates, "
                  f"F1 score -> {len(set(vals))} distinct values "
                  f"(min {min(vals)}, max {max(vals)}, zero {sum(1 for x in vals if x == 0)}), "
                  f"degree diff -> {len(set(deg_keys))} distinct")

    print("\n" + "=" * 100)
    print("  TIMING NOISE: spread of wall seconds across the five seeds")
    print("=" * 100)
    print(f"  {'graph':<10}{'method':<10}{'mean':>8}{'min':>8}{'max':>8}{'range':>8}")
    for name, entry in ARTIFACT["graphs"].items():
        for method, arm in entry["arms"].items():
            ws = [r["wall_seconds"] for r in arm["runs"]]
            print(f"  {name:<10}{method:<10}{statistics.fmean(ws):>8.2f}{min(ws):>8.2f}"
                  f"{max(ws):>8.2f}{max(ws) - min(ws):>8.2f}")

    print("\n" + "=" * 100)
    print("  HOW OFTEN DID THE FIRST SCANNED SWAP ALREADY IMPROVE?  (position == 1)")
    print("=" * 100)
    for name, entry in ARTIFACT["graphs"].items():
        for method, arm in entry["arms"].items():
            firsts = [s["first_improvement_position"] for run in arm["runs"]
                      for s in run["scan_log"] if not s["exhausted"]]
            if not firsts:
                continue
            print(f"  {name:<10}{method:<10} position 1 in {sum(1 for p in firsts if p == 1)}"
                  f"/{len(firsts)} improving scans;  scanned-before-accept mean "
                  f"{statistics.fmean(firsts):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
