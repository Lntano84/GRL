"""Summarise the targeted-objective validation artifact.

Reports, per cell: the sequential lift over the best static arm, whether the patience rule fired and
where, and the cost each arm paid --- all in the contracted units (``|positive ∩ D|`` and cascades),
so the table cannot repeat the whole-graph mistake.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC_ARMS = ("degree_static", "delta2_static")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "validation_targeted.json")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    if not cells:
        print("no cells")
        return 1

    print(f"  {len(cells)} cells   arm failures: {len(payload.get('arm_failures', []))}")
    print()
    print(f"  {'graph':<18}{'k':>4}{'|D|':>5}{'draw':>5}{'seed':>10}"
          f"{'seq':>8}{'best static':>12}{'lift':>9}{'seq casc':>10}")

    lifts: list[float] = []
    by_fraction: dict[float, list[float]] = {}
    for c in sorted(cells, key=lambda c: (c["graph"], c["budget"], c["pool_draw"],
                                          c["random_seed"])):
        seq = c["arms"].get("delta2_sequential")
        if not seq:
            continue
        statics = [c["arms"][a]["spread"] for a in STATIC_ARMS if a in c["arms"]]
        if not statics:
            continue
        best = max(statics)
        if abs(best) < 1e-9:
            continue
        lift = (seq["spread"] - best) / abs(best)
        lifts.append(lift)
        by_fraction.setdefault(round(seq["seed_fraction"], 3), []).append(lift)
        print(f"  {c['graph']:<18}{c['budget']:>4}{c['target_size']:>5}{c['pool_draw']:>5}"
              f"{c['random_seed']:>10}{seq['spread']:>8.2f}{best:>12.2f}"
              f"{lift*100:>8.2f}%{seq['mc_cascades']:>10}")

    print()
    print("=== Gate 1(b) on the CONTRACTED objective ===")
    if lifts:
        print(f"  cells                   : {len(lifts)}")
        print(f"  mean lift               : {statistics.fmean(lifts)*100:+.2f}%")
        print(f"  median lift             : {statistics.median(lifts)*100:+.2f}%")
        print(f"  positive                : {sum(1 for x in lifts if x > 0)}/{len(lifts)}")
        print(f"  threshold               : +5.00%")
        print(f"  verdict (median)        : "
              f"{'PASSES' if statistics.median(lifts) > 0.05 else 'DOES NOT PASS'}")
        for frac in sorted(by_fraction):
            v = by_fraction[frac]
            print(f"    |S|/n={frac:<6.3f} n={len(v):<3} median {statistics.median(v)*100:+7.2f}%"
                  f"  positive {sum(1 for x in v if x > 0)}/{len(v)}")

    print()
    print("=== the patience rule, now that it is actually enabled ===")
    print(f"  {'graph':<18}{'k':>5}{'seeds taken':>13}{'stopped early':>15}"
          f"{'patience spread':>17}{'fill-budget spread':>20}")
    fired = 0
    for c in sorted(cells, key=lambda c: (c["graph"], c["budget"])):
        p = c["arms"].get("delta2_patience")
        s = c["arms"].get("delta2_sequential")
        if not p:
            continue
        fired += 1 if p["stopped_early"] else 0
        print(f"  {c['graph']:<18}{c['budget']:>5}{p['seed_count']:>8}/{c['budget']:<4}"
              f"{str(p['stopped_early']):>15}"
              f"{(p['spread'] if p else float('nan')):>17.2f}"
              f"{(s['spread'] if s else float('nan')):>20.2f}")
    print(f"  patience fired in {fired} of the cells above")

    print()
    print("=== cost, in cascades (state acquisition included) ===")
    print(f"  {'arm':<22}{'cells':>7}{'median cascades':>18}{'median target count':>21}")
    by_arm: dict[str, list] = {}
    for c in cells:
        for name, entry in c["arms"].items():
            by_arm.setdefault(name, []).append((entry["mc_cascades"], entry["spread"]))
    for name in sorted(by_arm):
        group = by_arm[name]
        print(f"  {name:<22}{len(group):>7}"
              f"{statistics.median(g[0] for g in group):>18.0f}"
              f"{statistics.median(g[1] for g in group):>21.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
