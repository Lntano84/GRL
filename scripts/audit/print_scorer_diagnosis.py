"""Print the scorer diagnosis as a table.

Four columns answer the question "why does analytic screening not beat random pruning?":

``degree``
    Spearman of out-degree with the TRUE contracted marginal --- the zero-cost competitor.
``untgt``
    the closed form scored over every out-neighbour, i.e. the historical behaviour.
``targeted``
    the same score restricted to ``D``.
``hops1`` / ``hops3``
    the depth sensitivity of the targeted score.
``excl-act``
    the targeted score with already-positive targets removed.

If ``targeted`` is much worse than ``untgt``, restricting the score to the target set destroyed its
signal rather than sharpening it, and that alone explains the screening result.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "scorer_diagnosis.json")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"{args.input} not found")
        return 1
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    print(f"  {len(rows)} cells   objective: {payload.get('objective')}")
    print()
    header = ("graph", "|S|/n", "degree", "untgt", "targeted", "hops1", "hops3",
              "excl-act", "neg%", "|D| active")
    print("  " + "".join(f"{h:<11}" for h in header))
    for r in rows:
        values = (
            r["graph"], f"{r['fraction']:.2f}", f"{r['rho_degree']:+.3f}",
            f"{r['rho_untargeted_hops2']:+.3f}", f"{r['rho_targeted_hops2']:+.3f}",
            f"{r['rho_targeted_hops1']:+.3f}", f"{r['rho_targeted_hops3']:+.3f}",
            f"{r['rho_excluding_active_hops2']:+.3f}",
            f"{r['negative_marginal_share']*100:.0f}%",
            f"{r['share_targets_already_positive']*100:.0f}%",
        )
        print("  " + "".join(f"{str(v):<11}" for v in values))

    if rows:
        print()
        d_target = statistics.fmean(r["rho_targeted_hops2"] - r["rho_untargeted_hops2"]
                                    for r in rows)
        d_degree = statistics.fmean(r["rho_targeted_hops2"] - r["rho_degree"] for r in rows)
        print(f"  mean (targeted - untargeted) = {d_target:+.3f}"
              f"   {'targeting HURTS' if d_target < -0.05 else 'targeting helps' if d_target > 0.05 else 'no clear effect'}")
        print(f"  mean (targeted - degree)     = {d_degree:+.3f}"
              f"   {'the analytic score LOSES to free degree' if d_degree < 0 else 'analytic score ahead'}")
        act = statistics.fmean(r["share_targets_already_positive"] for r in rows)
        hit = statistics.fmean(r["top10_hitting_already_active"] for r in rows)
        print(f"  already-active targets: {act*100:.0f}% of |D|; top-10 pointing at one: {hit*100:.0f}%")
        print()
        print("  Reading: the analytic score only has to beat its competitors, and 'degree' is one")
        print("  of them at zero cost. Compare the columns against each other, not against 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
