"""Render the shortlist-headroom diagnostic as per-configuration tables.

Four questions per configuration, in the order they matter:

1. **How much headroom is there at all?**  The MC reference candidate's marginal, and the spread of
   batch-1 marginals across the pool.  If the best-of-50 marginal is within the noise of the rest,
   there is nothing to learn at this configuration and no scoring function can help.
2. **Does each score put the reference candidate in its top-8?**  If not, the score is discarding the
   candidate an independent Monte-Carlo pass would have picked.
3. **What does each method's chosen candidate actually achieve** on the *independent* batch, with the
   paired 95% CI of its gain over the reference.  Signed, never truncated at zero.
4. **How does random screening compare?**  Its mean and quantiles over draws, kept separate from the
   Monte-Carlo uncertainty of a single draw.

Configurations are shown separately; nothing is pooled across graphs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("degree", "static_delta2", "state_delta2")


def fmt(value: float | None, width: int = 9, digits: int = 3) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    return f"{value:>{width}.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "shortlist_headroom.json")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"{args.input} not found")
        return 1
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    print("=" * 108)
    print("SHORTLIST HEADROOM DIAGNOSIS")
    print("=" * 108)
    print(f"  objective        : {payload.get('objective')}")
    print(f"  code version     : {payload.get('code_version')}")
    print(f"  stream check     : {payload.get('stream_check')}")
    print(f"  complete         : {payload.get('complete')}")
    print(f"  wall seconds     : {payload.get('wall_seconds'):.0f}")
    print()

    for block in payload.get("blocks", []):
        if block.get("skipped"):
            print(f"  {block['graph']}: SKIPPED — {block.get('reason')}")
            continue
        rows = block["candidate_rows"]
        b1_means = [r["batch1"]["mean"] for r in rows]
        ref = block["reference_candidate"]

        print("-" * 108)
        print(f"  {block['graph']}   |S| = {block['seed_size']}  (|S|/n = {block['seed_fraction']:.3f})"
              f"   |D| = {block['target_size']}   candidates = {block['candidate_count']}"
              f"   MC = {block['mc_per_batch']} per batch")
        print(f"    purpose: {block['why']}")
        print()

        # 1 -- headroom
        ordered = sorted(b1_means, reverse=True)
        second = ordered[1] if len(ordered) > 1 else float("nan")
        print(f"    HEADROOM (batch 1, MC = {block['mc_per_batch']})")
        print(f"      best-of-pool marginal      : {ref['batch1_summary']['mean']:+.3f} "
              f"+-{ref['batch1_summary']['se']:.3f}  (candidate {ref['candidate']})")
        print(f"      runner-up marginal         : {second:+.3f}   "
              f"gap = {ref['batch1_summary']['mean'] - second:+.3f}")
        print(f"      pool mean / sd             : {statistics.fmean(b1_means):+.3f} / "
              f"{statistics.pstdev(b1_means):.3f}")
        print(f"      negative-marginal share    : "
              f"{sum(1 for m in b1_means if m < 0)/len(b1_means)*100:.0f}%  "
              f"(descriptive only; no CI attached, so not a conclusion)")
        print(f"      independent re-check       : reference batch2 "
              f"{ref['batch2_summary']['mean']:+.3f} "
              f"(batch1 {ref['batch1_summary']['mean']:+.3f})")
        print(f"      reference rank by score    : degree {ref['degree_rank']}"
              f"/{block['candidate_count']}, "
              f"static delta2 {ref['static_delta2_rank']}, "
              f"state delta2 {ref['state_delta2_rank']}")
        print()

        # 2 + 3 -- shortlist membership and realised gain
        print(f"    METHODS (shortlist of {block['shortlist_size']}, chosen by batch 1, "
              f"scored on batch 2)")
        print(f"      {'method':<14}{'chosen':>8}{'in top8?':>10}{'batch2':>10}"
              f"{'gain vs ref':>13}{'95% CI':>22}")
        for label in METHODS:
            m = block["methods"].get(label)
            if not m:
                continue
            gain = m["gain_vs_reference_batch2"]
            ci = f"[{gain['ci95_low']:+.3f},{gain['ci95_high']:+.3f}]"
            print(f"      {label:<14}{m['chosen']:>8}"
                  f"{('yes' if m['contains_reference'] else 'NO'):>10}"
                  f"{m['chosen_batch2_summary']['mean']:>10.3f}"
                  f"{gain['mean']:>13.3f}{ci:>22}")

        rnd = block["methods"].get("random")
        if rnd:
            q = rnd["batch2_mean_quantiles"]
            print(f"      {'random':<14}{'-':>8}{'-':>10}"
                  f"{rnd['batch2_mean_over_repeats']:>10.3f}"
                  f"{'':>13}{'':>22}")
            print(f"        over {rnd['repeats']} draws: mean {rnd['batch2_mean_over_repeats']:+.3f}"
                  f"  min {q['min']:+.3f}  p05 {q['p05']:+.3f}  median {q['median']:+.3f}"
                  f"  p95 {q['p95']:+.3f}  max {q['max']:+.3f}")
            print(f"        median MC CI width within one draw: "
                  f"{rnd['median_mc_ci_width']:.3f}")
        print()

        # headline readings
        state = block["methods"].get("state_delta2", {})
        if state and rnd:
            verdict = ("WORSE than random screening"
                       if state["chosen_batch2_summary"]["mean"]
                       < rnd["batch2_mean_over_repeats"] else
                       "better than random screening")
            print(f"    state delta2 chosen marginal "
                  f"{state['chosen_batch2_summary']['mean']:+.3f} vs random mean "
                  f"{rnd['batch2_mean_over_repeats']:+.3f}  ->  {verdict}")
        if not state.get("contains_reference"):
            print(f"    state delta2 MISSES the reference candidate; degree "
                  f"{'hits' if block['methods']['degree']['contains_reference'] else 'misses'}, "
                  f"static delta2 "
                  f"{'hits' if block['methods']['static_delta2']['contains_reference'] else 'misses'}")
        print()

    print("=" * 108)
    print("  None of the negative-marginal shares above carries a confidence interval, so none is")
    print("  a conclusion.  The gains are signed and untruncated: the reference candidate can be")
    print("  overtaken on the independent batch, and that is reported as a positive gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
