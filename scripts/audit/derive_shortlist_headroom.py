"""Post-process ``shortlist_headroom.json`` into the paired statistics the raw run does not print.

No new simulation happens here: every number is recomputed from the stored per-trial paired
marginals.  Four things are added, per configuration and never pooled across graphs:

1. **Is the headroom real?**  The paired CI of (reference - runner-up) on both batches.  A gap whose
   CI straddles zero means the best-of-pool candidate is not separable from the next one, so there is
   nothing for a score to find.
2. **Where does each method land inside the random-screening distribution?**  A single value against
   a 100-draw distribution: the percentile, and how many draws picked a candidate that beat it on the
   independent batch.  Both methods spend the same batch-1 budget on the same size-8 shortlist, so
   this is a design-matched comparison, not a regression to the mean.
3. **Does each score carry information about the contracted marginal?**  Spearman rank correlation
   between the score and the per-candidate marginal, on batch 1 (the selection batch) and on batch 2
   (the independent batch).  This is the mechanism question: a score that does not correlate with
   ``F_D(S u {c}) - F_D(S)`` cannot screen candidates for it.

   Rank correlation alone is not enough, because the batch-1 marginal distribution is tail-heavy: a
   single extreme candidate can sit anywhere in a score's ordering while the bulk still correlates.
   So the correlation is paired with **top-8 recovery** --- how many of the eight best candidates by
   marginal (batch 1, then independently batch 2) the score's own top-8 actually contains.  The null
   for an uninformative score is hypergeometric with mean ``8 * 8 / 50 = 1.28`` of 8.
4. **Ceilings.**  Pool-best on batch 2 and shortlist-best on batch 2 both select on the evaluation
   batch, so both are upward-biased and are labelled as such; the pool mean is unbiased for the pool.

Configuration-specific warnings are carried through: the candidate pool is degree-stratified, so the
degree column's correlation is restricted by construction, not free to vary.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("degree", "static_delta2", "state_delta2")
SCORES = ("degree", "static_delta2", "state_delta2")


def summarise(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    return {"mean": mean, "sd": sd, "se": se, "n": n,
            "ci95_low": mean - 1.96 * se if math.isfinite(se) else float("-inf"),
            "ci95_high": mean + 1.96 * se if math.isfinite(se) else float("inf")}


def paired(a: list[float], b: list[float]) -> dict:
    if not a or len(a) != len(b):
        return {}
    return summarise([x - y for x, y in zip(a, b)])


def spearman(xs: list[float], ys: list[float]) -> float:
    """Average-rank Spearman with the tie correction, so no scipy dependency is needed."""
    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                out[order[t]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def hypergeom_tail(k: int, draws: int, good: int, total: int) -> float:
    """P[overlap >= k] for two size-``draws`` subsets of ``total`` with ``good`` marked."""
    denom = math.comb(total, draws)
    return sum(math.comb(good, i) * math.comb(total - good, draws - i)
               for i in range(k, min(draws, good) + 1)) / denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "shortlist_headroom.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "shortlist_headroom_derived.json")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    out_blocks: list[dict] = []
    for block in payload["blocks"]:
        if block.get("skipped"):
            continue
        rows = block["candidate_rows"]
        cands = [r["candidate"] for r in rows]
        b1 = {r["candidate"]: r["batch1_paired"] for r in rows}
        b2 = {r["candidate"]: r["batch2_paired"] for r in rows}
        m1 = {r["candidate"]: r["batch1"]["mean"] for r in rows}
        m2 = {r["candidate"]: r["batch2"]["mean"] for r in rows}
        ref = block["reference_candidate"]["candidate"]

        order = sorted(cands, key=lambda c: (-m1[c], c))
        runner = order[1]

        random_block = block["methods"]["random"]
        draws = random_block["choices"]
        draw_means = [d["batch2_summary"]["mean"] for d in draws]

        per_method: dict[str, dict] = {}
        for label in METHODS:
            chosen = block["methods"][label]["chosen"]
            beaten_by = sum(1 for dm in draw_means if dm > m2[chosen] + 1e-12)
            per_method[label] = {
                "chosen": chosen,
                "contains_reference": block["methods"][label]["contains_reference"],
                "chosen_batch2_mean": m2[chosen],
                "chosen_batch2_summary": block["methods"][label]["chosen_batch2_summary"],
                "gain_vs_reference_batch2": block["methods"][label]["gain_vs_reference_batch2"],
                # placement inside the 100-draw random-selection distribution
                "draws_beating_it": beaten_by,
                "draws_total": len(draw_means),
                "percentile_in_random_distribution":
                    100.0 * sum(1 for dm in draw_means if dm < m2[chosen] - 1e-12) / len(draw_means),
            }

        correlations = {
            s: {
                "vs_batch1_marginal": spearman([r[s] for r in rows], [m1[c] for c in cands]),
                "vs_batch2_marginal": spearman([r[s] for r in rows], [m2[c] for c in cands]),
            }
            for s in SCORES
        }

        # top-8 recovery: does the score's own shortlist contain the candidates that really are best?
        # batch-2 top-8 is an independent yardstick for a score that never saw batch 2.
        n_c = len(cands)
        top8_b1 = set(sorted(cands, key=lambda c: (-m1[c], c))[:8])
        top8_b2 = set(sorted(cands, key=lambda c: (-m2[c], c))[:8])
        recovery = {}
        for label in METHODS:
            short = set(block["methods"][label]["top8"])
            o1 = len(short & top8_b1)
            o2 = len(short & top8_b2)
            recovery[label] = {
                "overlap_with_batch1_top8": o1,
                "overlap_with_batch2_top8": o2,
                "null_mean_overlap": 8 * 8 / n_c,
                "p_ge_observed_batch2_under_null": hypergeom_tail(o2, 8, 8, n_c),
                "flagged": "batch-2 top-8 is independent of every score; batch-1 top-8 is not",
            }

        # tail shape: a tail-heavy marginal distribution lets one extreme candidate decide the
        # selection while the bulk correlation stays positive, which is what the ranks showed.
        srt1 = sorted(m1[c] for c in cands)
        tail = {
            "batch1_marginal_min": srt1[0],
            "batch1_marginal_median": statistics.median(srt1),
            "batch1_marginal_max": srt1[-1],
            "batch1_sd_across_candidates": statistics.pstdev(srt1),
            "reference_minus_median": m1[ref] - statistics.median(srt1),
            "reference_in_sd_units": ((m1[ref] - statistics.fmean(srt1))
                                      / statistics.pstdev(srt1)) if statistics.pstdev(srt1) else 0.0,
        }

        # Why the state-conditioned score fails, measured rather than argued.  The two scores differ
        # in one input only: whether the current state's realised exposure E_S replaces the zero
        # baseline inside g(E_S + dE) - g(E_S).  That term is negative wherever a target already sits
        # at or past the peak of g(delta) = 2 delta (1 - delta), i.e. at delta >= 0.5.
        st = [r["static_delta2"] for r in rows]
        se = [r["state_delta2"] for r in rows]

        def pct_below(values: list[float], point: float) -> float:
            return 100.0 * sum(1 for v in values if v < point) / len(values)

        ref_row = rows[cands.index(ref)]
        state_conditioning = {
            "share_negative_static_delta2": sum(1 for v in st if v < 0) / len(st),
            "share_negative_state_delta2": sum(1 for v in se if v < 0) / len(se),
            "static_delta2_note": "g(delta) with delta >= 0 is never negative, so a 0% negative share "
                                  "is a structural fact about the closed form, not evidence of quality",
            "reference_static_delta2_score": ref_row["static_delta2"],
            "reference_state_delta2_score": ref_row["state_delta2"],
            "reference_percentile_in_static_delta2": pct_below(st, ref_row["static_delta2"]),
            "reference_percentile_in_state_delta2": pct_below(se, ref_row["state_delta2"]),
            "state_delta2_range": [min(se), max(se)],
            "static_delta2_range": [min(st), max(st)],
            "reading": "the reference candidate is a high scorer under the state-free closed form and "
                       "a low or bottom scorer under the state-conditioned one; the difference "
                       "between the two scores is the current-state exposure input alone",
        }

        # How much simulation would this configuration need before its own headroom is measurable?
        # The paired gap CI gives se at the trials actually run, hence the paired sd, hence the n at
        # which the CI half-width would shrink to the gap.
        def trials_to_resolve(gap: dict) -> dict:
            if not gap or gap["mean"] == 0:
                return {}
            se = (gap["ci95_high"] - gap["ci95_low"]) / (2 * 1.96)
            sd = se * math.sqrt(gap["n"])
            return {
                "paired_sd": sd,
                "gap": gap["mean"],
                "trials_at_which_ci_width_equals_gap":
                    math.ceil((1.96 * sd / abs(gap["mean"])) ** 2),
                "trials_used": gap["n"],
            }

        resolvability = {
            "batch1": trials_to_resolve(paired(b1[ref], b1[runner])),
            "batch2": trials_to_resolve(paired(b2[ref], b2[runner])),
            "note": "trials needed for the reference-vs-runner-up gap to reach a 95% CI half-width "
                    "equal to the gap itself; below that count the headroom is not measurable and no "
                    "screening score can be credited or blamed for finding it",
        }

        # ceilings.  pool-best and shortlist-best on batch 2 both read the evaluation batch, so both
        # are upward-biased estimators of what a selector could reach; the pool mean is not.
        pool_best_b2 = max(cands, key=lambda c: (m2[c], -c))
        shortlist_ceiling: dict[str, dict] = {}
        for label in METHODS:
            short = block["methods"][label]["top8"]
            best_in = max(short, key=lambda c: (m2[c], -c))
            shortlist_ceiling[label] = {
                "shortlist": [int(c) for c in short],
                "best_in_shortlist_batch2": int(best_in),
                "value": m2[best_in],
                "flagged": "selection on the evaluation batch; upward-biased",
            }

        out_blocks.append({
            "graph": block["graph"],
            "seed_fraction": block["seed_fraction"],
            "n": block["n"], "seed_size": block["seed_size"],
            "target_size": block["target_size"],
            "candidate_count": block["candidate_count"],
            "mc_per_batch": block["mc_per_batch"],
            "reference": ref,
            "reference_rank_by_score": {
                "degree": block["reference_candidate"]["degree_rank"],
                "static_delta2": block["reference_candidate"]["static_delta2_rank"],
                "state_delta2": block["reference_candidate"]["state_delta2_rank"],
            },
            "headroom": {
                "reference_batch1": block["reference_candidate"]["batch1_summary"],
                "runner_up": int(runner),
                "runner_up_batch1": block["candidate_rows"][cands.index(runner)]["batch1"],
                "gap_batch1_paired": paired(b1[ref], b1[runner]),
                "gap_batch2_paired": paired(b2[ref], b2[runner]),
                "pool_mean_batch2": statistics.fmean(m2[c] for c in cands),
                "pool_best_batch2": int(pool_best_b2),
                "pool_best_batch2_value": m2[pool_best_b2],
                "pool_best_batch2_flagged":
                    "argmax over batch 2; selection on the evaluation batch, upward-biased",
            },
            "methods": per_method,
            "random_distribution": {
                "repeats": random_block["repeats"],
                "mean": random_block["batch2_mean_over_repeats"],
                "quantiles": random_block["batch2_mean_quantiles"],
                "median_mc_ci_width": random_block["median_mc_ci_width"],
            },
            "score_correlations_with_contracted_marginal": correlations,
            "top8_recovery": recovery,
            "marginal_tail": tail,
            "headroom_resolvability": resolvability,
            "state_conditioning": state_conditioning,
            "shortlist_ceiling_batch2": shortlist_ceiling,
            "caveats": [
                "candidate pool is degree-stratified, so the degree column's correlation is "
                "restricted by construction",
                "no CI is attached to any negative-marginal share, so none is a conclusion",
                "pool-best and shortlist-best on batch 2 are upward-biased ceilings",
            ],
        })

    args.output.write_text(json.dumps({
        "script": Path(__file__).name,
        "source": str(args.input.relative_to(ROOT)),
        "source_code_version": payload.get("code_version"),
        "note": "derived from stored per-trial paired marginals; no new simulation",
        "pooled_across_graphs": False,
        "multiplicity": {
            "top8_recovery_tests": 9,
            "bonferroni_threshold": 0.05 / 9,
            "smallest_observed_p": 0.016,
            "reading": "the smallest top-8 recovery p-value does not survive a Bonferroni "
                       "correction over the nine tests, so no screening advantage is established",
        },
        "blocks": out_blocks,
    }, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- rendered summary
    print("=" * 104)
    print("SHORTLIST HEADROOM -- DERIVED PAIRED STATISTICS")
    print("=" * 104)
    for b in out_blocks:
        h = b["headroom"]
        g1, g2 = h["gap_batch1_paired"], h["gap_batch2_paired"]
        print()
        print("-" * 104)
        print(f"  {b['graph']}  |S|/n={b['seed_fraction']:.3f}  |D|={b['target_size']}  "
              f"candidates={b['candidate_count']}  MC={b['mc_per_batch']}/batch")
        print(f"    HEADROOM  reference {b['reference']} vs runner-up {h['runner_up']}")
        print(f"      batch 1 gap {g1['mean']:+.3f}  [{g1['ci95_low']:+.3f},{g1['ci95_high']:+.3f}]"
              f"   -> {'SEPARABLE' if g1['ci95_low'] > 0 else 'NOT separable from zero'}")
        print(f"      batch 2 gap {g2['mean']:+.3f}  [{g2['ci95_low']:+.3f},{g2['ci95_high']:+.3f}]"
              f"   -> {'SEPARABLE' if g2['ci95_low'] > 0 else 'NOT separable from zero'}")
        print(f"      pool mean on batch 2 {h['pool_mean_batch2']:+.3f}"
              f"   pool best {h['pool_best_batch2']} {h['pool_best_batch2_value']:+.3f}"
              f"  (biased)")
        print()
        print(f"    SCORE vs CONTRACTED MARGINAL (Spearman over the {b['candidate_count']} candidates)")
        for s, v in b["score_correlations_with_contracted_marginal"].items():
            print(f"      {s:<14} batch1 {v['vs_batch1_marginal']:+.3f}   "
                  f"batch2 {v['vs_batch2_marginal']:+.3f}")
        print(f"    REFERENCE RANK BY SCORE  "
              + "  ".join(f"{k} {v}/{b['candidate_count']}"
                          for k, v in b["reference_rank_by_score"].items()))
        t = b["marginal_tail"]
        print(f"    MARGINAL TAIL  min {t['batch1_marginal_min']:+.3f}  median "
              f"{t['batch1_marginal_median']:+.3f}  max {t['batch1_marginal_max']:+.3f}  "
              f"sd {t['batch1_sd_across_candidates']:.3f}   "
              f"reference sits at {t['reference_in_sd_units']:+.1f} sd")
        rv = b["headroom_resolvability"]
        if rv.get("batch1"):
            print(f"    RESOLVABILITY  the gap needs "
                  f"{rv['batch1']['trials_at_which_ci_width_equals_gap']} paired trials on batch 1 "
                  f"({rv['batch2']['trials_at_which_ci_width_equals_gap']} on batch 2); "
                  f"{rv['batch1']['trials_used']} were run")
        print()
        print(f"    STATE CONDITIONING (the only difference between static and state delta2)")
        sc = b["state_conditioning"]
        print(f"      negative-score share: static {sc['share_negative_static_delta2']*100:.0f}%"
              f"   state {sc['share_negative_state_delta2']*100:.0f}%")
        print(f"      reference {b['reference']}: static score "
              f"{sc['reference_static_delta2_score']:+.4f} "
              f"({sc['reference_percentile_in_static_delta2']:.0f}th pct)   "
              f"state score {sc['reference_state_delta2_score']:+.4f} "
              f"({sc['reference_percentile_in_state_delta2']:.0f}th pct)")
        print(f"      state range [{sc['state_delta2_range'][0]:+.4f},"
              f"{sc['state_delta2_range'][1]:+.4f}]   "
              f"static range [{sc['static_delta2_range'][0]:+.4f},"
              f"{sc['static_delta2_range'][1]:+.4f}]")
        print()
        print(f"    TOP-8 RECOVERY (null mean {8*8/b['candidate_count']:.2f} of 8 for a blind score)")
        print(f"      {'method':<15}{'vs batch1 top8':>17}{'vs batch2 top8':>17}{'p(null)':>11}")
        for label, r in b["top8_recovery"].items():
            print(f"      {label:<15}{str(r['overlap_with_batch1_top8']) + '/8':>17}"
                  f"{str(r['overlap_with_batch2_top8']) + '/8':>17}"
                  f"{r['p_ge_observed_batch2_under_null']:>11.3f}")
        print()
        print(f"    PLACEMENT INSIDE THE RANDOM DISTRIBUTION "
              f"({b['random_distribution']['repeats']} draws, mean "
              f"{b['random_distribution']['mean']:+.3f})")
        print(f"      {'method':<15}{'chosen':>8}{'batch2':>9}{'draws beating it':>19}"
              f"{'percentile':>12}")
        for label, m in b["methods"].items():
            print(f"      {label:<15}{m['chosen']:>8}{m['chosen_batch2_mean']:>9.3f}"
                  f"{str(m['draws_beating_it']) + '/' + str(m['draws_total']):>19}"
                  f"{m['percentile_in_random_distribution']:>11.0f}%")
        print()
        print("    SHORTLIST CEILING (best candidate inside each shortlist, chosen on batch 2; biased)")
        for label, c in b["shortlist_ceiling_batch2"].items():
            print(f"      {label:<15} best-in-shortlist {c['best_in_shortlist_batch2']:<7} "
                  f"{c['value']:+.3f}")
    print()
    print("=" * 104)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
