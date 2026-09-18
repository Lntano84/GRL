# Go/No-Go: the answer, and what it decides

> 2026-09-18. Covers the five items of the Go/No-Go plan. Every number is read from
> `docs/results/*.json`; none is transcribed. Reproduce with the commands at the end.
>
> **Verdict: NO-GO for the sample-efficiency thesis.** Detail below, per item.

---

## The pre-registered rule

`docs/DECISIONS.md`, 2026-09-15, set a gate with a stated consequence:

> Gate 1 (by 2026-10-13): (a) the Spearman correlation between true marginal gain and degree under
> overexposure collapses (≈0 or negative); (b) **sequential decision beats the static optimum by a
> material margin (>5%)**. If either fails, **DASFAA is dropped and the target moves to CIKM 2027.**

---

## Item 1 — sequential vs static (Gate 1(b)): **FAILS**

49 cells, 4 graphs, 3 seed fractions, 2 candidate pools, 2 random seeds, both weight normalisations.
**The largest lift of any sequential arm over the best static arm is +2.64%, against a +5%
requirement.**

| graph \| normalisation | cells | lift range | median | positive |
| --- | --- | --- | --- | --- |
| Congress-Twitter `sum_to_one` | 12 | **−2.19% … −0.34%** | −0.94% | **0/12** |
| Congress-Twitter `clip_to_one` | 12 | +0.03% … +1.65% | +1.16% | 12/12 |
| email-Eu-core `sum_to_one` | 12 | −9.69% … +2.12% | −0.17% | 5/12 |
| email-Eu-core `clip_to_one` | 12 | −0.79% … +0.39% | +0.02% | 6/12 |
| ca-GrQc `sum_to_one` | 1 | +2.64% | +2.64% | 1/1 |

**The sign of the result is set by the weight normalisation**, which is the sharpest consequence of
confound P1-3.8 and worse than the 50× exposure-scale figure we had. Same graph, same arms, same
seeds, same pools: `sum_to_one` gives 12/12 negative, `clip_to_one` gives 12/12 positive.

And the two are not equally valid for this question. `clip_to_one` only scales *down* nodes over 1;
Congress-Twitter's raw weights have `max total 1.648` (one node over) and `min nonzero 0.0011`, so
almost every weight keeps its raw tiny value, exposures stay below the windows, and spreads collapse
from 155–208 to 44–135. **The configuration where sequential "wins" is the one where the model is
barely over-exposed at all.** The honest reading is the `sum_to_one` column.

**The sequential arm is not strawmanned.** `delta2_patience` additionally uses observation to decide
*how many* seeds to take — a lever a static arm structurally lacks, and the mechanism the paper's own
framing proposes. It **never fires**: `stopped_early = False` and the seed count equals the budget in
every cell, because the realised spread is weakly increasing in the number of seeds, so a patience
rule on it never triggers.

---

## Item 2 — quality–cost curves: built, and the reference is not a ceiling

**Complete: 18 of 18 cells** (3 graphs × k ∈ {5, 10, 20} × 2 seeds, 85 minutes of compute). Arms:
`mc_greedy_mc50` (reference), `degree_static`, `delta2_static`, `delta2_sequential`,
`delta2_patience`, `random_pruning`, `selective_analytic`, `adaptive_selective`.

| arm | median cascade saving | median max-loss (target nodes) | worst max-loss |
| --- | --- | --- | --- |
| `delta2_static` | **100%** | **3.18** | 114.94 |
| `degree_static` | **100%** | 3.98 | 105.52 |
| `random_pruning` | 75% | 4.53 | 58.53 |
| `selective_analytic` | 74% | 17.40 | 104.19 |
| `adaptive_selective` | 71% | 17.40 | 104.19 |
| `delta2_sequential` | 99% | **34.07** | 169.86 |
| `delta2_patience` | 99% | **34.07** | 169.86 |

"max-loss" is the upper end of the paired 95% CI on `reference − method`, in target nodes: **the
largest loss the data still permits.** It is reported instead of a verdict because it survives any
choice of tolerance. The worst-case column is large because ca-GrQc is large — the reference there is
also the noisiest, see below.

**Three things follow.**

1. **The cost half of the criterion (≥30% fewer cascades) is satisfied by everything**, including
   every zero-cost arm. It does not discriminate.
2. **The free arms are also the most accurate.** `degree_static` costs 0 cascades and
   `delta2_static` costs 25 (one state read); their median worst-case loss is 3.2–4.0 target nodes,
   while the expensive shortlist arms are 4–5× worse at 71–75% of the cost. **There is no quality to
   buy with cascades at these budgets.**
3. **The sequential arms are the worst of the set** — median worst-case loss 34.07 nodes, roughly
   10× the free static arms, while costing 99% less than the reference only because they use a
   tiny state read. On the three graphs in this design the median lift is **−9.61%** with **0 of 18
   cells positive**.

And a fourth, which constrains every claim of this form:

4. **The Monte-Carlo reference is not a quality ceiling.** In **14 arm-cells** a cheaper arm beat it
   on spread — e.g. `delta2_static` at 25 cascades beating the reference's 9,750 on Congress-Twitter
   k=5, and `random_pruning` beating it by +2.02 at 4,500 against 18,250. This is expected at a
   finite MC budget, and it is why the arm's name carries `--reference-mc` and why **no claim of the
   form "within X of optimal" may be made against it.** It also means the "quality loss vs the
   reference" framing is itself shaky here: the reference is a noisy heuristic, not an optimum.

---

## Item 3 — coverage: 4 graphs, 3 fractions, 2 pools, 2 seeds, 2 normalisations

Satisfied for item 1 (49 cells: 4 graphs × 3 fractions × 2 pools × 2 seeds × 2 normalisations) and
for item 2 (18/18 cells: 3 graphs × 3 budgets × 2 seeds). Every cell records its contract, its
stopping rule, its cost breakdown and its pool/budget ratio. Three structural limits are recorded
rather than hidden:

* At `|S|/n = 0.20` on Congress-Twitter the pool had to be the whole eligible set, so the pool draws
  are not independent replicates — flagged as `pool_covers_all_eligible`.
* Cells with `k > 1600` are skipped with the reason recorded, because the sequential arm's cost is
  `O(k · state_mc)` and a 40% cell on NetHEPT would be hours of compute for one number.
* The quality–cost design deliberately uses small `k`, because the full-pool Monte-Carlo reference
  costs `O(k · |pool| · MC)`: at `k = 95`, `|pool| = 380`, `MC = 25` that is ~900,000 cascades for
  ONE cell. **The reference cannot be run in the saturated regime at all**, which is itself the
  sharpest statement of why a cheaper method is wanted — and simultaneously why the quality–cost
  curve can only be shown in the mild regime.
* ca-GrQc and NetHEPT were not finished for item 1: 49 cells cover Congress-Twitter and
  email-Eu-core fully plus one ca-GrQc cell. The remaining ca-GrQc cells cost 6–14 min each and the
  NetHEPT 40% cells are out of budget. The artifact says which cells are present.

---

## Item 4 — `sum_to_one` vs `clip_to_one`: the strategies differ, and it matters

The first version of the runner applied the strategy to an **already-normalised** graph, because the
obvious loader normalises internally. Every strategy was a no-op and the two columns were
byte-identical; the identical columns were the tell. Fixed, and the raw profiles are now reported
*before* a sweep runs on them (`scripts/audit/report_raw_weight_profiles.py`):

| graph | n | raw max in-total | min nonzero | nodes over 1 |
| --- | --- | --- | --- | --- |
| ca-HepPh | 12008 | **4.9100** | 0.0100 | 423 |
| Wiki-Vote | 7115 | 4.5700 | 0.0100 | 176 |
| Facebook | 4039 | 2.5100 | 0.0100 | 131 |
| email-Eu-core | 1005 | 2.1200 | 0.0100 | 30 |
| Congress-Twitter | 475 | 1.6483 | 0.0011 | 1 |
| ca-GrQc | 5242 | 0.8100 | 0.0100 | 0 |

Most graphs **violate the model's `Σω ≤ 1` requirement in raw form** (ca-HepPh by 4.9×), so
`as_given` would refuse them — correctly. And the effect is not subtle: Congress-Twitter at
`|S|/n = 0.10`, static degree, gives **174.97 under `sum_to_one` and 65.17 under `clip_to_one`**. A
2.7× difference in spread from a preprocessing choice that the earlier work did not report.

---

## Item 5 — the criterion: restated, because the relative form cannot be evaluated

The plan's "quality loss ≤ 1%" is **not evaluable in this regime**, and neither is an arbitrary "one
node per thousand". Calibrated on Congress-Twitter at k=5, MC=300
(`docs/results/gate_calibration.json`):

| contrast | paired SE | CI95 width | MC needed to resolve `\|D\|/1000` |
| --- | --- | --- | --- |
| reference vs `delta2_sequential` | 2.492 | 9.770 | **793,265** |
| reference vs degree | 1.247 | 4.887 | 198,502 |
| reference vs `delta2_static` | 0.795 | 3.118 | 80,770 |

Resolving one node per thousand needs ~790,000 paired trials per contrast against the 300 used — a
factor of ~2600. **One percent of a near-zero marginal is below the noise floor of any affordable
experiment.**

The replacement is an absolute tolerance in target nodes with a paired CI and three outcomes —
`PASS` / `FAIL` / `UNDECIDED`, where `UNDECIDED` means the CI *straddles* the tolerance and the
experiment lacks the power to decide. The number to publish is **`max_plausible_loss`**, the CI's
upper end, which does not depend on a tolerance someone chose.

---

## What this decides, and the three honest responses

Items 1 and 2 together say: **a sequential, state-conditioned, sample-efficient method is not needed
here.** It does not beat a static ranking by the margin the project required, it costs 24× more than
the static analytic arm when it re-ranks, the stopping lever that was its strongest argument never
fires, and the free heuristics already match an expensive Monte-Carlo reference at 100% saving.

Per the pre-registered rule, **(b) fails and the target moves to CIKM 2027**. That is the PI's call,
and the three responses available should be chosen deliberately:

1. **Apply the rule as written.** Retarget.
2. **Keep DASFAA with a different thesis.** Two contributions do not depend on sequential selection
   winning: the coverage result (`sections/coverage.tex`, exact arithmetic) and the measurement
   protocol (`sections/protocol.tex`). The weight-normalisation finding above is a genuinely new
   result that belongs to the protocol contribution and is arguably stronger than anything the
   sample-efficiency framing had.
3. **Re-scope to a conditional claim.** The lift tracks how misleading degree is — email-Eu-core
   (`ρ_degree = −0.236`) reaches +2.1% at 20% and moves monotonically with the fraction (−9.7% at
   5%, ≈0% at 10%, +2.1% at 20%); ca-GrQc (`ρ_degree = −0.583`) gives +2.64%; Congress-Twitter
   (`ρ_degree = −0.047`) is negative everywhere. A 2% effect predicted by `ρ_degree` is defensible
   but it is not the claim the project set out to make, and it needs the remaining ca-GrQc and
   NetHEPT cells finished first. Note the tension with item 2: the same sequential arm that is
   +2.1% at 20% on email-Eu-core is **−9.6% median in the quality–cost design**, where `k` is small
   — so the conditional claim is specifically about the saturated regime and must be stated that
   way, not as a general property of the method.

**What must not happen is reporting the `clip_to_one` column as the result.** It has the right sign
and it is the configuration in which the model is least over-exposed.

---

## Reproduce

```bash
# raw weights, before any strategy is applied
python scripts/audit/report_raw_weight_profiles.py

# what the experiment can resolve, before the gate is set
python scripts/audit/calibrate_gate_tolerance.py --graphs congress_twitter --budgets 5

# Gate 1(b)
python scripts/experiments/go_no_go.py \
  --graphs congress_twitter email_eu_core ca_grqc nethept \
  --fractions 0.05 0.10 0.20 --pool-draws 2 --seeds 20260917 20260918 \
  --normalisations sum_to_one clip_to_one \
  --arms delta2_sequential delta2_patience degree_static delta2_static \
  --eval-mc 200 --state-mc 25 --max-seeds 1600 --resume \
  --output docs/results/gate1b_sequential_vs_static.json

# quality-cost, including the random-pruning control
python scripts/experiments/go_no_go.py \
  --graphs congress_twitter email_eu_core ca_grqc --budgets 5 10 20 \
  --pool-size 40 --seeds 20260917 20260918 --normalisations sum_to_one \
  --reference-mc 50 --eval-mc 200 --state-mc 25 --shortlist 8 --resume \
  --output docs/results/quality_cost.json
python scripts/audit/analyse_quality_cost.py
```
