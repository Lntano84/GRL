# Candidate-screening headroom on three fixed configurations

> **Correction, 2026-09-19.**  §6 below ("Why the state-conditioned score fails") is arithmetically
> correct but was read as saying the state *information* is harmful.  That reading is not supported by
> this document's data and is now refuted at ca-GrQc on the objective itself.  The companion
> verification `docs/GRQC_RANK_REVERSAL.md` reproduces this document's scores bit-exactly and measures
> what §6 did not: the two candidates the reversal swaps are `+0.104 [−0.194, +0.403]` apart at 6000
> fixed trials, i.e. **not separable**.  This document establishes a **rank reversal in the score**,
> not a loss in the objective.  Read §6 together with that correction.  Nothing below has been altered.
>
> 2026-09-19.  This is the diagnostic requested after the corrected validation
> (`docs/VALIDATION_TARGETED.md`) showed that state-aware sequential screening does not stably beat
> static screening on the contracted objective.  It answers one question and nothing else: **at three
> fixed configurations, is there anything for a scoring function to find, and does the current
> analytic score find it?**
>
> No GNN was trained.  Stage A and Stage B were not re-run.  Nothing here pools across graphs, and no
> cross-graph average is reported or implied.

---

## 1. What was run

Three configurations, fixed in advance, 50 candidates each, one shortlist of 8:

| configuration | why this one | n | \|S\| | \|S\|/n | \|D\| |
|---|---|---|---|---|---|
| `congress_twitter` | where the current score performs poorly | 475 | 5 | 1.1% | 95 |
| `email_eu_core` | the existing positive ranking signal | 1005 | 50 | 5.0% | 201 |
| `ca_grqc` | the target-aware score's improvement | 5242 | 52 | 1.0% | 1048 |

Four screening methods, all spending the same budget:

- **`degree`** — out-degree of the candidate.  No cascades.
- **`static_delta2`** — the two-hop closed form `exposure_scores_delta` evaluated with an all-zero
  exposure baseline and no existing seeds, restricted to `D`.  Purely structural; no cascades.
- **`state_delta2`** — the same closed form evaluated on the realised mean exposure of the *current*
  seed set.  This is the paper's state-conditioned score.
- **`random`** — 100 independent draws of 8 candidates from the same 50, argmax by batch-1 marginal
  inside each draw.  Same budget, same shortlist size: a design-matched control, not a strawman.

Two independent Monte-Carlo batches of 1000 trials each:

- **batch 1** is used for selection only;
- **batch 2** is used for evaluation only and never touches selection;
- the state observation has its own third stream.

Disjointness was verified before the run, not after:

```
{"trials": 1000, "overlaps": {"state_batch1": 0, "state_batch2": 0, "batch1_batch2": 0},
 "disjoint": true}
```

Every number is `|P_final(S ∪ {c}) ∩ D| − |P_final(S) ∩ D|` — the **contracted** objective, end-state
counting, signed and untruncated.  The oracle keeps its level check (`[0, |D|]`) separate from its
marginal check (`[−|D|, |D|]`) precisely so that a negative marginal cannot be silently clipped, and
this diagnostic repeats both checks inline on every one of the 306,000 runs.  The reference is
labelled a **Monte-Carlo reference candidate**, never a true optimum: it is the batch-1 argmax of the
pool and it can be overtaken on batch 2, and where it is overtaken that is reported as a positive gain.

## 2. Provenance

| item | value |
|---|---|
| script | `scripts/audit/diagnose_shortlist_headroom.py` |
| derived statistics | `scripts/audit/derive_shortlist_headroom.py` |
| code version | `75b0635` |
| artifact | `docs/results/shortlist_headroom.json` |
| derived artifact | `docs/results/shortlist_headroom_derived.json` |
| renders | `docs/results/shortlist_headroom_render.txt`, `shortlist_headroom_derived.txt` |
| wall time | 3153 s (52.6 min) |
| `complete` | `true` |
| objective | `F_D(S) = E[|P_final(S) ∩ D|]` |
| stream namespaces | state `900_000`, batch 1 `1_100_000`, batch 2 `1_300_000` |

Simulation counts.  Per configuration: `(1 base + 50 candidates) × 1000 trials × 2 batches =
102,000` cascade runs, plus `2000` state-observation cascades.  Three configurations:
**306,000 marginal runs + 6,000 state runs**.  Split by configuration: `congress_twitter` 322 s,
`email_eu_core` 678 s, `ca_grqc` 2117 s.

```
python scripts/audit/diagnose_shortlist_headroom.py \
  --configs congress_twitter:0.01 email_eu_core:0.05 ca_grqc:0.01 \
  --mc 1000 --candidates 50 --shortlist 8 --random-repeats 100 --pool-size 120 \
  --output docs/results/shortlist_headroom.json
```

The candidate pool is degree-stratified (120 eligible nodes, first 50 used), so the `degree` column's
rank correlation is restricted by construction and cannot be read as a free estimate.

---

## 3. Configuration 1 — `congress_twitter`, `|S|/n = 1.1%`

| reading | value |
|---|---|
| MC reference candidate | 81: batch 1 **+0.321**, batch 2 **+0.317** |
| runner-up | 172: batch 1 +0.255 |
| **headroom gap** | batch 1 **+0.066 [−0.144, +0.276]** → **not separable** |
| | batch 2 **−0.170 [−0.411, +0.071]** → **not separable** |
| pool mean / sd on batch 2 | +0.166 / 0.094 |
| marginal shape | min −0.068, median +0.060, max +0.321; reference at **+2.6 sd** |
| negative-marginal share | 26% *(descriptive; no CI attached, so not a conclusion)* |
| resolvability | gap needs **10,079** paired trials on batch 1; 1000 were run |

| method | chosen | reference in top-8? | batch 2 | gain vs reference | 95% CI |
|---|---|---|---|---|---|
| `degree` | 172 | **no** | +0.487 | **+0.170** | [−0.071, +0.411] |
| `static_delta2` | 172 | **no** | +0.487 | **+0.170** | [−0.071, +0.411] |
| `state_delta2` | 28 | **no** | +0.277 | **−0.040** | [−0.265, +0.185] |
| `random` (100 draws) | — | — | +0.325 | min +0.103, p05 +0.153, median +0.317, p95 +0.487 | — |

| placement inside the random distribution | chosen | batch 2 | draws beating it | percentile |
|---|---|---|---|---|
| `degree` | 172 | +0.487 | **0/100** | 86% |
| `static_delta2` | 172 | +0.487 | **0/100** | 86% |
| `state_delta2` | 28 | +0.277 | **65/100** | 31% |

Score quality against the contracted marginal (Spearman over 50 candidates):

| score | vs batch-1 marginal | vs batch-2 marginal | reference rank | top-8 recovery vs batch-2 top-8 |
|---|---|---|---|---|
| `degree` | +0.239 | **+0.410** | 14/50 | 4/8 |
| `static_delta2` | +0.350 | +0.212 | 10/50 | 2/8 |
| `state_delta2` | **−0.202** | **−0.082** | 40/50 | 1/8 |

Reading.  There is no measurable headroom: the batch-1 argmax is not separable from the runner-up, and
on batch 2 the ordering actually inverts.  `degree` and `static_delta2` nevertheless land on a
candidate that no random draw beat — but that candidate is the *runner-up*, and its +0.170 advantage
over the reference has a CI straddling zero.  `state_delta2` is the only method that is worse than
random screening here, and it is the only score whose rank correlation with the contracted marginal is
negative on both batches.

## 4. Configuration 2 — `email_eu_core`, `|S|/n = 5.0%`

| reading | value |
|---|---|
| MC reference candidate | 733: batch 1 **+0.040 ±0.057**, batch 2 **+0.040** |
| runner-up | 219: batch 1 +0.026 |
| **headroom gap** | batch 1 **+0.014 [−0.117, +0.145]** → **not separable** |
| | batch 2 **−0.003 [−0.117, +0.111]** → **not separable** |
| pool mean / sd on batch 2 | **+0.001** / 0.017 |
| marginal shape | min −0.060, median **+0.000**, max +0.040; reference at **+2.6 sd** |
| negative-marginal share | 32% *(descriptive; no CI attached, so not a conclusion)* |
| resolvability | gap needs **87,922** paired trials on batch 1; 1000 were run |

| method | chosen | reference in top-8? | batch 2 | gain vs reference | 95% CI |
|---|---|---|---|---|---|
| `degree` | 733 | **yes** | +0.040 | 0.000 | [+0.000, +0.000] |
| `static_delta2` | 733 | **yes** | +0.040 | 0.000 | [+0.000, +0.000] |
| `state_delta2` | 78 | **no** | 0.000 | **−0.040** | [−0.141, +0.061] |
| `random` (100 draws) | — | — | +0.016 | min −0.028, p05 +0.000, median +0.013, p95 +0.043 | — |

| placement inside the random distribution | chosen | batch 2 | draws beating it | percentile |
|---|---|---|---|---|
| `degree` | 733 | +0.040 | 15/100 | 68% |
| `static_delta2` | 733 | +0.040 | 15/100 | 68% |
| `state_delta2` | 78 | **0.000** | **69/100** | 3% |

| score | vs batch-1 marginal | vs batch-2 marginal | reference rank | top-8 recovery vs batch-2 top-8 |
|---|---|---|---|---|
| `degree` | **−0.398** | +0.079 | 1/50 | 4/8 |
| `static_delta2` | **−0.439** | −0.018 | 8/50 | 4/8 |
| `state_delta2` | **+0.477** | +0.055 | 37/50 | **0/8** |

Reading.  The best candidate in the pool is worth **+0.040 target nodes out of |D| = 201** — 0.02% of
the target set, with a CI straddling zero, over a pool whose mean marginal is +0.001.  This
configuration has no headroom to screen for.  More diagnostic than that: **all three scores correlate
with the batch-1 marginal (state δ₂ best at +0.477) and none correlates with the independent batch
(+0.079, −0.018, +0.055).**  At this configuration the scores are fitting selection noise, which is the
expected signature of a configuration whose true marginal spread is smaller than the Monte-Carlo noise
that estimates it.

## 5. Configuration 3 — `ca_grqc`, `|S|/n = 1.0%`

| reading | value |
|---|---|
| MC reference candidate | 8150: batch 1 **+1.096 ±0.357**, batch 2 **+0.947** |
| runner-up | 15960: batch 1 +0.656 |
| **headroom gap** | batch 1 **+0.440 [−0.345, +1.225]** → **not separable** |
| | batch 2 **+0.530 [−0.317, +1.377]** → **not separable** |
| pool mean / sd on batch 2 | +0.264 / 0.252 |
| marginal shape | min +0.000, median +0.136, max +1.096; reference at **+3.5 sd** |
| negative-marginal share | 0% *(descriptive; no CI attached, so not a conclusion)* |
| resolvability | gap needs **3,181** paired trials on batch 1; 1000 were run |

| method | chosen | reference in top-8? | batch 2 | gain vs reference | 95% CI |
|---|---|---|---|---|---|
| `degree` | 3058 | **no** | +0.392 | **−0.555** | [−1.392, +0.282] |
| `static_delta2` | **8150** | **yes** | **+0.947** | 0.000 | [+0.000, +0.000] |
| `state_delta2` | 15960 | **no** | +0.417 | **−0.530** | [−1.377, +0.317] |
| `random` (100 draws) | — | — | +0.633 | min +0.079, p05 +0.102, median +0.631, p95 +0.947 | — |

| placement inside the random distribution | chosen | batch 2 | draws beating it | percentile |
|---|---|---|---|---|
| `degree` | 3058 | +0.392 | 76/100 | 15% |
| `static_delta2` | **8150** | **+0.947** | **3/100** | 70% |
| `state_delta2` | 15960 | +0.417 | **65/100** | 24% |

| score | vs batch-1 marginal | vs batch-2 marginal | reference rank | top-8 recovery vs batch-2 top-8 |
|---|---|---|---|---|
| `degree` | +0.418 | +0.415 | 13/50 | 3/8 |
| `static_delta2` | **+0.828** | **+0.772** | **1/50** | 2/8 |
| `state_delta2` | +0.634 | +0.515 | **50/50** | 3/8 |

Reading.  This is the only configuration with a visible raw headroom (+0.44 target nodes, reference at
+3.5 sd), and even here a 1000-trial paired comparison cannot separate the top candidate from the
runner-up: the CI is ±0.79 on a gap of +0.44.  The two delta2 scores disagree completely about which
candidate is best — `static_delta2` ranks the reference **1st of 50**, `state_delta2` ranks it **50th
of 50 (last)** — while they share every input except the current-state exposure.  On the independent
batch `static_delta2` lands on the reference and is beaten by only 3 of 100 random draws;
`state_delta2` is beaten by 65 of 100.

---

## 6. Why the state-conditioned score fails

The two delta2 scores differ in exactly one input: whether the realised current-state exposure `E_S`
replaces the zero baseline inside `g(E_S + ΔE) − g(E_S)`, where `g(δ) = 2δ(1−δ)`.  That term is
negative wherever a target already sits at or past the peak of `g` (δ ≥ 0.5) — and **those are exactly
the targets a high-marginal candidate reaches.**  Measured:

| configuration | negative-score share, `static` | negative-score share, `state` | reference's percentile |
|---|---|---|---|
| `congress_twitter` | 0% | **100%** | 80th → **20th** |
| `email_eu_core` | 0% | 44% | 84th → **26th** |
| `ca_grqc` | 0% | 6% | **98th → 0th** |

On `ca_grqc` the reference candidate's `state_delta2` score is `−0.0126`, the **minimum of the entire
state-δ₂ range** `[−0.0126, +0.1529]`: the state-conditioned closed form ranks the single best
candidate in the pool dead last.  This is not noise and not a tie-break artefact; it is the sign of the
`g(E_S + ΔE) − g(E_S)` term flipping for the targets that candidate activates.

The `static` 0% column is a structural fact (`g(δ) ≥ 0` for δ ≥ 0), **not** evidence of quality.  The
informative quantity is the percentile the reference occupies, and it inverts in all three
configurations.

**So the answer to "why does the score fail" is specific and it is not "the exposure proxy is
uninformative":**

1. The exposure proxy itself carries real information about the contracted marginal.  On `ca_grqc`
   `static_delta2` reaches **ρ = +0.772 against the independent batch** — the strongest positive
   correlation with the contracted objective measured anywhere in this project — and places the
   reference 1st of 50.
2. **The state conditioning is what destroys it.**  Feeding the current state's realised exposure into
   the same formula turns ρ = +0.828 into +0.634 on the selection batch and rank 1 into rank 50, and
   it puts the reference below 65–69 of 100 random draws on the independent batch in two of three
   configurations.
3. Where there is no headroom to find (`email_eu_core`), all three scores fit the selection batch and
   carry nothing into the independent batch — the correlation simply changes sign.

A fourth, separate point: rank correlation over the bulk does not buy the tail.  On `ca_grqc`
`state_delta2` has ρ = +0.634 with the batch-1 marginal *and* ranks the batch-1 argmax last.  The
decision is entirely in the tail — the reference sits at +2.6, +2.6 and +3.5 sd above the pool mean —
so a score can be positively correlated overall and still lose the only comparison that matters.

---

## 7. Top-8 recovery against the independent batch

How many of the eight best candidates by *batch-2* marginal each score's shortlist actually contains.
Batch 2 is independent of every score, so this is a clean yardstick; the null for a blind score is
hypergeometric with mean **1.28 of 8**.

| configuration | `degree` | `static_delta2` | `state_delta2` |
|---|---|---|---|
| `congress_twitter` | 4/8 *(p = 0.016)* | 2/8 *(p = 0.378)* | 1/8 *(p = 0.780)* |
| `email_eu_core` | 4/8 *(p = 0.016)* | 4/8 *(p = 0.016)* | **0/8** *(p = 1.000)* |
| `ca_grqc` | 3/8 *(p = 0.105)* | 2/8 *(p = 0.378)* | 3/8 *(p = 0.105)* |

`state_delta2` is at or below the blind null in **3 of 3** configurations, and on `email_eu_core` it
recovers **0 of 8** — the worst attainable value.

**Multiplicity.**  Nine overlap tests were performed.  The smallest p-value is 0.016, and the
Bonferroni threshold is 0.05/9 = 0.0056.  **No screening advantage survives correction**, so the
nominally positive `degree` and `static_delta2` entries above are *not* established results and must
not be written as such.  What *is* established is the negative direction: `state_delta2` never exceeds
the blind null in any configuration.

---

## 8. What this establishes, and what it does not

**Established.**

1. At all three configurations the headroom — the MC reference versus the runner-up — is **not
   statistically separable** at 1000 paired trials.  Two configurations need ≤ 10,079 paired trials to
   resolve it and one needs 87,922.  Until that budget is spent, "did screening improve" is not a
   measurable question at these configurations, and no score can be credited or blamed for finding a
   best candidate that is not distinguishable from the runner-up.
2. `state_delta2` is worse than random screening on the independent batch in **3 of 3** configurations
   (65/100, 69/100, 65/100 random draws beat it) and at or below the blind null in top-8 recovery in
   **3 of 3**.
3. Conditioning the exposure score on the current state makes it *worse* than the same score at an
   empty state at locating the reference: rank 40 vs 10, 37 vs 8, **50 vs 1**.
4. The mechanism is identified and measured (§6): the state term is negative for targets at or past
   the peak of `g`, which is the set high-marginal candidates reach.

**Not established — do not write these.**

- That candidate screening is impossible.  Only that at these three configurations the target quantity
  is unresolved at MC = 1000, and that the state-conditioned score is not the answer.
- That `degree` or `static_delta2` beats random screening.  Both nominal positives fail the
  multiplicity correction, and the pool is degree-stratified, which flatters `degree` by construction.
- Any negative-marginal share as a conclusion: none carries a confidence interval.
- Anything about three-hop exposure (not tested) or about whether a target is "inactive" by an
  activation-frequency threshold (not tested, and not a valid test of this model).
- Any pooled or averaged figure across the three graphs.  There is none in this document.

**One more consequence, stated plainly.**  The same resolvability problem applies to the existing
Gate 1(b) comparison: its 16 cells compared chosen seed sets with paired CIs that mostly straddled
zero.  The **FAIL verdict stands** — it rests on negative point estimates, and more trials cannot turn
a negative point estimate positive — but any reading of the form "sequential and static are the same"
is unresolved rather than supported, and the 4 nominally positive Stage B cells are unresolved in the
other direction.

---

## 9. Where this leaves the project

The corrected validation said state-aware screening does not stably beat static screening.  This
diagnostic says why, and the answer points away from the current design rather than toward a learned
model:

- The **exposure proxy is informative** (`static_delta2`: ρ = +0.772 on the independent batch,
  reference ranked 1st of 50 on `ca_grqc`).
- The **state-conditioned form of it is actively harmful**, for a reason that is a property of
  `g(δ) = 2δ(1−δ)` and therefore of the model, not of the estimator.
- Therefore the cheapest well-posed next question is **not** "train a GNN to predict the marginal".
  It is whether a screening score should be conditioned on the current state *at all* — and that is
  answerable with exactly the machinery already in this repository, with no learned component.
- Before any of that, the PI has to fix a **resolvability budget** alongside the Gate 1(b) tolerance:
  a configuration whose headroom needs 87,922 paired trials cannot be used to compare screening
  methods at 1000, and reporting a method ranking from it would be reporting noise.

---

## 10. Reproduce

```
python scripts/audit/diagnose_shortlist_headroom.py --check-only
python scripts/audit/diagnose_shortlist_headroom.py \
  --configs congress_twitter:0.01 email_eu_core:0.05 ca_grqc:0.01 \
  --mc 1000 --candidates 50 --shortlist 8 --random-repeats 100 --pool-size 120 \
  --output docs/results/shortlist_headroom.json
python scripts/audit/print_shortlist_headroom.py
python scripts/audit/derive_shortlist_headroom.py
```

`--check-only` exits non-zero if the three streams overlap.  The derived script adds no simulation: it
recomputes every statistic in §3–§7 from the per-trial paired marginals stored in the artifact, and its
Spearman implementation was checked against `scipy.stats.spearmanr` to three decimals on all nine
score/configuration pairs.
