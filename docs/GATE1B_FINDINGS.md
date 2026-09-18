# Gate 1(b): sequential vs static — the Go/No-Go answer

> Written 2026-09-18, from the sweep in flight. **Every cell completed so far is in the artifact**
> `docs/results/gate1b_sequential_vs_static.json`; the summary below covers the completed cells and
> says which graphs are still running. Numbers are from that file, not from memory.

## The question, and why it decides the submission

`docs/DECISIONS.md`, 2026-09-15, set a gate with a stated consequence:

> Gate 1 (by 2026-10-13): (a) Spearman correlation between true marginal gain and degree under
> overexposure collapses (≈0 or negative); (b) **sequential decision beats the static optimum by a
> material margin (>5%)**. If either fails, **DASFAA is dropped and the target moves to CIKM 2027.**

Gate 1(a) passes. This is (b).

## The answer

**Gate 1(b) does not pass.** Across every cell measured, under both weight normalisations, the
largest lift of a sequential arm over the best static arm is **+2.64%**, against a **+5%**
requirement. Most cells are between −10% and +2%.

| graph | normalisation | completed cells | lift range | median | positive |
| --- | --- | --- | --- | --- | --- |
| Congress-Twitter | `sum_to_one` | 12 | **−2.19% … −0.34%** | ≈ −1.1% | 0/12 |
| Congress-Twitter | `clip_to_one` | 12 | **+0.03% … +1.65%** | ≈ +1.0% | 12/12 |
| email-Eu-core | `sum_to_one` | 12 | −9.69% … +2.12% | see below | 4/12 |
| email-Eu-core | `clip_to_one` | 12 | −0.79% … +0.39% | ≈ 0% | 6/12 |
| ca-GrQc | `sum_to_one` | 1 | +2.64% | +2.64% | 1/1 |

Still running: the rest of ca-GrQc and NetHEPT.

## Two things this measurement establishes beyond the verdict

### 1. The result's SIGN depends on the weight normalisation

This is the sharpest consequence of confound P1-3.8, and it is worse than "a 50x exposure-scale
difference". On Congress-Twitter:

* `sum_to_one`: sequential loses in **12 of 12** cells (−0.34% to −2.19%);
* `clip_to_one`: sequential wins in **12 of 12** cells (+0.03% to +1.65%).

Same graph, same arms, same seeds, same pools — the sign of the finding is set by a preprocessing
choice that the earlier work did not report. And the two configurations are not equally valid for
this question:

* `sum_to_one` scales every node's in-weights to sum to 1, which is what the source model's datasets
  ship and what the previous work used;
* `clip_to_one` only scales *down* nodes that exceed 1. On Congress-Twitter the raw weights have
  `max total 1.648` (one node over) and `min nonzero 0.0011`, so almost all weights keep their raw
  tiny values. Exposures stay far below the windows, spreads collapse (44–135 against 155–208), and
  **the model is barely in the overexposure regime at all**.

So the one configuration where sequential "wins" is the one where there is little overexposure to
reason about. The +1% there is not evidence that sequencing helps under overexposure; it is evidence
that when overexposure is switched off, sequencing does nothing much either way. **The honest
reading is the `sum_to_one` column: under the model as the source specifies it, sequential selection
loses to static selection.**

### 2. Where sequential does help, it tracks how bad degree is

Under `sum_to_one` the lift is not uniform; it follows the regime:

| graph | `ρ_degree` at `\|S\|/n ≥ 20%` (from `tab:regime-fixed`) | sequential lift |
| --- | --- | --- |
| email-Eu-core | −0.236 | **+1.6% … +2.1%** at 20% |
| ca-GrQc | −0.583 | **+2.64%** at 5% |
| Congress-Twitter | −0.047 | **−0.3% … −2.2%**, negative at every fraction |

And the fraction matters in the direction the paper predicts: email-Eu-core under `sum_to_one` goes
**−9.7% at 5%**, to ≈0% at 10%, to **+2.1% at 20%**. Sequencing hurts most where a static ranking is
already fine, and helps where overexposure makes it stale — but it helps by ~2%, not by the 5% the
gate asks for, and on the densest graph it does not help at all.

## Why the sequential arm is not being strawmanned

Two sequential arms were run, and the weaker one is the pre-registered test:

* `delta2_sequential` — observation used **only to re-rank**. This is the most conservative reading
  of "sequential decision", because re-ranking is a lever a static arm may already have.
* `delta2_patience` — observation also used to decide **how many seeds to take**, by stopping when
  the realised spread stops improving. That is the mechanism the paper's own framing proposes
  ("analytic ranking + a patience stop on the observed spread"), and it is a lever a static arm
  structurally cannot use.

`delta2_patience` **never fires**. In every cell, `stopped_early = False` and the seed count equals
the budget: the realised spread is weakly increasing in the number of seeds, so a patience rule on
it never triggers. The arm costs 24–95 extra spread reads and returns the same set. So the stopping
lever, which was the strongest argument for a sequential method here, does not activate on this
data.

Cost accounting, at `k = 24` on Congress-Twitter (`state_mc = 25`):

| arm | cascades | what it buys |
| --- | --- | --- |
| `degree_static` | 0 | nothing |
| `delta2_static` | 25 | one state read |
| `delta2_sequential` | 600 | 24 state reads |
| `delta2_patience` | 624 | 24 state reads + 24 spread reads |

So the sequential arm is **24x** the cost of the static analytic arm and **infinitely** more than
degree, for a result 1–2% *worse* on the densest graph.

## What this implies for the venue decision

The pre-registered rule is explicit: failing either part of Gate 1 drops DASFAA and moves the target
to CIKM 2027. On the evidence above, **(b) fails**, and it fails by a wide margin on the graph the
paper's regime section is built around.

That is the PI's call, not mine, but three readings are available and they should be made
deliberately rather than drifted into:

1. **Apply the rule as written.** The sequential-vs-static thesis is not supported; retarget.
2. **Keep DASFAA with a different thesis.** The measurement-protocol contribution
   (`sections/protocol.tex`) and the coverage result stand on their own and do not depend on
   sequential selection winning. That is a narrower paper, but it is a true one, and the
   normalisation finding in §1 above is a genuinely new result that belongs to it.
3. **Re-scope the claim from "sequential beats static" to "sequential beats static where degree is
   misleading, by ~2%"**, and report the `ρ_degree` correlation as the predictor of when. This is
   supportable by the data above, but it is a 2% effect, it needs ca-GrQc and NetHEPT to confirm,
   and it is not the claim the project set out to make.

**What must not happen is reporting the `clip_to_one` column as the result.** It has the right sign
and it is the configuration in which the model is least over-exposed.

## Reproduce

```bash
python scripts/experiments/go_no_go.py \
  --graphs congress_twitter email_eu_core ca_grqc nethept \
  --fractions 0.05 0.10 0.20 --pool-draws 2 --seeds 20260917 20260918 \
  --normalisations sum_to_one clip_to_one \
  --arms delta2_sequential delta2_patience degree_static delta2_static \
  --eval-mc 200 --state-mc 25 --max-seeds 1600 \
  --output docs/results/gate1b_sequential_vs_static.json
```

Raw in-weight profiles, which is what makes §1 visible before a sweep is run:
`python scripts/audit/report_raw_weight_profiles.py`.
