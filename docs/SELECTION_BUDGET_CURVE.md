# Selection-budget curve: does more selection simulation improve the choice?

> 2026-09-19.  Tests the prediction made in `docs/GRQC_NEW_STATES.md` §7 instead of assuming it.
> Budgets 1000 / 3000 / 10000 trials per candidate, two selection methods, a new 6000-trial independent
> confirmation.  Nothing here is pooled across configurations, and no "refuted" / "equivalent" /
> "no effect" language is used: where an interval contains zero the result is recorded as *not
> detected*, together with the family of tests it belongs to.
>
> **Design note.**  An earlier attempt at this run was lost when the connection dropped: the workers
> wrote their output only at the end, so ~1.77M cascades of compute vanished with nothing on disk
> (§9).  The workers are now checkpointed every 200 trials and resume exactly; the resume path is
> tested (`RESUME IS EXACT: True`, and a mismatched file is refused rather than reused).

---

## 1. What is tested

`docs/GRQC_NEW_STATES.md` observed that the MC reference's selection-batch value exceeded its
independent-batch value by 0.13--0.86 target nodes and proposed an explanation: at 1000 selection
trials the argmax over 50 noisy estimates is largely the argmax of the noise.  That was explicitly
recorded as a **hypothesis**, because both batches are finite and the gap is a difference between two
estimates.

The hypothesis makes one testable prediction: **raising the selection budget should improve the chosen
candidate's independent gain.**  This document tests it.  It does not test, and cannot establish,
whether the original gap was selection noise.

## 2. Frozen, and the one thing changed

| held fixed | source |
|---|---|
| the four configurations' `S`, 50 candidates, and candidate order | read from `grqc_new_states_cfg{0..3}.json`, never regenerated |
| target set `D` | the same `D`, verified equal to the stored set |
| diffusion parameters, weight handling | unchanged |
| the `degree` top-8 | read from `grqc_new_states_choices_cfg{0..3}.json`, re-derived only to check it |
| state estimation | not re-run; no method here uses the state score |
| model training | none |

**Changed: the selection budget alone** --- 1000, 3000 and 10000 trials per candidate.

The three budgets are **nested prefixes of one stream**.  Trial `t` is computed from its own seed and
depends on no other trial, so the 3000-trial selection is the 10000-trial selection restricted to its
first 3000 trials.  That is what makes the levels comparable on identical data.  All three budgets were
computed and frozen **before any confirmation batch existed**, and the workers report progress counts
only, so no intermediate result could inform whether to go on to 10000.

The two methods, both reading the **same paired data** (one window draw per trial, shared by all 50
candidates):

| method | rule |
|---|---|
| `full50` | highest selection-batch mean over all 50 candidates |
| `degree8` | highest selection-batch mean **within the fixed degree top-8** --- degree filtering plus Monte-Carlo verification, **not** a zero-simulation method |

Ties break by ascending node id, fixed in advance and never adjusted to a result.

**Streams.**  Twenty streams are involved: the original configuration's four, the four new
configurations' historical state and evaluation streams (eight, which the first version of the check
omitted), and this experiment's selection and confirmation streams (eight).  All **190 pairs** were
checked at their actual lengths, before the run:

```
{"streams_checked": 20, "pairs_checked": 190, "overlapping_pairs": {}, "disjoint": true}
```

The 1000-trial selection stream is deliberately **not** counted again: it is a prefix of the
10000-trial selection stream, which is already checked.

## 3. Reproduction: the stored 1000-trial selections come back exactly

The first 1000 trials of the new stream are the stored selection batch, so recomputing them directly
checks the stream and the candidate order that decides how the activation coin flips are consumed:

| configuration | max abs difference, 50 candidates × 1000 trials | `full50@1000` = stored MC reference | `degree8@1000` = stored degree choice |
|---|---|---|---|
| cfg0 | **0.0** | yes | yes |
| cfg1 | **0.0** | yes | yes |
| cfg2 | **0.0** | yes | yes |
| cfg3 | **0.0** | yes | yes |

## 4. The finding worth keeping: 3000 and 10000 give the same `full50` choice

| configuration | `full50@3000` | `full50@10000` | identical? |
|---|---|---|---|
| cfg0 | 9713 | 9713 | **yes** |
| cfg1 | 3718 | 3718 | **yes** |
| cfg2 | 4960 | 4960 | **yes** |
| cfg3 | 5840 | 5840 | **yes** |

In **all four** measured configurations, tripling the `full50` selection budget from 3000 to 10000 did
**not change the selected candidate**.  `degree8@3000` selected the same candidate as `full50@10000` in
**three of the four** (cfg1, cfg2, cfg3; cfg0 differs).

This is the sharpest result of the run and it is a statement about **these four configurations
only**.  It must not be extrapolated to "3000 trials is generally enough": nothing here varies the
graph, the target set, the seed fraction or the pool size.  It is also specific to `full50` --- for
`degree8` the two budgets disagree in cfg0 (4960 → 7811).

## 5. The twelve-row table

Candidates and their **independent 6000-trial confirmation marginal**:

| configuration | budget | full50 candidate / gain / 95% CI | degree8 candidate / gain / 95% CI | paired difference full50 − degree8 |
|---|---|---|---|---|
| cfg0 | 1000 | 19244 / 0.456 [+0.285, +0.628] | 4960 / 0.542 [+0.333, +0.751] | −0.086 [−0.352, +0.181] |
| cfg0 | 3000 | 9713 / 0.784 [+0.546, +1.023] | 4960 / 0.542 [+0.333, +0.751] | +0.242 [−0.069, +0.554] |
| cfg0 | 10000 | 9713 / 0.784 [+0.546, +1.023] | 7811 / 0.506 [+0.300, +0.713] | **+0.278 [−0.030, +0.585]** |
| cfg1 | 1000 | 11640 / 0.731 [+0.501, +0.961] | 3718 / 0.669 [+0.468, +0.870] | +0.062 [−0.228, +0.351] |
| cfg1 | 3000 | 3718 / 0.669 | 3718 / 0.669 | **same choice** |
| cfg1 | 10000 | 3718 / 0.669 | 3718 / 0.669 | **same choice** |
| cfg2 | 1000 | 839 / 0.616 [+0.466, +0.765] | 839 / 0.616 | **same choice** |
| cfg2 | 3000 | 4960 / 1.264 [+0.963, +1.565] | 4960 / 1.264 | **same choice** |
| cfg2 | 10000 | 4960 / 1.264 [+0.963, +1.565] | 4960 / 1.264 | **same choice** |
| cfg3 | 1000 | 5840 / 0.750 [+0.550, +0.950] | 5840 / 0.750 | **same choice** |
| cfg3 | 3000 | 5840 / 0.750 | 5840 / 0.750 | **same choice** |
| cfg3 | 10000 | 5840 / 0.750 | 5840 / 0.750 | **same choice** |

"Same choice" means both methods selected the same candidate at that budget: the contrast is exactly
zero by construction and is **not** evidence that the methods agree.

## 6. Primary comparison: full50 versus degree8 at budget 10000

| configuration | contrast | p |
|---|---|---|
| cfg0 | 9713 − 7811 = **+0.278** [−0.030, +0.585] | 0.0767 |
| cfg1 | same choice | entered as **1.0** |
| cfg2 | same choice | entered as **1.0** |
| cfg3 | same choice | entered as **1.0** |

**Family: all four configurations.**  A same-choice tie is not a missing test --- it is a contrast that
came out exactly zero --- so it enters the family conservatively as `p = 1`.  Holm over the four:
**cfg0 `p_adj = 0.3067`**, not significant at 0.05.

**Answer: no.**  `full50` is **not established** to beat same-budget `degree8`.  In three of the four
configurations both methods selected the same candidate, and in the fourth the difference has an
interval containing zero and does not reach the corrected threshold.  Spending 50× the candidates
bought no measurable advantage over spending the same budget inside the degree top-8.

## 7. Auxiliary comparison: the budget effect on full50 (exploratory)

Prediction under test: raising the budget from 1000 to 10000 improves the chosen candidate.

| configuration | 1000-budget pick | 10000-budget pick | difference (10000 − 1000) | p |
|---|---|---|---|---|
| cfg0 | 19244 (+0.456) | 9713 (+0.784) | **+0.328 [+0.047, +0.609]** | 0.0222 |
| cfg1 | 11640 (+0.731) | 3718 (+0.669) | −0.062 [−0.351, +0.228] | 0.676 |
| cfg2 | 839 (+0.616) | 4960 (+1.264) | **+0.648 [+0.314, +0.983]** | 0.000145 |
| cfg3 | 5840 (+0.750) | 5840 | same choice | entered as **1.0** |

**Family: the same four configurations**, ties as `p = 1`.  Holm:

| configuration | p raw | factor | p adjusted | significant at 0.05 |
|---|---|---|---|---|
| cfg2 | 0.000145 | 4 | **0.00058** | **yes** |
| cfg0 | 0.0222 | 3 | **0.0667** | no |
| cfg1 | 0.676 | 2 | 1.0 | no |
| cfg3 | (tie) 1.0 | 1 | 1.0 | no |

**One configuration has evidence of a positive budget effect (cfg2); the others do not reach the
threshold.**  cfg0 is borderline and **not** significant after correction.  This comparison remains
**exploratory** --- it is not a confirmatory result and no claim rests on it.

**Accurate statement of direction.**  One configuration detected a positive budget effect and the
others detected no clear change.  **cfg1's point estimate is negative** (−0.062).  Nothing here
guarantees that increasing the budget cannot reduce the gain; the non-monotonicity is visible in the
raw cells, where the extra budget moved `degree8` in cfg0 from 4960 to 7811 (0.542 → 0.506) and
`full50` in cfg1 from 11640 to 3718 (0.731 → 0.669).  Neither is significant, and both are what an
argmax does as its estimates change: a bigger budget changes which candidate wins, it does not walk a
fixed ordering towards the truth.

## 8. Deployment cost and quality

**Cost per decision** is one base run per trial plus one run per candidate screened.  It is a **count
of cascades, not measured wall time**, and it excludes the independent confirmation, which exists only
to evaluate a decision and is not part of deploying one.

| strategy | cascades per decision |
|---|---|
| `degree8@1000` | 1000 × 9 = **9,000** |
| `degree8@3000` | 3000 × 9 = **27,000** |
| `full50@3000` | 3000 × 51 = **153,000** |
| `full50@10000` | 10000 × 51 = **510,000** |

Quality is this run's 6000-trial independent confirmation of each strategy's chosen candidate:

| configuration | `degree8@1000` (9,000) | `degree8@3000` (27,000) | `full50@3000` (153,000) | `full50@10000` (510,000) |
|---|---|---|---|---|
| cfg0 | 4960 / 0.542 | 4960 / 0.542 | 9713 / 0.784 | 9713 / 0.784 |
| cfg1 | 3718 / 0.669 | 3718 / 0.669 | 3718 / 0.669 | 3718 / 0.669 |
| cfg2 | 839 / 0.616 | 4960 / **1.264** | 4960 / 1.264 | 4960 / 1.264 |
| cfg3 | 5840 / 0.750 | 5840 / 0.750 | 5840 / 0.750 | 5840 / 0.750 |

### The core comparison: `degree8@3000` versus `full50@10000`

| configuration | cascades saved | gain difference (expensive − cheap) |
|---|---|---|
| cfg0 | 483,000 (**94.7%**) | **+0.242** [−0.069, +0.554] |
| cfg1 | 483,000 (94.7%) | **same candidate** (3718) → identical gain for this decision |
| cfg2 | 483,000 (94.7%) | **same candidate** (4960) → identical gain for this decision |
| cfg3 | 483,000 (94.7%) | **same candidate** (5840) → identical gain for this decision |

The cheap strategy costs **27,000** cascades against **510,000**, a reduction of **94.7%**.

Two things this table does **not** say, and they must not be written:

- The 94.7% is a reduction in the **number of cascades**.  It is **not a measured speed-up**; no wall
  time for a single decision was recorded, and the confirmation cost is excluded from it.
- It is **not** "no loss in all four configurations".  In cfg0 the cheap strategy's gain is lower by
  0.242, and the interval [−0.069, +0.554] **contains zero**, so the loss cannot be called negligible.
  In the other three the two strategies chose the same candidate, which is an exact tie, not an
  equivalence result.

### Provisional cheap baseline for future work

**`degree8@3000` (27,000 cascades per decision) is fixed as the cheap baseline that any new method must
beat.**  It was selected **from these results**: it matched `full50@10000`'s candidate in three of four
configurations, and in the fourth its gain was lower by an amount whose interval still contains zero.

It is a working reference, **not an established optimum**, and it has **not been validated on any
configuration other than these four**.  It must be re-checked on new configurations before being
treated as a fixed reference, and a future method that beats it on these four configurations has
beaten a baseline chosen on those same four.

## 9. Cost, at three levels

These answer different questions and are kept apart.

| level | cascades | what it is |
|---|---|---|
| **1. Method decision cost** | 9,000 / 27,000 / 153,000 / 510,000 | what one decision costs, per strategy (§8).  This is what a deployed method pays. |
| **2. Experimental confirmation cost** | 2,040,000 selection + 78,000 confirmation = **2,118,000** | what *this experiment* spent beyond a single decision, to learn whether the decisions were any good |
| **3. Interruption overhead** | **~1,770,000** (estimate) | compute spent and discarded when the connection dropped during the first attempt |

**Counting detail for level 2.**  The confirmation's chosen-candidate union is 4+2+2+1 = **9**
candidates.  The identity `marginal = newly_positive − lost_positive` is checked on
9 × 6000 = **54,000 trial-candidate pairs**; the confirmation **cost** of 78,000 cascades is those
54,000 candidate runs **plus 24,000 base runs** (4 configurations × 6000 trials).

**Basis for level 3, which is an estimate and not a measurement.**  Twelve workers had each reached
roughly 2,900 of their 3,334 trials when the processes were killed (the last observed checkpoints
ranged from 2,500 to 3,300), giving about 12 × 2,900 × 51 ≈ 1.77M cascades ≈ 15 CPU-hours.  The first
attempt's log files were overwritten by the second, so this cannot be tightened retrospectively.  It is
**execution overhead**, not part of the experiment's cost, and it is excluded from the level-2 total.

The estimate made before the run was ~2.004M (selection ~1.836M, confirmation ≤168,000).  The run is
larger because the first 1000 selection trials were **recomputed rather than literally reused**, which
is what makes the exact reproduction in §3 possible; that adds 204,000 cascades.  The confirmation is
smaller than its ceiling because three of the four unions collapsed to one or two candidates.

## 10. What this establishes, and what it does not

**Established.**

1. The stored 1000-trial selections are **reproduced exactly** by the new stream in all four
   configurations, for both methods.
2. Tripling the `full50` budget from 3000 to 10000 **did not change the selected candidate in any of
   the four** configurations, and `degree8@3000` matched `full50@10000` in three of the four.
3. `full50` is **not** established to beat same-budget `degree8` at budget 10000: one contrast of four,
   `p_adj = 0.3067` over the four-configuration family.
4. The selection budget **does** change the chosen candidate in a measurable way in at least one
   configuration (cfg2, auxiliary `p_adj = 0.00058`), so the budget is a real lever rather than an
   inert parameter.
5. `degree8@3000` reaches the same decision as `full50@10000` in three of four configurations at
   **5.3%** of the cascade count.

**Not established --- do not write these.**

- That expensive screening is worth its cost.  In three of four configurations it found the same
  candidate as the far cheaper degree-filtered search, and the one place it differed does not reach
  the corrected threshold.
- That `full50` and `degree8` are *equivalent*.  Three of the four comparisons are exact ties between
  identical candidate sets, which is not a test of equivalence; no equivalence test was run.
- That the budget effect is real in cfg0.  `p_adj = 0.0667`, exploratory.
- That increasing the budget cannot hurt.  cfg1's point estimate is negative; none of the detected
  changes rules a loss out.
- **That 3000 trials is generally enough.**  That is true of `full50` on these four configurations and
  nothing more.
- That the 94.7% is a speed-up, or that the cheap strategy loses nothing on all four configurations.
- That `degree8@3000` is optimal, or a validated baseline.
- That the selection-versus-evaluation gap in `docs/GRQC_NEW_STATES.md` was selection noise.  This run
  tested a *prediction* of that hypothesis; supporting a prediction is not decomposing the gap.
- Anything pooled across configurations, or about other graphs, fractions or budgets.
- Anything about `state_delta2`, which this run does not test at all.

## 11. Provenance

| item | value |
|---|---|
| script | `scripts/audit/selection_budget_curve.py` (driver: `scripts/audit/run_budget_curve.py`) |
| artifact | `docs/results/selection_budget_curve.json` |
| per-configuration files | `sbc_sel_cfg{0..3}_{00000_03334,03334_06668,06668_10000}.json`, `sbc_choices_cfg{0..3}.json`, `sbc_confirm_cfg{0..3}.json` |
| code version | `ee50a7c` (merged statistics revised in place; no simulation re-run) |
| configuration sources | `grqc_new_states_cfg{0..3}.json`, `grqc_new_states_choices_cfg{0..3}.json` |
| streams | selection `seed + 1_100_000` (10000 trials, nested prefixes); confirmation `seed + 1_700_000` (6000 trials); historical state `seed + 900_000` and evaluation `seed + 1_300_000` (1000 each) |
| stream check | 20 streams, 190 pairs, all disjoint |
| `complete` | `true` |

## 12. Reproduce

```
python scripts/audit/selection_budget_curve.py --check-streams-only
python scripts/audit/selection_budget_curve.py --select-chunk I START END   # 12 chunks, resumable
python scripts/audit/selection_budget_curve.py --assemble I                 # freeze, verify, hash
python scripts/audit/selection_budget_curve.py --confirm I                  # 6000 independent trials
python scripts/audit/selection_budget_curve.py --merge
```

Or all of it, in order, with waiting: `python scripts/audit/run_budget_curve.py`.

Every stage is resumable and idempotent.  `--assemble` refuses to run on an incomplete chunk and stops
if the first 1000 trials do not reproduce the stored selection exactly; `--merge` refuses to run unless
all four choice files exist, still hash to what was written before confirmation, and each confirmation
records the hash of the choices it was produced from.  `--merge` performs no simulation, so the
statistics in §6--§9 can be revised without re-running anything.
