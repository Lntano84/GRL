# Selection-budget curve: does more selection simulation improve the choice?

> 2026-09-19.  Tests the prediction made in `docs/GRQC_NEW_STATES.md` §7 instead of assuming it.
> Budgets 1000 / 3000 / 10000 trials per candidate, two selection methods, a new 6000-trial independent
> confirmation.  Nothing here is pooled across configurations, and no "refuted" / "equivalent" /
> "no effect" language is used: where an interval contains zero the result is recorded as *not
> detected*, with the family of tests it belongs to stated.
>
> **Design note.**  An earlier attempt at this run was lost when the connection dropped: the workers
> wrote their output only at the end, so ~1080 CPU-minutes vanished with nothing on disk.  The workers
> are now checkpointed every 200 trials and resume exactly; the resume path is tested (`RESUME IS
> EXACT: True`, and a mismatched file is refused rather than reused).

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
| the `degree` top-8 | read from `grqc_new_states_choices_cfg{0..3}.json`, and re-derived only to check it |
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

## 3. Reproduction: the stored 1000-trial selections come back exactly

The first 1000 trials of the new stream are the stored selection batch, so recomputing them is a direct
check of the stream and of the candidate order that decides how the activation coin flips are consumed:

| configuration | max abs difference, 50 candidates × 1000 trials | `full50@1000` = stored MC reference | `degree8@1000` = stored degree choice |
|---|---|---|---|
| cfg0 | **0.0** | yes | yes |
| cfg1 | **0.0** | yes | yes |
| cfg2 | **0.0** | yes | yes |
| cfg3 | **0.0** | yes | yes |

(`marginal = newly_positive − lost_positive` also holds on every one of the 78,000 confirmation
trial-candidate pairs.)

## 4. The twelve-row table

Candidates and their **independent 6000-trial confirmation marginal**, paired within a trial:

| configuration | budget | full50 candidate / gain / 95% CI | degree8 candidate / gain / 95% CI | paired difference full50 − degree8 | selection cascades |
|---|---|---|---|---|---|
| cfg0 | 1000 | 19244 / 0.456 [+0.285, +0.628] | 4960 / 0.542 [+0.333, +0.751] | −0.086 [−0.352, +0.181] | 51,000 |
| cfg0 | 3000 | 9713 / 0.784 [+0.546, +1.023] | 4960 / 0.542 [+0.333, +0.751] | +0.242 [−0.069, +0.554] | 153,000 |
| cfg0 | 10000 | 9713 / 0.784 [+0.546, +1.023] | 7811 / 0.506 [+0.300, +0.713] | **+0.278 [−0.030, +0.585]** | 510,000 |
| cfg1 | 1000 | 11640 / 0.731 [+0.501, +0.961] | 3718 / 0.669 [+0.468, +0.870] | +0.062 [−0.228, +0.351] | 51,000 |
| cfg1 | 3000 | 3718 / 0.669 | 3718 / 0.669 | **same choice** | 153,000 |
| cfg1 | 10000 | 3718 / 0.669 | 3718 / 0.669 | **same choice** | 510,000 |
| cfg2 | 1000 | 839 / 0.616 [+0.466, +0.765] | 839 / 0.616 | **same choice** | 51,000 |
| cfg2 | 3000 | 4960 / 1.264 [+0.963, +1.565] | 4960 / 1.264 | **same choice** | 153,000 |
| cfg2 | 10000 | 4960 / 1.264 [+0.963, +1.565] | 4960 / 1.264 | **same choice** | 510,000 |
| cfg3 | 1000 | 5840 / 0.750 [+0.550, +0.950] | 5840 / 0.750 | **same choice** | 51,000 |
| cfg3 | 3000 | 5840 / 0.750 | 5840 / 0.750 | **same choice** | 153,000 |
| cfg3 | 10000 | 5840 / 0.750 | 5840 / 0.750 | **same choice** | 510,000 |

"Same choice" means both methods selected the same candidate at that budget: the contrast is exactly
zero by construction and is **not** evidence that the methods agree.  Selection cascades are
`budget × 51` (one base run plus 50 candidates) and are **shared by both methods**, because they read
the same paired data; the 10000-trial run subsumes the smaller budgets, so the actual selection cost is
4 × 510,000 = 2,040,000, not the sum of the column.

## 5. Primary comparison: full50 versus degree8 at budget 10000

| configuration | contrast | result |
|---|---|---|
| cfg0 | 9713 − 7811 | **+0.278**, 95% CI [−0.030, +0.585], p = 0.077 |
| cfg1 | 3718 − 3718 | same choice |
| cfg2 | 4960 − 4960 | same choice |
| cfg3 | 5840 − 5840 | same choice |

**Holm over the four configurations:** only cfg0 contributes a test, so `p_adj = 0.077`, **not
significant at 0.05**.

**Answer: no.**  At a 10000-trial budget, `full50` is **not established** to beat same-budget
`degree8`.  In three of the four configurations both methods selected the same candidate, which means
the degree top-8 already contained everything the full-pool search found; in the fourth the difference
is +0.278 with an interval that contains zero.  Spending 50× the candidates bought no measurable
advantage over spending the same budget inside the degree top-8.

## 6. Auxiliary comparison: the budget effect on full50 (exploratory)

Prediction under test: raising the budget from 1000 to 10000 improves the chosen candidate.

| configuration | 1000-budget pick | 10000-budget pick | difference (10000 − 1000) | p |
|---|---|---|---|---|
| cfg0 | 19244 (+0.456) | 9713 (+0.784) | **+0.328 [+0.047, +0.609]** | 0.0222 |
| cfg1 | 11640 (+0.731) | 3718 (+0.669) | −0.062 [−0.351, +0.228] | 0.676 |
| cfg2 | 839 (+0.616) | 4960 (+1.264) | **+0.648 [+0.314, +0.983]** | 0.000145 |
| cfg3 | 5840 (+0.750) | 5840 | same choice | — |

Two of the four move up, one does not move detectably, and one has nothing to measure.  No
configuration moves down significantly.

**Multiplicity, stated rather than chosen after the fact.**  This family is exploratory, so its
definition changes the verdict and both readings are given:

- Over the **four distinct method × configuration budget effects** (including `degree8` in cfg0, whose
  pick moved 4960 → 7811, a nominal −0.035, p = 0.806): Holm gives cfg2 **p_adj = 0.00058
  (significant)** and cfg0 **p_adj = 0.067 (not significant)**.
- Over only the **pre-specified `full50` comparison** (three informative tests): Holm gives cfg2
  **0.00043** and cfg0 **0.044**.

So cfg2's improvement is robust to how the family is drawn; **cfg0's is borderline and its status
depends on the family definition.**  It must be reported as borderline, not as a second positive.

## 7. The mechanism, and what it costs

The §7 prediction is **partly supported and nowhere contradicted**: more selection budget did change
the chosen candidate in a measurable, sometimes significant way (cfg2), and did nothing detectable in
cfg1.  So the selection budget is a **live lever** --- not a dead one.

But the lever is not monotone, and this is worth recording: in cfg0 the extra budget made `degree8`
pick a *nominally worse* candidate (4960 → 7811, 0.542 → 0.506), and in cfg1 it made `full50` pick a
nominally worse one (11640 → 3718, 0.731 → 0.669).  Neither is significant, and both are the expected
behaviour of an argmax moving as its estimates change: a bigger budget does not shrink the estimate
towards the truth along a fixed ordering, it changes which candidate wins.

## 8. What this establishes, and what it does not

**Established.**

1. The stored 1000-trial selections are **reproduced exactly** by the new stream in all four
   configurations, for both methods.
2. `full50` is **not** established to beat same-budget `degree8` at budget 10000: one contrast of four,
   `p_adj = 0.077` (cfg0), and three exact ties.
3. The selection budget **does** change the chosen candidate in a measurable way in at least one
   configuration (cfg2, `p_adj = 0.00058` after Holm over the method × configuration family), so the
   budget is a real lever rather than an inert parameter.
4. The budget effect is **not monotone**: two cells moved nominally downward, neither significantly.
5. Restricting to the degree top-8 before spending the selection budget found the same candidate as
   the full-pool search in **3 of 4** configurations at every budget from 3000 up.

**Not established --- do not write these.**

- That expensive screening is worth its cost.  In 3 of 4 configurations it found the same candidate as
  the far cheaper degree-filtered search, and the one place it differed is not significant.
- That `full50` and `degree8` are *equivalent*.  Three of the four comparisons are exact ties between
  identical candidate sets, which is not a test of equivalence; no equivalence test was run.
- That the budget effect is real in cfg0.  It is borderline and family-dependent (§6).
- That raising the budget never hurts.  Two cells moved nominally down; neither is significant, and the
  6000-trial confirmation has a fixed budget that does not guarantee significance either way.
- That the selection-versus-evaluation gap in `docs/GRQC_NEW_STATES.md` was selection noise.  This run
  tested a *prediction* of that hypothesis, and finding the prediction partly supported does not
  identify how much of the original gap came from which batch.
- Anything pooled across configurations, or about other graphs, fractions or budgets.
- Anything about `state_delta2`, which this run does not test at all.

## 9. Cost

| item | cascades |
|---|---|
| selection (4 configurations × 10000 trials × 51 runs) | 2,040,000 |
| confirmation (6000 trials × (1 + union) per configuration; unions 4, 2, 2, 1) | 78,000 |
| **total** | **2,118,000** |

The estimate before the run was ~2.004M: selection ~1.836M and confirmation ≤168,000.  The run is
larger because the first 1000 selection trials were **recomputed rather than literally reused**, which
is what makes the exact reproduction in §3 possible; that adds 204,000 cascades.  The confirmation is
smaller than its ceiling because three of the four unions collapsed to one or two candidates.

## 10. Provenance

| item | value |
|---|---|
| script | `scripts/audit/selection_budget_curve.py` (driver: `scripts/audit/run_budget_curve.py`) |
| artifact | `docs/results/selection_budget_curve.json` |
| per-configuration files | `sbc_sel_cfg{0..3}_{00000_03334,03334_06668,06668_10000}.json`, `sbc_choices_cfg{0..3}.json`, `sbc_confirm_cfg{0..3}.json` |
| code version | `162163b` |
| configuration sources | `grqc_new_states_cfg{0..3}.json`, `grqc_new_states_choices_cfg{0..3}.json` |
| streams | selection `seed + 1_100_000` (10000 trials, nested prefixes); confirmation `seed + 1_700_000` (6000 trials) |
| stream check | 12 streams, 66 pairs, all disjoint; the original 6000-trial batch is checked at its full length, not its first 1000 |
| `complete` | `true` |

## 11. Reproduce

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
records the hash of the choices it was produced from.
