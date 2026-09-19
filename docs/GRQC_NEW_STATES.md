# Four new states on the same ca-GrQc configuration: is there stable screening headroom?

> **Correction, 2026-09-19 (after review).**  Three over-statements in the first version of this
> document are corrected in place, and the measurement itself is unchanged.  (a) The
> selection-versus-evaluation gap of 0.863 in cfg0 is an **observed difference between two finite
> batches**, not a decomposition into selection noise --- the independent evaluation has error of its
> own and the two are not separable from that comparison.  (b) The pool's cross-candidate standard
> deviation is computed from 1000-trial estimates and therefore contains Monte-Carlo noise; it cannot
> on its own show that the true candidate differences are large.  (c) The "3,346--595,136 trials"
> figures are a post-hoc extrapolation of **evaluation** precision for a fixed candidate difference
> --- not the sample size the **selection** stage needs, and not a formal power analysis.  §7 now
> treats the proposed mechanism as a hypothesis with a testable prediction rather than a finding.

> 2026-09-19.  The prerequisite experiment before deciding whether a learned screener is worth
> building.  One thing is changed --- the random draws that produce the existing seed set and the
> candidate pool.  Everything else is frozen.  One raw artifact plus the four-row table.
>
> **Nothing is pooled across configurations.**  The four are separate states, not replicates of one
> population, and the random method's 100 repetitions are a baseline *within* a configuration, never
> 100 study scenarios.
>
> **No "refuted", "equivalent", "harmless" or "no cost" language is used below.**  Where an interval
> contains zero the finding is recorded as *not detected*, with the power of the design stated.

---

## 1. The question

`docs/GRQC_RANK_REVERSAL.md` settled one configuration.  One configuration cannot answer the question
that decides whether to invest in learning a screener:

> With a different batch of existing seeds and candidates, does expensive Monte-Carlo screening still
> **stably** beat simple screening?

## 2. Frozen, and the one thing changed

| held fixed | value |
|---|---|
| graph | `ca_grqc`, n = 5242 |
| weight normalisation | `sum_to_one` |
| target set `D` | **the same `D` as before**, 1048 nodes; verified equal to the stored set |
| existing seed count | 52 |
| candidate count | 50 |
| shortlist | 8 |
| score formula, hop depth | unchanged |
| diffusion parameters | unchanged |
| seed and candidate generation rule | unchanged, including the degree-stratified pool of 120 |

| changed | value |
|---|---|
| the random seed driving `S` and the pool | four pre-fixed new seeds: **20261001, 20261002, 20261003, 20261004** |

The four seeds were fixed before any configuration was generated.  None was re-drawn, replaced or
dropped on the basis of a result.

**The configurations are genuinely different, and none repeats the original.**  The actual node lists
are stored in the artifact:

| configuration | duplicate of another? | repeats the original? | seeds shared with original | candidates shared with original |
|---|---|---|---|---|
| cfg0 (20261001) | no | no | 0 of 52 | 1 of 50 |
| cfg1 (20261002) | no | no | 2 of 52 | 2 of 50 |
| cfg2 (20261003) | no | no | 1 of 52 | 2 of 50 |
| cfg3 (20261004) | no | no | 1 of 52 | 0 of 50 |

`S ∩ D = ∅` in all four, as required by the contract.

**Streams.**  Sixteen streams are involved: the four original ones (state, batch 1, batch 2, batch 3)
and twelve new ones (state, selection, evaluation for each of the four configurations).  All **120
pairs** were checked before the run:

```
{"streams_checked": 16, "pairs_checked": 120, "overlapping_pairs": {}, "disjoint": true}
```

## 3. How each configuration was run

The existing two-batch framework, unchanged in shape:

1. **state** --- 1000 cascades, used only for the state-conditioned score;
2. **selection** --- 1000 cascades, paired marginals for all 50 candidates;
3. **choices frozen here** --- `degree` top-8, `static_delta2` top-8, `state_delta2` top-8 and the
   full-50 argmax each take their best candidate by the selection-batch marginal; `random` draws a
   top-8 100 times and takes its best by the same rule;
4. **evaluation** --- 1000 cascades on a disjoint stream, for all 50 candidates, with the per-trial
   arrays saved.

The frozen choices were written to a per-configuration file and hashed **before** that
configuration's evaluation batch was generated.  `--merge` refuses to produce the artifact if any of
the four files is missing or if its hash no longer matches, so no candidate could be changed after
the evaluation numbers existed.

The full-50 pick is the **MC reference candidate** --- the best of 50 under one finite selection pass,
never the optimum.  Per configuration: `1000 + 51 000 + 51 000 = 103 000` cascades, 2920--3034 s each,
four run in parallel.

## 4. The four-row summary table

Independent-evaluation marginal gain `Δ(v|S) = |P_final(S ∪ {v}) ∩ D| − |P_final(S) ∩ D|`, mean over
1000 trials:

| new configuration | degree | static | state | random-screening mean | MC reference |
|---|---|---|---|---|---|
| cfg0 (20261001) | 0.221 | 0.359 | 0.314 | **0.421** | 0.139 |
| cfg1 (20261002) | **1.067** | 0.448 | 0.465 | 0.566 | 0.759 |
| cfg2 (20261003) | 0.947 | 0.947 | 0.947 | 0.794 | 0.947 |
| cfg3 (20261004) | 0.433 | 0.433 | 0.201 | 0.299 | 0.433 |

Random-screening spread over the 100 draws (the same configuration each time):

| configuration | mean | p05 | median | p95 |
|---|---|---|---|---|
| cfg0 | 0.421 | 0.139 | 0.381 | 0.763 |
| cfg1 | 0.566 | 0.142 | 0.463 | 1.067 |
| cfg2 | 0.794 | 0.129 | 0.800 | 1.417 |
| cfg3 | 0.299 | 0.063 | 0.253 | 0.660 |

In cfg2 **all four methods selected the same candidate** (839), and in cfg3 `degree`, `static` and the
MC reference all selected 5840.  Where two methods chose the same candidate the contrast between them
is exactly zero by construction; it is an exact tie, **not evidence of agreement**.

## 5. The three paired differences

Paired per trial on the independent evaluation batch, same windows within a configuration:

| contrast | cfg0 | cfg1 | cfg2 | cfg3 | positive | interval excludes zero |
|---|---|---|---|---|---|---|
| **MC reference − degree** | −0.082 ±0.230 [−0.532, +0.368] | −0.308 ±0.415 [−1.122, +0.506] | 0 (tie) | 0 (tie) | **0/4** | **0/4** |
| **MC reference − static** | −0.220 ±0.427 [−1.056, +0.616] | +0.311 ±0.302 [−0.281, +0.903] | 0 (tie) | 0 (tie) | 1/4 | **0/4** |
| **state − static** | −0.045 ±0.409 [−0.846, +0.756] | +0.017 ±0.212 [−0.398, +0.432] | 0 (tie) | −0.232 ±0.217 [−0.656, +0.192] | 1/4 | **0/4** |

## 6. The answer

**No stable screening headroom was detected, and the prerequisite for building a learned screener is
not met by this measurement.**

- The **MC reference never beat `degree`** in any of the four configurations: 0 positive, 2 exact
  ties, 2 negative.
- **Not one** of the twelve contrast-by-configuration cells has an interval excluding zero.
- The best method changes with the configuration and is never the expensive one for a reason that
  generalises: random wins cfg0, `degree` wins cfg1, everything ties in cfg2, `degree`/`static` tie in
  cfg3.  `degree` is best or tied-best in **three of the four** configurations and in the fourth is
  0.200 below the best random draw --- **at zero cascade cost**.
- Random screening beat `degree` in cfg0 only (0.421 versus 0.221) and lost in the other three.

**But this is a "not detected", and the design is the reason.**  With 1000 trials the paired standard
errors are 0.21--0.43, so the 95% intervals span roughly ±0.4 to ±0.8 target nodes, while the effects
at stake are 0.02--0.31.  Four configurations with two informative cells each is also far too few for
a sign test: 0 of 2 informative cells positive against `degree` gives a one-sided sign-test p of 0.25.
This run cannot establish that expensive screening does *not* help; it establishes that at this budget
it does not show up.

**What the design would have needed** --- and this number must be read for exactly what it is: a
**post-hoc extrapolation of evaluation precision for a fixed candidate difference**.  It is *not* the
sample size the **selection** stage would need to find good candidates, and it is *not* a formal power
analysis.  It says only: had the two candidates' difference been this size and fixed, the interval on
it would have needed this many trials.  A contrast whose true effect is zero is not resolved by any
number of trials.

| contrast | cfg0 | cfg1 | cfg2 | cfg3 |
|---|---|---|---|---|
| MC reference − degree | 30,154 | 6,981 | tie | tie |
| MC reference − static | 14,439 | 3,626 | tie | tie |
| state − static | 316,658 | 595,136 | tie | 3,346 |

The cheapest contrast here needs **3,346** trials and the most expensive **595,136**, so the fixed
1000-trial budget is far short of what these particular intervals would need.  What the selection stage
needs is a different question, and it is the one the budget-curve experiment
(`docs/SELECTION_BUDGET_CURVE.md`) was built to answer.

## 7. The selection gap: an observation, and a hypothesis that is not yet tested

The MC reference is chosen as the maximum over 50 candidates on the selection batch, so its
selection-batch value is a maximum and is upward-biased by construction.  Comparing it against the
same candidate's value on the independent batch gives an **observed difference between two finite
batches**:

| configuration | selection-batch value | independent evaluation | observed gap | pool evaluation mean / sd |
|---|---|---|---|---|
| cfg0 | +1.002 | +0.139 | **+0.863** | +0.200 / 0.211 |
| cfg1 | +0.890 | +0.759 | +0.131 | +0.227 / 0.297 |
| cfg2 | +0.898 | +0.947 | −0.049 | +0.287 / 0.369 |
| cfg3 | +0.781 | +0.433 | +0.348 | +0.161 / 0.169 |

**That gap cannot be attributed entirely to selection noise.**  Both batches are finite samples, the
independent evaluation carries its own error, and the two errors are not separable from this
comparison alone: part of the 0.863 is the selection maximum being too high, and part is the
evaluation estimate being too low.  Reading the whole gap as "selection noise" would treat a noisy
measurement as if it were the truth.

**Nor does the cross-candidate spread settle it.**  The pool's per-candidate evaluation standard
deviation (0.17--0.37) is computed from 1000-trial estimates, so it contains Monte-Carlo noise as well
as any real heterogeneity.  It cannot on its own establish that the true candidate differences are
large; it bounds them from above, not from below.

**The mechanism is therefore a hypothesis, not a finding.**  "With 1000 selection trials the argmax
over 50 is largely the argmax of the noise" is a plausible and testable explanation of the observed
gap.  It predicts something specific: **raising the selection budget should improve the chosen
candidate's independent gain.**  That prediction is what the budget-curve experiment tests, and it
could fail --- if the budget curve is flat, the explanation is wrong and the gap has to come from
somewhere else.

## 8. What this establishes, and what it does not

**Established.**

1. Four genuinely distinct new states were built under the frozen rule, with the same `D`, the same
   budget and the same scoring, and none repeats the original or another.
2. All sixteen streams are pairwise disjoint, checked before the run.
3. Choices were frozen and hashed before evaluation in every configuration, and the merge re-verifies
   the hashes.
4. **The MC reference was never better than `degree`** (0/4 positive) and **no contrast in any
   configuration had an interval excluding zero** (0/12 cells).
5. The selection-batch value of the MC reference exceeds its independent-batch value by 0.13--0.86
   target nodes.  This is an observed between-batch difference; **how much of it is selection noise is
   not identified here**, because both batches are finite and the evaluation has error of its own.

**Not established --- do not write these.**

- **That the selection gap is selection noise.**  It is a difference between two finite estimates and
  the two error contributions are not separable from this comparison.  The "argmax of the noise"
  explanation is a *hypothesis* with a testable prediction, not a finding.
- That the pool's true candidate differences are large.  Its cross-candidate spread is measured from
  1000-trial estimates and therefore contains Monte-Carlo noise; it is an upper bound, not a lower one.
- That the "3,346--595,136 trials" figures describe what the selection stage needs, or that they
  constitute a power analysis.  They are a post-hoc extrapolation of *evaluation* precision for a
  fixed candidate difference.
- That expensive screening is useless, or that simple screening is as good.  Eleven of the twelve
  cells have intervals containing zero; this design is underpowered for the effects present.
- That any method is *equivalent* to any other.  No equivalence test was run, and "not detected" is
  not "equivalent".
- That there is no candidate-screening benefit at ca-GrQc.  What is unmeasured is whether any rule can
  identify the good candidates within a feasible budget.
- That `degree` is the best screener.  It is competitive in all four and best in one, which is a
  description of four configurations, not a general claim.
- Anything pooled across the four configurations, and anything about other graphs, fractions or
  budgets.
- Any reading of the random method's 100 draws as 100 independent scenarios.

## 9. Consequence for the learned-screener decision

The question "is it worth learning a screener?" needs a preceding answer: **is there a screening gain
that a feasible amount of simulation can capture?**  On this evidence the answer is not yet yes --- and
not no either.  What this run does show is that a learned screener trained on 1000-trial labels would
be trained on labels of exactly the quality that produced this null result.

The mechanism proposed in §7 makes a specific prediction --- raising the selection budget should
improve the chosen candidate's independent gain --- and that prediction is testable without training
anything.  `docs/SELECTION_BUDGET_CURVE.md` tests it over the same four configurations, the same
frozen `D`, the same candidate order and the same rule, changing only the selection budget, with a new
6000-trial independent confirmation.  If the budget curve is flat, the proposed explanation is wrong
and the gap must have another source; if it rises and separates from simple screening, there is a
measured target worth learning.  Either outcome is informative, and neither is assumed in advance.

## 10. Provenance

| item | value |
|---|---|
| script | `scripts/audit/verify_grqc_new_states.py` |
| artifact | `docs/results/grqc_new_states.json` |
| per-configuration outputs | `docs/results/grqc_new_states_cfg{0..3}.json` |
| frozen choices | `docs/results/grqc_new_states_choices_cfg{0..3}.json` |
| frozen SHA-256 | cfg0 `ca71d629…`, cfg1 `6a1f7a64…`, cfg2 `8594f66b…`, cfg3 `06f1c4d7…` |
| code version | `cdcc2db` |
| new seeds | 20261001, 20261002, 20261003, 20261004 |
| trials | 1000 per batch (state, selection, evaluation) |
| per configuration | 103,000 cascades; 2920--3034 s |
| `complete` | `true` |

## 11. Reproduce

```
python scripts/audit/verify_grqc_new_states.py --check-streams-only
python scripts/audit/verify_grqc_new_states.py --config-index 0    # ... 1 2 3
python scripts/audit/verify_grqc_new_states.py --merge
```

`--check-streams-only` exits non-zero if any two of the sixteen streams overlap.  `--merge` refuses to
run unless all four frozen-choices files exist and still hash to what was written before evaluation.
The four `--config-index` runs are independent and were executed in parallel; each writes its own
files, so there is no shared-state race.
