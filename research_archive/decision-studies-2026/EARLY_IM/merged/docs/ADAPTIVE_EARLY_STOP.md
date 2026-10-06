# Adaptive early stopping on paired gaps: a zero-simulation prototype

> 2026-09-19.  Tests one specific improvement --- cheap decisions for candidates that are easy to tell
> apart, more simulation for those that are not --- entirely by **replaying data already on disk**.
> No new diffusion simulation was run.
>
> **Status: exploratory offline replay on historical data.**  This is **not** an independent
> confirmation experiment.  The evaluation segment lies inside the same simulation stream as the
> decision prefix, and these trials have already been used for research analysis.  Nothing here is an
> established contribution.

---

## 1. What was tested

Candidates and screening method are **unchanged**: the four configurations' existing `degree` top-8.
Only the **budget** is allowed to vary.

| strategy | rule |
|---|---|
| fixed 1000 | highest mean marginal over the first 1000 trials |
| fixed 3000 | highest mean marginal over the first 3000 trials --- the current cheap baseline |
| **adaptive** | check at 300 and 1000; hard stop at 3000 |

The adaptive rule, fixed in advance and **not tuned**:

1. at a checkpoint, let `b` be the candidate with the highest mean marginal so far;
2. for every other candidate `v`, form the **paired** per-trial difference on the same trial,
   `d_t = Δ_t(b) − Δ_t(v)`, and compute `mean(d)` and `SE(d)`;
3. stop and choose `b` only if **every** competitor satisfies `mean(d) > 3 · SE(d)`;
4. if any `SE(d)` is exactly zero, **do not** stop --- a finite sample with no observed variation is
   not evidence of certainty;
5. if no checkpoint triggers, run to 3000, take the highest mean, ties by ascending node id.

**The `3 · SE` factor is a pre-fixed heuristic, not a proven confidence bound, and it carries no
95%-correct-selection guarantee.**  No other multiple was tried and no threshold was searched for on
these four configurations.

Fixed 1000 is included deliberately: without it, "we compute less" could be presented as if it were an
adaptive advantage.

## 2. The replay

| segment | trials | used for |
|---|---|---|
| `[0, 3000)` | 3000 | the decisions; a checkpoint may read only the prefix revealed at that point |
| `[3000, 4000)` | 1000 | **deliberately unused** |
| `[4000, 10000)` | 6000 | evaluation of the selected candidate |

Decisions were computed and written **before** the evaluation segment was read.  Because all 8
candidates were simulated for all 10000 trials, an early stop that selected a different candidate
would need no re-run --- its evaluation data already exists.  That is the whole point of using this
data for the prototype.

**Costing.**  One decision costs `9 × stopping trials` cascades (one base run per trial plus one run
per candidate screened): 300 → **2,700**, 1000 → **9,000**, 3000 → **27,000**.  This is a
**replay-derived decision cost, not a measured speed-up.**

## 3. The four-row table

| configuration | early-stop trial | early-stop candidate | same as fixed 3000? | decision cascades | gain difference vs fixed 3000 |
|---|---|---|---|---|---|
| cfg0 | 3000 | 4960 | **yes** | 27,000 | **same choice** |
| cfg1 | 3000 | 3718 | **yes** | 27,000 | **same choice** |
| cfg2 | 3000 | 4960 | **yes** | 27,000 | **same choice** |
| cfg3 | 3000 | 5840 | **yes** | 27,000 | **same choice** |

**The rule never triggered.**  In all four configurations it ran to the 3000 hard stop, so the adaptive
column is *identical* to fixed 3000 by construction; the contrast is exactly zero and is not evidence
of anything.  Early stops: **0 of 4**.

Fixed 1000, for comparison:

| configuration | fixed-1000 candidate | cascades | gain | fixed-3000 candidate | cascades | gain |
|---|---|---|---|---|---|---|
| cfg0 | 4960 | 9,000 | 0.388 | 4960 | 27,000 | 0.388 |
| cfg1 | 3718 | 9,000 | 0.784 | 3718 | 27,000 | 0.784 |
| cfg2 | **839** | 9,000 | **0.505** | **4960** | 27,000 | **1.326** |
| cfg3 | 5840 | 9,000 | 0.812 | 5840 | 27,000 | 0.812 |

## 4. Why it never stopped

At each checkpoint, how many of the **seven** competitors satisfied `mean(d) > 3 · SE(d)`:

| configuration | @300 | @1000 | @3000 (hard stop) |
|---|---|---|---|
| cfg0 | 0 / 7 | 0 / 7 | **2 / 7** |
| cfg1 | 0 / 7 | 0 / 7 | **4 / 7** |
| cfg2 | 0 / 7 | 2 / 7 | **5 / 7** |
| cfg3 | 0 / 7 | 1 / 7 | **2 / 7** |

The binding competitor is always the **closest** one, and its gap is small compared with its own
standard error:

| configuration | @300 binding | @1000 binding | @3000 binding |
|---|---|---|---|
| cfg0 | vs 7811, mean/SE = +0.02 | vs 6431, +0.39 | vs 7811, **+0.16** |
| cfg1 | vs 3718, +0.19 | vs 1608, +0.26 | vs 379, **+1.28** |
| cfg2 | vs 920, +0.25 | vs 4960, +0.35 | vs 839, **+1.48** |
| cfg3 | vs 2805, +0.91 | vs 2805, +0.53 | vs 2049, **+0.82** |

Two things follow, and they are the substance of this result:

1. **The stop condition was never satisfiable at any tested budget, including the maximum one.**  The
   runs ended because of the hard stop, not because the criterion was met --- at 3000 the best case is
   cfg2 with 5 of 7 competitors still short.  Requiring **all seven** gaps to clear 3 SE makes the
   closest pair decisive, and with 8 candidates there is nearly always one close pair.
2. **The leader is not stable at small budgets.**  In cfg1 the leader changed from 379 at 300 to 3718
   at 1000; in cfg2 it changed from 839 at 300 and 1000 to 4960 at 3000.  A decision taken at 300 or
   1000 would have been taken on a leader that later lost.

## 5. What the non-trigger bought, and what it cost

**Against fixed 1000** --- the comparison that shows whether the adaptive rule does anything right:

| configuration | adaptive vs fixed 1000, on the evaluation segment | cascades |
|---|---|---|
| cfg0 | same choice (4960) | costs **18,000 more** |
| cfg1 | same choice (3718) | costs **18,000 more** |
| cfg2 | **+0.820** [+0.493, +1.148] | costs **18,000 more** |
| cfg3 | same choice (5840) | costs **18,000 more** |

So the rule paid double in every configuration and bought something in **one** of four.  In cfg2 the
purchase was large and clear: fixed 1000 selected 839 (0.505) while 3000 selected 4960 (1.326).  The
rule's refusal to stop at 1000 is exactly what avoided that loss --- at 1000 the leader *was* 839, and
its gap to 4960 was only +0.35 with `SE = 0.37`.  **The conservatism was protective here.**

That is a counterfactual on a single configuration and it is not a measured saving.  It does say
something useful: the rule's failure to stop is not purely wasted work --- in cfg2 it was the
difference between a mediocre and a good choice.

## 6. Verdict for this round

The question was whether the improvement shows behaviour worth continuing, not whether anything is
significant.

- **Can it stop early in some configurations?**  **No --- 0 of 4.**
- **If it stops early, does it choose wrong, and how much is lost?**  No early stop occurred, so no
  loss was realised.  The counterfactual in cfg2 (+0.820 had it stopped at 1000) says the rule's
  conservatism can be protective, on one configuration.
- **Compared with simply fixing at 1000, does the adaptive rule do anything right?**  It is
  **exactly fixed 3000** on all four configurations.  Against fixed 1000 it paid 2× in every
  configuration and changed the outcome in one.

**On these four configurations the rule achieved zero savings.**  By the criterion set before the run
--- *if everything runs to 3000, the rule has not delivered a saving* --- that is the outcome, and it
is recorded as such.  The threshold was **not** adjusted afterwards.

**Structural diagnosis, not a tuning proposal.**  The condition is governed by the closest of seven
competitors, and the diagnostics show several of those gaps are genuinely small (cfg0's leader is ahead
of 7811 by +0.16 with `SE = 0.23` at 3000).  Relaxing the rule --- fewer candidates, a different test,
a different margin --- would be a **new rule** that has to be pre-registered and validated on **new
configurations**.  Tuning it on these four would fit the rule to the data that produced it.

## 7. The two constraints

Both were checked by re-running the pipeline, not argued:

| constraint | cfg0 | cfg1 | cfg2 | cfg3 |
|---|---|---|---|---|
| the decision is unchanged when the evaluation segment is replaced by garbage | **pass** | **pass** | **pass** | **pass** |
| a non-triggering run equals fixed 3000 exactly | **pass** | **pass** | **pass** | **pass** |

The first was tested by overwriting trials `[4000, 10000)` with uniform noise and requiring the chosen
candidate, the stopping trial and every checkpoint leader to be identical.  The second holds trivially
here because nothing triggered, but it is asserted rather than assumed.

## 8. What this establishes, and what it does not

**Established.**

1. The specified rule **does not stop early** on any of the four configurations, and its stop
   condition is unmet even at the 3000 hard stop (best case 5 of 7 competitors).
2. The rule therefore **saves nothing** relative to fixed 3000 on this data, and costs 2× fixed 1000
   for an outcome that differs in 1 of 4 configurations.
3. The leader at 300 and at 1000 is **not stable** in two of the four configurations.
4. Both pre-registered constraints hold: the decision never reads the evaluation segment, and a
   non-triggering run reproduces fixed 3000.

**Not established --- do not write these.**

- That adaptive early stopping cannot work.  One fixed rule, one checkpoint schedule, one multiplier,
  eight candidates and four configurations is a single data point about a family of rules.
- That the `3 · SE` margin is wrong or right.  No other margin was tried, deliberately.
- That any early-stop decision would have been wrong.  No early stop happened; cfg2's loss is a
  counterfactual.
- That the evaluation-segment gains are comparable to the confirmation-stream gains in
  `docs/SELECTION_BUDGET_CURVE.md`.  They come from different streams and must not be put in one
  table.
- Anything about `state_delta2`, other graphs, other fractions, or configurations not measured here.
- Any claim of a speed-up, a confidence guarantee, or a contribution.

## 9. Provenance

| item | value |
|---|---|
| script | `scripts/audit/adaptive_early_stop.py` |
| artifact | `docs/results/adaptive_early_stop.json` |
| log | `docs/results/adaptive_early_stop.log` |
| data source | `docs/results/sbc_sel_cfg{0..3}_{00000_03334,03334_06668,06668_10000}.json` (digests re-verified on load) |
| candidate pool | `grqc_new_states_choices_cfg{0..3}.json` → the existing `degree` top-8 |
| code version | `76b397a` |
| new simulation | **none** |
| `complete` | `true` |

## 10. Reproduce

```
python scripts/audit/adaptive_early_stop.py
```

The script re-verifies every selection chunk's digest and completeness before use, refuses to run on
an incomplete chunk, writes the decisions before reading the evaluation segment, and asserts both
constraints.  It takes well under a minute because it simulates nothing.
