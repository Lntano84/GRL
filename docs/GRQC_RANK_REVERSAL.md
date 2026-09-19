# ca-GrQc rank reversal: independent replication and input ablation

> 2026-09-19.  Independent replication of the one finding `docs/SHORTLIST_HEADROOM.md` singled out,
> plus an ablation that separates the two inputs the static and state scores change at once.
> Configuration frozen from `docs/results/shortlist_headroom.json`; **no candidate re-drawn, no
> target added, no existing seed changed**.  The confirmation batch is new, independent, and fixed at
> 6000 trials with no early stop.  Original artifacts are not overwritten.

---

## 0. Correction of the earlier conclusion

The earlier document reported, correctly, that on ca-GrQc 1% the state-conditioned closed form ranks
the Monte-Carlo reference candidate **50th of 50** where the state-free form ranks it **1st of 50**.
It then explained that as the state-conditioning term `g(E_S + dE) − g(E_S)` going negative, and that
explanation is arithmetically correct.  The mistake was in what it was taken to mean.

**What is established is a rank reversal in the score.  What is not established --- and is *not*
established by anything below either --- is that the state information is harmful.**  The correct
statement of the result is:

> The exposure input causes a reproducible rank change, but at this configuration **no quality
> difference between the two selections was detected** in 6000 independent confirmation trials.

"Not detected" is not "equivalent".  A paired interval that contains zero is insufficient evidence in
both directions, and it must not be written up as a demonstration that the two methods agree.

Three reasons the earlier reading was wrong, all of them visible in that document's own numbers:

1. **The two candidates were never separable.**  The earlier run measured the reference-versus-runner-up
   gap as `+0.440 [−0.345, +1.225]` on batch 1 and `+0.530 [−0.317, +1.377]` on batch 2 — both
   straddling zero.  "The score ranked the best candidate last" therefore never meant the score
   discarded a candidate known to be better; it meant the score reordered two candidates that one
   thousand paired trials could not tell apart.  Ranking cannot be called *wrong* against an ordering
   that was never measured.
2. **A negative score is not a negative gain.**  The score is a closed form over a proxy; the gain is
   `|P_final(S ∪ {v}) ∩ D| − |P_final(S) ∩ D|`.  The earlier document did not measure the gain of the
   reversal at all, and §6 of it should not have been read as if it had.
3. **The retracted reading.**  A handoff summary of that document said "what fails is the state
   conditioning, not the exposure proxy".  That sentence is **withdrawn here**: it attributed a
   quality failure on the strength of a score ordering.  What the earlier data supported was only that
   the exposure proxy correlates with the marginal and the state-conditioned form ranks differently.

This document measures the missing quantity.

---

## 1. What was run

Graph, weight normalisation (`sum_to_one`), target set `D`, existing seed set `S`, the 50 candidates
and the batch-1 marginals are **read** from `docs/results/shortlist_headroom.json` and re-derived only
to check them:

| item | value |
|---|---|
| graph | `ca_grqc`, n = 5242 |
| existing seed set `S` | 52 nodes (1.0% of n), reconstructed exactly |
| target set `D` | 1048 nodes (`degree-tail`, 20%), reconstructed exactly |
| candidates | the original 50, list identical to the stored one |
| shortlist | 8 |
| target/graph relation | `S ∩ D = ∅` (0 of 52 seeds inside `D`) |

The four versions differ **only in the score's inputs**.  Score function, hop depth (2), candidate
list, `D`, `S` are all held fixed, and every real simulation still starts from the actual `S`:

| version | exposure input | seed mask passed to the score |
|---|---|---|
| `A` | all-zero (empty state) | empty set — **reproduces the original static score** |
| `B` | all-zero (empty state) | the current `S` |
| `C` | current mean exposure | empty set |
| `D` | current mean exposure | the current `S` — **reproduces the original state score** |

The mean exposure was reproduced from the original state stream (`base_seed + 900_000`, 1000 trials,
`step=0`), not re-drawn; the empty-state exposure was verified to be exactly zero before use.

## 2. Reproduction check, before any evaluation

| check | result |
|---|---|
| `A` vs original `static_delta2`, max abs difference | **0.00e+00** |
| `D` vs original `state_delta2`, max abs difference | **0.00e+00** |
| `A` top-8 equals original static top-8 | **yes** |
| `D` top-8 equals original state top-8 | **yes** |
| rank of the reference candidate, `A` | **1** (original 1) |
| rank of the reference candidate, `D` | **50** (original 50) |

The reproduction is **bit-exact**.  The rank reversal is a reproducible property of the closed form,
not a transcription error, and it is not an artefact of a different graph, a different target set or a
different candidate list.

Stream disjointness was checked **before** the run, over all six pairs:

```
batch1_batch2 0   batch1_batch3 0   batch1_state 0
batch2_batch3 0   batch2_state 0   batch3_state 0      disjoint: true
```

## 3. Frozen candidate list

Each version's choice was decided by the **existing** batch-1 marginals, written to
`docs/results/grqc_rank_reversal_candidates.json`, and hashed before any new simulation:

| version | top-8 head | chosen |
|---|---|---|
| `A` | 8150, 15960, 13529, 11563, … | **8150** |
| `B` | 8150, 15960, 13529, 11563, … | **8150** |
| `C` | 15960, 9391, 5445, 4775, … | **15960** |
| `D` | 15960, 9391, 5445, 4775, … | **15960** |
| `degree` (kept for reference) | 733, 844, 1608, 2055, **3058**, … | **3058** |
| MC reference candidate | — | **8150** |

**`A` and `B` chose the same candidate; `C` and `D` chose the same candidate.**  Only **three** distinct
nodes needed evaluation — 3058, 8150, 15960 — so this is not a re-run of all 50 candidates.  The frozen
payload SHA-256 is `86e8b36a61e9c22a…`; the full value is recorded in the artifact, and the file was
regenerated identically by the evaluation run, so nothing was swapped after the numbers arrived.

## 4. Confirmation batch

| item | value |
|---|---|
| trials | **6000, fixed in advance, run to completion** |
| early stop | **none** — no stopping rule of any kind was applied |
| stream namespace | `base_seed + 1_500_000`, disjoint from state, batch 1 and batch 2 |
| cascades | 25,000 (4 × 6000 marginal runs + 1000 state reads) |
| windows | drawn once per trial and shared by every candidate |
| wall time | 616 s (replay pass; 518 s on the first pass, same numbers) |
| base level `\|P_final(S) ∩ D\|` | **186.926 ± 0.608** (sd 47.10, median 198, min 10, max 276) |

Old batches selected; batch 3 only confirms.  The two are **never pooled**.

## 5. Gain decomposition

Every candidate, every trial: `Δ(v|S)`, newly-positive targets, lost-positive targets.

| candidate | role | Δ mean | 95% CI | newly positive | lost positive | identity residual |
|---|---|---|---|---|---|---|
| 3058 | `degree` choice | +0.656 | [+0.463, +0.850] | 1.832 | 1.175 | 0.00e+00 |
| **8150** | `A`/`B` choice, MC reference | **+0.927** | [+0.658, +1.197] | 2.962 | 2.034 | 0.00e+00 |
| **15960** | `C`/`D` choice | **+0.823** | [+0.647, +0.999] | 1.526 | 0.703 | 0.00e+00 |

The identity **Δ = newly positive − lost positive holds exactly** (residual 0.00e+00) for all three,
asserted on every one of the 6000 trials rather than checked afterwards.

Non-monotonicity is large and directly visible: adding 8150 turns on 2.962 target nodes and turns
**off 2.034** on average, and it destroys at least one already-positive target in **2290 of 6000**
trials (15960: 1101/6000; 3058: 1414/6000).  The marginals are negative on 19.8% / 8.0% / 10.5% of
trials respectively.

## 6. The fixed comparisons

| contrast | what it changes | result |
|---|---|---|
| **`A − D`** (primary, pre-registered) | static vs state selection | 8150 − 15960 = **+0.104 ± 0.152**, 95% CI **[−0.194, +0.403]**, p = 0.49 |
| `A − B` | seed mask only | **same choice** (8150) — exactly zero by construction, **not evidence** |
| `C − D` | seed mask only | **same choice** (15960) — exactly zero by construction, **not evidence** |
| `A − C` | exposure input only | 8150 − 15960 = +0.104 ± 0.152, CI [−0.194, +0.403] |
| `B − D` | exposure input only | 8150 − 15960 = +0.104 ± 0.152, CI [−0.194, +0.403] |

**The primary comparison does not separate the two choices.**  No quality difference between the
state score's pick and the static score's pick was detected.  That is all it says.

**What the interval does and does not bound.**  The difference is defined as *static minus state*, so
the interval's two ends bound two different losses and must not be read as if they bounded one:

| quantity | bound from the CI `[−0.194, +0.403]` |
|---|---|
| how much worse the **state** pick could be than the static pick | **0.403 target nodes** (upper end) |
| how much worse the **static** pick could be than the state pick | **0.194 target nodes** (lower end) |

The bound on the state method's loss is **0.403, not 0.194**.  Both directions remain open, and neither
is converted into a verdict here.  In particular, **no gate tolerance is applied to this interval post
hoc**, and being a small fraction of `|D| = 1048` is **not** evidence of equivalence — that argument
would be a threshold invented after seeing the number, and it is not made.

**Multiplicity, handled explicitly.**  After removing the two same-choice contrasts, the remaining
contrasts (`A−C`, `B−D`) compare *the same candidate pair* as the primary — they are the primary
measurement restated, not additional tests.  There is therefore **exactly one independent
measurement**, no multiplicity correction is meaningful, and none is applied.  The four secondary
entries must not be reported as four findings.

**The comparison with the earlier point estimates matters.**  The earlier run put the same gap at
`+0.440` (batch 1) and `+0.530` (batch 2).  Both are consistent with the new `+0.104` — each old CI
contains it — but the new estimate is roughly eight times more precise.  The old point estimates were
not wrong; they were too imprecise to be read as a size, and the new estimate is still compatible with
a state-method loss as large as 0.403 target nodes.

## 7. The score side: what actually moved

| node | `A` static | `B` zero-exp + mask | `C` exposure, no mask | `D` state |
|---|---|---|---|---|
| **8150** | +0.4859, rank **1**, in top-8 | +0.4859, rank **1**, in | −0.0126, rank **50**, out | −0.0126, rank **50**, out |
| **15960** | +0.4081, rank 2, in | +0.4081, rank 2, in | +0.1529, rank **1**, in | +0.1529, rank **1**, in |
| **3058** | +0.2490, rank 8, in | +0.2490, rank 8, in | +0.0381, rank 16, out | +0.0381, rank 16, out |

Which input moves the ranking:

| pair | input changed | max abs score change | Spearman | candidates changing position | reference rank |
|---|---|---|---|---|---|
| `A` vs `B` | seed mask only (zero exposure) | **0.000000e+00** | +1.000 | **0 / 50** | 1 → 1 |
| `C` vs `D` | seed mask only (current exposure) | **0.000000e+00** | +1.000 | **0 / 50** | 50 → 50 |
| `A` vs `C` | exposure input only (empty mask) | 0.498464 | +0.759 | **47 / 50** | 1 → 50 |
| `B` vs `D` | exposure input only (mask = `S`) | 0.498464 | +0.759 | **47 / 50** | 1 → 50 |
| `A` vs `D` | both | 0.498464 | +0.759 | **47 / 50** | 1 → 50 |

**The rank change is caused entirely by the exposure input.**  The seed mask changes the scores by
exactly nothing here — not approximately, exactly.

That identity had to be explained rather than assumed, since a silently ignored argument would look
the same.  A liveness control, recorded in the artifact:

- `S ∩ D = ∅` by construction (seeds are drawn from `V \ D`), so the mask can never drop a summed term;
- only **3 of 50** candidates have a two-hop neighbourhood that touches a seed at all, and never more
  than one seed, so no maximising path is blocked;
- **control**: widening the mask with the candidates' own `D`-neighbours changes the scores by
  **0.485899** — proving the mask reaches the score.  The zero is a property of this configuration,
  not a dead code path.

**And the state score did not pick a bad candidate.**  15960's state score `+0.1529` is the *maximum*
of the state-δ₂ range, and its confirmed marginal is `+0.823 [+0.647, +0.999]` — statistically
indistinguishable from 8150's `+0.927 [+0.658, +1.197]`.  What the state input did was swap the top
two, not promote a worthless candidate.  Similarly 8150's static score `+0.4859` is the *maximum* of
the static range, so the state-free form is not obviously "right" either — it agrees with the batch-3
ordering only by an amount that is not established.

---

## 8. What this establishes, and what it does not

**Established.**

1. The rank reversal is **exact and reproducible**: versions `A` and `D` reproduce the original static
   and state scores to `0.00e+00`, including both top-8 lists and the reference's ranks 1 and 50.
2. The reversal is caused by the **exposure input**, not by the seed mask — which is *exactly* inert at
   this configuration, on a code path proven live by a control.
3. **No quality difference was detected.**  `A − D = +0.104 [−0.194, +0.403]`, p = 0.49, at 6000 fixed
   trials.  This is insufficient evidence in *both* directions: a state-method loss up to **0.403**
   target nodes and a static-method loss up to **0.194** remain compatible with the data.
4. The reversal is a **swap between two good candidates**, not the promotion of a bad one: both chosen
   candidates have positive confirmed marginals within ~0.10 of each other.
5. All three evaluated candidates are strongly **non-monotone** in the contracted objective: each
   destroys already-positive targets in a substantial minority of trials, 8150 in 38% of them.

**Not established — and these must not be written.**

- **That the state information is harmful.**  Not demonstrated, and equally **not excluded**: the
  interval leaves a state-method loss of up to 0.403 target nodes open.  Do not write "refuted", "no
  cost", "harmless" or "equivalent".
- That the state information is harmless.  The same interval leaves a static-method loss of up to 0.194
  open.  "Not detected" is not "equivalent", and no equivalence test was run.
- That either selection passes any quality gate.  No tolerance was applied to this interval, and the
  fact that 0.403 is a small fraction of `|D| = 1048` is **not** used here as evidence of anything —
  that would be a threshold chosen after seeing the result.
- That the static score is better than the state score.  `+0.104` is not distinguishable from zero.
- That the earlier `+0.44` / `+0.53` headroom figures were real.  The better-powered estimate is
  `+0.104`, inside noise; the honest reading of the earlier numbers is "unresolved", not "large".
- That 8150 is the best candidate in the pool.  It is the *batch-1* Monte-Carlo reference; batch 3 puts
  it above 15960 by an amount that is not established, and it was not compared against the other 47
  candidates here.
- Anything about other graphs, other fractions, other budgets, or three-hop exposure.  Nothing is
  pooled and nothing generalises from one configuration.
- Any inference from a score's sign or magnitude to a gain's sign or magnitude.  That inference is the
  exact error §0 corrects.
- That "there is no screening benefit at ca-GrQc".  This run tested two score *choices*, not whether any
  score can find a good candidate.  The three evaluated candidates have point estimates between
  `+0.656` and `+0.927`, but their CIs overlap and the 3058-versus-8150 contrast was **not** among the
  frozen comparisons, so no ordering among them is established here either.  The earlier top-8 recovery
  test found nothing that survives multiplicity, which is a separate and weaker statement than "no
  benefit exists".

## 9. Provenance

| item | value |
|---|---|
| script | `scripts/audit/verify_grqc_rank_reversal.py` |
| artifact | `docs/results/grqc_rank_reversal.json` |
| frozen candidates | `docs/results/grqc_rank_reversal_candidates.json` (sha256 `86e8b36a…`) |
| log | `docs/results/grqc_rank_reversal.log` |
| code version | `d06f719` |
| source artifact | `docs/results/shortlist_headroom.json` (code version `75b0635`) |
| state exposure vector | sha256 `9cf38f90…` (reproduced, not re-drawn) |
| streams | `base_seed = 20260917`; state `+900_000`, batch1 `+1_100_000`, batch2 `+1_300_000`, batch3 `+1_500_000` |
| `complete` | `true` |

## 10. Reproduce

```
python scripts/audit/verify_grqc_rank_reversal.py --check-only
python scripts/audit/verify_grqc_rank_reversal.py --trials 6000
```

`--check-only` performs the reproduction check, verifies stream disjointness and writes the frozen
candidate file, then exits without simulating.  It exits non-zero if `A`/`D` fail to reproduce the
original scores, if the stored configuration cannot be rebuilt, if the streams overlap, or if the
empty-state exposure is not all-zero.  The full run regenerates an identical frozen file before
evaluating, and the artifact records that file's hash.
