# Successive-halving candidate elimination: 8 → 4 → 2 → 1 at 17000 cascades

> 2026-09-19.  A **zero-new-simulation development diagnostic**.  The previous prototype asked whether
> the *whole decision* could stop early; it could not, and that line is archived.  This one asks a
> different question: can simulation be **withdrawn from trailing candidates**?
>
> **Not a novel algorithm.**  Progressively eliminating trailing candidates is the established
> successive-halving / racing idea, and budget-allocation methods of this kind are well documented.
> Nothing here is presented as new.
>
> **Status: development diagnostic on development data.**  These four configurations have already been
> used for research analysis.  Even a good result here would **not** be an independently validated
> rule.  The requirement to confirm on new configurations still stands and is not satisfied by
> anything below.

---

## 1. The rule, fixed in advance

Candidates and screening method are unchanged: the four configurations' existing `degree` top-8.

| stage | candidates computed | decision data | stage end |
|---|---|---|---|
| 1 | all 8 | trials `[0, 1000)` | keep top 4 by mean |
| 2 | the surviving 4 | cumulative `[0, 2000)` | keep top 2 by mean |
| 3 | the surviving 2 | cumulative `[0, 3000)` | pick 1 by mean |

Ties break by ascending node id.  An eliminated candidate **never returns**, and **no stage length or
ratio was adjusted** after seeing a result.

**The risk is explicit**: early noise can eliminate the candidate that would have won.  The point of
this run is to check whether that risk cancels the saving.

## 2. Cost, and why cost is not what is being tested

| method | cascades | breakdown |
|---|---|---|
| fixed 1000 | **9,000** | 1000 × 9 |
| **fixed 1888** | **16,992** | 1888 × 9 |
| **successive halving** | **17,000** | 8×1000 + 4×1000 + 2×1000 = **14,000** candidate runs, plus **3,000** shared base runs |
| fixed 3000 | **27,000** | 3000 × 9 |

Halving is **37.0% cheaper** than fixed 3000.  **That saving is guaranteed by the schedule** — there is
nothing to test about whether computation can be saved.  The only open question is **quality**.

**The near-equal-cost control is mandatory.**  `fixed1888` costs 16,992 cascades, **8 fewer** than the
halving schedule — a difference that comes only from integer rounding.  If uniform sampling at the same
cost performs about as well, then any gain **cannot be attributed to allocating the budget cleverly**.

## 3. Data split

Decisions read only trials `[0, 3000)`; evaluation uses `[4000, 10000)`; no new diffusion simulation.
A candidate eliminated at stage `s` must never have its later trials read, and that is **tested by
garbling them** rather than assumed: for every eliminated candidate, trials from its elimination point
to 3000 are overwritten with uniform noise, and the whole trajectory — every stage ranking and every
survivor set — must be unchanged.  It is, in **all four** configurations.

## 4. Elimination trajectories

`fixed 3000`'s final candidate is marked in each stage.

### cfg0 — degree top-8 `[115, 408, 820, 4554, 4960, 5355, 6431, 7811]`

| stage | trials read | candidate means | kept | eliminated |
|---|---|---|---|---|
| 1 | `[0, 1000)` | 4960 **0.819**, 6431 0.652, 7811 0.429, 5355 0.321, 820 0.155, 408 0.135, 115 0.000, 4554 −0.056 | 4960, 6431, 7811, 5355 | 820, 408, 115, 4554 |
| 2 | `[1000, 2000)` | 7811 **0.610**, 6431 0.577, **4960 0.531**, 5355 0.443 | 7811, 6431 | **4960** ← `fixed 3000`'s candidate, 5355 |
| 3 | `[2000, 3000)` | 7811 **0.546**, 6431 0.468 | **7811** | 6431 |

**`fixed 3000`'s candidate (4960) was eliminated at trial 2000.**  Halving's winner is 7811.

### cfg1 — `[379, 884, 1608, 1833, 1834, 1969, 3239, 3718]`

| stage | trials read | candidate means | kept | eliminated |
|---|---|---|---|---|
| 1 | `[0, 1000)` | **3718 0.613**, 1608 0.516, 1834 0.436, 379 0.414, 1833 0.224, 884 0.096, 1969 0.000, 3239 0.000 | 3718, 1608, 1834, 379 | 1833, 884, 1969, 3239 |
| 2 | `[1000, 2000)` | **3718 0.991**, 379 0.800, 1608 0.496, 1834 0.474 | 3718, 379 | 1608, 1834 |
| 3 | `[2000, 3000)` | **3718 1.081**, 379 0.754 | **3718** | 379 |

**`fixed 3000`'s candidate (3718) survived every stage.**

### cfg2 — `[839, 920, 2074, 2504, 2883, 3744, 3852, 4960]`

| stage | trials read | candidate means | kept | eliminated |
|---|---|---|---|---|
| 1 | `[0, 1000)` | 839 **0.898**, 4960 0.767, 920 0.731, 2504 0.569, 2074 0.502, 2883 0.296, 3744 0.000, 3852 0.000 | 839, 4960, 920, 2504 | 2074, 2883, 3744, 3852 |
| 2 | `[1000, 2000)` | **4960 1.104**, 839 0.769, 920 0.681, 2504 0.354 | 4960, 839 | 920, 2504 |
| 3 | `[2000, 3000)` | **4960 1.134**, 839 0.822 | **4960** | 839 |

**`fixed 3000`'s candidate (4960) survived every stage** — note it was *second* after stage 1 and took
the lead at stage 2.

### cfg3 — `[750, 2049, 2805, 3927, 4019, 4189, 5840, 6291]`

| stage | trials read | candidate means | kept | eliminated |
|---|---|---|---|---|
| 1 | `[0, 1000)` | **5840 0.781**, 2805 0.631, 4189 0.561, 2049 0.454, 750 0.253, 6291 0.234, 3927 0.178, 4019 0.000 | 5840, 2805, 4189, 2049 | 750, 6291, 3927, 4019 |
| 2 | `[1000, 2000)` | **5840 0.628**, 2049 0.555, 2805 0.473, 4189 0.469 | 5840, 2049 | 2805, 4189 |
| 3 | `[2000, 3000)` | **5840 0.654**, 2049 0.509 | **5840** | 2049 |

**`fixed 3000`'s candidate (5840) survived every stage.**

**Summary: `fixed 3000`'s final candidate was eliminated early in 1 of 4 configurations (cfg0).**

## 5. Cost, gain, and the comparisons

Evaluation segment `[4000, 10000)`:

| configuration | fixed 1000 (9,000) | fixed 1888 (16,992) | halving (17,000) | fixed 3000 (27,000) |
|---|---|---|---|---|
| cfg0 | 4960 / 0.388 | 6431 / 0.501 | **7811 / 0.587** | 4960 / 0.388 |
| cfg1 | 3718 / 0.784 | 3718 / 0.784 | 3718 / 0.784 | 3718 / 0.784 |
| cfg2 | 839 / 0.505 | 4960 / 1.326 | 4960 / 1.326 | 4960 / 1.326 |
| cfg3 | 5840 / 0.812 | 5840 / 0.812 | 5840 / 0.812 | 5840 / 0.812 |

**Halving minus fixed 1888 — the near-equal-cost control:**

| configuration | difference | 95% CI | p |
|---|---|---|---|
| cfg0 | **+0.085** | [−0.183, +0.354] | 0.534 |
| cfg1 | **same choice** (3718) | — | — |
| cfg2 | **same choice** (4960) | — | — |
| cfg3 | **same choice** (5840) | — | — |

**Halving minus fixed 3000:**

| configuration | difference | 95% CI | p |
|---|---|---|---|
| cfg0 | **+0.198** | [−0.076, +0.473] | 0.157 |
| cfg1 | **same choice** (3718) | — | — |
| cfg2 | **same choice** (4960) | — | — |
| cfg3 | **same choice** (5840) | — | — |

In three of the four configurations halving, fixed 1888 and fixed 3000 all select the **same
candidate**, so there is nothing to attribute.  **Only cfg0 carries information**, and there the
halving-versus-near-cost-control difference is +0.085 with an interval containing zero.

## 6. Verdict

**The elimination method shows no established advantage over uniform sampling at the same cost.**

Per the criterion set before the run, that is the outcome that **pauses this budget-allocation route**.
Two further points belong in the record:

- In cfg0 the halving method did **not** lose by eliminating `fixed 3000`'s candidate early — it
  gained, nominally.  Evicting 4960 at trial 2000 left 7811, whose evaluated gain (0.587) is above
  fixed 3000's own pick (0.388).  So on this one configuration the elimination risk did not cost
  anything; but the gain over the near-equal-cost control is not established, and one configuration is
  one configuration.
- **The saving came from spending less, not from spending smartly.**  Both ~17,000-cascade methods
  matched fixed 3000 in all four configurations — three exact ties each, and cfg0 nominally better in
  both cases.  That is a statement about the **budget level**, not about the allocation scheme, and it
  is consistent with the earlier budget-curve result that 3000 and 10000 gave `full50` the same
  candidate.  It is also a statement about four development configurations and nothing more.

**No ratio or stage length is adjusted here.**  Tuning the schedule on the same four configurations
that produced this table would fit it to its own data.

## 7. What this establishes, and what it does not

**Established.**

1. The schedule costs **17,000** cascades against fixed 3000's 27,000 — a **37.0%** reduction that is
   guaranteed by construction.
2. `fixed 3000`'s final candidate was eliminated early in **1 of 4** configurations.
3. Against the near-equal-cost control, halving's advantage is **not established**: three exact ties
   and one difference of +0.085 [−0.183, +0.354].
4. An eliminated candidate's later trials are **never read** — verified by garbling them in all four
   configurations.
5. At ~17,000 cascades, both the halving schedule and uniform sampling matched fixed 3000's choice in
   every configuration.

**Not established — do not write these.**

- That successive halving helps.  It cannot be separated from uniform sampling at the same cost on
  this evidence.
- That successive halving is novel.  It is the standard racing / successive-halving idea.
- That early elimination is safe.  It eliminated `fixed 3000`'s candidate in cfg0, and the fact that
  this happened to help once is not a safety property.
- That ~17,000 cascades is the right budget.  That observation comes from four development
  configurations, is a statement about the budget level rather than the allocation, and needs new
  configurations before anything is claimed.
- Any claim of a speed-up (these are cascade counts, not wall time), any significance claim, or any
  contribution.
- Anything about `state_delta2`, other graphs, other fractions, or configurations not measured here.
- Any comparison with the confirmation-stream gains in `docs/SELECTION_BUDGET_CURVE.md`: these gains
  come from a different segment of a different stream and the two tables must not be merged.

## 8. Provenance

| item | value |
|---|---|
| script | `scripts/audit/successive_halving.py` |
| artifact | `docs/results/successive_halving.json` |
| log | `docs/results/successive_halving.log` |
| data source | `docs/results/sbc_sel_cfg{0..3}_{00000_03334,03334_06668,06668_10000}.json` (digests re-verified on load) |
| candidate pool | `grqc_new_states_choices_cfg{0..3}.json` → the existing `degree` top-8 |
| code version | `a784f0a` |
| new simulation | **none** |
| `complete` | `true` |

## 9. Reproduce

```
python scripts/audit/successive_halving.py
```

It re-verifies every chunk digest before use, refuses to run on an incomplete chunk, performs the
garbling test on the eliminated candidates, and simulates nothing.
