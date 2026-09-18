# Claim ledger — every number traceable to a result file

> Purpose: the paper must not contain a number that cannot be traced to a committed result
> file. This ledger is the single source of truth for the writing. If a claim is not listed
> here, it does not go in the paper.
>
> All numbers below were read from the JSON files under `docs/results/`, not from memory.

---

## STATUS OF EVERY CLAIM IN THIS LEDGER

> Rewritten 2026-09-18 after two findings that invalidate most of the numerical record.
>
> **Finding 1 — the state machine was wrong.** Until commit `0248612` a positively activated node
> could never become negative again, and a node with `δ = θ^τ = 1` was denied activation. The
> correction removes 60–63% of the measured spread on Congress-Twitter (k=1: 299.35 → 111.44;
> k=5: 358.52 → 137.28; k=10: 365.95 → 143.43). Every spread, marginal gain, and calibration
> number in this ledger was produced with the broken machine.
>
> **Finding 2 — the ledger's own MC warning was ignored in the paper.** C2 carries an explicit
> warning that its rank correlations came from `mc-runs = 12` and that only the sign pattern may be
> used. `docs/GATE3A_REPORT.md` records the producing command and says the same. The paper quoted
> the magnitudes anyway (abstract, introduction, regime section), against the C5 constraint that a
> rank correlation in this regime requires `MC ≥ 300`. A re-measurement at `MC = 300` under the
> fixed model gives Congress-Twitter at `|S|/n = 20%`: raw negative share `4%` (table says `44%`),
> `0%` negative beyond noise, `ρ_degree = +0.039`, `ρ_δ₂ = +0.030`.
>
> | claim | status |
> | --- | --- |
> | C1 RR sampling has no domain | **WITHDRAWN** — the probe was mis-specified. See C1 below. |
> | C1′ no non-negative seed-independent coverage form matches `F_D` | **STANDS** — exact arithmetic, no simulation |
> | C2 out-degree anti-signal | **WITHDRAWN** — two independent defects; partially contradicted |
> | C3 exact MC greedy fails | **WITHDRAWN** — pre-fix spread |
> | C4 advantage is not a selection artefact | **WITHDRAWN** — pre-fix spread |
> | C5 measurement caveat (MC ≥ 300) | **STANDS** — and it is the constraint the paper violated |
> | C6 learned scorer does not beat the closed form | **WITHDRAWN as a number**; the direction of the negative result is the honest expectation but must be re-measured |
> | C7 surrogate gap, measured | **WITHDRAWN as a number**; the structural half stands |
> | C7′ `λ` is a difference of two submodular functions | **STANDS** — algebraic, no simulation |
> | `σ^κ`, `σ^τ` are each monotone submodular LT spreads | **STANDS**, but they are LT spreads under the *simplex* marginals `F^κ(x)=2x−x²` and `F^τ(x)=x²`, **not** uniform-threshold LT |
>
> Nothing in this ledger may be cited until it is re-derived under the frozen model contract
> (`src/grl/diffusion/contract.py`) with the eight confounds of
> `paper/dasfaa2027/src/dasfaa2027/sections/limitations.tex` paragraph 7 removed.

---

## C1. ~~Reverse-reachability sampling is structurally inapplicable~~ — WITHDRAWN

**This claim is withdrawn.** The probe it rested on required a predecessor's window to contain zero
and excluded the root, and it tested whether the *returned set* was non-empty rather than whether it
*intersected the seed set*. A reverse-reachability set for a root answers "which seeds can reach this
root"; it does not require a node to self-activate under zero incoming influence. The measured
all-zero RR sets were an artefact of the probe, not a property of the model.

The second argument recorded here — that `δ` is a deterministic set function given the windows, so
neither a live-edge graph nor a trigger set exists — is **also wrong as stated**: `σ^κ` and `σ^τ`
each fix one threshold per node and *are* monotone submodular, so the coverage identity applies to
each of them individually.

**Superseded by C1′.**

| original quantity | value | status |
| --- | --- | --- |
| nodes whose window contains 0 | 0 / 15233 | not evidence of anything |
| RR-set size, 300 reverse BFS draws | all 0 | probe artefact |
| true Monte-Carlo spread `σ(S)`, `\|S\| = 50` | 111.93 | pre-fix model |

---

## C1′. No non-negative seed-independent coverage function matches the objective — STANDS

**Source** `scripts/audit/verify_coverage_argument.py`, generated into
`paper/dasfaa2027/src/dasfaa2027/sections/coverage.tex`
**Method** exact arithmetic on a one-hop instance, closed form validated against 2-D numerical
integration (worst absolute difference `4.5e-4`). No simulation.

Single target `D = {t}`, candidates `a, b`, edges `a → t` and `b → t` both of weight `ω = 0.40`:

| `S` | `δ` | `F_D(S)` under the model | `F_D(S)` under uniform-threshold LT |
| --- | --- | --- | --- |
| `∅` | 0.00 | 0.0000 | 0.0000 |
| `{a}` | 0.40 | 0.4800 | 0.4000 |
| `{a,b}` | 0.80 | 0.3200 | 0.8000 |

`F_D({a,b}) = 0.3200 < F_D({a}) = 0.4800`, so the objective is non-monotone. For any random set `R`
independent of `S`, `S ↦ c·Pr[S ∩ R ≠ ∅]` is non-decreasing and submodular in `S`; a non-monotone
`F_D` therefore cannot equal such a function on this instance, for any `c` and any distribution
of `R`. The argument fails under uniform-threshold LT, where the same instance *is* monotone
(`0.4000 → 0.8000`) — which is exactly why the threshold distribution must be named.

**Scope.** This excludes the standard non-negative seed-independent coverage representation only.
It does not exclude signed decompositions, restricted instance classes, Monte-Carlo estimation, or
the possibility that a learned predictor helps.

---

## C2. ~~Out-degree degrades to an anti-signal in the saturated regime~~ — WITHDRAWN

**Two independent defects.**

1. **Pre-fix state machine.** These spreads were measured before commit `0248612`.
2. **`MC = 12`.** The producing command is `docs/GATE3A_REPORT.md`:
   `python scripts/experiments/evaluate_density_degree_signflip.py --candidates 50 --mc-runs 12`.
   The same report warns, in the same block, that this is only good for trends and that the
   correlation numbers must not be cited from it. The stored JSON
   (`docs/results/density_degree_signflip_20260917.json`) confirms `candidates: 50, mc_runs: 12`.
   C5 requires `MC ≥ 300` for a rank correlation in this regime.

**Re-measurement under the fixed model at `MC = 300`**, script
`scripts/audit/rederive_signflip_high_mc.py`, output
`docs/results/signflip_fixedmodel_mc300.json`:

| graph | `\|S\|/n` | neg% (table, MC=12) | neg% (MC=300) | neg% beyond noise | `ρ_degree` | `ρ_delta2` |
| --- | --- | --- | --- | --- | --- | --- |
| Congress-Twitter | 20% | 44.0% | **4.0%** | **0.0%** | **+0.039** | **+0.030** |
| email-Eu-core | 20% | 36.0% | see JSON | see JSON | see JSON | see JSON |

The negative-share column was largely Monte-Carlo artefact, and on Congress-Twitter there is no
separation between degree and `δ₂`. **The sign pattern itself is not yet established**; the full
eight-graph sweep is in the JSON above.

The original table, kept for the record:

| graph | n | `⟨k⟩` | `ρ_degree` | `ρ_delta2` | negative-gain share |
| --- | --- | --- | --- | --- | --- |
| NetHEPT | 15233 | 4.23 | +0.092 | +0.686 | 0.0% |
| p2p-Gnutella08 | 6301 | 6.59 | −0.094 | +0.756 | 8.0% |
| ca-GrQc | 5242 | 11.06 | −0.701 | +0.656 | 20.0% |
| Wiki-Vote | 7115 | 29.15 | −0.035 | +0.265 | 8.0% |
| ca-HepPh | 12008 | 39.48 | −0.431 | +0.481 | 21.0% |
| Facebook | 4039 | 43.69 | −0.402 | +0.388 | 21.0% |
| email-Eu-core | 1005 | 50.89 | −0.695 | +0.685 | 34.0% |
| Congress-Twitter | 475 | 55.95 | −0.049 | +0.050 | 35.0% |

> ⚠️ **Do NOT claim density causes the sign flip.** `Spearman(⟨k⟩, ρ_degree) = −0.183` over nine
> graphs, and Congress-Twitter (the densest) is a counterexample. Even the sign of this correlation
> is now in doubt, because both components are measured at `MC = 12`.

---

## C3. Exact Monte-Carlo greedy also fails in the saturated regime

**Source** `docs/results/overexposure_gate2_congress_paired_20260916.json`
**Setting** Congress-Twitter, budget 10, pool 60, **400 paired trials with per-trial shared
threshold windows**. Reported quantity is the marginal spread added by the budget over the
pre-existing seed set.

| `\|S\|` | `\|S\|/n` | degree top-10 | `delta2` top-10 | exact MC-greedy top-10 |
| --- | --- | --- | --- | --- |
| 0 | 0% | +362.51 ± 0.61 | +361.64 ± 0.59 | **+367.28 ± 0.58** |
| 24 | 5.1% | **−4.39 ± 0.48** | **+0.49 ± 0.39** | −0.01 ± 0.45 |
| 48 | 10.1% | **−1.61 ± 0.47** | **+2.24 ± 0.32** | +0.28 ± 0.46 |
| 95 | 20.0% | **−5.17 ± 0.40** | **+2.26 ± 0.24** | +0.42 ± 0.31 |

**Statements.**
* Unsaturated (`|S|/n = 0`), all three are within 1.5% of each other. Degree is already fine;
  the interesting regime is `|S|/n ≥ 5%`.
* Once saturated, adding the highest-degree nodes **reduces** spread by 1.6 to 5.2; at
  `|S|/n = 20%` that is `−5.17 ± 0.40`, i.e. **13 standard errors**.
* `delta2` beats the exact MC greedy as well. Mechanism: the greedy searches for the largest
  marginal using `MC = 25` estimates while the true marginal is near zero there, so noise
  drives its decisions; `delta2` is a closed form with no noise.

---

## C4. The analytic ranking's advantage is not a selection artefact

**Source** `docs/results/overexposure_selection_robustness_20260916.json`
**Setting** Congress-Twitter, budget 10, pool 60, **8 independent selection draws**, each scored
on a shared set of 100 evaluation windows (only the selection varies).

| `\|S\|/n` | degree | `analytic` mean [min, max] | wins |
| --- | --- | --- | --- |
| 0% | +362.72 | **+366.30** [364.29, 368.05] | **8/8** |
| 10% | −2.51 | **+1.19** [0.48, 2.10] | **8/8** |
| 20% | −5.55 | **+0.23** [0.09, 0.42] | **8/8** |

The degree baseline is itself stable across draws (+5.07 to +5.97 on the 10% block), so the
degradation is not a selection artefact. **No counterexample draw exists.**

**Companion result** `docs/results/overexposure_plateau_stop_20260916.json` (200 paired trials):
analytic ordering turns the degree damage into a gain at every saturated fraction
(5%: −4.59 → −0.50; 10%: −1.97 → +2.23; 20%: −5.06 → +1.00).

> ⚠️ **Graph-dependent.** `docs/results/overexposure_plateau_stop_nethept_20260916.json`:
> on NetHEPT degree stays beneficial at every fraction and **wins at `|S|/n = 20%`**
> (+15.70 vs +11.05). Do not generalise beyond the tested graphs.

---

## C5. Measurement caveat that constrains every ranking claim

**Source** Gate 3A diagnostic (see `docs/GATE3A_REPORT.md`), Congress-Twitter, 30 candidates,
paired labels with per-trial shared windows.

| `\|S\|/n` | MC | mean gain | sd(gain) | mean se | **SNR = sd/se** | `ρ_delta2` |
| --- | --- | --- | --- | --- | --- | --- |
| 10% | 60 | +0.084 | 0.514 | 0.525 | **0.98** | +0.010 |
| 10% | 250 | −0.245 | 0.338 | 0.249 | 1.36 | +0.588 |
| 10% | 800 | −0.179 | 0.349 | 0.132 | **2.65** | +0.613 |
| 20% | 60 | −0.018 | 0.521 | 0.383 | 1.36 | +0.469 |
| 20% | 800 | −0.106 | 0.268 | 0.096 | **2.79** | **+0.828** |

**Statement.** In the negative-marginal regime the label noise at `MC = 60` is as large as the
dispersion of the labels themselves. Any rank-correlation claim in that regime requires
`MC ≥ 300`; a claim about *spread* should use a paired evaluation instead, where common random
numbers cancel the noise.

---

## C6. Negative result: a learned scorer does NOT beat the analytic baseline

**Source** `docs/results/overexposure_ranker_congress_20260917.json`,
`docs/results/overexposure_feature_ablation_r1_20260917.json`,
`docs/results/overexposure_feature_ablation_r2_20260917.json`
**Protocol** Congress-Twitter, 480 records, paired labels at `MC = 400`,
**leave-one-context-out** (so every score is a cross-state generalisation score).
`regret = 1 − (model top-5 gain) / (true top-5 gain)`.

| feature subset | r1 regret (Δ) | r1 wins | r2 regret (Δ) | r2 wins |
| --- | --- | --- | --- | --- |
| `analytic_only` (= delta2) | **0.5025** | — | **0.3360** | — |
| `analytic+threshold` | 0.6696 (+0.167) | 3/12 | 0.6358 (+0.300) | 1/12 |
| `analytic+structural` | 0.4722 (−0.030) | 7/12 | 0.2290 (−0.107) | 5/12 |
| `analytic+structural+threshold` | 0.8102 (+0.308) | **0/12** | 0.6265 (+0.291) | 1/12 |
| `threshold_only` | 0.9669 | 1/12 | 0.8364 | 1/12 |
| `structural_only` | 1.7869 | 1/12 | 1.4022 | 1/12 |

End-to-end (`MC = 400` labels, `top-5`): `delta2` Spearman **+0.6932**, ridge +0.4255, MLP
+0.3652; regret **0.5025** / 0.6172 / 0.9662.

**Statements.**
* The single `analytic_score` column is the strongest single predictor; nothing added to it
  improves it on a majority of held-out contexts.
* The hypothesis that threshold information (distance to `θ^τ`, fraction of neighbours in the
  band) is signal the analytic score cannot see is **refuted on both replications**.
* Pooled feature-label correlations of `|r| ≈ 0.85–0.91` were **regime identification**, not
  incremental information: the dataset mixes regimes whose label scales differ by three orders
  of magnitude (≈ +200 unsaturated, ≈ 0 saturated).
* This is a **limitation**, and the paper should report it rather than hide it.

---

## C7. The surrogate gap, and an open question about Theorem 6

**Source** `docs/results/surrogate_gap_20260917.json`,
`docs/results/surrogate_bound_verification_20260917.json`
**Detail** `docs/SURROGATE_GAP_FINDING.md`
**Scripts** `scripts/experiments/measure_surrogate_gap.py`,
`scripts/experiments/verify_surrogate_bound.py`

The source model optimises a surrogate. Its Section 5.2 defines
`λ(·) = σ^κ(·) − σ^τ(·)` and Theorem 6 asserts `λ(S) ≥ σ(S)`. We measured the gap; the model never
reports it. **The whole table below is withdrawn** (pre-fix state machine), and the description
must be exact: `σ^κ` is the linear-threshold spread at the lower threshold
`θ^κ ∼ F^κ(x) = 2x − x²` and `σ^τ` at the upper threshold `θ^τ ∼ F^τ(x) = x²`. These are the
marginals induced by the model's simplex law; **neither is uniform**, so neither may be described
as standard uniform-threshold LT.

| graph | `\|S\|/n` | `E[\|A\|]` | `λ` | gap | gap/λ |
| --- | --- | --- | --- | --- | --- |
| Congress-Twitter | 5.1% | 367.17 | 441.40 | +74.23 | +0.170 |
| Congress-Twitter | 10.1% | 366.13 | 403.53 | +37.40 | +0.098 |
| Congress-Twitter | 20.0% | 363.57 | 342.23 | −21.33 | −0.065 |
| NetHEPT | 5.0% | 1469.87 | 1442.00 | −27.87 | −0.019 |
| NetHEPT | 10.0% | 2787.77 | 2329.97 | −457.80 | −0.197 |
| NetHEPT | 20.0% | 4672.73 | 2387.00 | **−2285.73** | **−0.958** |

Per-realisation verification (180 draws, 2 graphs × 3 fractions × 30 trials) — also pre-fix:

```
A_window ⊆ A^κ \ A^τ         holds in   0 / 180
|A| ≤ |A^κ| − |A^τ|           holds in  73 / 180
λ < σ in expectation         in 4 / 6 settings
```

**Can claim**
* `σ^κ` and `σ^τ` are each genuine monotone submodular LT influence functions, so RR sampling
  applies to each **individually**; what it does not reach is their difference.
* `λ` is a **difference** of two monotone submodular functions; differences do not preserve
  submodularity, so even a valid upper bound would not by itself license a sampling guarantee.
* Whether the surrogate is tight anywhere is an **open question** until re-measured.

**Must NOT claim**
* ~~"Theorem 6 is false."~~ Three innocent readings remain open (our interpretation of
  `σ^κ`/`σ^τ`; whether the two processes are coupled; whether the proof is illustrative). Report
  the measured gap and flag the discrepancy as an open question requiring clarification from the
  authors.

---

## What the paper can and cannot claim

**Can claim, today, without new measurement**
1. The objective is not representable by any non-negative seed-independent coverage function, hence
   the standard RR/RIS identity has no domain here (C1′). Exact arithmetic.
2. `σ^κ` and `σ^τ` are individually amenable to RR sampling, but their difference is not, and
   submodularity is not preserved under differences (C7′). Algebraic.
3. The threshold law must be named, because the same instance is monotone under uniform-threshold
   LT and non-monotone under the model's simplex law (C1′).
4. In the negative-marginal regime, a rank correlation must not be reported below `MC ≥ 300` (C5).
5. Until `0248612` the measured spread was inflated by 60–63% on Congress-Twitter; every number
   produced before that commit is withdrawn.

**Cannot claim**
* That RR sampling is structurally inapplicable in general, or that RR sets are empty (C1, withdrawn).
* That out-degree becomes an anti-signal once saturated, or any of its magnitudes (C2, withdrawn).
* That exact Monte-Carlo greedy fails there (C3, withdrawn).
* That the analytic ranking's edge is not a selection artefact (C4, withdrawn).
* That a learned scorer does or does not beat the analytic baseline (C6, numbers withdrawn).
* The measured surrogate gap, or that the bound is tight anywhere (C7, withdrawn).
* That density causes the sign flip (C2 caveat).
* That degree is universally harmful (C4 caveat: NetHEPT is a counterexample).
* Any rank-correlation magnitude measured below `MC = 300` (C5) — **the paper currently violates
  this in the abstract, the introduction and the regime section; those numbers are withdrawn in the
  draft and must not be restored.**
* Any number measured with the pre-fix state machine.
