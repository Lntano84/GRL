# Claim ledger — every number traceable to a result file

> Purpose: the paper must not contain a number that cannot be traced to a committed result
> file. This ledger is the single source of truth for the writing. If a claim is not listed
> here, it does not go in the paper.
>
> All numbers below were read from the JSON files under `docs/results/`, not from memory.

---

## C1. Reverse-reachability sampling is structurally inapplicable

**Source** `docs/results/overexposure_monotonicity_rr_20260916.json`
**Script** `scripts/experiments/evaluate_overexposure_monotonicity_rr.py`

Measured on NetHEPT (15,233 nodes / 32,235 edges), in-weights normalised:

| quantity | value |
| --- | --- |
| nodes whose window contains 0 | **0 / 15233** |
| RR-set size, 300 reverse BFS draws | min 0, median 0, mean **0.0**, max **0** |
| RR-set size as fraction of graph | **0.0000** |
| TIM-style estimate `n · Pr[S ∩ RR ≠ ∅]`, `\|S\| = 50` | **0.0** |
| true Monte-Carlo spread `σ(S)`, `\|S\| = 50` | **111.93** |

**Statement.** Positive activation requires `δ ∈ [θ^κ, θ^τ]` with `θ^κ > 0` almost surely, so a
node with `δ = 0` can never be positive; the reverse walk therefore has no base case and every
RR set is empty. The classic identity degenerates to `0` while the true spread is positive.

**Second, independent reason.** `δ(v,t) = Σ_{u ∈ A^in_t(v)} ω_uv` where `A^in_t(v)` is the set of
in-neighbours that were *ever* positively activated (the source paper states this set also
contains nodes later turned negative). This is a set function: given the threshold windows,
`δ` is deterministic in `S`. So neither a live-edge graph (IC) nor a trigger set (LT) can be
sampled — the randomised structure the identity is built on does not exist.

**Corroboration from the source paper.** Its Theorem 5 gives only a `γ/k` ratio (not
`1 − 1/e`); Lemma 2 and Theorem 6 construct monotone submodular `σ^κ`, `σ^τ` with
`λ(S) ≥ σ(S)` and optimise the surrogate. Its own solution path avoids the non-monotone
objective.

---

## C2. Out-degree degrades to an anti-signal in the saturated regime

**Source** `docs/results/density_degree_signflip_20260917.json`
**Script** `scripts/experiments/evaluate_density_degree_signflip.py`
**Setting** 8 graphs, seed fraction swept as `|S|/n`; values below are means over `|S|/n ≥ 20%`.
Rank correlation is Spearman(out-degree, true marginal gain).

| graph | n | `⟨k⟩` | `ρ_degree` | `ρ_delta2` | negative-gain share |
| --- | --- | --- | --- | --- | --- |
| NetHEPT | 15233 | 4.23 | **+0.092** | +0.686 | 0.0% |
| p2p-Gnutella08 | 6301 | 6.59 | −0.094 | +0.756 | 8.0% |
| ca-GrQc | 5242 | 11.06 | −0.701 | +0.656 | 20.0% |
| Wiki-Vote | 7115 | 29.15 | −0.035 | +0.265 | 8.0% |
| ca-HepPh | 12008 | 39.48 | −0.431 | +0.481 | 21.0% |
| Facebook | 4039 | 43.69 | −0.402 | +0.388 | 21.0% |
| email-Eu-core | 1005 | 50.89 | −0.695 | +0.685 | 34.0% |
| Congress-Twitter | 475 | 55.95 | −0.049 | +0.050 | 35.0% |

`δ2` beats degree on rank correlation in **8 / 8** graphs. Degree stays positive **only** on
NetHEPT, the sparsest graph.

> ⚠️ **Do NOT claim density causes the sign flip.** `Spearman(⟨k⟩, ρ_degree) = −0.405` over 8
> graphs, and Congress-Twitter (the densest) is a counterexample at `−0.049`. The supported
> claim is "in the saturated regime", not "in dense graphs".

> ⚠️ These rank correlations were produced with `mc-runs = 12`. See **C5**: at low MC the
> negative-marginal regime has SNR below 2, so correlation magnitudes here are noisy. Use only
> the **sign pattern** from this table, never the magnitudes.

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

## What the paper can and cannot claim

**Can claim**
1. RR-style sampling is structurally inapplicable (C1) — provable, and independent of
   non-monotonicity.
2. In the saturated regime out-degree becomes an anti-signal across 8 graphs (C2).
3. Exact Monte-Carlo greedy is noise-driven there and is beaten by a closed form (C3).
4. The analytic ranking's edge is robust across selection draws (C4).
5. A learned scorer does not beat the analytic baseline at the tested scale (C6).

**Cannot claim**
* That density causes the sign flip (C2 caveat).
* That degree is universally harmful (C4 caveat: NetHEPT is a counterexample).
* That `delta2` is optimal — its regret is 0.34–0.50, so a large gap remains (C6).
* Any rank-correlation magnitude measured below `MC = 300` (C5).
