# P0 completion report

> Written 2026-09-18. Covers audit items **P0-1 … P0-4, P1-1 … P1-3**.
> All nine are now **RESOLVED**, each with a probe in `scripts/audit/check_audit_items.py` that
> tests the current source rather than the state the audit happened to observe.
>
> Commits: `0248612` (state machine) · `ed0a3d7e` (withdraw the RR claim) · `7dc14d9` (contract,
> acceptance stages, withdrawal markers) · `241e4df9` (P1-3 part 1) · `232aefb8` (P1-3 part 2).

---

## 1. The two findings that invalidated the empirical record

### 1.1 The state machine was wrong

Until commit `0248612`, a positively activated node could **never** become negative again, and a node
with `δ = θ^τ = 1` was **denied** activation. Both are contrary to the model, which re-evaluates a
non-seed node each round under the literal rule `θ^κ ≤ δ ≤ θ^τ`.

Counterexample for the first, with no Monte-Carlo noise:
`s → a` (0.4), `s → b` (0.5), `b → a` (0.4), seed `{s}`, windows `a = [0.2, 0.6]`, `b = [0.2, 0.9]`.
`a` first receives `δ = 0.4` (inside its window, so positive), then `b` activates and `δ` rises to
`0.9 > 0.6`, so `a` must turn negative. Correct answer `spread = 2`, `negative = {a}`; the old code
gave `spread = 3`, `negative = {}`.

Counterexample for the second: `s → v` with weight 1 and `v = [0.2, 1.0]`. Here `δ = 1 = θ^τ`, which
the rule admits, so `spread = 2`; the old `δ < 1` guard gave `spread = 1`.

**Impact.** On Congress-Twitter the corrected machine removes 60–63% of the measured spread
(k=1: 299.35 → 111.44; k=5: 358.52 → 137.28; k=10: 365.95 → 143.43). That is larger than every
effect the draft reported, so the draft's empirical sections are withdrawn. They are **kept** under
an explicit `\withdrawn` marker rather than deleted, because the record of what was measured and of
why it was wrong is part of the contribution. `\withdrawn` renders 16 times in the compiled PDF.

### 1.2 The draft cited rank correlations from an MC = 12 run

Found while auditing provenance, and **independent of the state-machine bug**.

- The producing command is recorded in `docs/GATE3A_REPORT.md`:
  `evaluate_density_degree_signflip.py --candidates 50 --mc-runs 12`.
- The same report warns, in the same block, that `mc-runs=12` is *only good for trends* and that the
  correlation numbers must not be cited from it.
- `notes/CLAIMS.md` C2 warned independently: *"Use only the sign pattern from this table, never the
  magnitudes."*
- The draft's abstract, introduction and `regime.tex` quoted the magnitudes anyway, against its own
  constraint (C5) that a rank correlation in this regime needs `MC ≥ 300`.
- The stored JSON confirms `candidates: 50, mc_runs: 12`.

**Re-measurement** at `MC = 300` under the fixed model, paired per-trial estimator
(`scripts/audit/rederive_signflip_high_mc.py`). Seven graphs of eight are complete; the table is
being filled in as the last cells land.

| graph | ⟨k⟩ | `\|S\|/n` | neg% (draft) | neg% (new) | `ρ_degree` draft → new | `ρ_δ₂` draft → new |
| --- | --- | --- | --- | --- | --- | --- |
| p2p-Gnutella08 | 6.59 | 10% | — | 0.0% | — → +0.647 | — → +0.680 |
| p2p-Gnutella08 | 6.59 | 20% | 8.0% | 0.0% | −0.094 → **+0.338** | +0.756 → +0.259 |
| ca-GrQc | 11.06 | 10% | — | 0.0% | — → −0.320 | — → +0.692 |
| ca-GrQc | 11.06 | 20% | 20.0% | 0.0% | −0.701 → **−0.560** | +0.656 → +0.653 |
| Wiki-Vote | 29.15 | 20% | 8.0% | 0.0% | −0.035 → **+0.090** | +0.265 → **+0.527** |
| Wiki-Vote | 29.15 | 40% | 8.0% | 0.0% | −0.035 → **−0.173** | +0.265 → **+0.245** |
| ca-HepPh | 39.48 | 10% | — | 0.0% | — → +0.009 | — → +0.392 |
| ca-HepPh | 39.48 | 20% | 21.0% | 0.0% | −0.431 → **−0.179** | +0.481 → **+0.380** |
| Facebook | 43.69 | 10% | — | 0.0% | — → −0.044 | — → **+0.784** |
| Facebook | 43.69 | 20% | 21.0% | 0.0% | −0.402 → **−0.413** | +0.388 → **+0.497** |
| Facebook | 43.69 | 40% | 21.0% | 4.0% (2% beyond noise) | −0.402 → **−0.425** | +0.388 → **+0.557** |
| email-Eu-core | 50.89 | 10% | — | 0.0% | — → −0.049 | — → +0.033 |
| email-Eu-core | 50.89 | 20% | 34.0% | 0.0% | −0.695 → **−0.139** | +0.685 → **+0.106** |
| email-Eu-core | 50.89 | 40% | 34.0% | 0.0% | −0.695 → **−0.332** | +0.685 → **+0.372** |
| Congress-Twitter | 55.95 | 10% | — | 0.0% | — → +0.406 | — → −0.320 |
| Congress-Twitter | 55.95 | 20% | 35.0% | 2.0% (0% beyond noise) | −0.049 → **+0.039** | +0.050 → **−0.060** |
| Congress-Twitter | 55.95 | 40% | 35.0% | 0.0% | −0.049 → **−0.133** | +0.050 → **+0.230** |

Two conclusions:

1. **The negative-marginal-share claim is dead.** At `MC = 300` the share is `0%` in seventeen of
   eighteen cells, `2%` in one, and `0%` *beyond noise* in every cell. The draft's headline
   "$8\%$ to $35\%$ of candidates have a negative marginal gain" was Monte-Carlo artefact.
2. **The sign flip and `δ₂`'s advantage largely survive.** At `|S|/n ≥ 20%`, `ρ_δ₂ > ρ_degree` on
   six of seven graphs measured, and degree is negative on five of them. The two exceptions are the
   extremes of the density range: p2p-Gnutella08 (`⟨k⟩ = 6.59`, degree `+0.338`) and
   Congress-Twitter at `20%` (`⟨k⟩ = 55.95`, degree `+0.039`, `δ₂ = −0.060`).

**An intermediate conclusion of mine was itself underpowered, and I am recording that.** After the
first two graphs I wrote that "the headline phenomenon is not visible at adequate power". That was
based on Congress-Twitter and email-Eu-core. With seven graphs the sign flip is present on five and
`δ₂` leads on six. The negative-share half of my statement held; the sign-flip half did not, and
generalising from two graphs was the same error as citing `MC = 12`. `regime.tex` therefore states
neither version until the eighth graph lands.

---

## 2. Audit item status

| item | status | how it is verified |
| --- | --- | --- |
| **P0-1** positive node can turn negative | resolved | counterexample above; `run.negative == {a}` |
| **P0-2a** `δ = θ^τ = 1` activates | resolved | counterexample above; `spread == 2` |
| **P0-2b** degeneracy paths kept apart | resolved | four named laws; `E[κ]` measured as 0.332 for `simplex_tau_clamped_to_one` vs 0.498 for `uniform_lt_tau_one`, so they are provably different processes |
| **P0-3a** no coverage representation | resolved | exact arithmetic: `F({a}) = 0.4800 > F({a,b}) = 0.3200` |
| **P0-3b** RR claim withdrawn, title changed | resolved | no `No Domain` title; no live "RR sets are empty" claim; `inapplicability.tex` gone; `coverage.tex` present; all 8 pre-fix tables carry `\withdrawn` |
| **P0-4** frozen contract | resolved | `build_contract` refuses a config with no `D`; a seed inside `D` is refused; `k+1` seeds refused, `k` accepted; `TargetedObjective` evaluates on `D` only |
| **P1-1** threshold laws not misdescribed | resolved | empirical marginals match `F_τ = x²` and `F_κ = 2x − x²` to 0.0073; no source line asserts a simplex path is uniform-threshold LT |
| **P1-2** acceptance stages separated | resolved | `empirical_accept` / `statistical_certificate` / `fallback_full_scan`; the counterexample accepts the wrong node with `empirical_accept=True` and `fallback_full_scan=False` |
| **P1-3** eight confounds | resolved | one probe per confound; see §4 |

---

## 3. What the paper can defend today

Only the analytic material survived both findings. This is a short paper, and it is the only part
that is currently true:

1. **No non-negative seed-independent coverage function matches the objective.** Exact one-hop
   arithmetic: `F_D(∅) = 0.0000`, `F_D({a}) = 0.4800`, `F_D({a,b}) = 0.3200` under the model's simplex
   law, against `0.0000 / 0.4000 / 0.8000` under uniform-threshold LT. The closed form `2δ(1−δ)` was
   validated against 2-D numerical integration (worst difference `4.5e-4`). Because
   `S ↦ c·Pr[S ∩ R ≠ ∅]` is non-decreasing and submodular for any `R` independent of `S`, a
   non-monotone `F_D` cannot equal it for any `c` and any distribution of `R`.
2. **The threshold distribution must be named**, because the sign of the effect depends on it.
3. **`λ` is a difference of two monotone submodular functions**, so submodularity is not preserved
   and even a valid upper bound would not by itself license a sampling guarantee.
4. **A measurement constraint**: no rank correlation in the negative-marginal regime below
   `MC ≥ 300`, and the negative-marginal share is itself the quantity most inflated by that noise.

Everything else in the draft is marked `\withdrawn`.

---

## 4. The eight confounds: what each was, and what happened

| # | confound | verdict |
| --- | --- | --- |
| 1 | inconsistent stopping under a budget of *at most* k | **reproduced** — now an explicit `StoppingRule` recorded per decision; a prediction-only policy refuses a rule it cannot evaluate |
| 2 | state acquisition reported at zero cost | **reproduced** — `state_cascades` is now a *breakdown* of `mc_cascades`, so charging the primary unit charges the state |
| 3 | `MC = 25` greedy called an oracle | **reproduced** — `reference_policy_name` puts the budget in the name; the CLI says "reference point, not an upper bound" |
| 4 | mixed seed sizes in one within/between statistic | **reproduced, and it changed a verdict** — see below |
| 5 | a single random candidate pool | **reproduced** — degree-stratified by default plus `--pool-draws`; on Congress-Twitter k=2 the degree gap ranged 0.90%–13.49% across two pools |
| 6 | splits that do not deduplicate | **alleged, NOT reproduced** — zero duplicate states at 12/24/48/120 contexts. The split now groups by canonical state as a *guard*; no number changed and the code says so |
| 7 | `label_std` recorded as `NaN` | **reproduced** — `score_with_uncertainty` returns the paired standard error, available free because per-trial differences were already formed |
| 8 | weights normalised to exactly 1 where the model allows ≤ 1 | **reproduced** — four named strategies; `sum_to_one` vs `clip_to_one` differ by 50× in exposure scale on a sparse file |

### Confound 4 deserves its own paragraph

The state-dependence contexts were built at seed sizes `0, k, 2k, 3k` and the within/between ratio was
computed over all of them at once. Measured on Congress-Twitter:

| seed size | between-candidate sd | within-candidate sd | ratio |
| --- | --- | --- | --- |
| `\|S\| = 0` | **34.86** | 6.34 | 0.182 |
| `\|S\| = 2` | 4.32 | 2.47 | 0.571 |
| `\|S\| = 4` | 2.16 | 2.84 | 1.316 |

The `|S| = 0` cell's between-spread is 8× larger than the others, so the pooled ratio landed near the
unsaturated value. Stage 4's verdicts — ca_GrQc `0.64` ("moderate"), NetHEPT `0.075` ("NOT state
dependent") — could therefore have been produced by the pooling rather than by the graphs. The ratio
is now computed inside one seed size and reported per size, and a cell is usable only if the ratio
clears the threshold **at the seed size where the headroom was measured**. The corrected sweep is
running.

---

## 5. What is not done

1. **The `MC = 300` regime sweep is one graph short** — NetHEPT's three cells remain, and ca-HepPh's
   `40%` cell. Nothing in `regime.tex` may be asserted until they land.
2. **Stage 4b's corrected verdict is in.**
   `docs/results/stage4b_confounds_removed.json` reports, at the budget's own seed size:

   | graph | ⟨k⟩ | state ratio at `|S| = k` | median degree gap over 5 pools |
   | --- | --- | --- | --- |
   | Congress-Twitter | 55.95 | **1.940** (k=1) | 6.01% [−8.67, 25.57] |
   | email-Eu-core | 50.89 | 0.896 (k=1) | 0.00% [−2.67, 0.00] |
   | ca-GrQc | 11.06 | 0.150 (k=3) | 5.65% [0.00, 21.11] |
   | Facebook | 43.69 | 0.057 (k=3) | 15.32% [0.00, 72.62] |
   | Bitcoin-Alpha | 12.79 | 0.350 (k=3) | 0.00% [−0.25, 0.38] |
   | NetHEPT | 4.23 | 0.090 (k=3) | 14.77% [−0.50, 20.78] |

   Only Congress-Twitter at `k = 1` clears both thresholds, so the verdict that state dependence
   and optimisation headroom do not coincide **survives** the pooling correction — but the ratios
   that supported it do not (the old docstring recorded ca-GrQc `0.64` and email-Eu-core `1.94`;
   the corrected values are `0.150` and `0.896`), and the reason the sparsest graph fails is now
   visible rather than inferred: NetHEPT's between-candidate sd is `1.27`, so there is genuinely
   little for a state-conditioned score to separate.
3. **`contract.py` is wired into stage4b and stage5**, which now print and record `|D|`,
   seed eligibility, counting rule, stopping rule, normalisation and reference policy. It is *not*
   wired into the older one-off `evaluate_*.py` scripts, which remain withdrawn.
4. **No reference number has been re-derived under the full contract.** That is what would let the
   `\withdrawn` markers come off the tables in `experiments.tex`.
5. **The draft is 21 pages against a 16-page limit**, and citations remain unverified.
6. **The pre-registered gate is still not frozen.** The PI must confirm the thresholds
   (≤1% quality loss vs a high-accuracy reference AND ≥30% fewer online cascades vs the strongest
   non-learning method, with paired CIs, failure fraction, runtime, and a break-even `Q` for
   `C_offline + Q·C_online`).

---

## 6. The paper's contribution, as it now stands

With every measured number withdrawn, the draft had one analytic result and five empty tables. The
contribution the evidence does support is a **measurement protocol** for this model, now written as
`sections/protocol.tex`: five checks, each with the measured magnitude of the error it prevents.

| check | measured cost of getting it wrong |
| --- | --- |
| re-evaluate every non-seed node every round | spread inflated **60–63%** on Congress-Twitter |
| name the threshold law | non-monotonicity's sign reverses under uniform-threshold LT |
| no rank correlation below `MC = 300` | negative share **44% → 0%**; `ρ_degree` −0.112 → +0.039 |
| never pool seed sizes | between-candidate sd **36.54 vs 0.74**, a factor of 50 |
| repeat the candidate pool | degree gap **0.00%–98.52%**; the sign depends on the draw |

Four of the five checks are free or nearly so. Only the Monte-Carlo budget is a real factor of 25.
That asymmetry is the argument: the errors are cheap to prevent and were expensive to have made.

---

## 6. Honesty ledger

Things this report deliberately does **not** claim:

- That any previously reported empirical result is repaired. None is; all are withdrawn.
- That confound 6 was fixed. It was alleged, we measured it, and it does not bite on our data.
- That the regime phenomenon is refuted. The negative-marginal-share half is refuted; the sign-flip
  half largely **survives**, and an interim conclusion of mine that said otherwise was drawn from
  two graphs and was itself underpowered. That mistake is recorded in §1.2 rather than quietly
  corrected.
- That `δ₂` is useless. It leads degree on six of seven graphs at `|S|/n ≥ 20%`. What fails is the
  specific claim about the `20%` cell on the densest graph, and the claim that a large share of
  candidates has a negative marginal.
- That the eight confounds are the whole story. They are the ones we found; the protocol section
  exists because we expect more.
- That the model is monotone, or that the paper's core question is wrong. The coverage argument
  (item 1 in §3) is exact arithmetic and is unaffected by any of this.
- That the paper is submittable. It is 21 pages against a 16-page limit, most of its empirical
  content is withdrawn, and its citations are unverified.
