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
(`scripts/audit/rederive_signflip_high_mc.py`):

| graph | `\|S\|/n` | neg% (draft) | neg% (MC=300) | neg% beyond noise | `ρ_degree` draft → new | `ρ_δ₂` draft → new |
| --- | --- | --- | --- | --- | --- | --- |
| Congress-Twitter | 10% | — | 0.0% | 0.0% | — → **+0.406** | — → **−0.320** |
| Congress-Twitter | 20% | 44.0% | **4.0%** | **0.0%** | −0.112 → **+0.039** | +0.034 → **−0.060** |
| Congress-Twitter | 40% | 26.0% | 0.0% | 0.0% | +0.015 → **−0.133** | +0.065 → **+0.230** |
| email-Eu-core | 10% | — | 0.0% | 0.0% | — → −0.049 | — → +0.033 |
| email-Eu-core | 20% | 36.0% | 0.0% | 0.0% | −0.752 → **−0.139** | +0.726 → **+0.106** |
| email-Eu-core | 40% | 32.0% | 0.0% | 0.0% | −0.639 → **−0.332** | +0.643 → **+0.372** |

Two conclusions, both negative for the draft:

1. **The negative-marginal share was almost entirely Monte-Carlo artefact.** At `MC = 300` it is
   0–4%, with 0% negative *beyond noise* in every cell measured so far. The draft's headline
   phenomenon — "$8\%$ to $35\%$ of candidates have a negative marginal gain" — does not survive.
2. **`δ₂` does not separate from degree at the saturated cells.** At Congress-Twitter 20% the two are
   `+0.039` and `−0.060`, i.e. indistinguishable; at email-Eu-core 20% it is `−0.139` vs `+0.106`.

The remaining six graphs are still running; the JSON is rewritten after every cell.

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

1. **The `MC = 300` regime sweep is incomplete** — 3 of 24 cells for Congress-Twitter plus 3 for
   email-Eu-core, facebook in progress. Nothing in `regime.tex` can be asserted until it finishes.
2. **Stage 4b's corrected verdict is unknown.** The `|S| = 0` vs `|S| = 2` gap above suggests the old
   "no usable regime" conclusion may not survive; that run is in progress.
3. **`contract.py` is not yet wired into the experiment scripts.** It exists and is tested, but the
   scripts still declare their own target sets and budgets. Until that is done no re-derived number
   is contract-clean.
4. **No reference number has been re-derived under the full contract.** That is the next large unit
   of work, and it is what would let the `\withdrawn` markers come off the tables.
5. **The draft is 21 pages against a 16-page limit**, and citations remain unverified.
6. **The pre-registered gate is still not frozen.** The PI must confirm the thresholds
   (≤1% quality loss vs a high-accuracy reference AND ≥30% fewer online cascades vs the strongest
   non-learning method, with paired CIs, failure fraction, runtime, and a break-even `Q` for
   `C_offline + Q·C_online`).

---

## 6. Honesty ledger

Things this report deliberately does **not** claim:

- That any previously reported empirical result is repaired. None is; all are withdrawn.
- That confound 6 was fixed. It was alleged, we measured it, and it does not bite on our data.
- That the regime phenomenon is refuted. It is *unresolved* — the measurement that supported it was
  underpowered, and the adequately powered measurement contradicts it on the cells measured so far,
  but the sweep is incomplete.
- That `δ₂` is useless. At `|S|/n = 40%` on Congress-Twitter it reaches `ρ_δ₂ = +0.230`; the claim
  that fails is the specific one about the saturated `20%` cell.
- That the model is monotone, or that the paper's core question is wrong. The coverage argument
  (item 1 in §3) is exact arithmetic and is unaffected by any of this.
