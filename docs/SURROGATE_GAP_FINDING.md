# Surrogate bound: measured gap and an unresolved discrepancy with Theorem 6

> Date: 2026-09-17
> Source model: Inf. Sci. 744 (2026) 123375, Section 5.2 and Theorem 6
> **Status: report as an open question, NOT as a rebuttal.** See "How to describe this" below.

---

## What the source model claims

Its Section 5.2 defines an upper bound for the overexposure objective:

> *We suppose that `σ^κ(·)` (`σ^τ(·)`) is the objective value under the LT model, which implies
> that any node `v` becomes activated when `Σ_{u∈N⁺_a(v)} ω_uv > θ^κ_v` (`θ^τ_v`).
> Denote `λ(·) = σ^κ(·) − σ^τ(·)`.*
>
> **Theorem 6.** *For any `S ∈ V`, we have `λ(S) ≥ σ(S)`.*

The proof (Proof 5.2) is a single illustrative realisation (its Fig. 5(b)): with window
`[θ^κ_v, θ^τ_v]` and seed set `S = {u, v}`, the lower layer (threshold `θ^κ`) activates
`A^κ = {a, c, b, e}`, the upper layer (threshold `θ^τ`) activates `A^τ = ∅`, and therefore

> *Consequently, the positively activated set under the window model satisfies
> `|A| ≤ |A^κ| − |A^τ|`.* (line 531)

Note the logical shape: the **statement** is about `σ`, an expectation over window draws; the
**proof** establishes a containment for one fixed draw.

---

## What we measured

`docs/results/surrogate_gap_20260917.json`, `docs/results/surrogate_bound_verification_20260917.json`
Scripts: `scripts/experiments/measure_surrogate_gap.py`,
`scripts/experiments/verify_surrogate_bound.py`

All quantities share the same window draw within a trial, so the comparison is paired.

| graph | `\|S\|/n` | `E[\|A\|]` | `E[\|A^κ\|]` | `E[\|A^τ\|]` | `λ` | gap | subset viol. | per-real. viol. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Congress-Twitter | 5.1% | 367.17 | 467.97 | 26.57 | 441.40 | **+74.23** | 30/30 | 0/30 |
| Congress-Twitter | 10.1% | 366.13 | 467.53 | 64.00 | 403.53 | **+37.40** | 30/30 | 0/30 |
| Congress-Twitter | 20.0% | 363.57 | 466.53 | 124.30 | 342.23 | **−21.33** | 30/30 | 29/30 |
| NetHEPT | 5.0% | 1469.87 | 2680.13 | 1238.13 | 1442.00 | **−27.87** | 30/30 | 18/30 |
| NetHEPT | 10.0% | 2787.77 | 5025.93 | 2695.97 | 2329.97 | **−457.80** | 30/30 | 30/30 |
| NetHEPT | 20.0% | 4672.73 | 7362.40 | 4975.40 | 2387.00 | **−2285.73** | 30/30 | 30/30 |

Across the full sweep (4 graphs × 3 fractions × 30 trials = 360 draws, of which the tables above
cover 180):

```
A_window ⊆ A^κ \ A^τ           holds in   0 / 180 draws
|A| ≤ |A^κ| − |A^τ|             holds in  73 / 180 draws
λ < σ in expectation           in 4 / 6 settings
```

On the full four-graph sweep (`surrogate_gap_20260917.json`) the relative gap ranges from
**+0.396** (ca-GrQc, 5%) to **−0.958** (NetHEPT, 20%).

---

## How to describe this

**Do not write "Theorem 6 is false."** At least three innocent explanations remain open, and we
cannot distinguish them from the material available:

1. **Our reading of `σ^κ` / `σ^τ` may be wrong.** The paper says "the objective value under the LT
   model", which we took to mean: run a linear-threshold cascade with the fixed threshold vector
   `θ^κ` (resp. `θ^τ`), on the same graph and weights, under the same window draw. Other readings
   are possible (e.g. the two processes sharing an activation set, or `σ^κ`/`σ^τ` being
   single-realisation counts rather than expectations).
2. **The two processes may be intended to be coupled.** Our `A^κ` and `A^τ` are independent
   cascades at different thresholds. If the intended construction forces `A^τ ⊆ A^κ` by sharing
   state, the containment could hold.
3. **The proof may be informal.** It is explicitly "a concise illustrative example", and the
   containment is asserted for that figure rather than proven in general.

The second explanation is the most plausible and is also the most consequential: **`A^τ ⊆ A^κ`
is exactly what our measurements violate.** Every counterexample we see is a node that is active
in the `θ^τ` process but inactive in the `θ^κ` process — plausible because in a cascade a *higher*
threshold can, in principle, change the propagation order and admit a node by a different route.
Under independent cascades this is easy to produce and we observe it in 107 of 180 draws.

---

## Why this matters for our paper

Even setting the discrepancy aside, the measured gap is the number the source model never
reports, and it materially affects how the surrogate should be used:

* **The bound is tight at low coverage** (Congress-Twitter: +9.8% at `|S|/n = 10%`). A tight
  upper bound means optimising the surrogate is informative, and it is also *actionable*: `σ^κ`
  and `σ^τ` are genuine LT influence functions, so RR sampling applies to each of them
  individually.
* **It degrades and eventually inverts as the network saturates** (NetHEPT at 20%: `λ` is less
  than half of `σ`). In precisely the regime where the objective becomes non-monotone and where
  our earlier experiments show degree failing, the surrogate stops being an upper bound at all
  under our reading.
* **Even if `λ ≥ σ` held, `λ` need not be submodular.** `λ` is the *difference* of two monotone
  submodular functions, and differences do not preserve submodularity in general. The source
  model establishes submodularity of the objective only for outward-tree networks. So the step
  "RR applies to the surrogate, therefore the surrogate is solvable with guarantees" would need
  its own argument — and our measurement suggests the bound itself is the more basic problem.

---

## Recommended next step

**Write to the authors or read the published version's Section 5.2 in full before publishing
anything about this.** The check is cheap and the downside of being wrong is large. Specific
questions to resolve:

1. Is `σ^κ(·)` the expectation of an LT cascade at threshold `θ^κ`, or a single-realisation count?
2. Are the `κ` and `τ` processes coupled (shared activation set / monotone coupling), or
   independent?
3. Is Theorem 6 intended for a fixed window draw or in expectation?

The paper can report the *tightness* of the bound (which is informative regardless) and note the
discrepancy as an open question in a "threats to validity" or "discussion" paragraph. It should
not present it as a refutation.

---

## Reproduction

```powershell
$M = "C:\Users\windows\Desktop\_grl_merge\merged"; cd $M; $env:PYTHONPATH="$M\src"

python scripts/experiments/measure_surrogate_gap.py `
  --graphs congress_twitter email_eu_core ca_grqc nethept `
  --fractions 0.0 0.05 0.10 0.20 --trials 40 `
  --output docs/results/surrogate_gap_20260917.json

python scripts/experiments/verify_surrogate_bound.py `
  --graphs congress_twitter nethept --fractions 0.05 0.10 0.20 --trials 30 `
  --output docs/results/surrogate_bound_verification_20260917.json
```
