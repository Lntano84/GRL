# DASFAA 2027 Claim Ledger

Last updated: **2026-09-18**. This file is the claim source of truth for the manuscript.

## Claims supported by current artifacts

### C1 — Narrow coverage obstruction

On the exact one-hop instance with two candidate edges of weight `0.4` into one target, the
simplex-window law gives `F_D({a}) = 0.48` and `F_D({a,b}) = 0.32`. Therefore this objective is
not representable by a non-negative seed-independent coverage function, because every such
coverage function is non-decreasing and submodular. This does **not** rule out Monte-Carlo
estimation, signed decompositions, restricted-instance algorithms, or every possible RR-like
method.

### C2 — Corrected saturated-regime ranking result

The artifact `docs/results/signflip_fixedmodel_mc300.json` contains 8 graphs and 24 unique cells,
with paired MC=300 estimation. In the saturated cells (`|S|/n >= 20%`), degree is negatively
correlated with the true marginal on 7/8 graphs, while the state-conditioned `delta2` score is
better than degree on 8/8 graphs. NetHEPT is the graph-level exception for degree.

This is a regime-dependent observation, not a universal statement about degree, density, or every
graph family.

### C3 — Corrected negative-share result

The former 8--35% negative-marginal claim is false. Under MC=300, the saturated-cell mean is
0.5%, the maximum cell is 4.0%, and the mean share negative beyond two paired standard errors is
0.2%. Non-monotonicity remains possible by the exact C1 construction, but it is rare in the current
graph sweep; the practical issue is small marginal signal near the estimation noise floor.

### C4 — Measurement protocol

The paper can safely state five protocol requirements: re-evaluate non-seed state, name the
threshold law, use adequate MC for rank claims, never pool seed sizes in state-dependence ratios,
and repeat candidate pools. The broader audit also requires explicit stopping rules, charged state
acquisition, uncertainty reporting, and declared weight normalisation.

## Claims not established

- A post-fix learned scorer improves on degree or `delta2`.
- A post-fix sequential policy reduces oracle work while preserving near-oracle quality.
- A quality--cost Pareto curve with a random-pruning control.
- A general graph-level criterion explaining the sign flip.
- Any theorem about the tightness of the source model's surrogate bound.

## Permanently withdrawn numerical claims

The following must not be restored without a new artifact generated under the corrected contract:

- all spread, calibration, robustness, surrogate-gap, baseline and learned-ranking numbers from
  before commit `02486126`;
- all rank-correlation magnitudes from MC below 300 in the saturated regime;
- the claim that 8--35% of candidates have negative marginal gain;
- the claim that RR sets are empty or that RR/RIS is universally impossible;
- the claim that degree is universally harmful;
- the claim that a learned scorer does or does not beat the analytic baseline;
- the claim that the source model's theorem is false.

## Required evidence before an algorithmic sample-efficiency claim

Use the frozen contract and report at least three graphs, multiple budgets, repeated candidate
pools, independent seeds, Full-MC, degree, `delta2`, random pruning, online cascade cost, state
acquisition cost, paired uncertainty, and failure rate. The title and abstract must be changed if
these experiments do not establish a reproducible quality--cost advantage.
