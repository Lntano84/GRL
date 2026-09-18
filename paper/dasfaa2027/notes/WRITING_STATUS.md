# DASFAA 2027 Draft Status

Last updated: **2026-09-18**  
Target submission: **2026-11-25**  
Current title: *Sample-Efficient Seed Selection under Overexposure: Estimating Marginal Gains
without a Coverage Representation*

## Current manuscript state

- `abstract.tex`: updated with the narrow coverage obstruction and the corrected MC=300 regime
  result; no pre-fix baseline numbers.
- `introduction.tex`: aligned with the current claim ledger; invalid tables are no longer retained.
- `preliminaries.tex`: model and frozen objective contract.
- `coverage.tex`: exact one-hop counterexample and narrow scope statement.
- `protocol.tex`: five measurement checks and their methodological motivation.
- `regime.tex` and generated `regime_table.tex`: current 8-graph MC=300 result.
- `analysis.tex`: training-free state-conditioned score definition only; no withdrawn calibration
  table.
- `experiments.tex`: concise post-fix status and requirements for the pending quality--cost study;
  no pre-fix tables.
- `limitations.tex`: current limitations and audit confounds; no withdrawn surrogate table.
- `conclusion.tex`: must remain aligned with the conditional algorithmic claim and current ledger.

## Build and page budget

The rebuilt PDF is 16 pages after removing the withdrawn tables and tightening the conclusion and
bibliography. The PDF remains double-blind and contains no author-identifying repository links.

## Remaining blockers

1. Re-run sequential quality--cost experiments under the corrected contract.
2. Include Full-MC, degree, `delta2`, random-pruning, repeated pools and independent seeds.
3. Verify `sum_to_one` versus `clip_to_one` weight normalisation.
4. [done] Verified all 12 cited records, corrected the related-work prose, removed false/duplicate/
   uncited entries, and recorded provenance in `REFERENCE_AUDIT.md`.
5. Run the complete test suite in an environment with PyTorch, not only the lightweight targeted
   tests.

## Editorial rule

Only claims listed as supported in `notes/CLAIMS.md` may appear in the abstract, introduction,
results or conclusion. Historical withdrawn measurements may be mentioned only as a short audit
motivation, without tables or numerical comparisons.
