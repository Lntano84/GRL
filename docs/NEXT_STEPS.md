# GRL Next Steps

Last updated: **2026-09-18**

## P0 — repository and claim consistency

- [x] Freeze the corrected overexposure state-machine and objective contract.
- [x] Complete the MC=300, 8-graph regime sweep.
- [x] Withdraw all pre-fix numerical tables and the low-MC negative-share claim.
- [x] Add direct audit probes for P0-1 through P1-3 and targeted state-machine tests.
- [x] Keep `RESEARCH_STATE.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`, `CLAIMS.md`, and the paper
      introduction synchronized after every new result.
- [x] Remove duplicate rows from raw result artifacts, not only from table-generation scripts.

## P0 — paper-valid algorithmic evidence

1. Re-run the baseline and sequential-selection experiments under `contract.py`.
2. Compare Full-MC, degree, `delta2`, random pruning, audited selection, and fallback policies.
3. Use at least three graphs, multiple budgets, repeated candidate pools, and independent random
   seeds.
4. Report final quality, online cascades, state-acquisition cost, runtime, failure fraction, and
   paired uncertainty.
5. Add `sum_to_one` versus `clip_to_one` weight-normalisation sensitivity.
6. Do not restore any pre-fix table or any MC<300 rank-correlation magnitude.

## DASFAA Go/No-Go gate

The paper can keep the sample-efficiency title only if the corrected experiments show, in a
non-saturated regime, near-oracle quality (target loss about 1% or less) and a meaningful reduction
in expensive oracle work (target about 30% or more), with random-pruning controls and repeated
candidate pools. If this fails, move to a measurement/protocol framing and target a longer-cycle
venue such as CIKM or a suitable journal.

## Manuscript cleanup after the gate

- [x] Delete the pre-fix withdrawn tables from the submission PDF; retain only a concise threat/history
  paragraph.
- Keep the exact coverage counterexample and the corrected MC=300 regime table.
- [x] Verify every bibliography entry and remove `TODO(verify)` notes.
- [x] Rebuild the anonymous LNCS PDF and keep it at or below 16 pages.
- Run the complete test suite in an environment containing the declared requirements, including
  PyTorch-dependent oracle tests.
