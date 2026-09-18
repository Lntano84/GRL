# GRL Research State

Last updated: **2026-09-18**  
Remote HEAD: `744b2547869c9e6ce8288a8a1a4c84cd4972e933`  
Current target: **DASFAA 2027**, conditional on the post-fix algorithmic gates.

## Source of truth

This file is the current research-state source of truth. `DECISIONS.md` records durable route
choices, `EXPERIMENT_LOG.md` records experiment provenance, and `paper/dasfaa2027/notes/CLAIMS.md`
is the claim ledger. Older entries in those files are historical unless explicitly marked current.

## Current scope

The project studies targeted influence maximization under the threshold-dependent overexposure
model. A node is counted only when it is positive at the end of the process; a previously positive
node may later turn negative, while its historical activation still contributes to downstream
exposure. The literal sampled-window rule is `kappa <= delta <= tau`.

The model contract is implemented in `src/grl/diffusion/contract.py` and the state machine in
`src/grl/diffusion/overexposure.py`. The current default research question is whether a state-aware
proposal plus audited verification can reduce expensive Monte-Carlo oracle work in a setting where
standard coverage/RR guarantees do not apply.

## Confirmed post-fix findings

- The old state machine was wrong: it froze positive nodes and rejected the `delta=tau=1`
  boundary. The corrected implementation is covered by state-machine tests and the audit probes.
- The old pre-fix spread, calibration, surrogate-gap, learned-ranking and baseline tables are
  withdrawn and must not be used as current evidence.
- A one-hop exact counterexample supports the narrow statement that the objective has no
  non-negative seed-independent coverage representation. This does **not** prove that all RR-like
  algorithms, Monte-Carlo estimation, or signed decompositions are impossible.
- The corrected MC=300 sweep covers 8 graphs and 24 unique graph/fraction cells. In the saturated
  cells, degree is negatively correlated on 7/8 graphs and the state-conditioned `delta2` score is
  better on all 8.
- The old claim that 8--35% of candidates have negative marginal gain is false. The corrected
  average is 0.5%, the worst cell is 4.0%, and the mean beyond two paired standard errors is 0.2%.
- Corrected Stage 4b finds only one currently usable state-dependence/headroom cell:
  Congress-Twitter with `k=1`. This is not enough to claim a general learning advantage.

## What is not yet established

- No post-fix multi-graph quality--cost Pareto curve has established sample savings.
- Sequential-vs-static and random-pruning controls have not been re-derived under the frozen
  contract.
- The learning predictor has not shown an independent advantage over degree or `delta2` in a valid
  post-fix end-to-end experiment.
- Weight normalisation (`sum_to_one` versus `clip_to_one`) remains an external-validity choice and
  needs sensitivity evidence.

## Submission state

The DASFAA draft compiles, is anonymous, and contains the corrected analytic/measurement material.
The rebuilt PDF is 17 pages, so it is still one page over the 16-page limit; the pre-fix
withdrawn tables have been removed from the submission artifact. The current manuscript is
therefore not submission-ready. The title and algorithmic claims must remain conditional until
the post-fix quality--cost experiments pass.

## Immediate gates

1. Re-run sequential selection, Full-MC, degree, `delta2`, and random-pruning under the corrected
   contract.
2. Use at least three graphs, multiple budgets, repeated candidate pools, and independent seeds.
3. Require a pre-registered quality loss threshold (about 1%) and a meaningful online-cascade
   reduction (about 30%), with paired uncertainty and failure rates.
4. If those gates fail, remove the sample-efficiency framing and retarget the work as a model/
   measurement study or a longer-cycle journal/meeting paper.
