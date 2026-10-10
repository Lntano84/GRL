# GRID03 score normalization caveat

Installed Grid2Op 1.12.5 source (not modified):

- `utils/l2rpn_idf_2023_scores.py`: `ScoreL2RPN2023` combines operational, renewable and assistant components with default weights 0.60, 0.15 and 0.25.
- Parent `utils/l2rpn_2020_scores.py`, `_compute_episode_score`: both `load_p_rp` and `prod_p_rp` are read with `self.stat_no_overflow_rp.get("load_p")`.
- Therefore the reference expression `prod_p_rp.sum(axis=1) - ep_loads` equals zero, even when the physical reference grid has positive losses. This affects the interpolation endpoint for best operational score. It is a source-level observation, not a measured change in policy ranking.

The installed source hashes and hand fixtures are saved in `metric_unit_checks.json`. The same expression appears in the [official published source documentation](https://grid2op.readthedocs.io/en/latest/_modules/grid2op/utils/l2rpn_2020_scores.html).

GRID03 does not patch the library, claim to have reproduced the 2023 competition evaluator, or attribute earlier results to this expression. The helper smoke is labeled **installed-version API qualification**. It verifies expected reference scores on a bounded artificial horizon, not evaluator equivalence on full weeks.

Operational evaluation can still record the official unnormalized `L2RPNSandBoxScore` components and independently reconcile losses, dispatch, curtailment changes and storage throughput from adjacent delivered observations. In this installed version, the curtailment cost uses the change in total curtailment, not its current level; releasing curtailment can produce a negative contribution. Physical curtailed energy is retained as a separate field. The original stateless ledger failed at paired step 4 and was corrected with persistent/released/increased-curtailment fixtures, without modifying the library or tolerance. Survival and terminal renewable/assistant measures must remain separately visible. For premature blackouts, an episode's smaller raw accumulated cost is not a benefit; survival/missing-service treatment must precede cost comparison.

Before any main normalized-score claim, the precise evaluator version/formula and reference trajectories need to be fixed. If an independently corrected normalization is used, report it as a distinct metric and retain the installed helper result separately; do not silently call it the official competition score.

Official API description: [ScoreL2RPN2023](https://grid2op.readthedocs.io/en/latest/user/utils.html). Public environment description: [l2rpn_idf_2023](https://beta-grid2op.readthedocs.io/en/bd_dev/available_envs.html#l2rpn-idf-2023).
