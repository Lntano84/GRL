# GRID08

COMPLETE_AND_AUDITED: V0, V1, V2 and training-only guard qualification. No GRID08 jobs remain required.

- V0 train, fixed baselines and model evaluations have finished. Independent persisted-vector audit passed (36 episodes / 70,595 physical steps). Both V0 models reproduce ALWAYS_RESTORE on the four validation weeks.
- V1 GNN and MLP evaluation finished. Persisted-vector audit passed (24 episodes / 45,332 physical steps).
- Prior sessions 13296 / 21166 / 69833 disappeared without failure markers. Incomplete episodes and original updates are preserved; see resume_01.json. V2 restored model, Adam and Torch RNG from 11 complete episodes each and completed all 16 episodes and four fixed evaluations per model. Supervisor completed audits and reporting.
- Models evaluate only after their fixed training schedule finishes. Final test remains unopened.
- CPU updates are short relative to simulator time; no GPU or OS change is presently required.
- Original source, untrained initial checkpoints and failed first logging runs are preserved under v0_LOGGING_FAILED. The failed NumPy-mask V1 attempt is preserved under v1_MASK_FAILED. All successful versions use separate directories; see CHANGELOG.md and each design/freeze file.
- The cumulative three-version pilot cap is 220,000 physical steps / 1 GiB outputs; process wall cap remains 7,200 seconds. Actual counts must be reconciled in final delivery.
- Training-only ramp counterexample qualified: 131 prefix steps exactly matched, followed by two public forecast branches and two privileged actual branches. The restoring branch fails on the second step; holding survives those two steps. Extra physical steps: 135; forecast steps: 4. No active training config changed.
- Shared two-step forecast guard qualification completed on two training weeks × two fixed rules; audit passed. No veto occurred. October completed identically under both rules; January ended at step 1719 with a disconnected-grid/powerflow error under both. The local ramp counterexample is therefore not evidence of a general guard gain. It did not fit a model or use validation/final-test trajectories.
- Interrupted extra physical steps: 3,438 logged, conservative upper bound 3,441. Resume audit passed: all interrupted action/state prefixes match, completed update prefixes preserved. Cumulative physical count including all qualification/diagnostics: 204,027–204,030, within 220,000.
- Primary-source implementation review is in literature_failure_review.md (SMAAC, RL2Grid, runtime shielding preprint, official Grid2Op forecast API).
- No automation is active. Episode checkpoints, per-step logs and public-cost vectors are retained.

V2 GNN and MLP both reproduce ALWAYS_RESTORE exactly on all four evaluation weeks. Cost improvement versus NN20 is 63.8645% mean per-week reduction; additional learning improvement versus the fixed rule is zero. See GRID08_review_zh.md. No topic-wide or population conclusion follows.

Separate GRID09 controlled credit experiment is underway (its own design and resource ledger). GRID08 models, results and original assets remain unchanged. No automation resumed.
