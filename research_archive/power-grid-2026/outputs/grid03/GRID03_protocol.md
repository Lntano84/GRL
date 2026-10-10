# GRID03 — formal-data and scoring qualification

Frozen before archive download, 2026-10-08. No model training or policy tuning.

## Resource ceilings

- Official public l2rpn_idf_2023 archive, expected 5,358,993,257 bytes.
- TLS verified. GET Range, remote ETag and length validation; exact saved prefix is resumable.
- Cumulative application response body ceiling: 6 GiB. Acquisition invocation wall ceiling: 2 hours. No second full download.
- At initial check C: free space = 54,798,798,848 bytes. Extraction payload ceiling: 8 GiB; only a bounded scenario subset, plus necessary static configuration. No full automatic Grid2Op download/extraction.
- Archive index and selection must precede any agent-quality evaluation. Preserve archive and hashes.
- No new dependencies, training, hyperparameter changes, Top-k changes, or automations.

## Scope

1. Acquire and index the published archive. Inspect scenario naming and grouping without running agents. Freeze a deterministic development/validation/test subset and disclose that it is not proven unseen by the released LJN network.
2. Audit installed Grid2Op 1.12.5 ScoreL2RPN2023 source. Keep the author's MaxRhoReward search objective separate from operational evaluation rewards.
3. Run only bounded environment/scoring smoke checks, at most 100 physical environment steps in total. Do not run full official reference statistics on the dataset in this stage.
4. Verify static grid/action indexing and native configuration compatibility before reusing the baseline. Report qualification failures without silently changing native rules.

Detailed subset rule will be frozen after archive naming/size inspection but before scenario extraction or agent outcomes. This is a data qualification stage, not independent confirmation of a research hypothesis.

## Acquisition implementation amendment (before full archive available)

The first serial stream was stopped with its exact saved prefix retained because of slow throughput. Four disjoint Range streams will fetch the remaining bytes, using the same ETag, payload ceiling and a global 2-hour clock from archive creation. No complete redownload. Range-part files add at most one archive-sized temporary copy on disk. This changes transport concurrency, not dataset or evaluation selection.

## Formal-smoke clarification (before any formal environment step)

The paired passive-metric check uses the already released NN20 controller to choose the common action, so a DoNothing blackout cannot be confused with an interface failure. Tiny scripted curtailment/storage probes are applied only when max rho < 0.9. Both physical environments receive exactly the same fresh action vector. A separate clean NN20 prefix remains limited to 12 steps. Total stage envelope remains 100 physical steps; no outcomes are used to select scenarios.

The original execution-code hash snapshot is preserved; the pre-execution clarification revision is recorded separately. No formal policy trajectory existed when this clarification was made.

## Official initialization completion (after static construction, before any formal step)

The raw published archive lacks alerts_info.json. The initial static check therefore stopped with zero physical steps, reporting action/observation dimension differences. Grid2Op's official DownloadDataset._aux_download calls UpdateEnv._update_files after archive extraction. Complete this normal official initialization using updates.json and its three files pinned to grid2op-datasets commit 7020a556eab049366e74d28c23c9e2091ff33145. Apply them to a separate 48-scenario copy, preserving the raw extraction and failed static-check output. The official changes add alert metadata and alert-time windows, and remove duplicate area identifiers. They are not selected using policy outcomes. Keep the 100-step ceiling, frozen scenario and all policy/model assets unchanged. No global update_env invocation or modifications to the installed library.

## Metering correction after first paired prefix

The initialized paired prefix stopped at step 4 (8 physical environment calls) because the independent ledger treated current curtailment as the official cost term. Source review shows baseEnv._sum_curtailment_mw is the change from the previous curtailment level; L2RPNSandBoxScore uses its negative. Correct the ledger to use current minus previous public curtailment, separately retaining the current physical energy quantity. Add hand fixtures for persistent, released and increased curtailment. Preserve the failed initialized output and initial unit-check artifact. Repeat the same frozen scenario and policy with unchanged tolerance in a new output directory, saving observations/ledger before assertions. These 8 calls count toward the original 100-step ceiling; no change to dataset split, policy, score implementation or research criteria.
