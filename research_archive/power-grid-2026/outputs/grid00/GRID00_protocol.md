# GRID00 — Grid2Op / L2RPN entry qualification

Frozen before installing Grid2Op or running any environment on 2026-10-08.

## Scope

This is an engineering and task-definition check, not a learning experiment, algorithm comparison, or research-value verdict. No GNN/RL training, teacher generation, hyperparameter search, or large scenario download is authorized by this protocol.

The intended research platform is `l2rpn_idf_2023`, because the LJN 2023 released agent and action assets target that environment. Assets must not be silently transferred to a different grid. Official bundled test data may be used for smoke checks only, and must be explicitly marked as such. The complete research dataset and held-out evaluation remain unqualified.

## Resource envelope

- Create an isolated Python environment under `work/grid00/.venv`; do not change previous study environments or global packages.
- Use published wheels, with exact package versions and a dependency manifest.
- At most 200 MiB of dependency/test-asset downloads in this round; no full 5.4 GB IDF dataset.
- Environment smoke workload: at most four baseline trajectories, at most 24 executed steps each, with seed 0. Stop the smoke process at 180 seconds (excluding package installation and first import).
- Do-nothing and reconnection agents are interface controls, not strong research baselines.
- No model training or large model download.

## Required checks

1. Record source versions, package versions, asset hashes, grid dimensions, scenario IDs and native action permissions.
2. Verify reset/replay reproducibility for the same seed and actions on the bundled data.
3. Verify a legal forecast simulation does not advance physical environment state; simulation counters may legitimately change and must be recorded separately.
4. Record valid forecast horizons, actual step results, illegal/ambiguous flags, termination and timing. A forecast is not future ground truth and simulation agreement is not assumed.
5. Independently inspect strong-baseline source/requirements: LJN and continuous optimization components, including grid-specific assets and license availability.

## Outputs and interpretation

Save the probe source, results, environment metadata, source provenance, installation log, full dependency lock and a Chinese report. Distinguish environment execution, strong-baseline readiness, data realism, and novelty. A successful smoke check cannot establish comparative performance, benefit from learning, or a publication contribution. A dependency or asset failure is an engineering limitation, not a negative scientific result.

If bundled IDF data are absent, report that limit; a separately labelled small-grid interface test may be used, but does not qualify IDF or its trained agent. Do not silently replace the research task.
