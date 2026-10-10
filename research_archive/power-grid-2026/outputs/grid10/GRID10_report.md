# GRID10: paired critic training package

| Version | Reference | Jointly complete | Mean cost reduction | Positive / negative | Identical trajectories |
|---|---|---:|---:|---:|---:|
| raw | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| raw | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| raw | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| raw | V2_GNN | 4 | 0.0000% | 0 / 0 | 4 |
| normalized | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| normalized | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| normalized | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| normalized | V2_GNN | 4 | 0.0000% | 0 / 0 | 4 |
| normalized | RAW_HUBER | 4 | 0.0000% | 0 / 0 | 4 |

Same V2 actor and equal-budget critic warm-fits; native physical discount, reward and actor advantages retained. Fixed final checkpoints only.

{"raw": 23608, "normalized": 21724, "total_physical": 45332, "bytes": 69815742}
- One seed, four already-used development weeks; not independent confirmation.
- Fixed normalization and MSE loss change together; no adaptive PopArt implementation.
- Warm-fit targets are changing-policy training returns; not unbiased final-policy targets.
- Saved-vector arithmetic audit is not independent physics replay or retraining.
