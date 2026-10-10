# GRID09: decision-linked credit experiment

| Version | Reference | Jointly complete | Mean cost reduction | Positive / negative | Exact same trajectory |
|---|---|---:|---:|---:|---:|
| physical | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| physical | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| physical | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| physical | V2_GNN | 4 | 0.0000% | 0 / 0 | 4 |
| decision | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| decision | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| decision | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| decision | V2_GNN | 4 | 0.0000% | 0 / 0 | 4 |
| decision | PHYSICAL_CONTINUE | 4 | 0.0000% | 0 / 0 | 4 |

Both arms start from the same V2 GNN, reset Adam identically and complete two passes of the same training weeks. Decision-linked GAE retains gamma per physical step and applies lambda only on links to an actual next offered state; physical rollout boundaries and the original critic are retained. No candidate, safety screen, feature, reward or final-test change.

{"physical": 21724, "decision": 21724, "total_physical": 43448, "bytes": 67136106}

- One training seed and four repeatedly used development weeks; no independent confirmation.
- Only eligibility-trace timing changes between continuation arms. This is not a full semi-Markov or SMAAC replication.
- Costs ranked only jointly completed episodes; native costs are simulated benchmark accounting.
- Saved-vector audit is not independent power-flow rerun or retraining.
