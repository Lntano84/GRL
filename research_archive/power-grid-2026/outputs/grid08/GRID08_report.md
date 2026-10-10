# GRID08 training and adaptation delivery

Auxiliary binary GNN/MLP PPO chooses safe-continuous restoration over an immutable NN20 agent. This is actual training and complete-week replay, not a claim that GNN + RL itself is novel.

| Version | Model | Reference | Complete pairs | Mean cost reduction | Positive / negative | Identical trajectories |
|---|---|---|---:|---:|---:|---:|
| v0 | gnn | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| v0 | gnn | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| v0 | gnn | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| v0 | mlp | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| v0 | mlp | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| v0 | mlp | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| v1 | gnn | NN20 | 4 | 0.0000% | 0 / 0 | 4 |
| v1 | gnn | ALWAYS_RESTORE | 4 | -190.4942% | 0 / 4 | 0 |
| v1 | gnn | IMMEDIATE_COST | 3 | -73.1950% | 0 / 3 | 0 |
| v1 | mlp | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| v1 | mlp | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| v1 | mlp | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| v2 | gnn | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| v2 | gnn | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| v2 | gnn | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |
| v2 | mlp | NN20 | 4 | 63.8645% | 4 / 0 | 0 |
| v2 | mlp | ALWAYS_RESTORE | 4 | 0.0000% | 0 / 0 | 4 |
| v2 | mlp | IMMEDIATE_COST | 3 | 31.3110% | 3 / 0 | 0 |

## Interpretation

An improvement over NN20 may come from supplying the restoration action. Improvement over ALWAYS_RESTORE and IMMEDIATE_COST is the relevant evidence for added decision value from learning. Exact trajectory equality identifies a reproduced fixed rule; it is stronger than a nonsignificant comparison. Lower cumulative cost after earlier failure is never counted as quality gain.

V0 retains the original shared actor/critic and all rollout-step policy normalization. V1 isolates actor and critic, normalizes and optimizes the actor only on offered decisions, and makes critic bootstrap proposal-invariant. V2 shifts the V1 policy near the qualified fixed restoration rule and performs four additional training passes at half the learning rate, preserving critic and input-dependent logit differences. These are development adaptations; all checkpoints and failed engineering attempts are retained.

## Resources

{
  "successful_physical_steps": 192339,
  "failed_attempt_physical_steps": 259,
  "output_bytes": 401330897,
  "note": "Preflight physical steps are reported in its retained qualification record; counts above are run phases only. Simulated forecast power-flow calls are additional and logged per episode. Training/evaluation phase wall times include concurrent contention; they are not intrinsic speed comparisons.",
  "diagnostic_physical_steps": 135,
  "guard_qualification_physical_steps": 7472,
  "interrupted_extra_physical_steps_lower": 3438,
  "interrupted_extra_physical_steps_upper": 3441,
  "cumulative_physical_steps_including_qualification": 204030
}

## Limits

- Single training seed, four validation weeks; no population or publication guarantee.
- Author pretraining identities are unknown. Validation is disjoint from our PPO weeks, not proven disjoint from all author training.
- NN20 is instrumented with common proposal generation; wall time is not pure original-agent deployment latency.
- Independent audit checks saved cost vectors and accounting, not independent electrical replays or retraining.
- One-step proposal screens cannot guarantee later survival.
- V1 and V2 are disclosed development adaptations, not independent confirmations.
- Three processes were externally interrupted without a failure marker. Whole-episode checkpoint resumption retained model/Adam/Torch RNG; interrupted costs are counted with a conservative three-step uncertainty bound. Prior process and resumed wall times are not a single uninterrupted cap.
