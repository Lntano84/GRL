# GRID37 静态顺序与GNN的公平实际比较

32条；63616物理步；208592公开预测。

| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 643.951 | 45194 | 8/8 | 6/8 | 0.00000% |
| LOCAL_GNN0 | 549.889 | 14624 | 8/8 | 6/8 | 0.13612% |
| LOCAL_GNN1 | 559.584 | 14968 | 8/8 | 6/8 | 0.00081% |
| STATIC_ID | 548.469 | 15502 | 8/8 | 6/8 | 0.37200% |

| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |
|---|---|---:|---:|---:|
| STATIC_ID | FULL | 14.827% | 65.699% | 8/8 |
| LOCAL_GNN0 | FULL | 14.607% | 67.642% | 8/8 |
| LOCAL_GNN1 | FULL | 13.101% | 66.881% | 8/8 |
| LOCAL_GNN0 | STATIC_ID | -0.259% | 5.664% | 2/8 |
| LOCAL_GNN1 | STATIC_ID | -2.027% | 3.445% | 1/8 |

4 exposed development dates; 2 timing repeats are not independent date families. .5% is an investment quality tolerance, not statistical equivalence. Author base PPO and common caches remain in every arm. Saved-data audit is not independent AC feasibility solving. Unmatched earlier failure cannot count as speed win.

质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。
