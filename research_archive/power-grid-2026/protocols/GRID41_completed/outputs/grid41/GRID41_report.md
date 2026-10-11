# GRID41 近期SoftGNN适配与原模型的同条件闭环比较

20条；39760物理步；127047公开预测。

| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 307.513 | 22597 | 4/4 | 3/4 | 0.00000% |
| LOCAL_GNN0 | 272.249 | 7312 | 4/4 | 3/4 | 0.13612% |
| LOCAL_GNN1 | 273.679 | 7484 | 4/4 | 3/4 | 0.00081% |
| SOFTGNN0 | 267.580 | 8054 | 3/4 | 3/4 | 2.90814% |
| SOFTGNN1 | 276.756 | 8398 | 3/4 | 3/4 | 6.19453% |

| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |
|---|---|---:|---:|---:|
| LOCAL_GNN0 | FULL | 11.468% | 67.642% | 4/4 |
| LOCAL_GNN1 | FULL | 11.003% | 66.881% | 4/4 |
| SOFTGNN0 | FULL | 12.986% | 64.358% | 4/4 |
| SOFTGNN1 | FULL | 10.002% | 62.836% | 4/4 |
| LOCAL_GNN0 | SOFTGNN0 | -1.745% | 9.213% | 0/4 |
| LOCAL_GNN0 | SOFTGNN1 | 1.628% | 12.932% | 1/4 |
| LOCAL_GNN1 | SOFTGNN0 | -2.279% | 7.077% | 0/4 |
| LOCAL_GNN1 | SOFTGNN1 | 1.112% | 10.884% | 1/4 |

4exposed development dates,1timingrepeat. Fresh originalGNN and SoftGNN adaptation compared under common controller/caches/stop. Both seeds retained. .5%is investment quality tolerance,not equivalence. Official-core adaptation,not original WCCI trained agent. Saved audit is not independent ACsolve/refit. Incomparable failure not speedwin.

质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。
