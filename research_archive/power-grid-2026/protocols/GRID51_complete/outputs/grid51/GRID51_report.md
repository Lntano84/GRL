# GRID51 原快速控制下残差评分改动

20条；39760物理步；126810公开预测。

| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 328.615 | 22597 | 4/4 | 3/4 | 0.00000% |
| LOCAL_GNN0 | 295.480 | 7312 | 4/4 | 3/4 | 0.13612% |
| LOCAL_GNN1 | 279.749 | 7484 | 4/4 | 3/4 | 0.00000% |
| COMPRESSED_GNN0 | 278.915 | 7682 | 4/4 | 3/4 | 0.00000% |
| COMPRESSED_GNN1 | 281.646 | 7682 | 4/4 | 3/4 | 0.00000% |

| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |
|---|---|---:|---:|---:|
| LOCAL_GNN0 | FULL | 10.083% | 67.642% | 3/4 |
| LOCAL_GNN1 | FULL | 14.870% | 66.881% | 4/4 |
| COMPRESSED_GNN0 | FULL | 15.124% | 66.004% | 4/4 |
| COMPRESSED_GNN1 | FULL | 14.293% | 66.004% | 4/4 |
| COMPRESSED_GNN0 | LOCAL_GNN0 | 5.606% | -5.060% | 4/4 |
| COMPRESSED_GNN1 | LOCAL_GNN1 | -0.678% | -2.646% | 2/4 |
| COMPRESSED_GNN0 | LOCAL_GNN1 | 0.298% | -2.646% | 1/4 |
| COMPRESSED_GNN1 | LOCAL_GNN0 | 4.682% | -5.060% | 2/4 |

4exposed development dates,1timingrepeat. Original/soft-compressed residualGNN compared under unchanged fast controller/caches/stop. Both seeds retained. .5%is investment quality tolerance,not equivalence. Official-core adaptation,not original WCCI trained agent. Saved audit is not independent ACsolve/refit. Incomparable failure not speedwin.

质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。
