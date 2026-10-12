# GRID44 八个预留日期家族确认

96条串行回放；193632物理步；536370公开预测。

| 方法 | 完成 | 无提前失败 | 质量筛查 | 完成周最大成本差 | 对FULL宏平均控制时间减少 |
|---|---:|---:|---:|---:|---:|
| FULL | 16/16 | 16/16 | 16/16 | 0.00000% | 参照 |
| LOCAL_GNN0 | 16/16 | 16/16 | 14/16 | 0.88509% | 14.982% |
| LOCAL_GNN1 | 16/16 | 16/16 | 14/16 | 0.88509% | 14.265% |
| SOFTGNN0 | 16/16 | 16/16 | 14/16 | 2.46195% | 15.431% |
| SOFTGNN1 | 16/16 | 16/16 | 12/16 | 2.46195% | 12.014% |
| LOCAL_PPO | 16/16 | 16/16 | 14/16 | 0.88668% | 11.911% |

| 方法 | 对照 | 可比家族 | 宏平均提速 | 家族bootstrap95% | 时间正向家族 |
|---|---|---:|---:|---|---:|
| LOCAL_GNN0 | FULL | 8 | 14.982% | [11.443%, 18.360%] | 8 |
| LOCAL_GNN1 | FULL | 8 | 14.265% | [9.769%, 18.586%] | 8 |
| SOFTGNN0 | FULL | 8 | 15.431% | [8.709%, 22.457%] | 8 |
| SOFTGNN1 | FULL | 8 | 12.014% | [6.344%, 17.975%] | 8 |
| LOCAL_PPO | FULL | 8 | 11.911% | [8.331%, 15.668%] | 8 |
| LOCAL_GNN0 | LOCAL_PPO | 8 | 3.211% | [0.954%, 6.076%] | 7 |
| LOCAL_GNN0 | SOFTGNN0 | 8 | -1.424% | [-7.497%, 3.298%] | 4 |
| LOCAL_GNN0 | SOFTGNN1 | 8 | 2.930% | [-0.130%, 6.035%] | 6 |
| LOCAL_GNN1 | LOCAL_PPO | 8 | 2.481% | [-0.546%, 5.722%] | 6 |
| LOCAL_GNN1 | SOFTGNN0 | 8 | -2.102% | [-7.769%, 2.549%] | 3 |
| LOCAL_GNN1 | SOFTGNN1 | 8 | 2.248% | [-0.443%, 5.473%] | 5 |

两原模型种子均通过投资筛查：False。

First reserved date-family confirmation of frozen controller/model;8families from one IDF2023 grid. Timing repeats/model seeds are not independent families. FULL failed weeks retained; no speed ratio for unequal horizon; cost primary only on completed weeks. .5%screen is not equivalence. SoftGNN is resource-matched adaptation; authorPPO training exposure unknown. Audit uses saved vectors/feedback,not independent ACsolve or refit. No graph-only novelty established.

两次计时物理路径一致：48/48。
所有早失败/成本较差/较慢对照均保留；预留日期使用后不再宣称未暴露。
