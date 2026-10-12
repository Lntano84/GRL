# GRID52 八个预留日期家族确认

96条串行回放；193632物理步；661264公开预测。

| 方法 | 完成 | 无提前失败 | 质量筛查 | 完成周最大成本差 | 对FULL宏平均控制时间减少 |
|---|---:|---:|---:|---:|---:|
| FULL | 16/16 | 16/16 | 16/16 | 0.00000% | 参照 |
| SOFTGNN0 | 16/16 | 16/16 | 16/16 | 0.17696% | 15.934% |
| SOFTGNN1 | 16/16 | 16/16 | 16/16 | 0.16893% | 18.097% |
| LOCAL_PPO | 16/16 | 16/16 | 12/16 | 12.43418% | 17.164% |
| COMPRESSED_GNN0 | 16/16 | 16/16 | 16/16 | 0.08045% | 14.465% |
| COMPRESSED_GNN1 | 16/16 | 16/16 | 16/16 | 0.08045% | 14.748% |

| 方法 | 对照 | 可比家族 | 宏平均提速 | 家族bootstrap95% | 时间正向家族 |
|---|---|---:|---:|---|---:|
| SOFTGNN0 | FULL | 8 | 15.934% | [8.857%, 23.507%] | 7 |
| SOFTGNN1 | FULL | 8 | 18.097% | [10.368%, 25.793%] | 7 |
| LOCAL_PPO | FULL | 8 | 17.164% | [9.261%, 24.683%] | 7 |
| COMPRESSED_GNN0 | FULL | 8 | 14.465% | [6.988%, 22.219%] | 7 |
| COMPRESSED_GNN1 | FULL | 8 | 14.748% | [7.482%, 22.107%] | 7 |
| COMPRESSED_GNN0 | LOCAL_PPO | 8 | -3.570% | [-8.596%, 0.458%] | 4 |
| COMPRESSED_GNN0 | SOFTGNN0 | 8 | -1.888% | [-5.663%, 1.281%] | 4 |
| COMPRESSED_GNN0 | SOFTGNN1 | 8 | -4.603% | [-6.546%, -2.699%] | 0 |
| COMPRESSED_GNN1 | LOCAL_PPO | 8 | -3.243% | [-7.339%, 0.208%] | 3 |
| COMPRESSED_GNN1 | SOFTGNN0 | 8 | -1.613% | [-4.993%, 1.416%] | 4 |
| COMPRESSED_GNN1 | SOFTGNN1 | 8 | -4.343% | [-6.540%, -2.261%] | 0 |

两软压缩模型种子均通过投资筛查：True。

Fresh second date-family confirmation of soft-compressed candidate of frozen controller/model;8families from one IDF2023 grid. Timing repeats/model seeds are not independent families. FULL failed weeks retained; no speed ratio for unequal horizon; cost primary only on completed weeks. .5%screen is not equivalence. SoftGNN is resource-matched adaptation; authorPPO training exposure unknown. Audit uses saved vectors/feedback,not independent ACsolve or refit. No graph-only novelty established.

两次计时物理路径一致：48/48。
所有早失败/成本较差/较慢对照均保留；预留日期使用后不再宣称未暴露。
