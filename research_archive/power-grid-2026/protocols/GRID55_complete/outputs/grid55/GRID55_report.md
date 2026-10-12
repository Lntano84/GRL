# GRID55 八个已暴露日期的旧新配对与共有缓存比较

128条串行回放；258176物理步；853352公开预测。

| 方法 | 完成 | 无提前失败 | 质量筛查 | 完成周最大成本差 | 对FULL宏平均控制时间减少 |
|---|---:|---:|---:|---:|---:|
| FULL | 16/16 | 16/16 | 16/16 | 0.00000% | 参照 |
| SOFTGNN0 | 16/16 | 16/16 | 16/16 | 0.17696% | 16.330% |
| SOFTGNN1 | 16/16 | 16/16 | 16/16 | 0.16893% | 16.629% |
| LOCAL_PPO | 16/16 | 16/16 | 12/16 | 12.43418% | 14.048% |
| COMPRESSED_GNN0 | 16/16 | 16/16 | 16/16 | 0.08045% | 14.388% |
| COMPRESSED_GNN1 | 16/16 | 16/16 | 16/16 | 0.08045% | -63.316% |
| LOCAL_GNN0 | 16/16 | 16/16 | 14/16 | 1.74036% | 18.453% |
| LOCAL_GNN1 | 16/16 | 16/16 | 14/16 | 1.70438% | 18.625% |

| 方法 | 对照 | 可比家族 | 宏平均提速 | 家族bootstrap95% | 时间正向家族 |
|---|---|---:|---:|---|---:|
| SOFTGNN0 | FULL | 8 | 16.330% | [8.925%, 24.346%] | 7 |
| SOFTGNN1 | FULL | 8 | 16.629% | [9.342%, 24.058%] | 7 |
| LOCAL_PPO | FULL | 8 | 14.048% | [1.109%, 25.097%] | 6 |
| COMPRESSED_GNN0 | FULL | 8 | 14.388% | [6.537%, 22.862%] | 7 |
| COMPRESSED_GNN1 | FULL | 8 | -63.316% | [-225.517%, 22.828%] | 6 |
| LOCAL_GNN0 | FULL | 8 | 18.453% | [10.360%, 26.452%] | 7 |
| LOCAL_GNN1 | FULL | 8 | 18.625% | [10.797%, 26.341%] | 7 |
| LOCAL_GNN0 | LOCAL_PPO | 8 | 2.987% | [-1.692%, 9.322%] | 6 |
| LOCAL_GNN0 | SOFTGNN0 | 8 | 2.539% | [-0.329%, 5.756%] | 5 |
| LOCAL_GNN0 | SOFTGNN1 | 8 | 2.164% | [-1.147%, 6.292%] | 6 |
| LOCAL_GNN1 | LOCAL_PPO | 8 | 3.121% | [-1.387%, 9.374%] | 7 |
| LOCAL_GNN1 | SOFTGNN0 | 8 | 2.702% | [-0.033%, 5.561%] | 6 |
| LOCAL_GNN1 | SOFTGNN1 | 8 | 2.336% | [-0.829%, 6.128%] | 6 |
| COMPRESSED_GNN0 | LOCAL_PPO | 8 | -1.690% | [-8.733%, 5.092%] | 4 |
| COMPRESSED_GNN0 | SOFTGNN0 | 8 | -2.284% | [-5.246%, 0.905%] | 2 |
| COMPRESSED_GNN0 | SOFTGNN1 | 8 | -2.531% | [-4.691%, -0.606%] | 2 |
| COMPRESSED_GNN1 | LOCAL_PPO | 8 | -51.359% | [-148.254%, 0.032%] | 4 |
| COMPRESSED_GNN1 | SOFTGNN0 | 8 | -82.249% | [-244.976%, 0.887%] | 3 |
| COMPRESSED_GNN1 | SOFTGNN1 | 8 | -82.153% | [-243.652%, -0.009%] | 3 |
| COMPRESSED_GNN0 | LOCAL_GNN0 | 8 | -5.210% | [-9.825%, -1.223%] | 2 |
| COMPRESSED_GNN1 | LOCAL_GNN1 | 8 | -94.973% | [-278.154%, -1.782%] | 0 |

工程选型：NO_CLEAR_PREFERRED_VERSION；新对旧平均控制时间减少：-0.5009165561216382。

两软压缩模型种子均通过投资筛查：False。

Matched old/new frozen-model comparison on already exposed dates;shared equivalent prior cache;8families from one IDF2023 grid. Timing repeats/model seeds are not independent families. FULL failed weeks retained; no speed ratio for unequal horizon; cost primary only on completed weeks. .5%screen is not equivalence. SoftGNN is resource-matched adaptation; authorPPO training exposure unknown. Audit uses saved vectors/feedback,not independent ACsolve or refit. No graph-only novelty established.

两次计时物理路径一致：64/64。
所有早失败/成本较差/较慢对照均保留；预留日期使用后不再宣称未暴露。
