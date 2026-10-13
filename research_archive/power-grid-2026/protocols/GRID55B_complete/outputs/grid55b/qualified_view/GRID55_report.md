# GRID55 八个已暴露日期的旧新配对与共有缓存比较

128条串行回放；258176物理步；853352公开预测。

| 方法 | 完成 | 无提前失败 | 质量筛查 | 完成周最大成本差 | 对FULL宏平均控制时间减少 |
|---|---:|---:|---:|---:|---:|
| FULL | 16/16 | 16/16 | 16/16 | 0.00000% | 参照 |
| SOFTGNN0 | 16/16 | 16/16 | 16/16 | 0.17696% | 16.775% |
| SOFTGNN1 | 16/16 | 16/16 | 16/16 | 0.16893% | 17.062% |
| LOCAL_PPO | 16/16 | 16/16 | 12/16 | 12.43418% | 18.073% |
| COMPRESSED_GNN0 | 16/16 | 16/16 | 16/16 | 0.08045% | 15.438% |
| COMPRESSED_GNN1 | 16/16 | 16/16 | 16/16 | 0.08045% | 16.065% |
| LOCAL_GNN0 | 16/16 | 16/16 | 14/16 | 1.74036% | 18.336% |
| LOCAL_GNN1 | 16/16 | 16/16 | 14/16 | 1.70438% | 18.429% |

| 方法 | 对照 | 可比家族 | 宏平均提速 | 家族bootstrap95% | 时间正向家族 |
|---|---|---:|---:|---|---:|
| SOFTGNN0 | FULL | 8 | 16.775% | [9.355%, 24.536%] | 7 |
| SOFTGNN1 | FULL | 8 | 17.062% | [9.853%, 24.207%] | 7 |
| LOCAL_PPO | FULL | 8 | 18.073% | [10.237%, 25.608%] | 7 |
| COMPRESSED_GNN0 | FULL | 8 | 15.438% | [7.954%, 23.380%] | 7 |
| COMPRESSED_GNN1 | FULL | 8 | 16.065% | [8.559%, 23.893%] | 7 |
| LOCAL_GNN0 | FULL | 8 | 18.336% | [10.256%, 26.381%] | 7 |
| LOCAL_GNN1 | FULL | 8 | 18.429% | [10.550%, 26.269%] | 7 |
| LOCAL_GNN0 | LOCAL_PPO | 8 | 0.357% | [-2.051%, 1.978%] | 6 |
| LOCAL_GNN0 | SOFTGNN0 | 8 | 1.961% | [-0.776%, 5.220%] | 5 |
| LOCAL_GNN0 | SOFTGNN1 | 8 | 1.604% | [-1.439%, 5.685%] | 6 |
| LOCAL_GNN1 | LOCAL_PPO | 8 | 0.435% | [-1.690%, 1.880%] | 7 |
| LOCAL_GNN1 | SOFTGNN0 | 8 | 2.042% | [-0.539%, 4.905%] | 6 |
| LOCAL_GNN1 | SOFTGNN1 | 8 | 1.694% | [-1.176%, 5.422%] | 6 |
| COMPRESSED_GNN0 | LOCAL_PPO | 8 | -3.525% | [-9.463%, 1.319%] | 4 |
| COMPRESSED_GNN0 | SOFTGNN0 | 8 | -1.651% | [-4.831%, 1.534%] | 3 |
| COMPRESSED_GNN0 | SOFTGNN1 | 8 | -1.887% | [-4.374%, 0.335%] | 3 |
| COMPRESSED_GNN1 | LOCAL_PPO | 8 | -2.748% | [-8.465%, 1.684%] | 5 |
| COMPRESSED_GNN1 | SOFTGNN0 | 8 | -0.889% | [-3.946%, 1.971%] | 4 |
| COMPRESSED_GNN1 | SOFTGNN1 | 8 | -1.153% | [-3.830%, 1.173%] | 4 |
| COMPRESSED_GNN0 | LOCAL_GNN0 | 8 | -3.882% | [-8.808%, -0.089%] | 3 |
| COMPRESSED_GNN1 | LOCAL_GNN1 | 8 | -3.117% | [-7.300%, 0.075%] | 1 |

工程选型：COMPRESSED_CANDIDATE_QUALITY_SCREEN；新对旧平均控制时间减少：-0.03499309189662589。

两软压缩模型种子均通过投资筛查：True。

Matched old/new frozen-model comparison on already exposed dates;shared equivalent prior cache;8families from one IDF2023 grid. Timing repeats/model seeds are not independent families. FULL failed weeks retained; no speed ratio for unequal horizon; cost primary only on completed weeks. .5%screen is not equivalence. SoftGNN is resource-matched adaptation; authorPPO training exposure unknown. Audit uses saved vectors/feedback,not independent ACsolve or refit. No graph-only novelty established.

两次计时物理路径一致：64/64。
所有早失败/成本较差/较慢对照均保留；预留日期使用后不再宣称未暴露。
