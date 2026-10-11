# GRID39 一版目标修订的实际闭环评价

12条；23856物理步；81899公开预测。

| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 308.692 | 22597 | 4/4 | 3/4 | 0.00000% |
| REVISED_GNN0 | 274.638 | 7380 | 4/4 | 3/4 | 0.13612% |
| REVISED_GNN1 | 273.841 | 7381 | 4/4 | 3/4 | 0.13612% |

| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |
|---|---|---:|---:|---:|
| REVISED_GNN0 | FULL | 11.032% | 67.341% | 4/4 |
| REVISED_GNN1 | FULL | 11.290% | 67.336% | 4/4 |

4 exposed development dates and one timing repeat. Original GRID37 timings are historical, not paired to new models. Both seeds retained; .5% is a quality investment tolerance, not statistical equivalence. Common author PPO/caches remain in all arms. Saved-data audit is not independent AC feasibility solving. Incomparable earlier failure never counts as speed win.

质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。


## 与原版的质量/查询参照

| 种子 | 原/新查询 | 查询减少 | 成本变化（四周平均，以FULL成本归一化） |
|---|---:|---:|---:|
| 0 | 7312/7380 | -0.930% | 0.00000% |
| 1 | 7484/7381 | 1.376% | 0.03403% |

新旧属于不同计时批次，不计算配对速度增量。这版目标未建立稳定改进，暂不替换原配置；全部原/新权重、训练与失败/混合记录保留。下一步转向限定的近期论文对照适配，不继续扫描目标混合比例。
