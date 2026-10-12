# GRID47 既有作者阈值回补的质量与速度

24条；47712物理步；186502公开预测。

| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |
|---|---:|---:|---:|---:|---:|
| FULL | 321.672 | 22597 | 4/4 | 3/4 | 0.00000% |
| LOCAL_GNN0 | 310.548 | 14893 | 4/4 | 3/4 | 0.00000% |
| LOCAL_GNN1 | 310.316 | 14893 | 4/4 | 3/4 | 0.00000% |
| SOFTGNN0 | 301.067 | 15251 | 3/4 | 3/4 | 1.29005% |
| SOFTGNN1 | 313.987 | 16057 | 3/4 | 3/4 | 6.19453% |
| LOCAL_PPO | 306.968 | 14893 | 4/4 | 3/4 | 0.00000% |

| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |
|---|---|---:|---:|---:|
| LOCAL_GNN0 | FULL | 3.458% | 34.093% | 4/4 |
| LOCAL_GNN1 | FULL | 3.530% | 34.093% | 3/4 |
| SOFTGNN0 | FULL | 6.406% | 32.509% | 4/4 |
| SOFTGNN1 | FULL | 2.389% | 28.942% | 3/4 |
| LOCAL_GNN0 | SOFTGNN0 | -3.149% | 2.347% | 0/4 |
| LOCAL_GNN0 | SOFTGNN1 | 1.095% | 7.249% | 1/4 |
| LOCAL_GNN1 | SOFTGNN0 | -3.072% | 2.347% | 0/4 |
| LOCAL_GNN1 | SOFTGNN1 | 1.169% | 7.249% | 1/4 |
| LOCAL_PPO | FULL | 4.571% | 34.093% | 4/4 |
| LOCAL_GNN0 | LOCAL_PPO | -1.166% | 0.000% | 1/4 |
| LOCAL_GNN1 | LOCAL_PPO | -1.091% | 0.000% | 2/4 |

4exposed development dates,1timingrepeat. Fresh originalGNN,SoftGNN adaptation and authorPPO with common original-author-threshold completion/controller/caches. Both seeds retained. .5%is investment quality tolerance,not equivalence. Official-core adaptation,not original WCCI trained agent. Saved audit is not independent ACsolve/refit. Incomparable failure not speedwin.

质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。


预定开发联合筛查：False。原两模型实际轨迹逐步等同FULL，控制汇总减少3.46%/3.53%，拓扑搜索时间减少22.32%/22.77%；宏平均未达5%。PPO查询相同且略快，不宣称严格保护下图模型增量已经建立。
