# GRID29 整套控制器模型复用试点

| 重复/顺序 | 原控制时间 | 复用控制时间 | 减少 | 成本差 | 交付轨迹 |
|---|---:|---:|---:|---:|---:|
| 0 / 原→复用 | 216.509s | 192.846s | 10.93% | 0 | 完全相同 |
| 1 / 复用→原 | 176.785s | 176.090s | 0.39% | 0 | 完全相同 |

汇总控制时间减少6.19%；两组均提速：True。动作/观测/成本逐步相同，公开预测次数不变。

Complete decision/control time including DC cache lookup and all controls/forecasts. Setup, physical grid steps, archival separate.
Only exposed March, two balanced repetitions. Same source queries, delivery and cost; generic DPP engineering improvement shared with all baselines. Does not establish all-date robust speed or GNN contribution.

V0真实Parameter倒数表达式不满足DPP，资格检查拒绝；旧代码/日志保留。V1在DSL外计算倒数并按实际电抗值使缓存失效，保持原数学问题/solver/warm_start=False。130个保存状态流量差0，实时闭环仍单独核对。

保存数据审计4条回放、8068个物理记录，独立复核掩码/别名/模型排序/实际派发/交付成本；没有另做AC重放或独立拟合。

下一步将共有工程优化也用于PPO/FULL等强基线，然后再以平衡重复评估拓扑搜索增量。不能把本表减少与GRID28百分比直接相加。
