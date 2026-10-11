# GRID31 共有DC/安全QP编译复用完整控制试点

6条完整回放：12102个物理步骤和67950次预测，逐步交付/成本均与原受保护FULL相同；所有新拟合/RL/封存评价为0。

| 重复 | 原控制 | DC复用 | DC+安全QP复用 |
|---|---:|---:|---:|
| 0 | 184.153s | 166.174s | 140.898s |
| 1 | 180.612s | 166.366s | 142.079s |

| 比较 | 重复0减少 | 重复1减少 | 汇总点估计 |
|---|---:|---:|---:|
| ORIGINAL→DC_ONLY | 9.76% | 7.89% | 8.83% |
| DC_ONLY→DC_AND_SAFE | 15.21% | 14.60% | 14.90% |
| ORIGINAL→DC_AND_SAFE | 23.49% | 21.33% | 22.42% |

设置/物理步/归档单列；控制时间包括实际模型、查询、求解、恢复、缓存和相同组件计时开销。真实params始终刷新；原solver/资格/warm_start=False保留。不得相加局部影子百分比或择最快重复。

原作者连续QP可能触及迭代上限，其规则原样保留；恢复QP资格未放松。V0非DPP、复制器负值setter失败与预冻结准备错误全部保留，不等于方法价值否定。

One exposed March/2balanced repeats. Shared conventional engineering for all baselines; not GNN novelty, final heldout result or guaranteed universal speed.
