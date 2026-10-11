# GRID32 其余开发周共享编译复用兼容性

6条回放、11870个物理步骤、36726次公开预测均核验。动作/观测/成本逐步等同原FULL，未新增或提前失败；六月原有1901步失败保留，不能说全周完成。

| 日期 | 原控制 | 双复用控制 | 减少 | 步数 | 完成 | 数值路径状态/迭代差别数 |
|---|---:|---:|---:|---:|---|---:|
| 2035-06-25_8 | 149.844s | 55.046s | 63.26% | 1901 | False | 0 |
| 2035-09-03_4 | 161.379s | 61.503s | 61.89% | 2017 | True | 0 |
| 2035-10-01_5 | 169.387s | 43.254s | 74.46% | 2017 | True | 0 |

本批汇总控制时间减少66.75%，仅为3条单次配对的点估计。三月22.42%来自另2重复，不在这里当作更多独立日期。

3already-exposed development date pairs,one timed repeat each. Known June failure retained. No independent final generalization/GNN attribution or universal runtime guarantee.

共有DPP/程序复用应提供给所有强基线。GNN架构、RL增量和辅助搜索的独有提速尚需另行公平比较；原模型、质量取舍、全部失败与快慢计时保留。
