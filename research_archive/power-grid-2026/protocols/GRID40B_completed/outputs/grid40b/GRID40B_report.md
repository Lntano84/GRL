# GRID40B SoftGNN核心适配及训练优化检查

官方GAT和27维输入转换代码未修改。IDF2023、909动作、公开base连续动作上下文与共同搜索mask是明确适配；不是作者WCCI2022完整复现。

原论文学习率配置在148个小样本/批量8上训练进入平台；一版学习率0.001的修改只依据训练记录，原初始化/64轮/两种子/模型/dropout/数据不变，所有旧新模型保留。

| 种子 | 原训练KL | 修订训练KL | 静态频率KL |
|---|---:|---:|---:|
| 0 | 3.0739 | 1.1526 | 3.0717 |
| 1 | 2.9923 | 0.6785 | 3.0717 |

| 模型 | 26状态查询 | 原GNN查询 | 同FULL选择 | 最大rho差 | 关键1733匹配 |
|---|---:|---:|---:|---:|---|
| SOFTGNN0 | 8064 | 7358 | 22/26 | 0.0027224 | True |
| SOFTGNN1 | 8004 | 7530 | 22/26 | 0.0101787 | True |

Existing26exposed states, not closed-loop quality/time or independent transfer. Official-core resource-matched adaptation; both seeds retained. Training convergence reviewed separately.

训练拟合改善不是闭环质量提升；下一步以同一控制器和共有优化测完整质量—时间，不把公开方法适配版的有限结果外推到原论文。
