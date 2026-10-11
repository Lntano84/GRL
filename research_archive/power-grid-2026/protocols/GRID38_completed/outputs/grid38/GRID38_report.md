# GRID38 单一训练目标修订

原soft-rho目标保留一半，另一半均匀分配给实际FP32奖励最优的规范动作类；相同原初始化、数据顺序、优化器和64轮，不改变网络或部署规则。

| 模型 | 原查询 | 新查询 | 查询减少 | 原/新同FULL选择 | 最大rho差 | 找回关键动作 |
|---|---:|---:|---:|---:|---:|---|
| GNN0 | 7358 | 7490 | -1.794% | 24/24 | 0.0001174 | True |
| GNN1 | 7530 | 7426 | 1.381% | 25/24 | 0.0001174 | True |
| MLP0 | 7549 | 7663 | -1.510% | 25/24 | 0.0165976 | True |
| MLP1 | 7826 | 7651 | 2.236% | 26/24 | 0.0165976 | True |

26 exposed cached FULL states, not final generalization or closed-loop quality/speed. All four new models and original models preserved. Timing here is only isolated forward/ranking, not controller speed. Exact-original-choice comparison uses same historical feedback and stop.

这是一个配置假设检验，目标修改本身不能构成算法新意。闭环质量和完整控制时间尚未测，不能从离线查询量宣布提速。
