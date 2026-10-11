# GRID33 现成排序器同状态诊断

26个已暴露状态，共130个排序/停止比较。合法输入重新推理的GNN0/1顺序与历史52次逐项一致；独立实现从保存顺序重算130项查询和选择，无新AC仿真/拟合。

| 排序器 | 查询 | 同FULL选择 | 最大rho差 | 找回1733关键动作 |
|---|---:|---:|---:|---|
| GNN0 | 7358 | 24/26 | 0.0001174 | True |
| GNN1 | 7530 | 25/26 | 0.0000000 | True |
| MLP0 | 7549 | 25/26 | 0.0029137 | True |
| MLP1 | 7826 | 26/26 | 0.0000000 | True |
| PPO | 7823 | 24/26 | 0.0001174 | True |

全部现成排序器通过已知关键动作门槛，因此需公平闭环比较，不能把MLP省略。GNN0相对MLP0/PPO查询差仅为有限开发线索，不是图结构必要性、长期成本或封存泛化证据。

26exposed states; same FULL feedback replay diagnosis, not deployable oracle or closed-loop survival/cost proof. Timing includes first model loading and is not deployment latency.
