# GRID35 两个非学习顺序对照

所有方法共享既有停止规则，以下只为26个已暴露状态上的选择/查询诊断。

| 顺序 | 查询 | 同FULL选择 | 最大rho差 | 找回1733关键动作 |
|---|---:|---:|---:|---|
| STATIC_ID | 7879 | 21/26 | 0.0168152 | True |
| LOCAL_STRESS | 6965 | 17/26 | 0.0336139 | True |
| GNN0 (GRID33) | 7358 | 24/26 | 0.0001174 | True |
| GNN1 (GRID33) | 7530 | 25/26 | 0.0000000 | True |
| MLP0 (GRID33) | 7549 | 25/26 | 0.0029137 | True |
| MLP1 (GRID33) | 7826 | 26/26 | 0.0000000 | True |
| PPO (GRID33) | 7823 | 24/26 | 0.0001174 | True |

26exposed historical FULL states,not independent dates and not current LOCAL-policy state distribution. Cheap orders are publicly defined before read outcomes; no closed-loop quality/speed or GNN necessity claim.

失败只阻断当前顺序/停止组合；通过只允许闭环资格检查，不是证明廉价规则或GNN的整体速度。
