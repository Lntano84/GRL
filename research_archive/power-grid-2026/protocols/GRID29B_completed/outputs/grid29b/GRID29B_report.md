# GRID29B 实际状态配对诊断

保存数据核验通过：154组DC三向量，153组发生于实际控制中；流量最大差0MW，2017步交付/成本/查询与原轨迹相同。

同状态/同实际存储记忆隔离探针：原DC 34.611s，缓存DC 13.268s，减少61.67%。探针输出未派发；整个诊断回放含探针开销，不能作为整体提速。

| 阶段 | 对象 | solver | 次数 | 调用时间 | 编译时间 | 后端时间 | 迭代 |
|---|---|---|---:|---:|---:|---:|---:|
| actual_dc | restoration_optimizer | OSQP | 153 | 30.022 | 29.17269015312195 | 0.06957 | 7650.0 |
| actual_other | base_optimizer | OSQP | 34 | 67.682 | 7.4533936977386475 | 59.978269 | 2484500.0 |
| actual_other | restoration_optimizer | OSQP | 153 | 38.030 | 34.72405004501343 | 2.2733541 | 91450.0 |
| shadow_cached | shadow_cached | OSQP | 153 | 11.873 | 11.40942668914795 | 0.0683693 | 7650.0 |
| shadow_original | shadow_original | OSQP | 153 | 31.691 | 30.852704763412476 | 0.0698789 | 7650.0 |

缺失统计保持缺失，recursive solver=None调用不叠加到explicit子调用。

旧两组是平衡顺序测量，10.93%和0.39%均保留。下面只分解已测总时间；restore=True是实际交付恢复动作的物理步标记，并不代表所有尝试恢复/DC调用。

| 旧重复 | 整体节省 | 预测节省 | 非预测节省 | restore=True步节省 | 其余步节省 |
|---|---:|---:|---:|---:|---:|
| 0 | 23.664 | 0.197 | 23.467 | 25.416 | -1.752 |
| 1 | 0.695 | -2.235 | 2.930 | 10.320 | -9.625 |

Old pilot has per-step and forecast timing but no per-solver timing. Component shadow result cannot identify the old pairs individual DC savings.

下一步若需要解释幅度差异，应在完整原/缓存两臂中同时记录每次DC和连续QP的时间、编译/后端/迭代，保持实际动作完全一致，而不是把探针时间从整体时间机械相减。
