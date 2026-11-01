# GRID59 频次先验GNN有限闭环试点

裁决：WEAK_OR_MIXED_STOP_REVISION。

|方法|完成|质量通过|对FULL控制减少|对STATIC控制减少|对STATIC查询减少|查询正向日期|最大完整周成本变化/FULL|
|---|---:|---:|---:|---:|---:|---:|---:|
|STATIC_ID|3/4|4/4|15.336%|0.000%|0.000%|0/4|0.37200%|
|FREQUENCY_GNN0|3/4|4/4|14.848%|-0.635%|3.127%|2/4|0.37200%|
|FREQUENCY_GNN1|3/4|4/4|14.966%|-0.506%|2.675%|2/4|0.37200%|

Four olddevelopment dates;two overlapoldtraining families;single paired timing repeat;all seeds and common failure retained. Preliminary engineering screen,not significance/equivalence/unseenconfirmation or new RL.

早失败不作低成本成功；不同存活长度不算提速。实际控制时间包括图构造/先验/前向/仿真/连续优化，不用局部组件替代。频次先验是现成公开作者信息，原PPO先验版与所有失败均保留。
