# GRID63 等预算全局扩展闭环比较

冻结判读：GLOBAL_COMPETITIVE_NO_MATERIAL_STRUCTURE_GAIN。

Same seven exposed dates. Equal per-state extra budget, not guaranteed equal whole-week query totals. Saved-vector/cost/feedback audit, no independent AC solve. No GNN/RL.

|策略|质量日期|完整周日期|控制时间减少/FULL|查询减少/FULL|最大完整周成本增幅|
|---|---:|---:|---:|---:|---:|
|FULL|7/7|5/7|0.000%|0.000%|+0.00000%|
|FRAME|7/7|5/7|14.016%|74.092%|+0.37200%|
|GLOBAL_MATCH|7/7|5/7|13.721%|74.092%|+0.41877%|

|日期|FRAME质量|GLOBAL质量|GLOBAL减FRAME成本/FULL|FRAME时间减少/GLOBAL|两者平均查询|
|---|---|---|---:|---:|---|
|2035-03-05_0|True|True|+0.04677%|0.953%|3336.0 / 3336.0|
|2035-06-25_8|True|True|不可比|0.989%|1428.0 / 1428.0|
|2035-09-03_4|True|True|+0.00000%|-3.666%|2731.0 / 2731.0|
|2035-10-01_5|True|True|+0.00000%|0.657%|256.0 / 256.0|
|2035-04-09_3|True|True|+0.00782%|2.692%|1012.0 / 1012.0|
|2035-05-14_14|True|True|不可比|1.043%|0.0 / 0.0|
|2035-11-19_0|True|True|+0.00000%|-0.517%|384.0 / 384.0|

三重复物理路径一致组：21/21；共同输入状态等查询预算核对102项。计时电源事件：6，时间资格False。

63条回放来自7个已有日期，三次重复测量计时；不是63个独立场景。控制器各自形成后续轨迹，整周查询数可以不同。

完整逐日期质量、查询和时间离散程度见per_family.csv；全决策与搜索发生步的P95/P99/最大值见latency_per_run.csv；补查触发和改变动作见call_accounting.csv。

本次仅检验已冻结参数和共同频次排序下的结构补全增量，不追加参数、模型或日期。
