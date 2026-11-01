# GRID61 同排序框架机制闭环比较

裁决：LOCAL_COMPLETION_SUPPORTED_IN_MATRIX。

Four olddevelopment families,two timing repeats;same frequency order/sharedcache/basecontrol. Mechanism ablation,not new independent generalization or global optimality. First-safe reference has stronger best-admissible exhausted fallback;not full SoftGNN reproduction.

|策略|完成|质量通过|控制减少/FULL|查询减少/FULL|最坏完整周成本变化/FULL|
|---|---:|---:|---:|---:|---:|
|FULL|6/8|8/8|0.000%|0.000%|+0.00000%|
|FIRST_SAFE_BEST|6/8|4/8|7.032%|51.450%|+3.95164%|
|NO_LOCAL|6/8|6/8|14.251%|71.840%|+4.81017%|
|FRAME|6/8|8/8|13.059%|69.367%|+0.37200%|

同方法两重复路径完全一致：16/16；保存数据核对63616步、70358项匹配反馈。没有独立AC重求解。

局部补全在本矩阵有质量帮助：True；普通首个安全候选强参照具有竞争力：False。完整逐次成本、存活和查询见per_run.csv，不省略不利日期。

GNN不是此矩阵的排序输入；所有策略用同一作者频次顺序。各臂共享DC/safeQP编译复用，不将共有工程算作GNN或此搜索策略独有贡献。
