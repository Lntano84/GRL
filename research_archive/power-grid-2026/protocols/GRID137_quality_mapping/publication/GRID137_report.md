# GRID137：56条真实分支按原论文质量标准重新核对

本轮只添加按旧标准派生的评价字段；GRID96原SKIP/MIXED/KEEP标签、模型、概率和所有原文件不变。没有新训练、模型前向或物理回放。

## 对齐的标准与成本口径

严格沿用GRID91的已审核FULL参照映射与0.5%成本阈值。FULL完成时，取消分支必须完成、整周总成本不超过FULL的100.5%；FULL未完成时，只按原规则检查不早于FULL终止，但单独标为REFERENCE_LIMITED，不能作为整周成本达标正例。
全部56条的场景/电网环境/原FRAME参照与GRID91对应。独立核对每条实际FRAME重现、共同干预前缀和起始状态哈希；共同前缀成本加两条后缀分别等于整周总成本，不能把干预后剩余成本百分比当整周百分比。
对已完成分支，取消造成的成本增量应不超过1.005×FULL整周成本−FRAME整周成本。这个剩余空间依赖已有FRAME相对FULL的位置，不能再给FRAME额外叠加一个0.5%。

## 评价字段的分布

| 范围 | 例数 | 按原标准派生的状态 | 失败日期 |
|---|---:|---|---|
| all56 | 56 | {'PASS_COMPLETE': 45, 'REFERENCE_LIMITED': 9, 'FAIL_SURVIVAL': 1, 'FAIL_COST': 1} | ['2035-07-09', '2035-12-10'] |
| nonpilot53 | 53 | {'PASS_COMPLETE': 43, 'REFERENCE_LIMITED': 9, 'FAIL_COST': 1} | ['2035-12-10'] |
| action_changed | 17 | {'PASS_COMPLETE': 15, 'FAIL_SURVIVAL': 1, 'FAIL_COST': 1} | ['2035-07-09', '2035-12-10'] |

PASS_COMPLETE：有完整周证据且通过原筛查；FAIL_COST/FAIL_SURVIVAL：对应原成本/存活标准失败；REFERENCE_LIMITED：FULL本身未完成，仅相对存活可比。外部事件不一致或基线未通过的记录保留为单独未知状态。

## 与原多维标签的关系

- 原SKIP：{'PASS_COMPLETE': 35, 'REFERENCE_LIMITED': 9}。
- 原MIXED：{'PASS_COMPLETE': 10, 'FAIL_COST': 1}。
- 原KEEP：{'FAIL_SURVIVAL': 1}。

两个标签系统回答不同问题。原MIXED可能在论文完成/成本标准下通过，但过载或查询指标仍有权衡；不能改称所有维度都安全。原路径等价的共同提前结束，也不能因为原标签是SKIP就当完整周正例。风险原值全部保留在逐例账本。

## 全部失败与证据不足案例

- S003 old13/2035-06-25_8 step408：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1901/1901，对FULL成本差None。
- S004 old13/2035-06-25_8 step1003：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1901/1901，对FULL成本差None。
- S005 old13/2035-06-25_8 step1900：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1901/1901，对FULL成本差None。
- S025 old13/2035-07-09_2 step1901：FAIL_SURVIVAL；原KEEP；FULL/取消完成=True/False，步数2017/1906，对FULL成本差None。
- S028 old13/2035-12-10_6 step1170：FAIL_COST；原MIXED；FULL/取消完成=True/True，步数2017/2017，对FULL成本差0.011084379001464972。
- S036 G1/2036-03-03_9 step535：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1720/1720，对FULL成本差None。
- S037 G1/2036-03-03_9 step540：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1720/1720，对FULL成本差None。
- S038 G1/2036-03-03_9 step541：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1720/1720，对FULL成本差None。
- S047 G2/2036-03-03_10 step113：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1878/1878，对FULL成本差None。
- S048 G2/2036-03-03_10 step545：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1878/1878，对FULL成本差None。
- S049 G2/2036-03-03_10 step1742：REFERENCE_LIMITED；原SKIP；FULL/取消完成=False/False，步数1878/1878，对FULL成本差None。

原0.5%是实验筛查容差，不是物理安全保证。所有状态均为已暴露开发记录，单次干预后继续原FRAME；不能把这些标签复制给每批动作，也不能把多次取消的收益相加。

## 既有模型建议的回查

saved_readout_crosscheck.json按原0.5/0.95读数、原日历分组与old-to-generated划分，回查全部原模型和常数参照。不产生新概率、不挑最好读数、不训练新目标；旧概率是Pareto类别概率，不是本轮完整周通过概率。
这只是把已知建议与新评价字段并列，不能当一次预先设计的目标对齐学习实验，更不能据少量已知失败宣布部署可靠。

## 下一步判断

现在可明确区分“有完整质量证据”“原筛查失败”“仅相对存活可比”，而不再把即时G>0、MIXED或SKIP简单当训练答案。能否进行有意义的新训练，应看失败日期与非等价轨迹样本的覆盖，而不是56这个总数。
本轮不扩大场景、训练或修改论文；最终8测试仍封存。
