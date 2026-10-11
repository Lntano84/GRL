# GRID29 V0失败与V1资格修订

V0 ndarray玩具用例通过，但真实LJN的线路电抗是CVXPY Parameter，倒数表达式不满足DPP；首次真实缓存模型构造被断言拒绝，未产生有效缓存比较或闭环。缓存键也错误地将Parameter对象当数组，不能按真实数值失效。旧代码、设计、冻结和失败日志保留。

V1在DSL外计算数值倒数，并按实际电抗数值更新键。每项数学约束/目标及原求解器不变，warm_start=False不变；新增真实Parameter原地变化和三角电网手算，用实际流量变化验证键失效。V1尚未运行；失败不是GNN价值判断，也不是闭环质量损失。

CVXPY官方说明：https://www.cvxpy.org/tutorial/dpp/index.html 。
