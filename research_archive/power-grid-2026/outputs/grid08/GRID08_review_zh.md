# GRID08 实际训练与改进复核

**结论：辅助恢复模块在这四张开发评价周上有明显作用；当前 GNN/MLP 没有超过“通过共同筛选就始终恢复”的固定规则。V2 修复了 V1 GNN 的退化表现，但没有产生额外学习收益。电网方向继续，当前二选一策略的贡献仍未建立。**

| 版本 | GNN 的确定性行为 | 完整评价周 | 相对原 NN20 的逐周成本降幅均值 | 相对 ALWAYS_RESTORE |
|---|---|---:|---:|---:|
| V0 | 与始终恢复完全同轨迹 | 4/4 | 63.8645% | 0，4/4 动作与状态完全相同 |
| V1 | 与 NN20 完全同轨迹 | 4/4 | 0 | 成本高 190.4942%，按固定规则成本逐周归一化 |
| V2 | 与始终恢复完全同轨迹 | 4/4 | 63.8645% | 0，4/4 动作与状态完全相同 |

三个版本的 MLP 都与 ALWAYS_RESTORE 完全同轨迹。IMMEDIATE_COST 只完成 3/4 周，十一月在 1192/2017 步终止；不能给它按四周平均成本排名。V2 相对它的成本降幅只在双方完整的三个周计算，均值 31.3110%，另报告完整周数量的差异。

这里的费用是模拟器原生运营成本。63.8645% 是逐周相对降幅的平均值，不是现实电网节省金额，也不是 GNN 相对最强规则的提升。使用相同候选、信息、筛选和计量的固定规则已经获得了全部这部分收益。

## 已经改了什么

- V0 是原始共享 actor/critic 的 GNN/MLP PPO 试训；原始检查点与失败记录均保留。
- V1 将 actor 优化限定在有恢复选择的步骤，独立训练 critic，并统一 bootstrap 输入。GNN 出现“不恢复”退化；MLP 保持始终恢复。
- V2 从 V1 权重开始，用训练记录作概率偏置，使初始恢复概率接近 0.95；保留状态相关权重与 critic，重置 Adam、减半学习率、固定再训练四遍。没有选最好看的中间检查点。
- V2 的 GNN 与 MLP 各完成 16 个训练 episode 和 4 个评价 episode。末端训练样本上，两者确定性决策仍全部是恢复。

这是新构造的 GNN/MLP PPO 辅助接受器，并不是原 IM DQN 框架未经修改地移植到电网。原 LJN NN20、动作库和 unsafe QP 没有被覆盖。

## 失败反思已有具体依据

**安全筛选范围有限。** 对 V1 十月一次训练失败，前 131 步精确重现归档动作和状态；公开 forecast 的恢复→no-op 分支第二步报再调度不可行，hold→no-op 两步均存活。另列的真实未来反事实同样如此。这只是两步局部反例，不能推断 hold 活完整周。

两步 guard 在两个训练周的固定规则复核中没有触发任何 veto：十月两臂都完成，一月两臂都在 1719 步终止。一月终止反馈为断连电网/潮流失败，当步没有 offered 恢复选择。不能把这种故障和恢复过快混为同一个原因；也不能因为终止时无选择就排除更早决策的影响。

**回报归因与 critic 值得进一步检查。** 训练机会间隔 6–114 个物理步骤，现有 GAE 的额外 lambda 衰减也发生在没有选择权的步骤。准确 critic 仍可通过 bootstrap 传递长效应，因此这不是实现错误或已确定根因。[SMAAC 作者训练代码](https://github.com/sunghoonhong/SMAAC/blob/master/train.py) 提供了按高层决策组织交互的参考；不能把我们的动作掩码称为其复现。

训练记录的只读 critic 诊断显示，各周平均预测约 -15，而 realized 折扣回报均值约 -3 至 -119。静态归一化 MSE 的训练集拟合探针将 RMSE 从 105.76 降到 88.18，但 MAE 从 31.10 增到 35.26；原 Huber 续拟合 RMSE 为 102.50、MAE 为 25.91。这些来自变化中的训练策略且只评价拟合数据，不证明价值预测泛化、故障原因或控制收益。相关成熟方案见 [PopArt 原论文](https://papers.neurips.cc/paper_files/paper/2016/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)；当前控制器没有被这些 critic-only 探针替换。

## 续跑、审计与资源

三个进程曾消失，没有失败标记；具体外部原因未确定。保留中断 episode 与原更新日志后，从最近完整 episode 的模型、Adam 和 Torch RNG 恢复。1838/1199/401 步的中断前缀全部重现动作和状态哈希。额外物理开销为 3438 个已记录步骤，保守上界 3441；不能把它们算成有效训练 episode，也不称为中途电网状态无缝续跑。

V0/V1/V2 共 100 个完整归档 episode 的向量和账目已审计，另有四次 guard 运行。含资格检查、失败尝试、中断重跑和诊断，总物理步数为 **204027–204030**，在 GRID08 的 220000 上限内。公开 forecast 的计算调用另列，不能视为零成本。审计是独立向量算术与记录核对，不是独立电力仿真重跑或独立神经网络重训。

最终测试周没有用于拟合、调参或评分；作者原预训练轨迹身份未知，因此不能认证我们的划分与作者所有训练数据都不重叠。单训练种子、四个反复使用的开发评价周，不支持录用概率或总体泛化保证。

## 当前推进

GRID09 已单独冻结并启动普通 PPO 续训与 decision-linked GAE 两臂：同一个 V2 GNN、同初始权重、同预算、同奖励、候选和安全筛选，只改变 lambda 的决策链接。物理 gamma 与所有中间奖励保留。它是受控开发比较，不是完整 semi-Markov/SMAAC 复现，也不是新意认证。GRID08 裁决不会因后续结果被改写。

完整数字：[机器可读汇总](C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl/outputs/grid08/delivery.json)、[版本表](C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl/outputs/grid08/GRID08_report.md)、[续跑复核](C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl/outputs/grid08/resume_audit.json)、[一手代码与失败复核](C:/Users/windows/Documents/Codex/2026-09-24/influence-maximization-im-ccf-b-grl/outputs/grid08/literature_failure_review.md)。
