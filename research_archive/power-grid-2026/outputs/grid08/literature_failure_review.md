# GRID08：针对实际训练困难的一手实现复核

查阅日期：2026-10-09。正在运行的 V2 配置保持不变；本记录不把文献中的方法自动当作已复现，也不读取封存最终测试成绩。

## 已核验来源与具体启发

1. **SMAAC，ICLR 2021，作者实现**：[README](https://github.com/sunghoonhong/SMAAC)、[图网络](https://github.com/sunghoonhong/SMAAC/blob/master/models.py)、[策略与评论器](https://github.com/sunghoonhong/SMAAC/blob/master/agent.py)、[交互与训练](https://github.com/sunghoonhong/SMAAC/blob/master/train.py)。Actor 与 critic 使用 GAT；高层决策通过低层动作队列执行，安全时可以不动作。`TrainAgent.interaction` 在两个高层决策间累计奖励，作者代码还裁剪这一累计训练奖励；因此不能把普通逐步 PPO 的动作掩码称为 SMAAC 复现。旧实现依赖 Grid2Op 0.9.4，不直接覆盖正在使用的 1.12.5 环境。
2. **RL2Grid，作者论文与代码**：[论文](https://arxiv.org/html/2503.23101v1)、[项目](https://github.com/emarche/RL2Grid)、[环境构造](https://github.com/emarche/RL2Grid/blob/main/env/utils.py)、[规则包装器](https://github.com/emarche/RL2Grid/blob/main/env/heuristic.py)。公开实现明确纳入 idle/recovery 强规则并比较其辅助作用。它研究的是既有基准任务，不是我们这套 NN20 上的连续恢复接受器；它的热线路规则也不能直接证明机组爬坡安全。
3. **2026 runtime safety shielding 预印本**：[全文](https://arxiv.org/html/2604.14032v1)。正文公式 (4)、(10) 主要按一步预测线路负载过滤动作。该文不是“多步机组爬坡风险已经解决”的证据；本次未定位、复现其完整可运行实现，也不宣称其 formal guarantee 可迁移到我们的动作与预测误差条件。
4. **Grid2Op 官方预测接口**：[官方文档](https://grid2op.readthedocs.io/en/latest/model_based.html)，本地 `Observation/baseObservation.py:get_forecast_env` 已核验。接口构造由当前公开 forecast 驱动的环境，可以顺序执行候选动作与后续动作；与复制真实环境读取未来 chronics 的离线反事实必须分开记账。
5. **2026 图表示比较预印本**：[论文](https://arxiv.org/html/2609.02538v1)、[作者代码](https://github.com/KIT-IAI-DRACOS/L2RPNGraphReprComparison)。固定 GNN-PPO 下比较多种图与 MLP，紧凑 busbar/substation 图优于一些更丰富的表示；没有支持“图越复杂越好”。本题已经采用动态 busbar 图，不因单次失败就堆加 PTDF/LODF 全连接边。该研究的 case14 拓扑任务不等同于我们的连续恢复辅助任务，不能迁移其数值优势。
6. **官方爬坡辅助工具**：[BaseAction.limit_curtail_storage 文档](https://grid2op.readthedocs.io/en/stable/user/action.html#grid2op.Action.BaseAction.limit_curtail_storage)。该工具按公开 `gen_margin_up/down` 限制储能与限发变化，并预留 MW margin；文档明确提醒对未来生产变化仍有不确定性。不能把环境的 `LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION` 开关与它混同：前者可用环境真实下一步信息。当前版本没有改这个环境开关，也未把该工具未经验证地叠加进训练。

## 困难、已处理的部分与尚未解决的部分

| 实际困难 | 来源启发 | 本轮处理与限制 |
|---|---|---|
| 绝大多数物理步骤没有恢复选择，策略梯度稀疏 | SMAAC 在高层决策之间累计奖励；RL2Grid 使用规则包装器 | V1 只在 offered 步优化 actor，critic 单独训练。但回报仍按物理步骤计算；不是 semi-Markov 算法复现 |
| 随机探索容易破坏已有好规则 | RL2Grid 将规则作为正式对照，而不是默认弱基线 | V2 从 V1 权重开始，用训练记录的概率作 logit 平移，使初始恢复概率约 0.95，再继续 PPO；不声称这是某篇论文的完整算法 |
| V0 学成始终恢复，虽胜 NN20，却不胜固定恢复 | 作者公开强规则对照的做法 | ALWAYS_RESTORE 与相同合法信息的 IMMEDIATE_COST 保留；不能把动作模块收益记为学习收益 |
| 一步预测通过、下一步爬坡不可行 | 官方多步 forecast API；单步 shielding 的范围限制 | 单独重放 V1 的训练失败，比较恢复/不恢复两条两步预测，并与特权真实分支分开记录；不改变正在运行的 V2 |
| 输入只有当前图与候选的两个全局预测摘要 | SMAAC 的 Q 网络明确接收动作，afterstate/分层建模区分控制效果 | 当前 GNN 没有逐机组候选动作通道；候选可由当前状态的 QP 推导，不等于信息不可辨识，但可能增加学习难度。这是代码层面的潜在改进点，不是已证实的失败原因；本轮不临时修改 V2 输入 |

RL2Grid 的 `env/heuristic.py` 已读到具体实现：它在规则步骤中累计折扣奖励用于 episode 记录，但 `step()` 返回变量 `reward` 是首个 agent 步的奖励。因此本轮没有把该包装器当作可以未经审计直接复制的 semi-Markov 回报实现；采用其规则对照与接口设计思路，并独立核对我们自己的时间与奖励语义。

## 已完成的训练失败复核

`ramp_diagnostic.json` 精确核对训练轨迹前 131 个物理步骤的观测、动作哈希。第 132 步恢复动作通过原一步筛选；第 133 步为 no-op。公开两步 forecast 对恢复分支报 `ImpossibleRedispatching`，对 hold 分支两步均不终止；复制真实环境的特权反事实同样只有恢复分支第二步失败。

这是一例**合法两步预测能辨识的局部失败**，不证明 hold 能活完整周，也不证明统一两步筛选的总体收益。实际未来分支不进入训练标签、策略输入或正在运行版本。额外物理步数 135，forecast 步数 4；首次接口调用因 `from_vect` 原地修改、返回 None 而失败，0 个物理步骤，原脚本和错误记录保留。

## 后续采用原则

先完成已冻结的 V2 训练与比较。如果新增两步风险屏蔽，规则与 GNN/MLP 必须享有相同预测信息、相同筛选和完整计算成本；单纯让模型获得额外预测而基线没有，是不公平比较。先证明屏蔽能修复具体风险，再研究学习能否超过屏蔽后的强固定规则。不能把增加 GAT、SAC、历史输入等模块本身当作贡献，也不以单次失败否定整个电网应用。

## 续跑后的具体诊断与单项改动

V2 两个模型各 16 个训练 episode 已完成。训练末端 GNN 的恢复概率范围约 0.976–0.988，MLP 约 0.912–0.945；在保存的训练候选上，确定性决策均为恢复。整周评价仍单独执行，不把训练概率当评价成绩。

`credit_diagnostic.json` 显示实际 offered 时点间隔 6–114 个物理步。现有物理步 GAE 的额外 lambda 衰减也发生在没有选择权的步骤上。在 gamma=0.995、lambda=0.95 下，96 步对应的 trace 权重为 0.00449；若恰好每六步一个决策、lambda 只跨决策应用，则示例权重为 0.272。该代数差异不是实际梯度贡献，也不证明旧实现不能借助准确 critic 学到延迟效应。

因此 GRID09 单独冻结两条同起点、同预算的 GNN 续训：普通物理步 GAE 和 decision-linked GAE。后者在下一状态没有 offered 决策时令该 lambda link 为 1，仍按每个物理步应用 gamma、保留全部奖励和原 rollout 截断。没有额外 forecast、输入通道、动作或安全筛选。这是受 SMAAC 决策时点建模启发的受控改编，不是完整 semi-Markov/SMAAC 复现；普通 PPO 续训控制额外训练量。

另外，`v2/gnn_critic_diagnostic.json` 对最后一轮训练记录作只读诊断：critic 的各周平均预测约 -14.8 至 -16.0，而实际折扣回报均值约 -3.2 至 -118.7；不同周的 realized-return explained variance 接近零或为负。该对照来自训练期间不断变化的随机策略，不能当作当前策略价值的无偏检验，更不能直接宣布失败根因，但给出了需要检查 critic 拟合的具体证据。

处理回报量级差异的成熟方法包括 [PopArt 原论文，NeurIPS 2016](https://papers.neurips.cc/paper_files/paper/2016/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)；[DI-engine 的 PPO 实现](https://github.com/opendilab/DI-engine/blob/main/ding/bonus/ppof.py) 也提供价值归一化选项。此处只完成来源复核，未安装 DI-engine、未将归一化叠加进正在运行的 GRID09。若后续采用，必须核对归一化与反归一化、bootstrap、价值输出保持、终止边界；不能简单裁掉终止惩罚来获得漂亮回报。
