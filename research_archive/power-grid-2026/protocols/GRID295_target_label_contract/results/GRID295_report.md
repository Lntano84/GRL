# GRID295：训练目标与已有标签资格

裁决：不按现有标签直接重训“整周成本模型”。现有queryNET已有查询代价，并非无成本分类；但它的收益是负载率安全反馈的改善，不是整周运行费用。检查的学习格式未提供补查/停止两种完整控制动作的运行费用或长期配对标签。本阶段只查静态来源，未改目标、生成标签或拟合。

## 已核对的精确目标

- 有效模拟反馈MaxRhoReward=2−max(rho)；错误/非法/歧义时为−1。每个已付费候选得到该安全反馈，教师局部轨迹的gain是当前最佳合法反馈的增量。
- 原停止/继续目标为NET=(gain−cost_per_query×next_queries)/gain_scale；查询代理价来自自身训练日期的既定统计规则，源码明确其不是实际秒数或货币。
- LocalSearchEnv的position前进一步，是在同一电网时刻再买一批最多64候选的反馈，不是电网向下一个物理时刻推进。有限查询链和gamma=1不能称为完整周的控制收益。
- 208已预测回本概率×正gain再减已知查询费；261继续用absNET加权NLL、原父偏置和经验保护。286关于“已经成本敏感，不能换名重训”的裁决直接复用，没有重复其拟合、梯度或合成试验。

这些原目标并没有被本阶段判为错误：它们回答的是安全反馈收益—查询代价问题。现有100.5%FULL整周费用标准是另一个交付约束，不能把局部NET改善自动当满足这一约束。也不能因为费用单位不同，就不披露地重设gain、费用或奖励来救结果。

## 拓扑候选不是全部交付动作

make_agent_topoNN静态绑定LJNAgentTopoNN。其act先合并重连/拓扑候选，再模拟安全状态；若仍不安全或异常，会调用optim.get_act处理连续控制，最终返回完整act。这是原LJN基底控制器的组合层，与FACTOR局部查询停止网络不是同一个“NN”。
因此294的“搜索chosen_combined_hash不等于最终delivered action_hash”不应直接叫接口bug，也不能把最终再调度费用全归因于一个选候选或STOP决定。静态代码证明存在这条组合路径，但不凭源码单独证明它在每条实际分岔都执行过；本阶段未新挂hook/前向/运行控制器。

## 现有学习文件是否有合格标签

| 既有记录 | 静态格式已有内容 | 尚不能据此得到的标签 |
|---|---|---|
| GRID102 cases | states、gains、queries、chosen、sample/date/partition/physical group | 两种补查决定的完整联合动作运行费用或不同物理后续轨迹 |
| GRID113 public | context_action/context_feedback/next_action/anchor_action/global41/known_query_cost | 未揭示候选的未来费用或连续优化器响应 |
| GRID113 targets | 原queryNET target、训练sampling_probability，原日期/样本角色metadata | 整周成本差、长期因果收益或新增货币成本标签 |
| 实际物理步账本 | 已交付一个动作后观测到的原运行费用及其分项 | 另一种动作/停止决定若被执行会怎样的反事实标签 |

这里只是检查的8份实际生产源码/schema的结论，未反序列化pickle、读取NPZ数值或搜索全盘历史数据；不能扩大成“整个档案绝无可恢复信息”。源call及其他已发布文件是否保留额外可复用字段，仍须按精确来源盘点。也不能把已暴露March外层的结果贴回训练行，冒充自身独立长期标签。
294还表明22/42条首分岔当步运行费更低、最终整周费更高。所以即使将来补齐某个单步候选费用，也不能直接声称它足以解决整周费用；标签必须说明联合动作、连续响应和时间范围，验证仍要保留完成/成本/查询/控制时间及不利结果。

## 8份静态证据与边界

源码和8份固定文档先发布d4244c59，然后才做文本资格；未执行其中任何prepare、Gym、simulate、torch、优化器或模型代码。读取的第三方LJN源码保留原版权/许可证头，没有修改原控制器。

| 来源 | 可确立内容 | 资格限制 |
|---|---|---|
| 286_existing_contract | Existing208p*mu-charge and261absNET riskNLL are already query-cost aware. Reuse prior verdict; no renamed classifier or repeated synthetic qualification. | Does not identify monetary/physical operating-cost or delayed full-week loss. |
| query_environment | Continuation advances a saved local query-batch position and pays(gain-queryfee)/gain_scale; it does not advance physical power-system time. | Finitequery-horizon gamma1 is not a whole-week grid-control return. |
| teacher_case_schema | Cases persiststates/gains/queries/chosen and identity/date roles; gain is best-admissible reward increment. Queryfee is a train-date proxy, explicitly not seconds or currency. | This constructor does not persist per-alternative joint operating-cost/redispatch/storage or matched physical-time counterfactual trajectories. Sourcecall may have additional data not certified by this eight-document scope. |
| relational_training_schema | Public six-field inputs and targets remain original queryNET; training targets/sampling_probability, validation target, and role metadata are explicit. | No new operating-cost paired target in this checked learning schema; numerical arrays/pickles were not opened. |
| native_reward | MaxRhoReward returns2-rho.max() on valid transitions, reward_min=-1 on error/illegal/ambiguous. | This is a security-loading feedback, not the separate operating-cost auxiliary metric. |
| base_agent | LJNAgentTopoNN adds topology to act, then may replace/extend act via optim.get_act if forecast remains unsafe/exception; returns the full action. | Static branch existence explains why proposed topology and delivered-vector layers differ, not proof of every actual optimizer invocation or complete cost causality. |
| base_factory | make_agent_topoNN instantiates LJNAgentTopoNN with original topk20/PPO asset; this is originalbase control, distinct from FACTOR local query-stop network. | Factory path is not a new policy or tensor equality check; no environment/Gym/NN construction was executed. |
| operating_cost_ledger | Original passive operating cost is marginal price times stored losses+absolute redispatch+curtailment change+storage throughput; physical curtailment level separate. | A realised single delivered-action ledger is not an alternative-action/cancel-versus-continue long-horizon label. |

## 后续只准备一份可审阅数据方案

先限定原自身训练范围、沿既有source/manifest查是否有不需新模拟即可恢复的字段、精确物理调用/日期/角色与原控制器状态是否足够。若可恢复，先写清目标/单位/时域、隔离和全部操作；若不可恢复，明确缺哪些字段及采集成本，不盲训一个代理目标来代替。
下一步只能有一份有限的“既有信息恢复/缺失标签采集与验证方案”，不是继续追加模型资格串。方案必须给具体计数、资源估计、完整操作和停止条件，数据/目标变更显式披露；新的物理或标签生成尚无自动启动授权，不能把用户“继续”或原192条4小时批准当额外预算。正式采集前必须将方案做成可审阅结果再判断必要范围决定。
当前208/261、原NET标签/费用/奖励/控制器/排序/别名/初始锁站/128/64/停滞2/.9保持。残差取消/森林/阈值风险特征seed扫描关闭，主锚208/261和230历史保留。不扩日期/FRAME/电网/论文/最终8，不安装依赖。历史190拟合与6续训8285updates、604.703实际/678.156授权另列。
同一暴露开发资料、相关前缀/折/seed复用与own12父/内部cal4祖先限制不变，不称全链OOF/盲测。没有新模型成绩或整周提速结论。

文件：source_documents保留8份静态原文，static_contract_excerpts.json逐行摘录，target_label_matrix.json精确来源/hash/行号，review/decision/complete/NEXT_PROTOCOL_PLAN；D协作副本FRAME_目标标签资格_GRID295。
