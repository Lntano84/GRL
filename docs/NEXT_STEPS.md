# Next steps — ACM e-Energy 2027 provisional

1. GRID94固定可预测性诊断完成：简单模型尚不能可靠取消无效补查；没有部署。下一步先汇报无效标签与长期有害标签的区别。如继续，优先明确有限分支价值采集的状态、完整控制器记忆、后续策略和风险/成本目标；不直接加模型、扫阈值或做整周标签广播。
2. 整理稿件时以前缀初始化、反馈控制深度、结构引导补查位置解释现有方法。默认q_min256冗余已验证；旧冻结源码与所有不利结果保留。不要把搜索一步奖励改善写成整周成本必然改善。
3. 第二电网与比例预算是后续待确认阶段：先资格检查动作库、别名、预测与控制底座；参数只由源网确定，不在目标测试集重调。本轮未自动进入。
4. CCMA仍是领域学习基线候选，作者实现/权重尚未定位；不贸然训练或按会议名称替换META/ExpertOp，旧比较保留。
5. 机制定型及必要验证后编写e-Energy稿件；不追加普通对照堆数量，也不自动启动学习/智能体探索。

当前材料：research_archive/power-grid-2026/protocols/GRID93_trigger/complete/GRID93_report.md；前一轮三问见research_archive/power-grid-2026/paper/GRID92_completion_analysis_20261020/GRID92_report.md。
此前计划保存在docs/history/NEXT_STEPS_before_grid92_20261020.md。

## GRID93_TRIGGER_complete

单条站内支持否决规则完成：去掉最佳后以成对奖励胜率0.5作负证据门控，无拟合或扫描。185个FULL和179个实际FRAME调用筛查，19场景选择不变复用、6条新闭环。判读REJECT_STATION_SUPPORT_TRIGGER，原达标而新规则不达标的场景1个。一步搜索价值与长期动作采用分开，不加第二条规则救结果。所有混合结果保留，无新模型训练或同批提速主张。

先向用户汇报本条触发条件是否成立；当前不自动搜索新阈值或改动作采用目标。

## FRAME_TRIGGER_READING_20261020

原文方法与结论已核对，材料paper/frame_trigger_reading_20261020。新触发设计先区分任务风险、候选代理奖励及计算价值；当前没有启动下一条规则试验。
