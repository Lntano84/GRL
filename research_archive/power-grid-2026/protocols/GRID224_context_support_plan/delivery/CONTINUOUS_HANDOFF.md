# GRL持续推进交接：GRID224资格完成，下一步GRID225

用户授权持续自主研究和训练，无需逐轮确认，直到用户喊停。工作区D:/Codex/GRL研究；唯一Git集成在work/GRL-results-integration。grl当前聊天每30分钟跟进。不恢复旧自动化或归档聊天。

## 已完成（不要重复）

GRID222为旧教师路径上的成本质量折中：偏置回本保留39/55，查询比208少1.978%，代理净效用少0.172%，1正2负，原208仍主参考。源220模型未改。
GRID223只读完成610状态和22步长限制支持：16误删=11不同状态/7调用；Jan19_101call005占误删价值58.94%，前2调用73.81%。公开距离不支持简单“更远所以错”；低于正例最小b同时出现于9/16误删与63/111无利取消，不能据此做外层触发器。报告outputs/grid223_remaining_support/GRID223_report.md。
GRID224只做资格，无拟合。唯一可用于未来训练版本是outputs/grid224_context_support_plan/revision02；读取design.json、qualification.json、qualified_freeze.json。V1均分结构权重后乘absNET使辅助域占风险95.6-99.0%，尚未拟合即保留并修订。V2分别按域归一化结构权重乘absNET，再固定两域各0.5；标签/费不变，但明确改变辅助任务风险混合，不冒充原无条件合并效用目标。

## 下一步立即实施GRID225固定支持比较

原控制复用GRID222 ANCHORED（STOP_ONLY）与GRID208 GATE，不重训。
只新训MIXED_CONTEXT一个ANCHORED支线：11线性模型+11原STOP正例保护步长，共22个拟合/校准，一次600秒原窗口（含拟合和核验），不延长、不重跑完整项。f0_s1保持原KEEP，不因其辅助正例而改规则。
模型每池125输入/126参数、零初始化、logit=b+X@theta[:-1]+theta[-1]。目标调用work/grid219_prior_objective/run.py:objective，即加权平均逻辑损失+.5*theta全126维平方和（含截距），单位惩罚固定。scipy L-BFGS-B maxiter2000/gtol1e-6/ftol1e-12/maxls50；所有最终模型成功且grad_inf<1e-3才评分。所有种子0/1/2和4个外层保持。
只消费revision02/qualified_inputs/f{fold}_s{seed}.npz：X,b,ids,y,weight为混合拟合；calibration_X/calibration_b/calibration_y/calibration_ids为原STOP输入，和旧220完全相同。**仅原STOP正例**校准，不拿CONT正例加保护约束。用work/grid221_step_plan/run.py:choose_alpha求最大保护步长；b到新终点的方向，舍入仅为浮点可行性。没有按外层标签选参数。
11活跃池原STOP3338行90正例；辅助2927相关行1145正/1782非正。不是新增独立场景；添加暴露更多、域先验不同需报告。所有辅助来自原214实际NET停止之前继续状态，且内层公共b>0；正负全收，不用收益筛样本。

## 冻结、评分和核验

先写225显式源码/方案/日期/源模型/输入哈希并推送，再拟合一次。固定源：224 revision02设计、资格和哈希；220 public_held_inputs（纯公共125输入和b及ids/base_CONT）；222及208已发表控制、102 cases和任务配置用于最后连接核验。训练阶段不读控制结果标签。
全部新模型+步长封存推送后，才算12个池的4681个停止点新预测；f0_s1KEEP沿用。全部预测封存推送后再连接标签并按原保存序列重放。主比较MIXED_CONTEXT vs旧222 ANCHORED，次级208；完整控制不重新求解物理。可参考218 join的summarize和222/220重放器，但不要错误地让它重训或期望两个新臂；新旧所有情况保留。
检查目标/梯度/输入与原STOP保护约束、模型和数据未变、公共已观察候选、最多原同一批、best observed、保存动作/查询/增益/代理NET及日期/种子完整。所有改善都区分多查询与等预算，未另定预算实验就不称等预算优势。原外层反复暴露只称开发结果。

## 防止工程错误与边界

当前无后台runner，也无GRID225已开始拟合；先检查进程和当前docs/NEXT_STEPS再行动，若已有225完整项或进程则跟进不重复。旧224资格曾把选中STOP日期误作整个外层12日期，仅修正检查集合；原错误/6份部分输入/源码在outputs/grid224_context_support_plan_attempt01，6份输入修复前后完全相同，拟合0。另一个finalizer括号错误在解析时被捕获，副本与收据保留。两者不是方法失败。
归档路径必须读取实际新225源码A常量，不能猜名字；旧220实际为GRID220_anchored_logit。不要改、移动或覆盖旧冻结源/数据/结果。新阶段每个有意义单元用git_push.cmd提交推送，按AGENTS更新RESEARCH_STATE/DECISIONS/EXPERIMENT_LOG/NEXT和归档首页，核对协作副本与远端SHA。
权限只申请D:/Codex/GRL研究写和现有Git网络，自动拒绝不绕过，完成可做的只读部分并具体通知阻碍。最终8不生成/不读标签，不扩物理/电网、论文、新FRAME或安装依赖救结果。不设全赢/5%新门槛，不按中间结果扫惩罚、阈值或挑种子。阶段发现简短汇报；没有实质新发现则安静。用户喊停则停止新增工作并暂停grl。
