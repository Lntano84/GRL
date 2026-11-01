"""Conclude static target/label qualification without generating or changing targets."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid295_target_label_contract';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID295_target_label_contract';C=R/'论文协作资料/FRAME_目标标签资格_GRID295'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'complete.json').exists();D=read(O/'design.json');assert len(D['documents'])==len(read(O/'static_contract_excerpts.json'))==8
definitions=[
 ('286_existing_contract',[9,17,23,25,41],'Existing208p*mu-charge and261absNET riskNLL are already query-cost aware. Reuse prior verdict; no renamed classifier or repeated synthetic qualification.','Does not identify monetary/physical operating-cost or delayed full-week loss.'),
 ('query_environment',[1,19,34,35,36,37],'Continuation advances a saved local query-batch position and pays(gain-queryfee)/gain_scale; it does not advance physical power-system time.','Finitequery-horizon gamma1 is not a whole-week grid-control return.'),
 ('teacher_case_schema',[20,28,37,38,39,45,46,47,49,51,54,55,72,73,82],'Cases persiststates/gains/queries/chosen and identity/date roles; gain is best-admissible reward increment. Queryfee is a train-date proxy, explicitly not seconds or currency.','This constructor does not persist per-alternative joint operating-cost/redispatch/storage or matched physical-time counterfactual trajectories. Sourcecall may have additional data not certified by this eight-document scope.'),
 ('relational_training_schema',[19,21,26,29,34,35,36,40,41,42,43,84],'Public six-field inputs and targets remain original queryNET; training targets/sampling_probability, validation target, and role metadata are explicit.','No new operating-cost paired target in this checked learning schema; numerical arrays/pickles were not opened.'),
 ('native_reward',[15,21,22,23,25,26],'MaxRhoReward returns2-rho.max() on valid transitions, reward_min=-1 on error/illegal/ambiguous.','This is a security-loading feedback, not the separate operating-cost auxiliary metric.'),
 ('base_agent',[191,194,200,232,234,235,237,238,241,254],'LJNAgentTopoNN adds topology to act, then may replace/extend act via optim.get_act if forecast remains unsafe/exception; returns the full action.','Static branch existence explains why proposed topology and delivered-vector layers differ, not proof of every actual optimizer invocation or complete cost causality.'),
 ('base_factory',[24,35,39,42],'make_agent_topoNN instantiates LJNAgentTopoNN with original topk20/PPO asset; this is originalbase control, distinct from FACTOR local query-stop network.','Factory path is not a new policy or tensor equality check; no environment/Gym/NN construction was executed.'),
 ('operating_cost_ledger',[1,12,17,20,21,26,27,28,31,32,33,37],'Original passive operating cost is marginal price times stored losses+absolute redispatch+curtailment change+storage throughput; physical curtailment level separate.','A realised single delivered-action ledger is not an alternative-action/cancel-versus-continue long-horizon label.')]
matrix=[]
for ident,lines,known,limit in definitions:
 m=next(m for m in D['documents'] if m['id']==ident);matrix.append(dict(id=ident,source=m['source'],sha256=m['sha256'],line_refs=lines,established=known,qualification_limit=limit))
save(O/'target_label_matrix.json',matrix)
decision=dict(status='NO_JUSTIFIED_NEW_OPERATING_COST_FIT_FROM_CHECKED_SCHEMA',original_queryNET_valid_and_unchanged=True,renamed_cost_classifier_closed=True,operating_cost_target_distinct=True,immediate_cost_not_equivalent_to_fullweek=True,checked_schema_lacks_alternative_joint_or_long_horizon_operating_cost_targets=True,not_global_archive_absence_claim=True,no_target_change_executed=True,new_labels=0,new_fits=0,new_predictions=0,new_physics=0,next='One bounded existing-source recovery/collection proposal only: exact own-training provenance/counts/controller-state availability, reuse versus missing-label judgment and full operation/resource/validation contract. No automatic labels, fitting or physical generation.')
save(O/'decision.json',decision)
lines=['# GRID295：训练目标与已有标签资格','',
 '裁决：不按现有标签直接重训“整周成本模型”。现有queryNET已有查询代价，并非无成本分类；但它的收益是负载率安全反馈的改善，不是整周运行费用。检查的学习格式未提供补查/停止两种完整控制动作的运行费用或长期配对标签。本阶段只查静态来源，未改目标、生成标签或拟合。','',
 '## 已核对的精确目标','',
 '- 有效模拟反馈MaxRhoReward=2−max(rho)；错误/非法/歧义时为−1。每个已付费候选得到该安全反馈，教师局部轨迹的gain是当前最佳合法反馈的增量。',
 '- 原停止/继续目标为NET=(gain−cost_per_query×next_queries)/gain_scale；查询代理价来自自身训练日期的既定统计规则，源码明确其不是实际秒数或货币。',
 '- LocalSearchEnv的position前进一步，是在同一电网时刻再买一批最多64候选的反馈，不是电网向下一个物理时刻推进。有限查询链和gamma=1不能称为完整周的控制收益。',
 '- 208已预测回本概率×正gain再减已知查询费；261继续用absNET加权NLL、原父偏置和经验保护。286关于“已经成本敏感，不能换名重训”的裁决直接复用，没有重复其拟合、梯度或合成试验。','',
 '这些原目标并没有被本阶段判为错误：它们回答的是安全反馈收益—查询代价问题。现有100.5%FULL整周费用标准是另一个交付约束，不能把局部NET改善自动当满足这一约束。也不能因为费用单位不同，就不披露地重设gain、费用或奖励来救结果。','',
 '## 拓扑候选不是全部交付动作','',
 'make_agent_topoNN静态绑定LJNAgentTopoNN。其act先合并重连/拓扑候选，再模拟安全状态；若仍不安全或异常，会调用optim.get_act处理连续控制，最终返回完整act。这是原LJN基底控制器的组合层，与FACTOR局部查询停止网络不是同一个“NN”。',
 '因此294的“搜索chosen_combined_hash不等于最终delivered action_hash”不应直接叫接口bug，也不能把最终再调度费用全归因于一个选候选或STOP决定。静态代码证明存在这条组合路径，但不凭源码单独证明它在每条实际分岔都执行过；本阶段未新挂hook/前向/运行控制器。','',
 '## 现有学习文件是否有合格标签','',
 '| 既有记录 | 静态格式已有内容 | 尚不能据此得到的标签 |','|---|---|---|',
 '| GRID102 cases | states、gains、queries、chosen、sample/date/partition/physical group | 两种补查决定的完整联合动作运行费用或不同物理后续轨迹 |',
 '| GRID113 public | context_action/context_feedback/next_action/anchor_action/global41/known_query_cost | 未揭示候选的未来费用或连续优化器响应 |',
 '| GRID113 targets | 原queryNET target、训练sampling_probability，原日期/样本角色metadata | 整周成本差、长期因果收益或新增货币成本标签 |',
 '| 实际物理步账本 | 已交付一个动作后观测到的原运行费用及其分项 | 另一种动作/停止决定若被执行会怎样的反事实标签 |','',
 '这里只是检查的8份实际生产源码/schema的结论，未反序列化pickle、读取NPZ数值或搜索全盘历史数据；不能扩大成“整个档案绝无可恢复信息”。源call及其他已发布文件是否保留额外可复用字段，仍须按精确来源盘点。也不能把已暴露March外层的结果贴回训练行，冒充自身独立长期标签。',
 '294还表明22/42条首分岔当步运行费更低、最终整周费更高。所以即使将来补齐某个单步候选费用，也不能直接声称它足以解决整周费用；标签必须说明联合动作、连续响应和时间范围，验证仍要保留完成/成本/查询/控制时间及不利结果。','',
 '## 8份静态证据与边界','',
 '源码和8份固定文档先发布d4244c59，然后才做文本资格；未执行其中任何prepare、Gym、simulate、torch、优化器或模型代码。读取的第三方LJN源码保留原版权/许可证头，没有修改原控制器。','',
 '| 来源 | 可确立内容 | 资格限制 |','|---|---|---|']
for r in matrix:lines.append(f"| {r['id']} | {r['established']} | {r['qualification_limit']} |")
lines+=['','## 后续只准备一份可审阅数据方案','',
 '先限定原自身训练范围、沿既有source/manifest查是否有不需新模拟即可恢复的字段、精确物理调用/日期/角色与原控制器状态是否足够。若可恢复，先写清目标/单位/时域、隔离和全部操作；若不可恢复，明确缺哪些字段及采集成本，不盲训一个代理目标来代替。',
 '下一步只能有一份有限的“既有信息恢复/缺失标签采集与验证方案”，不是继续追加模型资格串。方案必须给具体计数、资源估计、完整操作和停止条件，数据/目标变更显式披露；新的物理或标签生成尚无自动启动授权，不能把用户“继续”或原192条4小时批准当额外预算。正式采集前必须将方案做成可审阅结果再判断必要范围决定。',
 '当前208/261、原NET标签/费用/奖励/控制器/排序/别名/初始锁站/128/64/停滞2/.9保持。残差取消/森林/阈值风险特征seed扫描关闭，主锚208/261和230历史保留。不扩日期/FRAME/电网/论文/最终8，不安装依赖。历史190拟合与6续训8285updates、604.703实际/678.156授权另列。',
 '同一暴露开发资料、相关前缀/折/seed复用与own12父/内部cal4祖先限制不变，不称全链OOF/盲测。没有新模型成绩或整周提速结论。','',
 '文件：source_documents保留8份静态原文，static_contract_excerpts.json逐行摘录，target_label_matrix.json精确来源/hash/行号，review/decision/complete/NEXT_PROTOCOL_PLAN；D协作副本FRAME_目标标签资格_GRID295。']
(O/'GRID295_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
plan=dict(id='GRID296_BOUNDED_EXISTING_INFORMATION_AND_DATA_PROPOSAL',status='PROPOSAL_PREPARATION_ONLY_NOT_EXECUTED',objective='Produce one concrete reviewable reuse-or-collect plan for a genuinely distinct joint-action operating-cost/time-horizon question; no more renamed cost fits or open-ended engineering stages.',scope='At most10 directly referenced source/schema/JSONmanifest documents resolved via existing constants/archives; no NPZ/model arrays or pickle/trajectory reading. Only own-training member/date/physical-call/role counts and stored-field/controller-state availability; no new gain/cost target values computed.',deliver=['Exact existing source/member manifest and source-context/controller-state adequacy; record missing proof rather than guessing an artifact alias','Can already-published own-training data supply valid matched joint-action costs/time-horizon labels without forecasts? Preserve originalNET, identify a separate target only if justified','If recovery is feasible: exact finite operation count and fixed source/input/label publication barriers, train-role isolation and all original seeds/pools/KEEP; no execution in this proposal stage','If recovery is impossible: concrete missing fields, smallest finite protocol options with actual counts/resource estimates from existing runtimes and validation contract; no new pilot forecasts to estimate budget','Explicit changes to target/units/horizon/exposure and scientific limitations; do not fit whole-week returns to individual decisions by assumption','One recommended route and stopping condition; if no identifiable/feasible target, close and request only a necessary concrete scope decision after the proposal exists'],forbidden=['Any labels/physical observations/NNhead predictions or fitting','UseMarch outer failures as own training targets or choose samples by outcome','Automatic extra budget/date/FRAME/final8 or changing existingcontroller','Hypothetical policy returns, safety promises from per-step cost or hidden larger search'],operation_caps=dict(new_fits=0,new_labels=0,new_predictions=0,new_physics=0,max_documents=10),stop='One reviewable plan or explicit infeasibility decision, not another indefinite qualification chain. New physics is a separate unapproved scope; any future learning component <=22models/scalars and one600s window, no extensions/blind sweeps.')
save(O/'NEXT_PROTOCOL_PLAN.json',plan);save(O/'review.json',dict(passed=True,source_published=read(O/'qualification_started.json')['source_published'],static_documents=8,original286_result_reused=True,new_labels=0,new_fits=0,new_predictions=0,new_physics=0,model_or_trajectory_arrays_read=0,claim_scope='Checked exact learning schema, not global archive absence or full cost causal proof.'));save(O/'complete.json',dict(passed=True,qualification_complete=True,new_fit_qualified=False,original_queryNET_unchanged=True,next296_executed=False,new_fits=0,new_labels=0,new_predictions=0,new_physics=0))
msg='GRID295源d4244c59后8静态来源资格完成，0新数组/标签/拟合/预测/物理。MaxRhoReward=2-maxrho；原NET=(最佳合法安全反馈增量-查询代理费)/scale，LocalSearchEnv推进的是查询批次不是物理时间，query价原定义非秒/货币；286已有成本敏感合同直接复用不换名重训。LJN基底act可在拓扑后接连续optim，候选不等完整交付。102/113被查学习格式未存联合替代动作费用/长期配对标签，实际单动作账本不等反事实；不扩大为全档案无信息。22/42首分岔即时费低却周费高，不能直接训即时成本头救整周。下一仅一份有限既有信息恢复/采集验证可审阅方案，具体计数资源/角色/全部操作，未授权标签生成或额外物理。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-11-01 — GRID295目标/既有标签可识别性\n\n'+msg+'\n';p.write_text('# 当前：GRID295未资格新成本拟合，先形成数据方案\n\n'+msg+'\n\n以下保留历史。\n\n'+t if name=='RESEARCH_STATE.md' else t+entry,encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前 — 一份有限数据/目标方案，不盲训\n\n1. 294保存成本/共同前缀读出及295八静态来源资格完成，不重做轨迹/数组/旧286合同试验或模型预测。读outputs/grid295_target_label_contract/GRID295_report.md和target_label_matrix/decision/complete。\n2. 下一296按NEXT_PROTOCOL_PLAN，只用最多10份直接引用source/schema/JSONmanifest做精确自身训练成员/日期/调用/角色和字段/控制器状态可恢复性盘点，形成一份具体可审阅恢复或采集/验证方案；不读取NPZ/pickle/轨迹、计算新标签、拟合或物理预测。\n3. 方案给确切操作计数、资源估计、目标/单位/时域和验证边界，不以即时费或单动作真实账本冒充长期配对标签。不可恢复则明示数据缺口，必要新物理范围仅在可审阅方案后决定，不自动新增。\n4. 保持原208/261/NET费用奖励控制器与全部池seedKEEP/反例；取消/森林/阈值风险seed修补关闭，最终8/新日期FRAME电网论文依赖不动；291186/192预算到限不补跑。\n5. 按AGENTS完成本轮Git/D协作/清单/具体blob/远端核对，同步交接和自动跟进。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID295明确queryNET与运行费用标签边界\n\n[报告](protocols/GRID295_target_label_contract/results/GRID295_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
hand='# GRID295完成：没有可直接启动的新成本拟合\n\n'+msg+'\n\n先读integration AGENTS/docs状态/NEXT，本目录报告/target_label_matrix/review/decision/complete/NEXT_PROTOCOL_PLAN。当前8静态来源已读完，不重开旧286或294审计，不加载数组或执行源代码。下一296只准备一份精确数据恢复/采集验证方案（最多10直接引用静态source/schema/JSONmanifest），不生成标签/预测/训练/物理，不无限追加资格。原192条批准已用尽，不将用户继续解释额外物理范围。若存在新目标/标签/时域，要明确披露和全部成本；若缺必要信息给明确数据缺口裁决，不按March结果改旧NET或盲训。\n'
(O/'CONTINUOUS_HANDOFF.md').write_text(hand,encoding='utf8');S=A/'results';S.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
names=['GRID295_report.md','target_label_matrix.json','static_contract_excerpts.json','decision.json','review.json','complete.json','NEXT_PROTOCOL_PLAN.json','CONTINUOUS_HANDOFF.md','design.json','source_freeze.json']
for name in names:shutil.copy2(O/name,S/name);shutil.copy2(O/name,C/name);assert sha(O/name)==sha(S/name)==sha(C/name)
shutil.copy2(W/'finalize.py',S/'finalize.py');shutil.copytree(O/'source_documents',C/'source_documents');save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()});save(C/'delivery_hashes.json',{p.relative_to(C).as_posix():sha(p) for p in C.rglob('*') if p.is_file()});print('GRID295_TARGET_DATA_DECISION_READY')
