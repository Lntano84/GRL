"""Conclude one data route proposal; nothing here executes the proposed collection."""
from pathlib import Path
import hashlib,json,shutil
R=Path('D:/Codex/GRL研究'); W=Path(__file__).parent; O=R/'outputs/grid296_data_route_proposal'; I=R/'work/GRL-results-integration'; A=I/'research_archive/power-grid-2026/protocols/GRID296_data_route_proposal'; C=R/'论文协作资料/FRAME_后续数据方案_GRID296'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'complete.json').exists(); assert read(O/'inspection_complete.json')['passed']
counts=read(O/'membership_counts.json'); pools=read(O/'fixed_pool_roles_read.json')['data']; assert len(pools)==12
members=read(O/'source_documents/own_members.json')['samples']; own=[m for m in members if m['partition']=='train']
matrix=[]
for p in pools:
 fit=set(p['fit_dates']); cal=set(p['calibration_dates']); outer=set(p['outer_dates']); assert len(fit)==8 and len(cal)==len(outer)==4 and not fit&cal and not(fit|cal)&outer
 assert fit|cal|outer==set(counts['own_dates']); rows=[m for m in own if m['calendar_group'] in fit|cal]
 matrix.append(dict(pool=p['pool'],KEEP=p['pool']=='f0_s1',fit_dates=sorted(fit),cal_dates=sorted(cal),outer_dates_excluded=sorted(outer),own12_teacher_members=len(rows),own12_calls=len({m['physical_state_group'] for m in rows}),fit8_calls=len({m['physical_state_group'] for m in rows if m['calendar_group'] in fit}),cal4_calls=len({m['physical_state_group'] for m in rows if m['calendar_group'] in cal})))
save(O/'pool_role_counts.json',matrix)
fields=[
 dict(source='grid89 adapter reveal',saved=['pool_id','rho','reward','done','flags','strict_admissible','forecast_s','phase'],missing=['full candidate successor observation','per-alternative original operating-cost ledger','post-LJN continuous optimizer joint delivered action','paired physical-time suffix'],lines='adapter.py:29-31,39-40',verdict='Security reward/candidate time only; cannot reconstruct joint monetary-equivalent costs from rho/reward.'),
 dict(source='grid55 worker',saved=['metadata library/checkpoint hashes, dt and generator price','call observation vector and base_action vector','delivered observation/action vectors and one realised step ledger'],missing=['full environment/backend/chronics/RNG snapshot','agent and continuous-optimizer internal state snapshot','other intervention closed-loop physical suffix'],lines='worker.py:52-56,62-64,101-113,120-136,149',verdict='Vectors establish provenance and observed path; not a complete restorable counterfactual simulator/controller checkpoint.'),
 dict(source='grid102/113',saved=['own date/scenario/call membership','query-prefix gains/queries/chosen','six public input fields','original queryNET target'],missing=['matched STOP versus one-more-batch joint-action weekly-cost labels'],lines='prepare102:23-47;prepare113:21-43,84',verdict='Reuse member identities and original NET; do not manufacture a new cost label from its target.'),
 dict(source='fixed roles',saved=['12 pool fit8/cal4/outer4 memberships'],missing=['independent upstream retraining/calibration isolation'],lines='fixed_pool_roles.json role fields only',verdict='Own12 date exclusion can be enforced; upstream own12 and historical exposure remain.')]
save(O/'stored_field_matrix.json',fields)
# Exact static proposal. All arithmetic below is resource arithmetic, not a target or return.
plan=dict(status='REVIEWABLE_SCOPE_ONLY_NOT_IMPLEMENTED_OR_AUTHORIZED',recommendation='Freeze208/261 and stop direct cost fitting. If cost research is funded, collect one matched whole-suffix intervention dataset; do not run a single-step cost-head substitute.',selection='Each of12 original pools, each of its fixed own12 dates, scenario100/envseed0. On the unchanged261 closed-loop prefix take the first nativeNN STOP with a nonempty original next batch. Public eligibility only, never cost/outcome selection. If absent, retain NO_ELIGIBLE_EVENT and do not substitute a date/call.',pairs=144,episodes=288,fit_role_pairs=96,cal_role_pairs=48,KEEP_pairs=12,active_nonKEEP_pairs=132,maximum_physical_steps=288*2017,maximum_total_forecasts=9000000,total_window_s=28800,per_episode_cap_s=1200,serial=True,repeats_per_arm=1,arms=['STOP at the selected nativeSTOP event','CONT exactly one original next batch <=64 at the same event'],pre_event='Both independent episodes follow identical frozen261 controller up to the public trigger. Require exact common before-observation, base-action, ordered visible IDs/alias/station and original model/config hashes. A mismatch invalidates that pair; no restore from incomplete saved vectors.',post_event='Assemble the original full LJN joint action in each arm including continuous optimizer response; after this one intervention revert to the identical original261 policy and run to original week end or failure. No repeated forced interventions.',target='Separate diagnostic physical-time target C_CONT-C_STOP over matched suffix, using unchanged original operating-cost units/ledger; completion, illegal/ambiguous/done and safety separate. Negative means cheaper CONT only when same full horizon is valid. Do not collapse early failures into cheap cost or substitute for original queryNET.',future_inputs='Only fields available before reveal; no suffix cost/reward/optimizer response enters model features.',timing='Record full feature/NN/head/residual/optimizer/forecast ctl.propose and loading separately; one repeat is label collection, not robust latency/P95 improvement confirmation.',isolation='Per-pool fit8 only may fit; cal4 only predeclared calibration/support; own outer4 prohibited in this collection. KEEP included for records but never newly fitted. Original validation4/final8 excluded. Source own12 parent ancestry still prevents full-chain independence.',publication=['Implementation+all model/input/role/selection references and exact scope authorization before first simulator call','All matched source receipts/observations/delivered-action identities before deriving the separate cost target','Label/data freeze before any uniquely justified model fit','All models before scoring; all predictions before held result join'],stops=['One28800s total manager window including loading/startup/failed work,1200s per episode,9000000 forecasts and580896 physical steps; stop at first exceeded cap, preserve every partial/missing pair, no automatic resume or replacement.','No common-prefix integrity -> mark invalid and stop that pair; cannot repair by changing control/numeric settings.','No valid matched full-horizon labels, or insufficient support across original date roles -> close cost-learning route; do not choose a successful subset.','No labels/outcome from heldMarch are added to ownfit. No fitting included in collection authorization.'],new_fits=0,new_labels_this_stage=0,new_physics_this_stage=0,future_training='Not prequalified: at most11 models+11 scalars <=22/one600s, only if this distinct target is identifiable and a unique model/negative-control/loss/stop contract is separately frozen. No model form chosen from current exposed cost failures.',validation='A new learned intervention would later need all original4 development dates x12 pools x2 arms x2 reverse repeats=192 physical episodes, its own fixed budget (old14400s did not complete). Not bundled or authorized by data collection, not final8/blind validation.')
assert plan['maximum_physical_steps']==580896
save(O/'data_collection_proposal.json',plan)
runtime=dict(source='Frozen GRID291_report.md: original14400s window,186 accepted,345066 physical steps,1106184 forecasts; mixed dates include partial-horizon May and setup/recovery.',envelope_s_per_accepted=14400/186,proposed_288_episode_simple_s=288*14400/186,proposed_288_episode_simple_hours=288*14400/186/3600,step_normalized_simple_s=580896*14400/345066,step_normalized_simple_hours=580896*14400/345066/3600,requested_cap_s=28800,requested_cap_hours=8,previous_scope_exhausted=True,forecast_cap_is_resource_choice_not_expected_usage=True,qualification='Two coarse historical extrapolations only, not guarantees or resource passed. NativeSTOP intervention effects and training dates/controller calls differ; retaining partial results is mandatory. No pilot simulation to estimate runtime.')
save(O/'resource_estimates.json',runtime)
decision=dict(status='EXISTING_CHECKED_DATA_CANNOT_IDENTIFY_MATCHED_JOINT_SUFFIX_COST_WITHOUT_NEW_PHYSICS',source_scope='Ten fixed static sources/JSON metadata only; not a global absence claim.',new_cost_fit_qualified=False,original_NET_unchanged=True,one_recommendation='No further proxy cost-head training or static qualification chain. Review the fixed144pair/288episode/8h collection only if the user elects new physical budget; otherwise archive and close this route.',new_collection_authorized=False,next_action='Only implementation and source/model binding of this exact reviewable proposal is reversible preparation; first simulator call requires separate scope approval. No new experimental phase to keep automation busy.',new_fits=0,new_predictions=0,new_labels=0,new_physics=0)
save(O/'decision.json',decision)
report='''# GRID296：先补对问题的数据，再决定是否学习

结论：这条主线暂不训练新成本头。固定10份来源表明，已有记录足以追踪自身训练成员和原安全反馈，却不能在不新模拟的情况下，恢复“停止/再买一批”两种完整控制动作的配对后续费用。建议保留208/261，若继续研究整周费用，只做下述一次有限配对采集；否则关闭这条新成本学习路线。

## 已有数据与缺口

原2968个教师成员中2378是自身训练成员，来自16日期、32场景、222个物理调用；3582条训练前缀仍是这222调用的相关展开，不是3582次独立实验。12池固定fit8/cal4/outer4日期逐池列在pool_role_counts.json；本次只数成员/角色，不读取原收益、向量、模型或轨迹。旧257的2859候选及2725活跃支持直接复用，不重新筛选或拟合。

原collector对每个候选模拟后只保存rho、reward、合法/结束标志和耗时；没有保存候选完整后继观测、原运行费用账本或最终连续优化后联合动作。worker另存调用前observation和base_action，以及实际交付的一条动作/观测/费用轨迹。这些向量不是环境后端、时序/RNG、原agent及连续优化器全部内部状态的恢复快照，也没有另一条干预后缀。故不能从已付费候选reward反推联合动作长期费用，也不能把单条已执行账本当配对标签。结论只限检查的源码格式，不声称全盘没有其他资料。

## 唯一建议：有限的完整后缀配对数据

拟用全部12原pool（含KEEP），各自原own12日期、scenario100、环境seed0，共144配对/288条独立从周初运行的episode，fit角色96对/cal角色48对；KEEP12对只保留记录，不训练。每条共同前缀使用冻结261，按公开规则取首次原nativeNN STOP且有原待买批次的调用；两臂分别立即停和仅多买原下一批至多64候选，之后恢复同一原控制器。无合格调用记缺失，不换日期或调用。成员规则不使用March结果或费用好坏。

必须核对干预前观测、base动作、候选顺序/别名/锁站和原模型hash相同；分别记录最终联合动作（含原连续优化器），从同一物理时刻继续到原周末或失败。新诊断目标为后缀C_CONT−C_STOP，保留原费用单位和全部分项、完成/安全/异常；未同完整时域的失败不能当低费用赢家。它不同于原queryNET，也不等于一般最优长期价值，原NET/奖励/费用不修改。只采每日期首个合格事件，不能外推所有前缀。

拟定硬上限：一次28800秒（8小时，含加载启动和失败工作）、单条1200秒、最多580896物理步和900万总forecast，串行，不补跑、不延窗。原291为14400秒接受186条，简单按episode估约6.19小时，按物理步线性估约6.73小时；包含旧修复和不同日期/未完May，均只是粗估，不是能完成的保证，也没有为估预算试跑新物理。

这一采集尚未实现或授权启动，本阶段0新标签/拟合/预测/物理。实施前需把这份固定scope、全部日期角色/模型引用、唯一管理器与截止守卫做成可审阅代码，源及授权先发布；配对原始输出先发布后派生独立费用标签。不能继承原291耗尽的4小时。没有合格完整配对或跨日期训练支持不足即关路线，不挑成功seed/调用救结果。

## 学习与验证不预支结论

没有在这里批准新模型或选择架构。只有配对数据证明目标可识别并有训练支持，才另固定唯一模型/对照/误停代价，最多11模型+11标量、一次600秒；没有支持就不训练。未来若学出候选，原4开发日期×12池×两臂×两逆序仍需192条独立验证预算，未包含在288条数据采集里，不隐去后续费用。完整成本/质量/控制时间分别报告，query不当真实提速。

同一own12父模型、内部cal4祖先、已暴露开发日期、相关前缀/seed/折复用限制保持，非全链独立OOF或盲测。全部池/日期/KEEP/不利和缺失结果保留，不拼成功子集。原261历史55/55及微小查询改善保留；真实291仅省2query且未稳定提速，不因这份方案升级结论。森林/阈值/风险/副本/位/叶频扫描和换名成本分类器保持关闭。最终8、新日期、电网、FRAME、论文和依赖不动。

文件：stored_field_matrix.json、membership_counts.json、pool_role_counts.json、data_collection_proposal.json、resource_estimates.json与decision.json。源49d12b2c先发布；第十份固定角色清单补充c7b42d30先发布后只提取角色，前9份未重读执行。没有重跑旧审计、数组或模型。
'''
(O/'GRID296_report.md').write_text(report,encoding='utf8')
save(O/'review.json',dict(passed=True,static_documents=10,member_counts_checked=True,role_dates_exact=True,existing_arrays_or_trajectories_read=0,new_labels=0,new_fits=0,new_predictions=0,new_physics=0,physical_authorization_not_inherited=True,resource_extrapolation_not_guarantee=True))
save(O/'complete.json',dict(passed=True,proposal_complete=True,collection_started=False,new_fit_qualified=False,new_fits=0,new_predictions=0,new_labels=0,new_physics=0))
handoff='''# GRID296完成：固定数据方案，未启动采集

用户2026-11-02要求持续推进，本轮10静态来源方案完成；原数据不能无预测恢复联合动作配对后缀费用。288条/144配对/8小时的有限方案已形成，0标签拟合预测物理；旧291预算不继承。208/261冻结，取消/森林微调关闭。

先读integration AGENTS/四docs及GRID296_report、decision、data_collection_proposal/resource_estimates/review/complete/publication_review。不要重做本次静态盘点或旧294/295/291审计。不因自动推进添加新实验。唯一可推进的是本方案的可审阅实施准备和引用绑定；新采集必须另获明确scope授权才启动，未同意不模拟，若无预算选择直接关该新成本支线。

定时grl重新ACTIVE两次被自动审批拒绝，理由是未识别本轮明确用户恢复授权；无绕过，仍PAUSED，异步确认待答。当前轮授权方案已完成；不是后台已恢复。用户回答后才用官方automation_update处理，不写自动化文件绕过。保留原控制器/费用NET/日期seed/KEEP/最终8/祖先开发限制，源码实际A=GRID296_data_route_proposal。
'''
(O/'CONTINUOUS_HANDOFF.md').write_text(handoff,encoding='utf8')
save(O/'NEXT_PROTOCOL_PLAN.json',dict(status='PROPOSAL_DONE_NO_NEW_RESEARCH_JOB_AUTOMATICALLY_QUEUED',one_allowed_preparation='Reviewable implementation/model-role binding for exact data_collection_proposal.json, zero simulator calls; no new feature/model search or repeat audits.',execution='Only after separate explicit approval of288episodes/8h and published concrete source+model references+authorization. No collection in GRID296.',stop='If no funded or feasible matched-label collection, close operating-cost learning route; do not invent a proxy or run more static stages.'))
msg='GRID296十静态来源完成，0标签/拟合/预测/物理。2378自身教师成员/3582前缀来自222物理调用16日期；格式缺候选联合动作及配对物理后缀费用，不能直接成本拟合。唯一可审阅方案原12池各own12日期scenario100/两臂144对288条、8h/580896步/900万forecast；预估6.19或6.73h非保证，未实施未授权启动，未来学习和192验证预算不隐去。保留208/261、关闭微调；定时grl恢复被自动审批拒绝仍PAUSED，确认待答。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name; old=p.read_text(encoding='utf8'); entry='\n\n## 2026-11-02 — GRID296有限数据路线裁决\n\n'+msg+'\n'
 p.write_text('# 当前：GRID296方案完成，新物理未启动\n\n'+msg+'\n\n以下保留历史。\n\n'+old if name=='RESEARCH_STATE.md' else old+entry,encoding='utf8')
(I/'docs/NEXT_STEPS.md').write_text('# 当前 — 数据方案已完成，停止无标签模型修补\n\n'+msg+'\n\n1. 读296真实报告/字段矩阵/角色计数/资源估计/采集方案，不重复旧盘点或物理。\n2. 新物理未批准；只可准备这份固定方案的reviewable实现/引用和守卫，首次模拟须独立明确288条/8h授权。无采集预算则关成本学习路线，不继续制造资格阶段。\n3. 全部模型日期seedKEEP/原NET费用控制器和最终8保持；未来模型最多22/600，尚未资格。\n4. 定时恢复自动审批未通过，用户确认待答，grl仍PAUSED，禁止绕过。\n',encoding='utf8')
p=I/'research_archive/power-grid-2026/README.md'; p.write_text('# 最新：GRID296固定后续数据方案\n\n[报告](protocols/GRID296_data_route_proposal/results/GRID296_report.md)。'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
S=A/'results';S.mkdir(exist_ok=False);C.mkdir(exist_ok=False)
for p in O.iterdir():
 if p.is_file():shutil.copy2(p,S/p.name);shutil.copy2(p,C/p.name)
shutil.copy2(W/'finalize.py',S/'finalize.py');save(S/'archive_manifest.json',{p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()});save(C/'delivery_hashes.json',{p.relative_to(C).as_posix():sha(p) for p in C.rglob('*') if p.is_file()})
print('GRID296_FINITE_DATA_ROUTE_PROPOSAL_COMPLETE')
