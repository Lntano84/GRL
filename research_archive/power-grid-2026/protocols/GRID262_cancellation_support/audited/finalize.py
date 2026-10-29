"""First saved-results integrity check and report for GRID262; no research refits."""
from pathlib import Path
import sys,json,shutil,collections
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();head=C.published('public');s=C.rd(O/'summary.json');rows=C.rd(O/'focus_support_details.json');pub=C.rd(O/'public_focus_pairs.json');meta=C.rd(C.META);assert s['passed'] and len(rows)==len(pub)==77
for r,p in zip(rows,pub):
 assert r['pool']==p['pool'] and r['metadata_id']==p['metadata_id']
 assert abs(r['published_score']-(r['parts']['parent_b']+r['parts']['alpha']*sum(r['parts'][k] for k in ['public38','parent_b_feature','STOP_feature','intercept'])))<1e-12
 z=np.load(C.Q/'fit_inputs'/f"{r['pool']}.npz");h=np.load(C.P/'outer_parent_public'/f"{r['pool']}.npz");j=int(np.flatnonzero(h['ids']==r['metadata_id'])[0]);x=h['X'][j].copy();x[:38]=np.tanh(x[:38])
 for n in r['own_training_neighbors']:
  if n.get('missing'):continue
  i=n['fit_index'];stop=n['domain']=='STOP';allowed=[ii for ii,idx in enumerate(z['ids']) if bool(z['domain_STOP'][ii])==stop and int(idx)!=r['metadata_id'] and meta[int(idx)]['physical_state_group']!=r['call']]
  ds=np.sum((z['X'][allowed]-x)**2,axis=1);assert allowed[int(np.argmin(ds))]==i and abs(n['distance']-float(np.sqrt(ds.min())))<1e-12
  assert n['y']==int(z['y'][i]) and n['actual_NET']==float(z['actual_NET'][i]) and n['risk_weight']==float(z['weight'][i])
assert sum(r['actual_next_queries'] for r in rows)==122 and all(r['actual_next_gain']==0 for r in rows)
assert abs(sum(r['family_macro_query_saved'] for r in rows)-s['macro_query_saved'])<1e-14
assert abs(sum(r['family_macro_NET_saved'] for r in rows)-s['macro_NET_saved'])<1e-14
assert s['neighbor_support']['STOP']['positive']==s['neighbor_support']['CONT']['positive']==1
contra=[dict(pool=r['pool'],sample_id=r['sample_id'],call=r['call'],position=r['position'],score=r['published_score'],neighbors=r['own_training_neighbors']) for r in rows if any(n.get('y')==1 for n in r['own_training_neighbors'])];assert len(contra)==1
C.save('contrary_neighbor_case.json',contra)
C.save('independent_audit.json',dict(passed=True,source_hashes=True,public_pair_gitblobs_published=head,all77_additive_scores_and_cost_conservation=True,all154_nearest_pairs_reproduced_excluding_samephysicalgroup=True,ownfit_labels_weights_exact=True,contrary_case_retained=True,newfits=0,newalpha=0,new_policy_predictions=0,new_NN_or_headforward=0,no_full261_reevaluation=True))
nextplan=dict(id='GRID263',scope='Training-only residual risk/cost qualification;0fits/alpha/newouterpolicyscores or labels',allowed_inputs='262frozen260 ownfit8 arrays y/NET/weight/domain/ids/X/b;261fixedmodels/steps;113metadata nextqueries/date/physicalgroup. No outerpublic/predictions/outcomes read by263 computation.',question='Do fixed261ownfit8 falseCONT nonprofitable rows retain material knownquerycost/risk/loss? SeparateSTOP/CONT and positive/negative, physicalcall/date concentration;objectiveNLL+L2 contribution from originalweights, no reweighting or thresholdsearch.',fixed='Use existing deployed zero threshold and existing41theta/alpha only. Recomputing owntraining scores is diagnostics, no new outerstrategy. All11active/KEEP/alltrainingcounterexamples retained. No model or finite candidate yet.',following='Only after owntraining evidence, qualify ONElimitedlearning hypothesis with completeoperationcount<=22/600s;do not scanriskratios/alpha/transform/seed or infer causal benefit from exposedouter.',boundaries='Final8/newphysics/grid/FRAME/paper/install untouched;priorown12/internalcal4 ancestry and proxies remain;branchfitsstill62plus6historicalcontinuations.')
C.save('NEXT_PROTOCOL_PLAN.json',nextplan)
C.save('decision.json',dict(retain261_small_qualitypreserving_candidate=True,no_single_call_dominates=True,nearest_negative_support76of77_descriptive=True,one_contrary_case_keep=True,not_proven_generalization_or_causality=True,no_newpolicy=True,next='GRID263ownfit8residualriskcostqualificationonly',do_not_immediately_reweight_or_addnetwork=True))
report=f'''# GRID262：省查询的来源与训练支持

本阶段只解释GRID261已经保存的77次取消。没有新模型、步长拟合、神经网络/收益头forward、策略比较或电网回放；没有重复GRID261完整评分和审计。

## 节省从哪里来

77次取消均为零即时增益补查，合计省122次原始候选查询，日期宏平均少{s['macro_query_saved']:.9f}次。这个宏平均与GRID261减公共GATE的查询差逐项守恒；原始122次与宏平均分母不同，不能混用。

这些取消分布在42个物理调用、14个旧日期。最大一个调用占宏查询节省{s['top_call_macro_share']:.2%}，前五个合计{s['top5_call_macro_share']:.2%}；按原始查询计，最大调用占{s['top_call_raw_share']:.2%}。收益没有集中到单一调用，但仍是相关的开发记录，不能当作独立确认。最多的宏贡献来自2037-11-02_100的call003，5次取消省8次原始查询。

按同样本/位置合并为71个状态，其中6个状态在两个种子重复取消，没有三个种子都取消的状态。三个种子总体同向省查询，并不意味着每个种子学习了相同取消集合。

## 模型为什么停

原父先验b在77条记录上全部为正，中位数{s['parent_b']['median']:.6f}；学习修正全部为负，中位数{s['correction']['median']:.6f}。最终停止裕度（负分数的绝对值）最小{s['stop_margin']['min']:.6f}、中位数{s['stop_margin']['median']:.6f}、最大{s['stop_margin']['max']:.6f}，说明取消不是浮点零附近的正负舍入。这里是描述现有分数，未扫描任何新阈值。

全部38公共项合计在这77条上均为负，中位数{s['public38_sum']['median']:.6f}；父b特征、STOP位和截距也按原式逐条分解保留。不能仅凭负项占比将变换解释为因果机制，或从中挑几列再试新策略。11个活跃模型的alpha均为1，当前这些取消由学到的端点实现，未依赖缩小步长。

## 是否有自身训练支持

固定有界40维欧氏距离，每条取消记录分别找自身fit8中最近的STOP和CONT记录各一个，排除自身和同一physical_state_group。公开配对与距离6b31b592先发布，再连接相应原训练标签，未按标签挑邻居、学习距离、扫描k或距离门槛。

- 最近STOP邻居：76/77是不回本标签，1条回本；距离中位数{s['neighbor_support']['STOP']['distance']['median']:.6f}。
- 最近CONT邻居：76/77是不回本标签，1条回本；距离中位数{s['neighbor_support']['CONT']['distance']['median']:.6f}。
- {s['outside_ownfit_any']}/77条取消至少一坐标超出自身fit范围；所有记录均保留，未把范围检查转成新门控。

两个相反标签邻居都对应同一取消记录：种子2、f2、2037-06-01_100的call008位置0。其最近训练邻居来自2037-05-18_101的call008，原NET约0.014250；而该取消记录自身保存的即时增益为零。该反例单列在contrary_neighbor_case.json。最近邻是支持描述，不能当作正确性的证书，当前距离也不是校准置信度。

原训练STOP正例已占各池STOP风险约87.94%–98.11%；因此不能简单说“正例权重不足”，也不能用这些外层取消结果去挑新的风险比例。当前小改善有来自自身训练的局部支持，值得继续研究，但尚未证明能扩大到更多省查询状态。

## 下一步与边界

保留GRID261这个质量保留的内部候选，以及250低查询/230历史端点。下一GRID263只看自身fit8：固定现有模型，检查不回本却仍继续的训练记录有多少、花多少查询、承担多少原风险与损失，按STOP/CONT、调用和日期分解。该步骤不读取新的外层结果，不立即改权重、网络、alpha或阈值；先确认剩余空间来自哪里，再资格固定唯一有限方案。

本阶段拟合0，数据基础分支仍累计62拟合；6次历史续训8285更新、原NN604.703秒/用户批准678.156秒上限继续披露。父own12/内部cal4祖先局限不变；16旧日期已暴露，前缀/种子/用途相关。局部代理与候选查询不等整周质量或真实时延。最终8未生成/读取，无新物理/电网/论文/FRAME/安装，费用/标签/控制器/排序别名/锁站/128/64/停滞2/.9保持。
'''
(O/'GRID262_report.md').write_text(report,encoding='utf8')
C.save('complete.json',dict(passed=True,focused77=True,fixed_public_pairs154=True,models=0,alpha=0,new_policy_scores=0,new_NNforward=0,final8_absent=True))
C.archive('audited')
D=R/'论文协作资料/FRAME_取消补查支持_GRID262';D.mkdir(exist_ok=False)
for name in ['GRID262_report.md','summary.json','decision.json','independent_audit.json','complete.json','contrary_neighbor_case.json','NEXT_PROTOCOL_PLAN.json','focus_support_details.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,collaboration=str(D),actual_archive=str(A),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID262仅只读77取消/raw省122query成本与训练支持完成：最大调用宏节省8.51%、前五32.66%，分散42调用14旧日期；71状态仅6跨两种子重复。既有stop裕度min.004104/median.117288，public38项合计全负、alpha全1。固定有界40维最近1、排除同调用，公开154配对先发后自身fit标签；STOP/CONT各76不回本/1回本，相反两邻居落同一Jun1seed2记录，保留反例。10/77至少一维越fit范围，不作OOD门/距离置信度。原STOP正风险87.94%-98.11%，不直接说正例无权重或按外层改比例。0模型/alpha/新策略/NNforward；保留261小改善，下一263仅自身fit8固定模型残余误继续的原查询/风险/损失和调用集中度资格，不读新outer/不重权重或加网络，分支仍62。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;text=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID262取消补查支持诊断\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':title,rest=text.split('\n',1);text=title+'\n'+entry+rest
 else:text+=entry
 p.write_text(text,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID262后\n\n1. '+msg+' 完整261/262结果与审计不重复。\n2. GRID263仅消费自身fit8原输入/标签/风险和261固定模型，按已封存NEXT_PROTOCOL_PLAN分解不回本误继续的已知query、原NLL与风险/调用日期集中。固定零阈值，无新theta/alpha/outer输入预测或结果，不扫权重比例/变换/距离门。先源码输入封存发布。\n3. 根据自身训练证据再资格唯一有限后续方案，仍<=22拟合/一次600s，不盲扫/拆搜索。保留全部反例/seed，最终8/新物理/电网/论文/FRAME不动，不安装；旧开发/祖先/时延边界和62计数保持。Githelper串行、路径白名单、实际A归档协作远端核对。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID262取消补查支持诊断\n\n[报告](protocols/'+A.name+'/audited/GRID262_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID262_AUDITED',json.dumps(dict(passed=True,cost77=True,pairs154=True,counterexample_kept=True,next='263ownfitonly'),ensure_ascii=False))
