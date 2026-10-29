"""First saved readonly guard-coverage/leaf-support audit; no cutoff fit."""
from pathlib import Path
import sys,json,shutil,pickle
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();pubsha=C.published('public');cov=C.rd(O/'guard_coverage.json');sel=C.rd(O/'guard_selection.json');meta=C.rd(C.META);roles={r['pool']:r for r in C.rd(C.ROLES)}
assert cov['legacy_positive']==85 and cov['current_positive']==500 and cov['shared']==42 and cov['missing_legacy']==cov['added_reachable']==43 and cov['legacy_parent_STOP']==0 and cov['union_positive']==543 and cov['all_shared_semantics_aligned']
added=[]
for r in sel:
 pool=r['pool'];old=np.load(C.LEGACY/'calibration_inputs'/f'{pool}.npz');new=np.load(C.Q/'qualified_inputs'/f'{pool}_cal.npz');base=np.load(C.Q/'qualified_inputs'/f'{pool}_base.npz');u=np.load(O/'union_inputs'/f'{pool}.npz');ol={int(i):j for j,i in enumerate(old['ids']) if old['y'][j]>0};nu={int(i):j for j,i in enumerate(new['ids']) if new['y'][j]>0};assert set(r['legacy_ids'])==set(ol) and set(r['current_ids'])==set(nu) and set(r['added_reachable_ids'])==set(ol)-set(nu) and set(u['ids'])==set(ol)|set(nu)
 ox=old['X'].copy();ox[:,:38]=np.tanh(ox[:,:38])
 for k,mid in enumerate(u['ids']):
  mid=int(mid);isnew=mid in nu;i=nu[mid] if isnew else ol[mid];src=new if isnew else old;xx=src['X'][i] if isnew else ox[i];assert np.array_equal(u['X'][k],xx) and np.array_equal(u['X32'][k],xx.astype(np.float32)) and u['y'][k]==1 and u['actual_NET'][k]==src['actual_NET'][i]>0 and u['fit_role'][k]==src['fit_role'][i] and u['origin_current'][k]==isnew and u['origin_legacy'][k]==(mid in ol)
  score=u['b'][k]+float(base['alpha'][0])*((u['b'][k]+np.sum(xx*base['theta'][:-1])+base['theta'][-1])-u['b'][k]);assert u['parent261_score'][k]==score>=0 and meta[mid]['date'] not in roles[pool]['outer_dates'] and meta[mid]['date'] in set(roles[pool]['fit_dates'])|set(roles[pool]['cal4_dates'])
 for mid in r['added_reachable_ids']:added.append(dict(pool=pool,id=mid,date=meta[mid]['date'],call=meta[mid]['physical_state_group']))
 assert r['shared_X64_maxdiff']==0 and r['union_semantics_aligned']
assert len(added)==43 and len({r['id'] for r in added})==23 and len({r['call'] for r in added})==11
members=C.rd(O/'leaf_public_memberships.json');cache={}
for key,m in members.items():
 pool=m['pool']
 if pool not in cache:
  z=np.load(C.Q/'qualified_inputs'/f'{pool}_fit.npz');b=pickle.loads((C.N/'models'/f'{pool}.pkl').read_bytes());cache[pool]=(z,b,[t.apply(z['X32']) for t in b['model'].estimators_])
 z,b,leaf=cache[pool];assert m['fit_indices']==np.flatnonzero(leaf[m['tree']]==m['leaf']).tolist()
assert len(C.rd(O/'leaf_public_focus.json'))==232 and len(C.rd(O/'focus_leaf_support_details.json'))==232
s=C.rd(O/'summary.json');assert s['new_model_fits']==s['new_cutoff_fits']==s['new_policy_evaluations']==s['new_NNforward']==s['new_forest_scores_for_oldonly_guard']==0
common=C.rd(O/'common_active_dates.json')['dates'];exposure=C.rd(C.N/'per_seed_preservation_and_exposure.json');assert set(common)==set.intersection(*(set(r['active_forest_dates']) for r in exposure)) and len(common)==12
C.save('added_native_guard_records.json',added);C.save('independent_audit.json',dict(passed=True,legacy85_current500_shared42_missing43_union543=True,all43reachable_same_original_targets_roles_features=True,perpool_union_excludes_corresponding_outer4=True,added43_poolrecords_23states_11calls=True,all_public_leaf_memberships_verified=True,focus232_only_not_full268_reaudit=True,common12_by_presetavailability_notquality=True,new_model_fits=0,new_cutoff_fits=0,new_policy_evaluations=0,new_forest_scores_for_oldonly_guard=0,publicselection_published=pubsha))
plan=dict(id='GRID270',scope='QualificationONLYofONEfixednative/referencepositiveguardunion;0fit/newcutoff/newforestscoreon43added/outerpolicy',evidence='269publicselection frozen:legacy85/current500 intersection42,43legacy-only allreachable;union543(active507+KEEP36),11activepoolsaffected. These are43pool-records/23globalstates/11calls/7olddates,notindependentnewscenarios.',selection='Useexact269union_inputs andguard_selection,per(pool,id) dedup/currentprecedence,alloriginal500 preserved. No outerfailureID insertion ordateexception;eachpool own12 and correspondingouter4 excluded. Legacyunreachable list preserved(emptyactual).',qualification='Verify269public_freeze/input/model/publicselection Gitblob lineage;543positive-onlyinputsguards represent selectedconstraints,not originalall2054calrows. Check yNET>0/role/finite/X64/X32/parent261entry,source249nativeSTOPbit versusreferenceNN8selection,canonicalpublicfeaturedefinitions. Resolveexisting256public_full cache/map via sourceconstants to crosscheck added43first39coordinates/b whenavailable,keepactualNET12STOPbit1 from249proof;no newNN/heads/data generation. If missingcache or semanticambiguity,reportandresolve beforefit,do not invent values.',model='FreezeALL268forests/261parent andsampleweights,0forestthetaNNrefit;reuse267runtimepositive_score/protect_cutoff/apply_gate unchanged. Stillown12/internalcal4empiricalguard,notindependentvalidation.',future271='After270qualification separately11newcutoffs/one600s,1KEEP. All11affectedpools recompute unique maximal preservingunion positive cutoff=minfrozenforestscore on543,do not choosebyobservedcutoffchange. No tree/depth/risk/fee/tau sweep;no .fit. Old500subset ensuresnewtau<=oldtau,onlyrestore268CONT within261entry;couldusemorequeries anddoesnotguaranteeouterNET/wholeweek.',barriers='271owncodeinputs firstpublish;ALL12cutoff/model-referencebundles publishedbefore newdecisions. Byte-reuse268publishedforest_score andparent261_continue/ids,so0newouterforestforward needed;ALL12predictionspublishedbeforeoldresults. Primary268,261qualityanchor/265historicalreuse,allseedsdatescounterexamples.',count='2700/currentbranch95. Separate271+11=>106,forests704unchanged. Historical6continuations8285update/604.703actual/678.156cap retained. Addedconstraintlabel exposure disclosed,notsamecalibrationsamplecomparison.',limits='Do not retrospectively change or invalidate268fixed500protocol/results. No final8/physics/grid/paper/FRAME/install/controller/order/alias/station/128/64/stagnation2/.9change. Oldexposedcorrelation/ancestry andquery-not-actualtime limitsremain.')
C.save('NEXT_PROTOCOL_PLAN.json',plan);C.save('decision.json',dict(retain261_main_qualitycandidate=True,retain268_localsignals_andalladverse=True,guard_domains_nonnested_confirmed=True,proposedunion_has43newreachable_trainingguards=True,no_claim_this_caused_or_fixes_allouterlosses=True,next='GRID270zero_fit_union_qualification',no_newcutoff_in269=True))
gloss=s['leaf_support']['profitable_loss'];gneg=s['leaf_support']['nonprofitable_cancel'];cm=s['common_existing_policy_comparison']
report=f'''# GRID269：保护集合覆盖与固定树叶支持

本阶段新增模型拟合0、截点拟合0、替代策略评估0。没有计算补充记录的森林分数或并集新截点。

## 发现明确的集合覆盖缺口

历史部署原NET12停止点的85条回本经验记录，来自GRID250实际Q常量所指的GRID249校准输入；GRID249已验证它们对应原NET的停止决定。当前500条来自参考NN8停止点采样。来源沿250→249→246→242的冻结源码常量和清单解析，没有按外层失误重新挑记录。

|集合关系（池×记录口径）|数量|
|---|---:|
|旧部署停止点正例|85|
|当前参考停止点正例|500|
|两者共有|42|
|旧集合有、当前没有|43|
|固定去重并集|543|

当前500不是旧85的超集。43条缺失记录全部能通过当前冻结GRID261入口，没有继承性父停止需要偷偷排除。共有记录的NET标签、内部角色、float64输入和float32森林输入完全一致。逐池并集成员、输入及选择规则已先发布b92109f6；选择只用了各池own12记录及原入口，没有加入对应外层4日期的标签或失误ID。

43个池×记录条目合并后是23个全局前缀记录、11个物理调用、7个旧日期，存在折/种子复用，不能说43个独立新场景。保留原500并补43后得到543条正例约束，其中活跃507、KEEP36。这是已有训练/校准记录的经验约束修订，不是新仿真数据。

参考网络决定了训练采样，部署入口仍由原网络决定；保护范围需要明确覆盖这些来源。该缺口是一个可检验的解释，但尚未证明它导致或能修复全部外层回本损失。GRID268按封存的500条规则执行正确，旧结果继续保留，不追改旧截点或将不利结果作废。

## 树叶支持没有给出手工安全门

11条回本误停经过的64棵树中，没有正例训练记录的树数量中位数为{gloss['trees_without_positive']['median']:.0f}；221条不回本取消的对应中位数为{gneg['trees_without_positive']['median']:.0f}。每个状态所落叶的训练行数中位数，分别为{gloss['median_leaf_training_rows']['median']:.0f}和{gneg['median_leaf_training_rows']['median']:.0f}。分布有重叠，不能用某个树叶计数、距离或事后阈值筛掉反例。

这些统计由已发表的固定树路径先确定成员，再连接自身fit8原标签/风险；固定叶加权类频率平均精确重现原公开分数。跨64树重复出现的训练记录不当独立样本或新的支持量。

## 保留共同作用范围内的小改善

共同启用的12日期由预定森林/KEEP可用范围交集决定，先固定再描述已有结果；不是按质量挑日期，也没有生成新子策略。

|种子|共同12日期代理净效用变化（减261）|查询变化|即时增益变化|
|---|---:|---:|---:|
'''
for r in cm:
 d=r['delta'];report+=f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|\n"
report+='''
共同作用范围中，种子0和1保持原增益/动作一致率并少用查询，种子2仍有损失。这些小信号保留，同时保留完整16日期的退化结果，不能以这个子集替换完整结果或挑种子部署。

## 下一固定方案

下一GRID270只资格已固定的543条正例并集：核对每条输入定义、原角色、原NET与父入口，补查已有公共缓存的数值来源。训练0，不预计算补充记录的森林概率或新截点。资格后另GRID271才在完全冻结的11森林上各校准一个最大可行截点，一次600秒，1KEEP；原704树不重训。

新截点应不高于旧截点，因为原500仍在并集中；新决定只能恢复部分GRID268继续决定，并受GRID261入口限制。实际能恢复多少回本、增加多少无利查询，需要全参数先发布、全预测先发布后单独比较，不能提前声称改善。全部种子/日期/反例保留，不扫截点或加入外层失败记录。

本阶段分支仍95；若后续271执行，另11、累计106。新增43条约束暴露须披露，不能称同校准样本对照。6次旧续训8285更新、原NN604.703秒/用户678.156秒上限另列。父own12/内部cal4祖先、已暴露16日期与相关前缀/种子用途局限不变，非全链OOF/盲测/整周质量或实际时延证据。最终8、新物理/电网/论文/FRAME未动，无安装；原费用/NET标签/基本控制器/排序别名/锁站/128/64/停滞2/.9保持。
'''
(O/'GRID269_report.md').write_text(report,encoding='utf8');C.save('complete.json',dict(passed=True,readonly=True,new_model_fits=0,new_cutoff_fits=0,new_policy_evaluations=0,coverage85_500_42_43_543=True,branch95=True,final8_absent=True));C.archive('audited')
D=R/'论文协作资料/FRAME_保护域覆盖_GRID269';D.mkdir(exist_ok=False)
for name in ['GRID269_report.md','guard_coverage.json','guard_selection.json','added_native_guard_records.json','summary.json','independent_audit.json','decision.json','NEXT_PROTOCOL_PLAN.json','complete.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID269只读确认保护域不嵌套：旧NET12部署STOP85正例与当前referenceNN8STOP500仅交42、缺43，43全可达，逐池共享NET/角色/X64/X32完全一致，去重并集543(活跃507/KEEP36)。43池记录是23全局前缀/11调用/7旧日期，非43独立场景；来源250Q→249→246→242常量/旧nativeSTOP资格及META哈希核对，选择own12排对应outer4，b92109f6先发成员/输入，0新增森林分数/截点。11回本损失无正例树数中位44/64、221取消49/64，不能作手工安全门。共同12日期由KEEP可用范围先固定，seed0/1增益动作同且少query、seed2仍损失；保留完整16日期反例，不挑子集部署。下一270仅资格543固定并集及缓存来源，0真实概率/截点；另271才冻结11森林校准11截点/600s/1KEEP，全部参数/预测先发后结果，原500保留、只可能恢复268CONT受261入口限制。缺口非因果或效果保证，旧268500协议结果不追改。当前95，后续新11若执行106；祖先/暴露/时延/旧续训边界保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-29 — GRID269保护域覆盖\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+e+rest
 else:t+=e
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID269后\n\n1. '+msg+' 完整覆盖/支持诊断和归档不重做。\n2. 下一270仅消费NEXT_PROTOCOL_PLAN和269public固定union_inputs/guard_selection，核对543正例、原500完整保留、43新增来源/角色/输入语义、261可达、704树权重与缓存来源。可沿256/257冻结源解析已有full公共缓存验证前39坐标与b，nativeNET12STOPbit1沿249旧证明，不新NNforward/补数据/改字段救对齐；0新概率/截点/outer策略。\n3. 资格后独立271最多11截点/一次600s，冻结268模型/261入口，预测时可复用268公开forest_score和parent261_continue，全参数/预测先发后结果，不扫tau或按外层失误加ID。新增约束与106潜在累计计数披露，原95/旧续训时间不隐藏；最终8/新物理/电网/论文/FRAME不动，不安装。Githelper串行白名单/实际A归档协作远端核验保持。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID269保护域覆盖\n\n[报告](protocols/'+A.name+'/audited/GRID269_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID269_AUDITED',json.dumps(dict(legacy85=True,current500=True,shared42=True,added43=True,union543=True,newfits0=True)))
