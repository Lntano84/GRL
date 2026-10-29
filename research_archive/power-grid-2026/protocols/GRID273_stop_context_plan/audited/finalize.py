from pathlib import Path
import sys,shutil,numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
O,A,P=C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();rr=C.rd(O/'qualified_pools.json');prov=C.rd(O/'augmentation_provenance.json');assert len(prov)==16 and len(rr)==12
for r in rr:
 p=r['pool'];old=np.load(C.F/'qualified_inputs'/f'{p}_fit.npz');new=np.load(O/'qualified_inputs'/f'{p}_fit.npz');n=len(old['ids'])
 for key in old.files:assert np.array_equal(new[key] if key=='fallback_KEEP' else new[key][:n],old[key])
 assert C.sha(O/'qualified_inputs'/f'{p}_cal.npz')==C.sha(C.U/'qualified_inputs'/f'{p}.npz') and C.sha(O/'qualified_inputs'/f'{p}_base.npz')==C.sha(C.F/'qualified_inputs'/f'{p}_base.npz')
 for a in [x for x in prov if x['pool']==p]:
  j,k=a['original_index'],a['new_index'];assert new['ids'][k]==old['ids'][j]==a['id'] and new['y'][k]==1 and new['actual_NET'][k]==old['actual_NET'][j] and new['weight'][k]==old['weight'][j];assert np.array_equal(new['X'][k,:39],old['X'][j,:39]) and new['X'][k,39]==1 and old['X'][j,39]==0
C.save('independent_audit.json',dict(passed=True,alloriginal2859rows_andlabelsweights_preserved=True,all16copies_onlycontext01_andinheritedNETweight=True,no_cal4_outer4_added=True,all543guardbyteunchanged=True,nine_changed_two_unchanged_oneKEEP=True,newmodel_scalar_fits=0,newprobability=0))
plan=C.rd(C.Q/'NEXT_PROTOCOL_PLAN.json');plan['id']='GRID274';plan['qualified_inputs']='Only GRID273 qualified_inputs and qualified_freeze/runtime267/model_references/qualified_pools. 274source inputs published before first fit. Nine changed pools one forest+one cutoff, two unchanged model/cutoff reuse, oneKEEP. All18operations one600s global manager watchdog.';C.save('EXECUTION_PLAN.json',plan);C.save('decision.json',dict(passed=True,ready_for_one_fixed274=True,model_fits9_scalar_fits9=True,newfits0=True));C.save('complete.json',dict(passed=True,newfits=0,newprobabilities=0,qualified_rows2875=True,active2741=True,copies16=True,cap_future600=True))
(O/'GRID273_report.md').write_text('# GRID273：自身训练停止上下文副本资格\n\n资格通过，模型/截点拟合、真实概率和新外层策略均为0。原2859训练行逐字段保留，增加16条正例STOP上下文副本后共2875行，活跃2741行；来自9个池、6调用4日期，非16个独立新样本。\n\n全部副本只取各池自己的fit8日期，排除内部cal4及对应outer4；原NET/标签/权重精确继承，其他39连续字段相同，仅上下文0→1。原CONT记录和所有负例保留，未引入27条cal-only记录。无相反标签精确输入冲突。原543条并集经验保护输入字节不变，未重新求截点。\n\n新副本继承的权重和为0.14013055289338797（跨9个池求和，不是单池比例）；这是明确增加正例经验风险和暴露，不称原两域风险比例不变。逐池旧新风险和见qualified_pools.json。source_mass仅保存来源，不声称新增独立结构质量。\n\n下一独立GRID274：一次600秒窗口9森林+9相同543正例截点=18操作。原64树/depth4/leaf10/gini/种子等配方固定，576新树、部署总704。两个无增补的活跃池复用原森林和271截点，f0_s1KEEP；不重复拟合未改变的旧项。全部模型/引用与截点先发布，九新森林计算外层分数，其他池复用；全预测先发布后连接旧开发结果。主匹配271及261质量锚/GATE，全部日期种子和反例保留，不扫描参数。\n\n资格不说明改善；输入样本暴露/经验风险发生变化，非纯上下文位消融。父own12/内部cal4祖先、已暴露16日期/相关前缀种子限制不变，非全链OOF/盲测/整周质量或实际时延保证。当前分支106，未来独立18至124；6旧续训8285更新/NN604.703实际与用户678.156上限另列。最终8、新物理/电网/论文/FRAME未动，未安装，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9不变。\n',encoding='utf8');C.archive('audited')
D=C.R/'论文协作资料/FRAME_停止上下文资格_GRID273';D.mkdir(exist_ok=False)
for n in ['GRID273_report.md','qualification.json','independent_audit.json','complete.json','decision.json','EXECUTION_PLAN.json','qualified_pools.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID273资格通过，0拟合/概率/策略。原2859训练行标签权重保留，16自身fit8 STOP副本后2875/活跃2741，全部排cal4/outer4、仅context0->1、无相反标签冲突，543guard字节不变。副本权重继承原值，正例风险增加总.140130553跨9池，显式目标/暴露修订不称原半半比例不变。下一274仅9森林+9截点/一次600秒，2未改活跃模型/截点复用+1KEEP，576新树总704，NNtheta0；全模型先发、全预测先发后旧结果，全部日期seed保留，当前106未来124。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID273停止上下文副本资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID273后\n\n1. '+msg+' 完整资格/审计/归档不重做。\n2. 下一274先核对273qualified_freeze/runtime/modelrefs/EXECUTION_PLAN具体Gitblob与远端，独立封存源码输入发布，9森林+9校准一次600秒，原配方不调参。两未改池复用271权重/截点/分数，KEEP不变，0NNtheta重训。\n3. 全12模型或引用截点先发，九新森林outer分数先发后旧结果；主匹配271/261/GATE，全日期seed和反例保留。最终8/新物理电网论文FRAME不动，不安装，实际A归档协作远端核验，Githelper串行白名单保持。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID273停止上下文副本资格\n\n[报告](protocols/'+A.name+'/audited/GRID273_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID273_AUDITED_NO_FIT')
