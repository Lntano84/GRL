"""First saved conditional-pool qualification audit; no fitting/prediction."""
from pathlib import Path
import sys,json,shutil
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,P=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();q=C.rd(O/'qualification.json');assert q['passed'];rr=C.rd(O/'qualified_pools.json');nsel=nexc=naug=0
for r in rr:
 p=r['pool'];old=np.load(C.F/'qualified_inputs'/f'{p}_fit.npz');sel=np.load(O/'qualified_inputs'/f'{p}_fit.npz');exc=np.load(O/'excluded_inputs'/f'{p}_fit.npz');pub=np.load(C.Q/'public_entry'/f'{p}_fit.npz');E=pub['eligible'];assert np.array_equal(sel['original_row_index'],np.flatnonzero(E)) and np.array_equal(exc['original_row_index'],np.flatnonzero(~E)) and np.array_equal(np.sort(np.r_[sel['original_row_index'],exc['original_row_index']]),np.arange(len(old['ids'])))
 for key in old.files:
  if key=='fallback_KEEP':assert np.array_equal(sel[key],old[key]) and np.array_equal(exc[key],old[key])
  else:assert sel[key].dtype==old[key].dtype and sel[key].tobytes()==old[key][E].tobytes() and exc[key].dtype==old[key].dtype and exc[key].tobytes()==old[key][~E].tobytes()
 for role in ['cal','base']:assert C.sha(O/'qualified_inputs'/f'{p}_{role}.npz')==C.sha(C.F/'qualified_inputs'/f'{p}_{role}.npz')
 assert (old['y'][~E]==0).all() and old['y'][E].sum()==old['y'].sum() and sel['is_augmented'].sum()==old['is_augmented'].sum() and abs(r['selected_risk']+r['excluded_risk']-r['old_risk'])<1e-10
 nsel+=len(sel['ids']);nexc+=len(exc['ids']);naug+=int(sel['is_augmented'].sum())
assert nsel==2544 and nexc==331 and naug==16 and C.sha(O/'runtime276.py')==C.sha(C.RUNTIME)
C.save('independent_audit.json',dict(passed=True,exact278publicE_rowindex_subset=True,alloriginalfitfields_partition_into_selected_and_excluded_byteexact=True,selected2544_excluded331=True,allpositive_and16copies_preserved=True,all543cal_base_byteunchanged=True,no_weightnormalization_dedup=True,all11changedclasses_nonempty=True,originalruntime_byteunchanged=True,newfits=0,newpredict=0,newouterstrategy=0))
plan=C.rd(C.Q/'NEXT_PROTOCOL_PLAN.json');plan['id']='GRID280';plan['scope']='One fixed invocation-conditioned39D forest experiment after279qualificationpublication;11forest+11same543cutoff/one600s/1KEEP. Not executed in279.';plan['qualified_source']='Only279qualified_inputs/qualified_freeze/runtime276/model_references. Excluded_inputs/provenance preserved but never enter fit. New280source+inputs publish before first fit;all12models publish before scores/all12predictions before old outcomejoin.';C.save('EXECUTION_PLAN.json',plan);C.save('decision.json',dict(passed=True,ready_for_one_fixed280=True,newfits0=True,future11forests11cutoffs=True));C.save('complete.json',dict(passed=True,newfits=0,newpredict=0,newstrategy=0,qualified2544_active2410=True,positive534_negative1876=True,excluded331_preserved=True,all16copies_guard543_preserved=True))
(O/'GRID279_report.md').write_text('# GRID279：固定调用入口条件训练子集资格\n\n资格通过，模型/截点拟合、真实概率和新外层策略均为0。只按GRID278预先发布的E逐行保留样本，不按负标签、外层失误或种子筛选。\n\n原2875条候选训练记录保留2544条，活跃2410条（534正例、1876负例），KEEP134条不变。331条被排除记录完整另存excluded_inputs，含原行索引/ID/原因；与保留行按索引合并可逐字段恢复全部原输入。16条相关副本和全部正例均保留，没有ID去重或合并权重。训练日期继续排除cal4/outer4，全部11个改变数据的活跃池两类均非空。\n\n保留行的X39/X32/X40来源、NET/y/weight/source_mass及角色字段均取原字面值；不重新按域归一化或重配风险。原543条保护输入及base文件字节不变，39维runtime原样复用。筛选会改变条件经验风险及样本暴露，不能声称原两域风险比例不变。排除记录占原活跃数12.08%但权重仅0.16071%，不称主要误停原因或331次线上节省。\n\n下一独立280固定11森林+11同543经验正例截点，一次600秒/22操作、1KEEP、704新树；原39D/64树depth4leaf10/gini及全部种子配方不变，0NN/theta重训。全部模型/截点先发布，再评分；全部12组预测先发布后连接旧开发结果。主要匹配277，保留274/271历史、26155/55质量锚/GATE及全种子/16日期/KEEP反例。更多查询换得质量不能称等预算或实际提速。\n\n当前分支146，279新增0，未来280独立22至168；原16副本相关、500→543约束暴露、6续训8285更新/NN604.703实际及用户678.156上限另列。入口E是教师前缀在假设部署STOP1上下文的固定资格，不是新物理闭环。父own12/内部cal4祖先、已暴露相关数据局限保持，经验保护非独立外层保证，非全链OOF/盲测/整周质量或实际时延。最终8、新物理/电网/论文/FRAME未动，无安装，原费用NET标签/控制器/排序别名/锁站/128/64/停滞2/.9保持。\n',encoding='utf8');C.archive('audited')
D=R/'论文协作资料/FRAME_条件训练资格_GRID279';D.mkdir(exist_ok=False)
for n in ['GRID279_report.md','qualification.json','independent_audit.json','complete.json','decision.json','EXECUTION_PLAN.json','qualified_pools.json','exclusion_provenance.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID279固定公开E子集资格通过，0拟合/概率/策略。候选2544/活跃2410=534正1876负，KEEP134，原331排除行和index/id/原因另存，逐字段与保留行可恢复原全部输入。原NET/y/weight/source_mass字面值保留，不重权重/归一化/去重，所有正例/16副本保留，543cal/base字节不变/39D runtime原样，11改池类均非空。条件风险/暴露修订明确，入口支持不当物理/因果证据。下一独立28011森林+11同543截点/600秒/1KEEP/704新树，NNtheta0，主匹配277/261锚和全部历史端点/种子日期反例；当前146未来168，原祖先开发/旧续训/整周时延边界保持。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID279条件训练子集资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID279后\n\n1. '+msg+' 完整资格/审计/归档不重做。\n2. 下一280先核对279qualified_freeze/runtime/modelrefs/EXECUTION_PLAN/decision具体Gitblob远端，独立封存源码输入推送后11森林+11截点一次600秒；只qualified_inputs拟合，excluded_inputs仅留档。原配方/39D/全seed/543guard，不改E或权重。\n3. 全模型截点先发、原249first38tanh切39评分、全12预测先发后旧结果；主匹配277，保留274/271/261/GATE和全部反例，最终8/新物理电网论文FRAME不动，不安装，实际A归档协作远端/Git串行白名单保持。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID279条件训练子集资格\n\n[报告](protocols/'+A.name+'/audited/GRID279_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID279_AUDITED_ZERO_FITS')
