"""First saved-input qualification audit and delivery for GRID267; no fitting."""
from pathlib import Path
import sys,json,shutil
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'independent_audit.json').exists();C.verify();q=C.rd(O/'qualification.json');assert q['passed'] and q['actualfits0'] and q['realforestpredictions0'] and q['newouterinputs_or_results0']==0
assert len(q['pools'])==12 and sum(not p['KEEP'] for p in q['pools'])==11 and sum(p['guard_positive'] for p in q['pools'])==500
for p in q['pools']:
 pool=p['pool']
 for role,folder in [('fit','fit_inputs'),('cal','calibration_inputs')]:
  old=np.load(C.Q/folder/f'{pool}.npz');new=np.load(O/'qualified_inputs'/f'{pool}_{role}.npz');assert set(new.files)==set(old.files)|{'X32'}
  for key in old.files:assert np.array_equal(new[key],old[key])
  assert new['X32'].dtype==np.float32 and np.array_equal(new['X32'],old['X'].astype(np.float32))
 assert C.sha(O/'qualified_inputs'/f'{pool}_base.npz')==C.sha(C.M/f'{pool}.npz')
 assert p['constructor_only_no_estimators'] and p['roles_outerexcluded'] and p['old261_all_guarded_continue']
for rel,h in C.rd(O/'qualified_freeze.json').items():assert C.sha(O/rel)==h
maxcast=max(max(p['fit_cast_maxabs'],p['cal_cast_maxabs']) for p in q['pools']);dups=sum(c['duplicate_groups'] for p in q['pools'] for c in p['collisions']);conf=sum(c['opposite_label_groups'] for p in q['pools'] for c in p['collisions'])
C.save('independent_audit.json',dict(passed=True,all36qualifiedinput_files_originalfields_unchanged=True,X32only_numerical_cache_same40dimensions=True,all12baseweights_byteidentical=True,guard500_active464=True,constructor_only_11active_1KEEP=True,actualfits0=True,realforestprediction0=True,actualcutoff0=True,cast_maxabs=maxcast,duplicate_groups=dups,opposite_label_groups=conf))
execution=dict(id='GRID268',requires='267qualified_freeze/runtime_freeze andall36inputs/runtime267specificGitblobs publication;new268sourcefreeze publishedbeforefit',params=C.RT.PARAMS,models='11newExtraTreesClassifier with originalseed0/1/2;f0_s1KEEP. All data from267 qualified_inputs:pool_fit.npz X32/y/weight, pool_cal.npz X32/y andpool_base.npz frozen261. No newscaler/field/label/weight.',calibration='For each forest,positive_score calibratedonALLsameown12selectedSTOPy>0;one tau=minpositive score, tiescontinue. Parent261empiricalpositive entryalreadychecked. KEEP/emptyclass/emptyguardkeeps261;no sweep. One riskweightedNETscore,not calibratedprobability.',budget='11forests+11cutoffs=22fits,704newtrees,one600sstrictwindow includingloads/fit/calprobabilityandchecks,1KEEP;0theta/NNfits or oldmodelrefits,no extension/retrycompleteditems.',checks='Eachforest64trees/depth<=4/minleaf10, weightedleafclassfrequencies vsnativepredict_proba, classes0/1, modelparameters/seed/rowids/weights andALL500positiveempiricalprotection. Keep original26x labels/ancestry caveats. Verify installed sklearn version==qualification1.7.2, noinstall.',score='ALL12model/cutoff bundles firstpublish. Then249rawouterX first38tanh exactly as261,castsame40featuresfloat32 via runtime267.as_input. Parent261continue/ids/logit byte-reused;newcont=parent261continue&(forestscore>=tau),KEEPidentical261. No265delta;no newNN/thetaforward. All12predictions firstpublish beforeoldresultjoin.',evaluation='Primaryfrozen261;sharedGATE/265historicaltradeoff reused. All3seeds16exposeddevelopmentdates/adversecases. Include quality/profitable preservation/actionmatch/gain/query/proxyNET/capacity andinferencecostlimitation,not onlineactualspeed.',change='Gini/forestcapacity differs fromNLL+unitL2;not samecapacity/purecalibration. SamecurrentNET/risk/inputrows/fees,own12guardnot independentOOF. 268separate+22,branch73->95;6historicalcontinuations8285update/604.703sactual/678.156cap separate.',limits='No final8/newphysical/grid/paper/FRAME/install;originalbaselinecontroller/order/aliases/station/128/64/stagnation2/.9 unchanged. No outerchosenrecipe/cutoff/risk/seed,allnegativeoutcomesretain.')
C.save('EXECUTION_PLAN.json',execution);C.save('decision.json',dict(passed=True,ready_for_one_fixed268=True,actual267fits0=True,new_model_capacity_and_giniobjective_disclosed=True,no_claim_of_quality_or_latency=True))
(O/'GRID267_report.md').write_text(f'''# GRID267：条件NET浅树执行前核查

资格通过，真实森林拟合0、真实森林预测0、真实截点校准0、新外层策略0。本阶段没有训练模型，也没有产生改善结果。

12池的原fit8/cal输入和GRID261固定权重已核对发布Git blob。共2859拟合候选行（活跃2725、KEEP134），500条原own12回本STOP保护记录（活跃464、KEEP36）保持。所有原标签、风险权重、样本ID、角色、原40维输入及原模型字段逐项不变；只额外缓存同40维的float32数值表示，供新树模型一致训练和预测。

float32转换最大绝对误差{maxcast:.10g}，新精度下重复输入组{dups}、相反标签精确碰撞组{conf}。该检查不是无信息损失或质量保证。原GRID261入口仍使用原float64保存决定，不改基本控制器。

已有scikit-learn {q['sklearn_version']}的构造接口与固定参数通过，未调用fit或predict_proba，也未安装依赖。配方与旧203完全一致：64树、深度至多4、叶子至少10、max_features=1.0、bootstrap=False、单线程、gini、class_weight=None、warm_start=False；沿用0/1/2种子。只借用配方，不重启旧任务或使用旧事件权重。

下一独立GRID268若执行：11个分类森林+11个原经验正例保护截点，共22拟合/一次600秒、1KEEP。拟合只用自身fit8原NET>0标签与原风险；截点只用原500保护记录。新过滤器仅能进一步取消GRID261的继续决定。全模型截点先发布、全部外层预测先发布后，才能连接旧开发结果；不能按结果调截点、选种子或扩参数网格。

函数类、容量及gini目标相较原NLL+单位L2改变，非同容量或纯校准消融。风险加权叶子频率不等校准概率。新增森林推理未计时，候选查询省不等实际提速。经验保护不保证外层回本，全部不利结果必须保留。

267新增拟合0，当前分支73；未来268另22、累计95。旧6次续训8285更新与原NN604.703秒/用户678.156秒上限另列。父own12/内部cal4祖先、已暴露16日期和相关前缀/种子用途局限不变，非全链OOF/盲测/整周质量证据。最终8、新物理、电网、论文、FRAME未动，无安装。
''',encoding='utf8')
C.archive('audited');D=R/'论文协作资料/FRAME_条件浅树资格_GRID267';D.mkdir(exist_ok=False)
for n in ['GRID267_report.md','qualification.json','independent_audit.json','decision.json','complete.json','EXECUTION_PLAN.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID267唯一条件NET浅树执行资格通过、0真实拟合/森林预测/截点/outer策略。36输入保留原260全部字段、2859候选/2725活跃/134KEEP、500保护(活跃464)和261权重；仅缓存同40维X32，max cast2.98e-8，无精确重复或相反标签碰撞。sklearn1.7.2构造接口和旧203固定64树depth4leaf10单线程gini配方一致，未fit/安装。下一独立268才11森林+11截点/一次600s/1KEEP，704树，全部参数先发/预测先发后结果；原261入口仅进一步取消，outer原249先tanh38再float32，不使用265delta。容量/gini目标与风险频率非校准概率披露，资格不代表效果。当前73，未来另22至95，原祖先/暴露开发/整周时延边界及6旧续训授权时间保持。'
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/n;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-29 — GRID267条件NET浅树资格\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+e+rest
 else:t+=e
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID267后\n\n1. '+msg+' 完整资格/归档不重做。\n2. 下一268只消费已发布EXECUTION_PLAN及qualified_inputs/runtime267，先验证36输入和runtime具体Gitblob/远端，再独立源码输入封存发布后11森林+11截点；总窗口600s含加载/校准预测/核验，不扩窗或重复完整项。所有12组合参数先发，outer249 X同样tanh38后float32，261入口字节复用；全12预测先发后连接结果。\n3. 固定配方/标签权重/全部seed日期/KEEP，禁止扫tau/树数/深度/叶子/风险/费用/正则或按Jan5例外，所有反例保留。原控制/最终8/新物理/电网/论文/FRAME不动，不安装；容量/目标/数值类型变更和95累计计数披露。Githelper串行/路径白名单/实际A归档协作远端核验。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID267条件NET浅树资格\n\n[报告](protocols/'+A.name+'/audited/GRID267_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID267_AUDITED_ZERO_FITS')
