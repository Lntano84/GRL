"""First saved-input audit for zero-fit GRID270 union qualification."""
from pathlib import Path
import sys,json,shutil,pickle
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'independent_audit.json').exists();C.verify();q=C.rd(O/'qualification.json');ctx=C.rd(O/'added_context_semantics.json');refs=C.rd(O/'model_references.json');assert q['passed'] and q['actual_model_fits']==q['actual_cutoff_fits']==q['actual_forest_probabilities']==q['new_outer_policy_evaluations']==0
assert len(refs)==12 and len(q['pools'])==12 and q['total_positive543']==543 and q['active507']==507
for r in q['pools']:
 pool=r['pool'];src=C.Q/'union_inputs'/f'{pool}.npz';dst=O/'qualified_inputs'/f'{pool}.npz';assert C.sha(src)==C.sha(dst);u=np.load(dst);assert len(u['ids'])==r['positive_guards'] and (u['y']==1).all() and (u['actual_NET']>0).all() and np.array_equal(u['X32'],u['X'].astype(np.float32)) and (u['X'][:,39]==1).all() and r['exact39coordinates_b_roles_cache'] and r['parent261_all_continue'] and r['own12_excludes_outer4']
for r in refs:
 assert C.sha(R/r['model_relative_to_R'])==r['model_sha256'] and C.sha(R/r['public_prediction_relative_to_R'])==r['public_prediction_sha256'];b=pickle.loads((R/r['model_relative_to_R']).read_bytes());assert b['sklearn_version']=='1.7.2' and b['tau']==r['old_tau'] and b['KEEP']==r['KEEP']
assert len(ctx['records'])==43 and ctx['reference_bit0']==43 and ctx['reference_auxCONT']==43 and ctx['reference_bit1']==ctx['reference_afterSTOP']==0
for r in ctx['records']:assert r['actual_NET12_STOPbit']==1 and r['reference_NN8_nativebit']==0 and r['all39continuous_coordinates_exact'] and r['original_field_not_overwritten']
for rel,h in C.rd(O/'qualified_freeze.json').items():assert C.sha(O/rel)==h
assert C.sha(O/'runtime267.py')==C.sha(C.RUNTIME)
C.save('independent_audit.json',dict(passed=True,all12qualifiedguardfiles_byteidentical_to269=True,positive543_active507_KEEP36=True,original500_added43_unchanged=True,all43continuous39_exact_refbit0_actualbit1_explained=True,all12oldmodel_andpublicprediction_hashes_unchanged=True,sklearn_version_1_7_2=True,actual_model_fits=0,actual_cutoff_fits=0,actual_forest_probabilities=0,new_outer_policy_evaluations=0))
execution=dict(id='GRID271',requires='Validate270qualified_freeze/runtime_freeze/model_references andEXECUTION_PLAN/decision specificGitblobs/remote,thennew271sourceinputfreeze published BEFORErealcalibration',data='Only270qualified_inputs pool.npz positive-only543(current500+legacy43);507active/36KEEP,allparent261reachable. SameX32/ids/yNET/roles,original500subset intact;43fixedown12source beforeoutcomeanalysis,notouterfailureIDs.',models='ReferenceALL12immutable268modelbundles bymodel_references SHA/Gitpath,704trees unchanged,0.fit/0thetaNN training. Runtime267positive_score/protect_cutoff/apply_gate unchanged;sklearn1.7.2 pinned. f0_s1KEEP oldtau0.',calibration='All11activepools once,one600swindow includesmodel loads/unionforestpredict/checks. Newtau=minscore onALLselectedpositiveguard;tiescontinue. No poolchosenbyobservedtauchange, no tau/ratio/tree/risk/fee/seed sweep. Verify543empiricalpreservation andnewtau<=oldtau;KEEPidentity. Calibrationforestscoresonlyinside this component,not during270.',count='11newscalarcalibrations+1KEEP,0forestthetaNNfits. Distinct+11,branch95->106. 6oldcontinuations8285updates/NN604.703actual/678.156cap separately retained;addedconstraintsampleexposure500->543 disclosed.',publication='ALL12newcutoff+immutablemodelreference bundles published beforeouter decisions. Byte-reuse268publishedforest_score/ids/parent261_continue andoldcontinue,0newouterforest/NNforward. NewCONT=parent261CONT & score>=newtau,KEEPidentical261;assertold268CONTsubsetnewCONTsubset261CONT. ALL12predictionspublishedbeforeoldresultjoin.',controls='Primaryfrozen268,261qualityanchor/GATE/265historicalreuse,no refits ornewFRAME. All3seeds16exposeddevelopmentdatesandadversecases;KEEPexposurecaveatremains. No seed/subsetdeploymentselection.',numerics='InputsX32 unchanged. Existingfull256cache continuousfirst39/b/roleexact forALL543;43legacyaddedreferenceNN8CONTbit0 butactualNET12STOPbit1from249nativeproof,sourcecontextbit intentionallyretained,notundocumentedfieldchange. No newfeatures/scaler/target/weights.',limits='Sameown12/internalcal4/upstreamancestry andexposedcorrelateddata,empiricalguardnotouterquality/NET/wholeweek/timeguarantee. Morequerypossible. No final8/physical/grid/paper/FRAME/install,baselinecontroller/order/aliases/station/128/64/stagnation2/.9 unchanged.')
C.save('EXECUTION_PLAN.json',execution);C.save('decision.json',dict(passed=True,ready_for_one_fixed271=True,actual_model_fits=0,actual_cutoff_fits=0,actual_forest_probabilities=0,not_a_quality_result=True))
(O/'GRID270_report.md').write_text('''# GRID270：543条并集保护输入资格

资格通过。本阶段模型拟合0、真实森林分数计算0、真实截点拟合0、新外层策略0，没有新的效果结果。

原500条正例与固定来源43条补充按逐池ID去重后为543条正例约束，活跃507、KEEP36。12份并集输入与GRID269已发布文件字节一致；原NET正标签、内部角色、父入口可达性和对应外层4日期排除保持。它们是正例约束行，不是原校准全集2054行，也不是新仿真样本。

## 数值来源与停止上下文

沿GRID256实际O/A和GRID257 PUBLIC常量解析已有public_full缓存，12份缓存的Git blob与原generated_freeze核对。全部543条的前39个连续坐标、父b、内部角色逐项完全一致，float32转换也一致；没有重新推理网络、收益头或修改字段来凑对齐。

补充43条在参考NN8缓存中全部属于采样CONT、停止位为0；在旧GRID249已验证的原NET12部署停止点中，停止位为1。不同来源的网络停止判定不同，正是集合不嵌套的来源之一。并集保留实际部署STOP上下文1，明确单列这个含义；没有把参考位偷偷改成1后宣称40列全部相同。其他39列保持原值。原500参考停止点记录也完整保留。

原GRID268的12份模型引用、704棵树和旧公开预测哈希保持，未做新森林预测。已有runtime267数值函数原样冻结。合成数据验证了：扩大正例保护集合时，新最大可行截点不高于旧截点，决定只能恢复部分GRID268继续、仍受GRID261入口限制。这不是外层质量或净效用保证。

## 下一独立校准

GRID271若执行，只在完全冻结的森林上校准11个截点，一次600秒，1KEEP。11个活跃池都执行同一规则，不先看截点是否变化再选择池，不扫阈值、比例或种子。全部12组新截点及模型引用先发布，才用GRID268已经公开的森林分数产生新决定；全部预测先发布后再连接旧开发结果，不需要新增外层森林推理。

新增43条约束暴露与截点经验目标修订必须披露，不能称同校准样本对照。当前分支95拟合，270新增0；后续271另11、累计106，原森林不重训。6次旧续训8285更新、原NN604.703秒/用户678.156秒上限另列。父own12/内部cal4祖先、已暴露16日期和相关前缀/种子用途局限保持。最终8、新物理、电网、论文、FRAME未动，无安装；候选查询代理不等整周质量或真实时延。
''',encoding='utf8')
C.archive('audited');D=R/'论文协作资料/FRAME_并集保护资格_GRID270';D.mkdir(exist_ok=False)
for name in ['GRID270_report.md','qualification.json','added_context_semantics.json','independent_audit.json','decision.json','complete.json','EXECUTION_PLAN.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID270固定543正例并集输入资格通过、0模型/真实森林分数/新截点/outer策略。原500+43、活跃507/KEEP36，12输入与269字节一致；543前39连续坐标/b/角色对已有256public_full逐项精确，float32一致。43补充在参考NN8均为采样CONT(bit0)，旧实际NET12STOP为bit1；保留部署STOP上下文1并明确语义差异，未偷偷改其他字段。704树/12旧模型和旧预测hash不变，runtime267与合成单调性通过。下一独立271仅冻结森林校准11截点/一次600s/1KEEP，所有活跃池同规则，先全截点模型引用发布、再复用268公共分数新决定、全预测先发后旧结果，0新outer森林forward；不扫tau/择seed。当前95，未来另11至106、43约束暴露/旧续训授权时间披露。资格非效果/独立校准保证，原祖先/暴露/整周时延边界保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-29 — GRID270并集保护输入资格\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+e+rest
 else:t+=e
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID270后\n\n1. '+msg+' 完整资格/核验/归档不重做。\n2. 下一271只消费270qualified_inputs/model_references/runtime267/EXECUTION_PLAN，先核对12输入/原模型引用/旧预测/runtime具体Gitblob与远端，再独立源码输入封存发布后11截点一次600s。原704树不fit；真实543森林分数只在271校准窗口内计算，不能另加预览拟合。\n3. 全12新截点/模型引用先发后复用268forest_score和parent261_continue生成决定，全预测先发后结果；保证old268CONT子集newCONT子集261CONT，KEEP一致。不扫tau/风险/费用/树/seed或加入外层损失ID，所有反例保留；当前95未来106和43约束暴露/旧续训时间披露，最终8/新物理/电网/论文/FRAME不动，不安装。Githelper串行白名单/实际A归档协作远端核验保持。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID270并集保护输入资格\n\n[报告](protocols/'+A.name+'/audited/GRID270_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID270_AUDITED_ZERO_FITS')
