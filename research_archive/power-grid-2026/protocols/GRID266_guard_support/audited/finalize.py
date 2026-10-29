"""First saved readonly GRID266 diagnosis audit and delivery."""
from pathlib import Path
import sys,json,shutil
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();pubsha=C.published('public');s=C.rd(O/'summary.json');rows=C.rd(O/'focus_support_details.json');meta=C.rd(C.META);assert len(rows)==191 and s['passed']
for r in rows:
 z=np.load(C.Q/'fit_inputs'/f"{r['pool']}.npz");raw=np.load(C.RAW/f"{r['pool']}.npz");j=int(np.flatnonzero(raw['ids']==r['metadata_id'])[0]);x=raw['X'][j].copy();x[:38]=np.tanh(x[:38]);assert r['newscore']==r['oldscore']-r['delta'] and r['bounded_correction']==r['oldscore']-r['parent_b']
 assert abs(r['oldscore']-(r['parent_b']+r['alpha']*(r['public38_sum']+r['parent_b_feature']+r['STOP_feature']+r['intercept'])))<1e-12
 for n in r['neighbors']:
  if n.get('missing'):continue
  stop=n['domain']=='STOP';allowed=[i for i,mid in enumerate(z['ids']) if bool(z['domain_STOP'][i])==stop and int(mid)!=r['metadata_id'] and meta[int(mid)]['physical_state_group']!=r['call']];dd=np.sum((z['X'][allowed]-x)**2,axis=1);i=allowed[int(np.argmin(dd))]
  assert i==n['fit_index'] and abs(n['distance']-float(np.sqrt(dd.min())))<1e-12 and n['y']==int(z['y'][i]) and n['actual_NET']==float(z['actual_NET'][i])
assert sum(r['profitable'] for r in rows)==6 and sum(r['actual_next_queries'] for r in rows)==723
assert sum(r['profitable'] and r['bounded_correction']>0 for r in rows)==5
assert s['guardlimiter_fit8']==6 and s['guardlimiter_cal4']==5
C.save('independent_audit.json',dict(passed=True,focused191additiveparts_and_costconservation=True,fixed382nearestpairs_andownlabels_verified=True,all6adversecasesretained=True,old_guard_limiters6fit8_5cal4=True,source_and_publicpair_blobs_verified=pubsha,newfits=0,newscalars=0,newpolicy=0,new_NNforward=0))
params=dict(n_estimators=64,max_depth=4,min_samples_leaf=10,max_features=1.0,bootstrap=False,n_jobs=1,criterion='gini',class_weight=None,warm_start=False)
plan=dict(id='GRID267',stage='QualificationONLY0fits/realforestprobability/calibration/newouterpolicy',hypothesis='ONEconditionalNETclassifiershallowExtraTrees filteronfrozen261entry, rather than uniformdelta. Trainingbasis is263ownfit8 STOP804falseCONT/962negative with222positive retained andconvergedmeanNLL+L2;not choosearchitecturebyouterJan5.',recipe_source='Exact existing203PARAMS andexistingseeds0/1/2;newweights/NETtarget/current40inputs,not reused203eventmodel or performance.',params=params,future_candidate='11ExtraTreesClassifier trainedonlyoriginal260ownfit8 X40/yNET>0/originalsampleweight. Freeze261 networks/theta/alpha/parententry; candidateonlycancels261CONT. f0_s1KEEP.',calibration='11singlecutoffs tau=max t preservingALLsameown12selectedSTOPpositives: tau=min forestweightedclass1score onthosepositive records;tiescontinue. 500positive/active464 sameguard. Newscoreisriskweightedleafclassfrequency,NOT calibratedprobability. No thresholdgrid or tauchosenbyouter;emptypool/guardorclassabsentpredeclaredKEEP,not replace dates/seeds.',qualification267='Onlycheck publishedmodels/data/roles/labels/weights/dtypes/capacity/libraryinterface,0fit. Verify267usesidenticalownfitrows/y/weights from260,originalguard andgate261. Explicit numericalfloat32fit/predict conversion inherited sklearn,record cast error,do not fit scaler or change target. Existing sklearn constructoronly,noinstall.',future268='Ifqualified,11forest+11cutoff=22fits/one600swindow,1KEEP,0newNN/theta. Onefixedrecipe no depth/leaf/trees/seed sweep. All12modelcutoff bundlespublishedbeforeouter,ALL12outer predictionspublishedbeforeoldresultjoin. Primary261,265historicaltradeoffreferenceonly. Extra forestinferencecost unmeasured,querysavingnotruntime.',change='Differentfunctionclass/capacity andginiobjective versus219meanNLL+unitL2;not samecapacity or purecalibration ablation. Keep originalNETlabels/fees/two-domainriskweights/40feature definitions;no newfeature/ratio/NN.',count='2670,currentbranch73. If268executes separate+22=>95,not hiddenwithin26511 ororiginal40. Historical6continuations8285updates/604.703sactual/678.156cap retained.',limits='Own12guard/internalcal4andparentlabelancestry remain;couldstillloseouterprofitable cases,allretained. Exposedcorrelateddevelopment,not wholechainOOF/blind/fullweek/time. No final8/physics/grid/paper/FRAME/install or baselinecontroller/order/aliases/station/128/64/stagnation2/.9change.')
C.save('NEXT_PROTOCOL_PLAN.json',plan);C.save('decision.json',dict(retain261_qualitypreserving_candidate=True,retain265_smallqueryqualitytradeoff_andseed1positive=True,uniformshift_not_stateconditional=True,nearestsupport_not_safehandrule=True,next='GRID267qualificationONLYofonefixedconditionalNETforest',actualnewfits0=True,no_immediate_threshold_adjustment=True))
lost=s['categories']['profitable'];good=s['categories']['nonprofitable']
report=f'''# GRID266：经验保护为何没有保住外层回本

本轮只描述GRID265已经发表的191处取消，不生成替代偏置、分位数、混合策略或新的策略回报，训练/标量拟合0。

## 统一偏置抹去了哪些区别

6条回本损失中，5条原本已被GRID261有界修正向上提高分数，随后又被统一向下偏置取消。6条的原分数范围{lost['oldscore']['min']:.6f}–{lost['oldscore']['max']:.6f}；185条不回本取消范围{good['oldscore']['min']:.6f}–{good['oldscore']['max']:.6f}，区间重叠。按各池偏置归一后的比值也重叠：回本损失中位数{lost['score_to_delta']['median']:.6f}，不回本取消中位数{good['score_to_delta']['median']:.6f}。

这些跨池描述不是线性不可分证明，也不能据此反向找一个“安全比例”。它说明原有向上修正与统一停止力度是不同作用；继续调一个标量无法自动利用完整状态差异。当前6条全部在Jan5两个调用，但不能设日期或调用例外。

## 保护集合与状态支持

11个已拟合偏置各由一条原经验正例卡住：6条来自fit8角色，5条来自cal4角色。所有500条原经验正例仍保留，这是原规则的结果，并非新的独立验证。cal4不是整链独立，原父own12祖先局限沿用GRID235/253既有结论，本轮没有重做宽泛祖先审计。

固定有界40维欧氏最近1、排除同一物理调用，先发布382个公开配对813a40e9，再连接对应自身fit8标签：

|已有取消类别|记录数|最近STOP邻居回本 / 不回本|最近CONT邻居回本 / 不回本|
|---|---:|---:|---:|
|回本误停|6|2 / 4|0 / 6|
|不回本取消|185|8 / 177|4 / 181|

不能直接把最近邻标签变成新手工门控：它会漏掉4条回本误停，也会阻止部分实际有利的取消。距离不是校准置信度，相关前缀不独立。

## 成本与反例保持

185条不回本取消分布79调用、15日期，新增宏净效用+0.000122637；最大调用占这部分宏查询节省{good['maxcall_query_share']:.2%}，前五{good['top5call_query_share']:.2%}。6条回本损失宏净效用-0.001343689，主要由种子2的call001较大收益主导。两类合计仍与GRID265减261逐项守恒，不能用广泛小节省掩盖少数大损失。

保留GRID261质量候选与GRID265查询/质量取舍、种子1小正线索；全部反例完整保存，不选择成功种子部署。

## 下一固定问题

只根据GRID263自身训练里804条不回本误继续、222条回本保留和数值已收敛的事实，下一GRID267资格一个按状态区分的浅树过滤器。固定复用旧GRID203的64树、深度4、叶子至少10、单线程等参数；新目标为原NET>0，输入仍40维，权重与样本保持当前260版本。不是重启旧203实验，也不沿用它的事件模型权重。

原GRID261入口保持，浅树只允许进一步取消。未来若执行268，11个分类森林+11个原own12经验正例保护截点，共22拟合/一次600秒，1KEEP；全模型截点先发、全部外层预测先发后接结果。它改变函数类/容量及训练目标（gini对比原NLL+L2），不称同容量或纯校准消融；风险加权叶子频率也不等校准概率。267只检查数据、角色、已有库接口和固定配方，训练0，不预设能保住外层回本。

当前分支仍73拟合，267新增0；若268执行则另22、累计95。6次旧续训8285更新与原NN604.703秒/用户678.156秒上限另列。最终8、新物理、电网、论文、FRAME未动，不安装依赖。父own12/内部cal4祖先与已暴露16日期、相关前缀/种子用途局限不变，代理查询不等整周质量或实际时延。
'''
(O/'GRID266_report.md').write_text(report,encoding='utf8');C.save('complete.json',dict(passed=True,focused191=True,publicpairs382=True,alladverse6retained=True,newfits0=True,newscalars0=True,newpolicy0=True,branch73=True,final8_absent=True));C.archive('audited')
D=R/'论文协作资料/FRAME_经验保护缺口_GRID266';D.mkdir(exist_ok=False)
for n in ['GRID266_report.md','summary.json','focus_support_details.json','existing_guard_distribution.json','guard_limiting_records.json','independent_audit.json','decision.json','NEXT_PROTOCOL_PLAN.json','complete.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID266只读191变化/382固定近邻完成：6回本损失中5条原有界修正为正，统一偏置随后取消；回本原分数.075739-.423533与185不回本.004545-.511294重叠，不作不可分或安全比例证明。11偏置限制记录6fit8/5cal4，500经验保护非独立保证。最近STOP邻居回本/不回本：6损失2/4，185取消8/177；CONT0/6与4/181，不能当手工安全门。185取消79调用15日期，最大宏查询占5.61%/前五24.04%，小节省广泛但大损失主导，保留261及265seed1小正和全部反例。下一267仅资格唯一固定条件NET浅树，依据263自身训练804误继续/222正保留，复用203固定64树depth4leaf10单线程配方；原40输入/NET标签/权重，冻结261入口，未来268仅11森林+11正例截点/600s、1KEEP。容量/gini目标变更与风险频率非校准概率披露，非重启203，不按outer挑参数。当前73，未来另22至95；0新拟合/策略，原祖先/暴露/整周时延边界保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID266经验保护支持缺口\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID266后\n\n1. '+msg+' 完整诊断核验不重复。\n2. 下一267仅资格消费NEXT_PROTOCOL_PLAN：核对原260fit/cal行与NET标签/权重/保护集合、261固定入口、固定ExtraTrees配方与已有sklearn构造接口、float32训练/预测一致转换；0真实森林/截点/outer预测。禁止扫描配置/截点/风险比或按Jan5例外。\n3. 资格后另268最多11森林+11截点/一次600s，全部模型参数先发/全预测先发后连旧结果；1KEEP，不动旧控制/最终8/新物理/电网/论文/FRAME，不安装。新容量/目标/数值类型变更披露，current73未来95及旧续训时间另列。Githelper串行/路径白名单/实际A归档协作远端核验。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID266经验保护支持缺口\n\n[报告](protocols/'+A.name+'/audited/GRID266_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID266_AUDITED_NO_FITS')
