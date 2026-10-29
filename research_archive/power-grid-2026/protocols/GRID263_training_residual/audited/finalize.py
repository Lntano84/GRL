"""First saved own-fit diagnostic check and delivery for GRID263; no optimizer."""
from pathlib import Path
import sys,json,collections,shutil
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.verify();s=C.rd(O/'summary.json');rows=C.rd(O/'training_row_diagnostics.json');pools=C.rd(O/'pool_risk_loss_gradient.json');meta=C.rd(C.META);assert len(rows)==2859 and len(pools)==12
for p in pools:
 z=np.load(C.Q/'fit_inputs'/f"{p['pool']}.npz");m=np.load(C.M/f"{p['pool']}.npz");rr=[r for r in rows if r['pool']==p['pool']];theta=m['theta'];alpha=float(m['alpha'][0]);end=z['b']+np.sum(z['X']*theta[:-1],axis=1)+theta[-1];score=z['b']+alpha*(end-z['b']);objz=z['X']@theta[:-1]+theta[-1]+z['b'];ll=np.logaddexp(0,objz)-z['y']*objz
 for i,r in enumerate(rr):
  assert r['metadata_id']==int(z['ids'][i]) and r['label']==int(z['y'][i]) and r['actual_NET']==float(z['actual_NET'][i])
  assert r['score']==float(score[i]) and r['continue_']==bool(score[i]>=0) and r['next_queries']==meta[int(z['ids'][i])]['next_queries']
  assert abs(r['weighted_NLL']-float(z['weight'][i]*ll[i]/z['weight'].sum()))<1e-14
 assert abs(sum(r['weighted_NLL'] for r in rr)-p['original_mean_NLL'])<1e-12 and abs(.5*(theta@theta)-p['full_unit_L2'])<1e-12
 assert sum(g['rows'] for g in p['groups'])==len(rr)
 if not p['KEEP']:assert p['gradient_inf']<1e-3
active=[r for r in rows if not r['KEEP']];bad=[r for r in active if r['label']==0 and r['continue_']];miss=[r for r in active if r['label']==1 and not r['continue_']]
assert len(active)==2725 and len(bad)==1816 and len(miss)==1 and miss[0]['domain']=='CONT'
assert s['domain']['STOP']['falseCONT_negative']==804 and s['domain']['STOP']['falseSTOP_positive']==0 and s['domain']['STOP']['falseCONT_raw_queries']==6844
assert s['domain']['CONT']['falseCONT_negative']==1012 and s['wrongneg_raw_queries']==21938
C.save('independent_audit.json',dict(passed=True,all2859_savedownfitrows_checked=True,all12objectives_and_groups_conserve=True,gradient11converged=True,STOP804falseCONTand0falseSTOP=True,CONT1012falseCONTand1falseSTOP=True,negative_query21938=True,onlyownfit8_andfixedmodels_meta=True,newfits=0,newalpha=0,newouterpolicyscores_or_results=0,no_NNheadforward=True))
nextplan=dict(id='GRID264',stage='Qualification only;0fits/alpha/newthreshold applied/newouterpolicy scores or outcomes',hypothesis='ONE empirical positive-guarded downward SCORE bias on frozen261, after qualification. This explicitly changes stopping operating point;equivalent to raising score threshold, NOT claimed unchanged threshold or original NLL optimization.',fixed='Freeze261theta/alpha/tanh38/parentb/nativeSTOP/basegate. Keepf0_s1delta0. Same own12newselectedSTOPpositive500(active464) used by original221;no newlabels/weights/risk ratios/seeds. No newnegative querylabel definition.',proposed_single_scalar='For each11activepool:delta=max d>=0 subjectto s261(calpositive)-d>=0 for ALL own12selectedSTOPpositives. Closed form min published-style s261 amongpositive records, equality continues. Maxdelta minimizes any nonnegative known querycost along this monotone one-parameter family. Not asserting outerquality.',qualification264='Verify own12 input/model lineage and role counts; all oldpositive scores nonnegative/finite, exact same floating expression as deployment, delta0KEEP, monotone parentgate, ties and emptyguard predeclared KEEP. Synthetic algebra/unit checks only;do NOT compute actual fitted delta/minimum/maximizer or evaluate shifted policies in264.',future265_ifqualified='11scalarcalibrations+1KEEP,0theta/NNfits,one600swindow,noextension/search. Unique maximalguard rule and exactfloating subtraction s261-delta;no threshold grid or choosingbyouter. ALL12scalars published before4681publicouter scores,ALL12predictionspublishedbefore results. Main261 plus sharedGATE historical,allseedsdatescounterexamples.',accounting='New distinct operating-point hypothesis: if265executes +11,branch62->73,notwithin old40or26122. 6historicalcontinuations/8285updates and604.703s/678.156cap remain separate.',limits='Same own12/cal4ancestry and exposeddevelopment;empirical positiveguard notindependent guarantee. May lose unseen/exposedouterprofitable outcomes,all adverse retained. 0newphysics/final8/grid/paper/FRAME/install. Originalqueryfee/labels/controllers/order/aliases/station/128/64/stagnation2/.9 unchanged.')
C.save('NEXT_PROTOCOL_PLAN.json',nextplan)
C.save('decision.json',dict(passed=True,retain261_quality_preserving_small_gain=True,material_owntraining_residual_negative_query_support=True,not_optimizer_incomplete=True,not_zero_negative_loss=True,no_reweight_or_network_now=True,qualify_one_explicit_guarded_operatingpoint_hypothesis_next=True,actual_newfits0=True))
stop=s['domain']['STOP'];cont=s['domain']['CONT'];share=stop['falseCONT_meanpool_weightedNLL']/stop['all_meanpool_weightedNLL']
text=f'''# GRID263：训练中的残余查询与损失

本阶段只读取原自身fit8输入、标签、风险、固定GRID261模型和训练元数据。没有读取新的外层输入、预测或结果，没有训练、步长拟合或电网回放。

## 剩余空间确实存在

11个活跃池共有2725条相关训练记录，另134条属于预定KEEP池，单列保存。

|训练上下文|回本/不回本记录|不回本但继续|回本但停止|误继续涉及原始查询|占本域不回本查询|
|---|---:|---:|---:|---:|---:|
|STOP|222 / 962|804|0|6844|{stop['falseCONT_query_fraction_of_negative']:.2%}|
|CONT（辅助）|296 / 1245|1012|1|15094|{cont['falseCONT_query_fraction_of_negative']:.2%}|

STOP中尚有804条不回本记录被判继续；222条回本记录全部保留。原始6844查询是这些训练记录对应的已知候选成本，并不是预计可以全部省下的在线查询。辅助CONT单列，不能混成部署STOP性能。重复前缀、种子和模型池用途不独立，不能把1816条误继续当1816个独立场景。

这些误继续合计涉及127个物理调用、12个训练日期，最大调用占原始查询{s['topcall_rawqueryshare']:.2%}、前五占{s['top5calls_rawqueryshare']:.2%}，并非仅一处异常。

## 为什么不能直接说训练没好

固定模型的11个原始梯度最大值{s['active_gradientinf']['max']:.3g}，当前目标已数值收敛。平均原NLL为{s['active_mean_NLL']:.6f}，完整单位L2为{s['active_mean_L2']:.6f}。这不是优化器尚未完成造成的残差。

STOP不回本误继续记录占平均总风险约{100*stop['falseCONT_meanpool_risk']:.2f}%，但贡献STOP域NLL的{share:.2%}；辅助CONT对应平均总风险约{100*cont['falseCONT_meanpool_risk']:.2f}%。因此原目标确实在惩罚这些错误，不能说“模型完全没看到负例”；同时，当前风险与单位L2的取舍仍保留了大量误继续。这里是训练现象，不是证明某一个损失项导致失败。

唯一回本误停出现在辅助CONT：f2_s1、2037-01-19_100的call016，原NET约0.834347、分数约-0.009134。该反例完整保存在positive_falseSTOP.json，不能把STOP无误停扩写成所有上下文无误停。KEEP池134条及全部错误也已保留。

## 下一步固定一个小问题

保留GRID261已确认的小改善。下一GRID264先资格一个明确的新假设：冻结当前模型，只研究是否可用原own12回本STOP经验记录约束一个向下分数偏置，从而在该约束下增加停止。每池只允许唯一的最大可行偏置规则，不扫描多个阈值、风险比例或种子。

这在数学上等价于提高当前分数的停止阈值，属于显式的策略工作点与经验约束修订；不能换个名字就声称阈值没变，也不是继续优化原NLL。264只核对输入祖先、记录集合、浮点表达式、单调性和空集/并列处理，实际拟合0，不提前求真实偏置或新策略分数。资格通过后另阶段265至多11标量、1KEEP、一次600秒窗口，全部标量先发布、全部外层预测先发布后才能连接结果。own12经验保护不保证外层不丢回本，所有反例必须保留。

263新增拟合0，分支仍62；若以后执行265，则明确另加11、累计73。6次旧续训8285增量更新及原NN604.703秒/用户678.156秒上限另列。当前16日期已暴露、原父own12/内部cal4祖先与前缀相关局限不变，不称全链OOF/盲测。查询代理不等整周质量或实际时延，最终8、新物理/电网/论文/FRAME未动，未安装依赖。
'''
(O/'GRID263_report.md').write_text(text,encoding='utf8')
C.save('complete.json',dict(passed=True,ownfitrows2859=True,active2725_KEEP134=True,models=0,scalars=0,newouterpolicyscores=0,newouterresults=0,NNheadforward=0,final8_absent=True))
C.archive('audited');D=R/'论文协作资料/FRAME_训练残余查询_GRID263';D.mkdir(exist_ok=False)
for n in ['GRID263_report.md','summary.json','independent_audit.json','decision.json','complete.json','NEXT_PROTOCOL_PLAN.json','positive_falseSTOP.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID263只自身fit8诊断完成：2859行=2725活跃/134KEEP，STOP222回本/962不回本中804仍继续、0回本误停，涉及6844原始query/95.83%本域不回本query；辅助CONT296回本/1245不回本中1012仍继续、1回本误停，15094query。误继续127调用12训练日期，最大调用4.67%/前五20.08%，非单点。11梯度max7.29e-7，均NLL.215018/L2.049779，STOP误继续占总风险2.73%但STOP域NLL24.32%，不说没看到负例或未收敛。保留辅助CONT大NET误停反例。0拟合/新outer输入预测结果。下一264仅资格唯一冻结261的经验正例保护向下偏置；明确等价提高停止阈值/新工作点目标，不冒充阈值不变；不求真实偏置或新策略。资格后265才11标量/600s/1KEEP，全发布屏障；若执行分支62->73，原祖先/暴露开发/整周时延边界保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID263训练残余查询与损失\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':title,rest=t.split('\n',1);t=title+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID263后\n\n1. '+msg+' 完整263诊断/审计不重做。\n2. GRID264只消费已封存NEXT_PROTOCOL_PLAN，资格冻结261模型后的唯一正例保护向下分数偏置：祖先/own12正例集合与浮点表达式、合成单调性/并列/空集KEEP，0真实偏置求解/新策略/outer标签。明确等价阈值提高与目标变更，不能靠改名绕过。\n3. 资格后另265至多11标量/一次600s，先源输入、全标量、全预测分阶段发布后结果；不扫描偏置/alpha/比例/正则/种子。原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，最终8/新物理/电网/论文/FRAME不动，不安装。现分支62、未来加11为73，不隐藏6续训与授权时间。全部反例保留，小改善不设全赢5%门槛。Githelper串行/路径白名单/实际A归档协作远端核对。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID263训练残余查询与损失\n\n[报告](protocols/'+A.name+'/audited/GRID263_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID263_AUDITED',json.dumps(dict(passed=True,trainonly=True,newfits0=True,STOP_falseCONT804=True,STOP_loss_fraction=share,next='264qualificationonly'),ensure_ascii=False))
