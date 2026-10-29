"""First saved-result audit; no fitting, no forest or NN inference."""
from pathlib import Path
import sys,pickle,json,collections,shutil,hashlib
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,P=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.scored();assert C.read(O/'review.json')['passed'];rr=C.read(O/'calibration_receipts.json');refs=C.read(C.Q/'model_references.json');protected=active=rows=changedtau=0
for ref in refs:
 p=ref['pool'];u=np.load(C.Q/'qualified_inputs'/f'{p}.npz');b=C.read(O/'cutoffs'/f'{p}.json');cal=np.load(O/'calibration_predictions'/f'{p}.npz');oldcal=np.load(C.N/'calibration_predictions'/f'{p}.npz');old=np.load(R/ref['public_prediction_relative_to_R']);new=np.load(O/'outer_predictions'/f'{p}.npz');keep=ref['KEEP'];tau=b['tau'];pos=cal['positive']
 assert C.sha(R/ref['model_relative_to_R'])==ref['model_sha256']==b['model_sha256'] and b['model_git_path']==ref['model_git_path'] and C.sha(C.Q/'qualified_inputs'/f'{p}.npz')==b['guard_input_sha256']
 assert b['KEEP']==keep and b['old_tau']==ref['old_tau'] and np.array_equal(cal['ids'],u['ids']) and pos.all() and np.isfinite(cal['score']).all() and ((cal['score']>=0)&(cal['score']<=1)).all() and np.array_equal(cal['parent_continue'],u['parent261_score']>=0)
 assert np.array_equal(cal['continue_'],C.RT.apply_gate(cal['parent_continue'],cal['score'],tau,KEEP=keep)) and cal['continue_'].all()
 if keep:assert p=='f0_s1' and tau==0 and (cal['score']==1).all()
 else:
  assert tau==float(cal['score'].min()) and tau<=b['old_tau'] and np.any(cal['score']<np.nextafter(tau,np.inf));active+=len(pos)
 idx={int(v):i for i,v in enumerate(oldcal['ids'])}
 for j,mid in enumerate(u['ids']):
  if u['origin_current'][j]:assert oldcal['positive'][idx[int(mid)]] and cal['score'][j]==oldcal['score'][idx[int(mid)]]
 for key in ['ids','forest_score','base_CONT','parent261_continue','parent261_logit']:assert new[key].dtype==old[key].dtype and new[key].shape==old[key].shape and new[key].tobytes()==old[key].tobytes(),(p,key)
 assert np.array_equal(new['gate_margin'],old['forest_score']-tau) and np.array_equal(new['continue_'],C.RT.apply_gate(old['parent261_continue'],old['forest_score'],tau,KEEP=keep)) and not np.any(old['continue_']&~new['continue_']) and not np.any(new['continue_']&~old['parent261_continue'])
 if keep:assert np.array_equal(new['continue_'],old['continue_'])
 protected+=len(pos);rows+=len(new['ids']);changedtau+=tau<ref['old_tau']
assert protected==543 and active==507 and rows==4681 and len(rr)==12 and sum(not r['KEEP'] for r in rr)==11
for phase,freeze,pubname in [('calibrated','calibrated_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
 pub=C.read(O/pubname);assert pub['passed'] and C.git('show',pub['git_sha']+':'+(A/phase/freeze).relative_to(P).as_posix())==(O/freeze).read_bytes()
 for rel,h in C.read(O/freeze).items():assert hashlib.sha256(C.git('show',pub['git_sha']+':'+(A/phase/rel).relative_to(P).as_posix())).hexdigest()==h
tr=C.read(O/'oof_trajectories.json');cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];fresh=[r for r in tr if r['method'].startswith('UNION_FOREST_')]
assert len(fresh)==7134 and len(tr)==14268 and [r for r in tr if r['method'].startswith('GATE_')]==[r for r in C.read(C.BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')]
for r in fresh:
 c=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:end].sum()) and r['queries']==int(c['queries'][:end].sum()) and r['chosen_pool_id']==c['chosen'][end] and r['batches']==end and r['actions']==[1]*end+([0] if end<len(c['states']) else []) and r['action_match']==(r['chosen_pool_id']==c['chosen'][-1]);assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
metrics=C.read(O/'search_metrics.json');family=C.read(O/'family_summary.json');nf=family['UNION_FOREST'];new=C.read(O/'gate_predictions.json')
def match(folder,arm,name):
 oldpred={(r['fold'],r['seed'],r['sample_id'],r['position']):r for r in C.read(folder/'gate_predictions.json')};prevmetrics=C.read(folder/'search_metrics.json');oldfam=C.read(folder/'family_summary.json')[arm];changes=[];paired=[]
 for r in new:
  z=oldpred[(r['fold'],r['seed'],r['sample_id'],r['position'])]
  for k in ['base_continue','actual_next_NET','actual_next_gain','actual_next_queries']:assert r[k]==z[k]
  if r['continue_']!=z['continue_']:
   sign=int(r['continue_'])-int(z['continue_']);changes.append(dict(**r,old_continue=z['continue_'],new_continue=r['continue_'],NET_delta=sign*r['actual_next_NET'],query_delta=sign*r['actual_next_queries'],gain_delta=sign*r['actual_next_gain']))
 for seed in range(3):
  n=metrics[f'UNION_FOREST_seed{seed}'];z=prevmetrics[f'{arm}_seed{seed}'];d={k:n['macro'][k]-z['macro'][k] for k in n['macro']};nd={v['date']:v['cases'] for v in n['per_date']};ss=[v for v in changes if v['seed']==seed]
  for k,f in [('mean_net_utility','NET_delta'),('mean_queries','query_delta'),('mean_gain','gain_delta')]:assert abs(sum(v[f]/(16*nd[v['date']]) for v in ss)-d[k])<1e-12
  paired.append(dict(seed=seed,macro_delta=d))
 categories=[]
 for label,sign,positive in [('profitable_restorations',1,True),('nonprofitable_restorations',1,False),('profitable_cancellations',-1,True),('nonprofitable_cancellations',-1,False)]:
  ss=[v for v in changes if (int(v['new_continue'])-int(v['old_continue']))==sign and (v['actual_next_NET']>0)==positive];v=dict(category=label,records=len(ss),raw_query_delta=sum(t['query_delta'] for t in ss),calls=len({t['call'] for t in ss}),dates=len({t['date'] for t in ss}),macro_NET_delta=0.,macro_query_delta=0.)
  for t in ss:
   nd={r['date']:r['cases'] for r in metrics[f"UNION_FOREST_seed{t['seed']}"]['per_date']};den=3*16*nd[t['date']];v['macro_NET_delta']+=t['NET_delta']/den;v['macro_query_delta']+=t['query_delta']/den
  categories.append(v)
 assert abs(sum(v['macro_NET_delta'] for v in categories)-(nf['mean_net_utility']-oldfam['mean_net_utility']))<1e-12 and abs(sum(v['macro_query_delta'] for v in categories)-(nf['mean_queries']-oldfam['mean_queries']))<1e-12
 result=dict(old_family=oldfam,new_family=nf,family_delta={k:nf[k]-oldfam[k] for k in nf},per_seed=paired,changes=len(changes),categories=categories,raw_query_delta=sum(v['query_delta'] for v in changes))
 C.save('matched'+name+'_changes.json',changes);C.save('matched'+name+'_summary.json',result);return result,changes
m268,c268=match(C.N,'NET_FOREST','268');m261,c261=match(R/'outputs/grid261_bounded_basis','BOUNDED','261');assert all(r['new_continue'] for r in c268) and all(not r['new_continue'] for r in c261)
restored=[r for r in c268 if r['actual_next_NET']>0];nonprofit=[r for r in c268 if r['actual_next_NET']<=0];misses=[r for r in new if r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0];C.save('profitable_restored.json',restored);C.save('nonprofitable_restored.json',nonprofit);C.save('profitable_misses.json',misses)
counts=C.read(O/'gate_counts.json');assert counts['profitable_retained']==45 and len(restored)==1 and len(misses)==10
oldpaths={(int(r['method'].split('seed')[-1]),r['sample_id']):r for r in C.read((R/'outputs/grid261_bounded_basis')/'oof_trajectories.json') if r['method'].startswith('BOUNDED_')};exposure=[]
for seed in range(3):
 ss=[r for r in fresh if r['method']==f'UNION_FOREST_seed{seed}'];exp=dict(seed=seed,paths=len(ss),gain_changed=sum(r['gain']!=oldpaths[seed,r['sample_id']]['gain'] for r in ss),chosen_id_changed=sum(r['chosen_pool_id']!=oldpaths[seed,r['sample_id']]['chosen_pool_id'] for r in ss),active_dates=sorted({r['date'] for r in ss if f"f{r['fold']}_s{seed}"!='f0_s1'}),KEEP_dates=sorted({r['date'] for r in ss if f"f{r['fold']}_s{seed}"=='f0_s1'}));exposure.append(exp)
assert exposure[1]['gain_changed']==exposure[1]['chosen_id_changed']==0 and len(exposure[1]['active_dates'])==12
C.save('per_seed_preservation_and_exposure.json',exposure)
C.save('independent_audit.json',dict(passed=True,frozen_model_hashes12_704trees_unchanged=True,original500_scores_exactly_reproduced_from_saved_arrays=True,all543guard_rows_active507=True,all11cutoffs_maxfeasible_on_saved_scores=True,old268CONT_subset_newCONT_subset261CONT=True,all4681outer_arrays_byte_reused=True,all7134new_savedpaths_checked=True,oldGATE_reused=True,two_matched_macroconservations=True,allpublication_blobs_verified=True,new_forests_theta_NNfits=0,new_outer_forestforward=0,new_calibration_inference_in_audit=0,all3seeds16dates_adversecases_retained=True))
C.save('decision.json',dict(keep261_quality_maincandidate=True,retain271_single_local_recovery_signal=True,do_not_promote271_over261=True,profitable45of55=True,changed_cutoffs=changedtau,seed1local_preservation_with_KEEP_caveat=True,next='GRID272readonly_fixedunion_limits_and_training_support',branch106=True))
plan=dict(id='GRID272',scope='Readonly published271changes/residuals andownfit/calibrationsupport. 0fit/scalar/alternativepolicy/forestNNforward.',focus='Only13restorations/1profitable12nonprofitable and10remainingprofitablecancellations vs261;fixedold/newcutoffs andunion limitingrows. Explain why added43 onlylowered2cutoffs,do notclaim coveringoldSTOP solvesouterloss.',support='Ownpool270543positiveguard+saved271calibrationscores andold267fitinputs/labels/weights;public40contextsemantics/roles/source500vs43. Fixedtreeweights can be inspected structurally, no new predictions/leaf policy. No broadancestryreaudit;reuse235/253.',rules='No alternative tau/min/quantile/threshold/unionselection/leafgate or returns,no outerfailureIDsexceptions/seedchoice. Only owntraininglabels andpublicsupport toqualify ONEfinitehypothesis beforefutureexecution;do not immediately addNN/features or scan.',count='Current106=40+22+11+22+11;6historiccontinuations8285updates/604.703actual/678.156cap separate. Anynewlearning<=22/one600s andcounts disclosed.',limits='Expandedguard500->543 notsamecalibrationsample,own12/internalcal4 notindependent,old16developmentdatescorrelated/exposed,queryproxy notfullweek/actualtime. No final8/physics/grid/paper/FRAME/install.')
C.save('NEXT_PROTOCOL_PLAN.json',plan)
lines=['# GRID271：冻结森林的并集正例保护校准','','冻结GRID268全部704棵树，只将保护集合从500条扩到预先固定的543条（活跃507、KEEP36）。11个活跃池各校准一次最大可行截点，另1池KEEP；无森林、网络或线性修正器重训。一次600秒窗口实际%.6f秒。'%C.read(O/'fits.json')['wall_s'],'','## 相对GRID268','','11池都执行同一规则，仅f0_s2和f1_s2两个截点降低，其余不变。13处决定变化全部恢复补查，其中1条回本、12条不回本；没有新增取消或回本损失。回本保留44/55→45/55，仍少于质量锚GRID261的55/55。','', '|种子|相对268代理净效用变化|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for v in m268['per_seed']:
 d=v['macro_delta'];lines.append(f"|{v['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|")
lines+=['',f"宏平均NET {m268['old_family']['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}；候选查询{m268['old_family']['mean_queries']:.6f}→{nf['mean_queries']:.6f}，原始净增{m268['raw_query_delta']}次。恢复收益伴随更多查询，不能称等预算收益。",'', '|恢复类别|记录数|原始查询变化|宏NET变化|','|---|---:|---:|---:|']
for v in m268['categories']:
 if v['records']:lines.append(f"|{v['category']}|{v['records']}|{v['raw_query_delta']}|{v['macro_NET_delta']:+.9f}|")
lines+=['','恢复回本记录：'+json.dumps(restored,ensure_ascii=False),'','## 相对质量锚GRID261','',f"仍取消{len(c261)}条记录，含10条回本。宏NET {m261['old_family']['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}；候选查询{m261['old_family']['mean_queries']:.6f}→{nf['mean_queries']:.6f}。以下全部种子保留，不择优替换：",'', '|种子|相对261 NET变化|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for v in m261['per_seed']:
 d=v['macro_delta'];lines.append(f"|{v['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|")
lines+=['','种子1的2378条保存路径增益与最终候选ID保持261，省查询线索保留；但f0_s1预定KEEP，四个日期未启用森林，实际森林启用12日期，不能宣称16日期全由新森林保住质量或纯随机种子对照。种子2仍主导大收益损失。全部10条残余见profitable_misses.json，不能因这些外层失误再加对应ID救校准。','', '并集缺口修复在开发比较中只恢复1条回本，说明该覆盖缺口不是全部损失的解释；本轮是一个小的局部恢复，不替换261。43条补充约束不等43独立新样本，当前原500仍保留。参考NN8 CONT位0与实际NET12 STOP位1的语义差异沿270明确披露；其他39坐标保持一致，输入未随结果修改。','','## 核验与边界','','全部543条经验正例保护、11唯一最大可行截点/旧新单调性、旧500分数与保存结果完全一致、704树哈希不变、4681条公开分数/父入口数组字节复用、7134条新路径动作/候选/成本/查询及两匹配宏贡献守恒均通过。审计不重算森林概率，不重训或重新评分外层。源19be870b→全部截点0cee1a26→全部预测b1f7700d严格先发布后结果连接。','', '本轮新增11标量，分支95→106。新增500→543约束暴露不是同校准样本对照；6次历史续训8285更新/原NN604.703秒实际及用户678.156秒上限另列。父own12/内部cal4祖先局限、已暴露16日期及相关前缀/种子用途保持，经验保护不是外层质量或独立校准保证，不称全链OOF/盲测。代理净效用/查询不等整周成本或实际控制时延。最终8、新物理/电网/论文/FRAME未动，未安装，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9不变。','', '下一GRID272只读解释13恢复、10残余与新增约束的限制记录/自身训练支持；不拟合新阈值、叶子门、混合策略或替代回报，先固定唯一有限后续假设。']
(O/'GRID271_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.save('complete.json',dict(passed=True,new_scalar_fits=11,newforestthetaNNfits=0,KEEP=1,wall_s=C.read(O/'fits.json')['wall_s'],guard543=True,scores4681=True,new_savedpaths7134=True,profitable45of55=True,branch106=True,final8_absent=True));C.archive('audited')
D=R/'论文协作资料/FRAME_并集保护校准_GRID271';D.mkdir(exist_ok=False)
for name in ['GRID271_report.md','matched268_summary.json','matched261_summary.json','profitable_restored.json','profitable_misses.json','decision.json','independent_audit.json','complete.json','NEXT_PROTOCOL_PLAN.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg=f"GRID271冻结704树校准11并集截点/1KEEP完成({C.read(O/'fits.json')['wall_s']:.6f}s)，0森林thetaNN重训；原500->543经验正例保护。仅2截点降低，13决定全恢复(1回本12不回本)，44->45/55，原始query+{m268['raw_query_delta']}，NET{m268['old_family']['mean_net_utility']:.9f}->{nf['mean_net_utility']:.9f}，query{m268['old_family']['mean_queries']:.6f}->{nf['mean_queries']:.6f}。seed0/1不变，seed2小正伴随query增加；相对261仍10回本误停/NET整体较低，保留261质量锚与271小恢复，不择seed/改tau。全543保护/4681数组复用/7134路径/单调性/宏守恒核验，源19be870b/截点0cee1a26/预测b1f7700d先发后结果。下一272只读13恢复/10残余及own训练约束支持，不新策略或阈值/叶子门；分支106和43新增约束暴露、旧续训时间披露。祖先/暴露/整周/实际时延边界保持。"
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID271冻结森林并集校准\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID271后\n\n1. '+msg+' 完整校准/评分/审计/归档不重做。\n2. 下一272按NEXT_PROTOCOL_PLAN只读固定13恢复/1回本12无利和10残余、两个截点的新限制正例/来源/角色/自身训练支持；0拟合/真实新概率/替代tau/策略。复用235/253祖先结论，不重复宽泛审计；不得添加外层失误ID或择seed/调用例外。\n3. 仅自身训练标签/公开支持资格唯一下一有限假设，<=22/600秒，不盲扫/拆搜索，不立即加特征/网络。保留261质量锚、268/265取舍与258/250/230历史，全部反例/KEEP范围保留；最终8/新物理/电网/论文/FRAME不动，Githelper串行白名单/实际A归档/协作远端核验。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID271冻结森林并集校准\n\n[报告](protocols/'+A.name+'/audited/GRID271_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID271_AUDITED',json.dumps(dict(changed_cutoffs=changedtau,counts=counts,matched268=m268,restored=restored,misses=len(misses)),ensure_ascii=False))
