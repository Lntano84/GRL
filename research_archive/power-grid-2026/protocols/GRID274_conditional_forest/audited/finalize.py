"""First audit of saved fixed context augmentation comparison, no refit/inference."""
from pathlib import Path
import sys,json,pickle,shutil,collections
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,P=C.R,C.O,C.A,C.REPO
assert not (O/'complete.json').exists();C.scored();assert C.read(O/'review.json')['passed'];rr=C.read(O/'fit_receipts.json');refs=C.read(C.Q/'model_references.json');rows=protected=trees=newtrees=0;newnodes=newleaves=0
for ref in refs:
 p=ref['pool'];base=np.load(C.Q/'qualified_inputs'/f'{p}_base.npz');cal=np.load(C.Q/'qualified_inputs'/f'{p}_cal.npz');fit=np.load(C.Q/'qualified_inputs'/f'{p}_fit.npz');b=pickle.loads((O/'models'/f'{p}.pkl').read_bytes());saved=np.load(O/'calibration_predictions'/f'{p}.npz');pred=np.load(O/'outer_predictions'/f'{p}.npz');parent=np.load(C.PUBLIC/f'{p}.npz');oldpred=np.load(R/'outputs/grid271_union_guard/outer_predictions'/f'{p}.npz');keep=ref['KEEP'];changed=ref['changed'];tau=b['tau'];r=next(v for v in rr if v['pool']==p)
 assert r['new_forest_fit']==r['new_cutoff_fit']==changed and b['context_augmented']==changed and b['KEEP']==keep and b['original_model_sha']==ref['model_sha256'] and b['base_model_sha']==C.sha(C.Q/'qualified_inputs'/f'{p}_base.npz') and np.array_equal(b['base_theta'],base['theta']) and np.array_equal(b['base_alpha'],base['alpha'])
 if not keep:
  clf=b['model'];assert len(clf.estimators_)==64 and all(clf.get_params()[k]==v for k,v in C.RT.PARAMS.items()) and clf.get_params()['random_state']==b['seed']
  for t in clf.estimators_:
   leaf=t.tree_.children_left==-1;assert t.tree_.max_depth<=4 and (t.tree_.n_node_samples[leaf]>=10).all()
   if changed:newnodes+=int(t.tree_.node_count);newleaves+=int(leaf.sum())
  trees+=64
  if changed:newtrees+=64;assert tau==float(saved['score'].min()) and np.any(saved['score']<np.nextafter(tau,np.inf))
 else:assert p=='f0_s1' and b['model'] is None and tau==0
 if not changed:
  old=pickle.loads((R/ref['model_relative_to_R']).read_bytes());oldcal=np.load(R/'outputs/grid271_union_guard/calibration_predictions'/f'{p}.npz');assert tau==ref['fixed271_cutoff']['tau'] and np.array_equal(saved['score'],oldcal['score'])
  if not keep:
   for a,z in zip(b['model'].estimators_,old['model'].estimators_):
    assert a.tree_.node_count==z.tree_.node_count
    for field in ['feature','threshold','children_left','children_right','value','n_node_samples','weighted_n_node_samples']:assert np.array_equal(getattr(a.tree_,field),getattr(z.tree_,field))
 assert np.array_equal(saved['ids'],cal['ids']) and saved['positive'].all() and np.isfinite(saved['score']).all() and ((saved['score']>=0)&(saved['score']<=1)).all() and np.array_equal(saved['parent_continue'],cal['parent261_score']>=0) and np.array_equal(saved['continue_'],C.RT.apply_gate(saved['parent_continue'],saved['score'],tau,KEEP=keep)) and saved['continue_'].all()
 for field,pf in [('ids','ids'),('base_CONT','base_CONT'),('parent261_continue','continue_'),('parent261_logit','logit')]:assert np.array_equal(pred[field],parent[pf])
 assert np.isfinite(pred['forest_score']).all() and ((pred['forest_score']>=0)&(pred['forest_score']<=1)).all() and np.array_equal(pred['gate_margin'],pred['forest_score']-tau) and np.array_equal(pred['continue_'],C.RT.apply_gate(parent['continue_'],pred['forest_score'],tau,KEEP=keep))
 if not changed:
  for field in pred.files:assert np.array_equal(pred[field],oldpred[field])
 protected+=len(cal['ids']);rows+=len(pred['ids'])
assert protected==543 and rows==4681 and trees==704 and newtrees==576 and sum(r['new_forest_fit'] for r in rr)==9
for phase,name,pub in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
 h=C.read(O/pub)['git_sha'];assert C.git('show',h+':'+(A/phase/name).relative_to(P).as_posix())==(O/name).read_bytes()
 for rel,s in C.read(O/name).items():assert __import__('hashlib').sha256(C.git('show',h+':'+(A/phase/rel).relative_to(P).as_posix())).hexdigest()==s
tr=C.read(O/'oof_trajectories.json');fresh=[r for r in tr if r['method'].startswith('CONTEXT_FOREST_')];assert len(fresh)==7134 and len(tr)==14268;cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
for r in fresh:
 z=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(z['gains'][:end].sum()) and r['queries']==int(z['queries'][:end].sum()) and r['chosen_pool_id']==z['chosen'][end] and r['batches']==end and r['actions']==[1]*end+([0] if end<len(z['states']) else []) and abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
assert [r for r in tr if r['method'].startswith('GATE_')]==[r for r in C.read(C.BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')]
metrics=C.read(O/'search_metrics.json');nf=C.read(O/'family_summary.json')['CONTEXT_FOREST'];new=C.read(O/'gate_predictions.json');exec((Path(__file__).parent/'match_saved.py').read_text(encoding='utf8'))
m271,c271=match(R/'outputs/grid271_union_guard','UNION_FOREST','271');m261,c261=match(R/'outputs/grid261_bounded_basis','BOUNDED','261');counts=C.read(O/'gate_counts.json');restored=[r for r in c271 if r['new_continue'] and r['actual_next_NET']>0];lost=[r for r in c271 if not r['new_continue'] and r['actual_next_NET']>0];misses=[r for r in new if r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0]
for n,v in [('profitable_restored.json',restored),('profitable_losses.json',lost),('profitable_misses.json',misses)]:C.save(n,v)
exposure=[];oldpaths={(int(r['method'].split('seed')[-1]),r['sample_id']):r for r in C.read((R/'outputs/grid271_union_guard')/'oof_trajectories.json') if r['method'].startswith('UNION_FOREST_')}
for seed in range(3):
 ss=[r for r in fresh if r['method']==f'CONTEXT_FOREST_seed{seed}'];exposure.append(dict(seed=seed,gain_changed=sum(r['gain']!=oldpaths[seed,r['sample_id']]['gain'] for r in ss),chosen_id_changed=sum(r['chosen_pool_id']!=oldpaths[seed,r['sample_id']]['chosen_pool_id'] for r in ss),KEEP_dates=sorted({r['date'] for r in ss if f"f{r['fold']}_s{seed}"=='f0_s1'})))
C.save('per_seed_preservation_and_exposure.json',exposure)
plan=dict(id='GRID275',scope='Limitedreadonly new274vs271fixedchanges andowntrainingeffect;0newfit/cutoff/forestNNforward/alternativepolicy.',focus='Usepublishedscores/tau todecompose fixed margins: delta(score-tau)=deltaforestscore-deltatau onALLpublishedchanges,not hybriddecisions/returns. All3seed/endpoints/adversecases retained. Check16copy trainingrisk/nodecount/contextsplit changes structurally,no newapply/predict.',support='273ownfit inputs/provenance/weights,274fitreceipts andfrozen old/newtrees;readowntraininglabels only. Explainwhether minorseed1querysaving comes fromdirectionorcutoff, whypositivecopyhypothesisdidnotrecover additionalprofitablecases. No broadre-audit235/253.',rules='No outerdate/seed/IDexceptions,no newtau/quantile/leaf/neighbor gate,no weights/riskscale/feature/hyperparamsweep. Onlyowntraining/publicsupport tofixONEfinite nextqualification after evidence,do not immediately doublecopies oraddNN.',count='Current124=106+9forests+9cutoffs,576newtrees/704deployed;qualification2730. Original6continuations8285updates/604.703actual/678.156cap separately retained.',limits='16correlatedpositivecopies changeempiricalrisk/exposure,not purebitablation. Same543guard/parentown12/internalcal4/exposed16date limitations;localproxy notfullweek/controltime. No final8/physics/grid/paper/FRAME/install,feeNETlabels/controller/order/aliases/station/128/64/stagnation2/.9 unchanged.')
C.save('NEXT_PROTOCOL_PLAN.json',plan);C.save('independent_audit.json',dict(passed=True,newforest9_newcutoff9=True,newtrees576_deployed704=True,all273qualifiedfit_prefix_andownfit_provenance_prechecked=True,all12base_theta_alpha_unchanged=True,all543_empiricalguarded=True,reuse2active_models_trees_andcal_scores_exact=True,KEEP_identity=True,all4681score_gate_inputarrays_consistent=True,new7134paths_cost_action_candidate_checks=True,publication_blobs_verified=True,bothmatched_macroconservations=True,newfits_in_audit=0,newforestNNforward_in_audit=0))
C.save('decision.json',dict(keep261_qualityanchor=True,keep271_main_forest_endpoint=True,retain274_seed1_smallquery_signal=True,do_not_promote274_meanNET_lower=True,profitable_retained=counts['profitable_retained'],newfits18=True,branch124=True,next='GRID275readonly_direction_cutoff_andownfit_effect'))
lines=['# GRID274：自身训练STOP上下文副本比较','','资格273通过后，固定补齐16条自身训练正例的STOP上下文副本，只重训9个受影响森林并校准9个原543并集保护截点。两个未改活跃池复用原森林/271截点，f0_s1 KEEP。新576棵树，部署仍704；原网络与线性修正器不重训。一次600秒窗口实际%.6f秒。'%C.read(O/'fits.json')['wall_s'],'','## 相对固定GRID271','',f"回本保留{45}/55→{counts['profitable_retained']}/55，新增回本恢复{len(restored)}条、新损失{len(lost)}条。没有继续找回剩余回本。整体代理净效用略降，保留一个种子的小省查询线索，不替换271或261质量锚。",'', '|种子|代理NET变化|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for r in m271['per_seed']:
 d=r['macro_delta'];lines.append(f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.12f}|")
lines+=['',f"三种子日期宏NET {m271['old_family']['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}；候选查询{m271['old_family']['mean_queries']:.6f}→{nf['mean_queries']:.6f}，原始查询净变{m271['raw_query_delta']:+d}。决定变化共{len(c271)}条，全部类别与宏守恒见matched271_changes.json及summary。",'', '|变化类别|记录数|原始查询变化|宏NET变化|','|---|---:|---:|---:|']
for r in m271['categories']:
 if r['records']:lines.append(f"|{r['category']}|{r['records']}|{r['raw_query_delta']:+d}|{r['macro_NET_delta']:+.9f}|")
lines+=['','种子1保留收益/候选选择并少量减少查询的改善线索继续保存；其他种子的反向结果全部保留，不选种子1拼接部署。f0_s1仍按预定KEEP，四个日期未启用森林，不能将全16日质量保持都归因于新森林。较小负NET记录与微小正增益也保留，不把“回本45条未变”误说成所有路径增益均不变。','','## 与261质量锚及解释边界','',f"当前相对261仍有{len(misses)}条回本误停。全日期均值NET为{nf['mean_net_utility']:.9f}，261为{m261['old_family']['mean_net_utility']:.9f}；保留261的55/55和原小省查询结果。16条上下文扩充没有在本固定方案中带来进一步回本恢复，不能据此断言上下文无用，也不能说补齐位差异已经解决泛化。",'', '16条副本只来自9池自身fit8、6调用4日期，与原记录相关。原所有行、标签和权重保留，副本继承正标签/原权重，正例经验风险和样本暴露增加；不是单因素语义消融或同样本暴露。没有使用27条cal-only记录拟合，没有把外层失误ID加进训练。原543经验保护集合不变，经验约束不是外层保证。','','## 核验与下一步','',f"新9森林/9截点、576新树（{newnodes}节点/{newleaves}叶）、总704树、原固定配方/深度/最小叶样本、未改池树结构/保存校准分数完全一致、原261theta/alpha、全部543经验正例、4681分数与父入口/阈值表达式、7134新保存路径以及两个匹配的宏贡献守恒均通过。保存结果审计没有重新训练或做新森林/网络推理。",'', '发布顺序：源码f0ba38da→全模型/截点c0fabe79→全预测0fe0369a→首次连接结果。实际归档A来自冻结源码，为GRID274_conditional_forest；W/O名为grid274_stop_context。准备发布白名单最初猜成后者，0拟合时被本地检查挡住，仅改白名单，冻结科研源码不变，note保留。','', '下一GRID275只读分解固定决定变化的模型分数方向与截点变化，以及16副本的训练风险/固定树结构作用；不产生替代阈值/混合策略回报，不重复模型或评分。不立即加倍副本、扫描权重或加网络，只用自身训练/公开支持固定下一有限假设。','', '本轮独立18操作，分支106→124。原6次续训8285更新/NN604.703秒实际及用户678.156秒上限另列。父own12/内部cal4祖先、暴露16日期/相关前缀种子限制不变，非全链OOF/盲测/整周质量或实际时延证据。最终8、新物理/电网/论文/FRAME未动，无安装，原费用NET标签/控制器/排序别名/锁站/128/64/停滞2/.9不变。']
(O/'GRID274_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.save('complete.json',dict(passed=True,new_forest_fits=9,new_scalar_fits=9,newthetaNN=0,KEEP1=True,reusedactive2=True,newtrees576_deployed704=True,wall_s=C.read(O/'fits.json')['wall_s'],guard543=True,scores4681=True,newpaths7134=True,profitable_retained=counts['profitable_retained'],branch124=True));C.archive('audited')
D=R/'论文协作资料/FRAME_停止上下文学习_GRID274';D.mkdir(exist_ok=False)
for n in ['GRID274_report.md','matched271_summary.json','matched261_summary.json','matched271_changes.json','profitable_misses.json','independent_audit.json','decision.json','complete.json','NEXT_PROTOCOL_PLAN.json','per_seed_preservation_and_exposure.json']:shutil.copy2(O/n,D/n)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg=f"GRID274固定16自身fit8 STOP副本训练9森林+9截点/1KEEP/2未改活跃复用完成({C.read(O/'fits.json')['wall_s']:.6f}s)，576新树/部署704，NNtheta0。对271回本45->45/55，恢复{len(restored)}失{len(lost)}，{len(c271)}变化；NET{m271['old_family']['mean_net_utility']:.9f}->{nf['mean_net_utility']:.9f}，query{m271['old_family']['mean_queries']:.6f}->{nf['mean_queries']:.6f}，raw{m271['raw_query_delta']:+d}。seed1保收益动作小省query/NET小正，seed0/2反向保留，不择种子；仍10回本误停不替换261质量锚/271端点。全543保护/4681分数/7134路径/宏守恒核验，源f0ba38da/模型c0fabe79/预测0fe0369a先发后结果。下一275仅只读分数方向/截点分解和16副本自身训练/树结构作用，不替代策略或扫描。原风险目标/暴露改变，分支124及旧续训时间、祖先/开发/整周时延边界保持。"
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=P/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID274停止上下文学习比较\n\n'+msg+'\n'
 if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+rest
 else:t+=entry
 p.write_text(t,encoding='utf8')
(P/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID274后\n\n1. '+msg+' 完整拟合/评分/审计/归档不重做。\n2. 下一275按NEXT_PROTOCOL_PLAN只读固定分数差减截点差的代数分解、全部变化与16副本自身风险/固定树结构；0新策略/概率/拟合。不给外层失误ID/seed/日期设例外，不盲加权/加倍副本/立即加网络。\n3. 只用自身训练标签/公开支持固定唯一有限下一方案，<=22/600秒，保留全部端点/反例/KEEP范围；最终8/新物理电网论文FRAME不动。实际A=GRID274_conditional_forest，协作/清单/远端核验；Githelper串行白名单保持。\n',encoding='utf8')
p=P/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID274停止上下文学习比较\n\n[报告](protocols/'+A.name+'/audited/GRID274_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID274_AUDITED',json.dumps(dict(counts=counts,matched271=m271,restored=len(restored),lost=len(lost),exposure=exposure),ensure_ascii=False))
