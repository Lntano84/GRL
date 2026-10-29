"""First saved-result audit for fixed GRID265; no repeat fitting."""
from pathlib import Path
import sys,json,pickle,collections,shutil,hashlib
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO;N=C.N
assert not (O/'complete.json').exists();C.scored();assert C.read(O/'review.json')['passed'];receipts=C.read(O/'fit_receipts.json');protected=activeprotected=scores=0
for r in receipts:
 p=r['pool'];cal=np.load(C.Q/'qualified_inputs'/f'{p}_cal.npz');old=np.load(C.Q/'qualified_inputs'/f'{p}_model.npz');m=np.load(O/'models'/f'{p}.npz');delta=float(m['delta'][0])
 for k in ['theta','alpha','fallback_KEEP']:assert np.array_equal(m[k],old[k])
 s=C.RT.base_score(cal['X'],cal['b'],m['theta'],float(m['alpha'][0]));pos=cal['y']>0;assert delta==r['delta'] and np.all((s-delta)[pos]>=0);protected+=int(pos.sum())
 if r['fitted']:
  activeprotected+=int(pos.sum());assert delta>=0 and np.any((s-delta)[pos]==0) and np.any((s-float(np.nextafter(delta,np.inf)))[pos]<0)
 else:assert p=='f0_s1' and delta==0
 prev=np.load(C.PUBLIC/f'{p}.npz');pred=np.load(O/'outer_predictions'/f'{p}.npz');assert np.array_equal(pred['ids'],prev['ids']) and np.array_equal(pred['logit'],prev['logit']-delta) and np.array_equal(pred['continue_'],prev['base_CONT']&(pred['logit']>=0));assert not np.any(pred['continue_']&~prev['continue_']);scores+=len(pred['ids'])
assert protected==500 and activeprotected==464 and scores==4681
for phase,name,pubname in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
 pub=C.read(O/pubname);assert pub['passed'] and C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
 for rel,h in C.read(O/name).items():assert hashlib.sha256(C.git('show',pub['git_sha']+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==h
tr=C.read(O/'oof_trajectories.json');cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
oldgate=[r for r in C.read(C.BASE/'oof_trajectories.json') if r['method'].startswith('GATE_')];assert [r for r in tr if r['method'].startswith('GATE_')]==oldgate
fresh=[r for r in tr if r['method'].startswith('SHIFT_')];assert len(fresh)==7134 and len(tr)==14268
for r in fresh:
 case=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(case['gains'][:end].sum()) and r['queries']==int(case['queries'][:end].sum()) and r['chosen_pool_id']==case['chosen'][end];assert r['action_match']==(r['chosen_pool_id']==case['chosen'][-1]) and r['actions']==[1]*end+([0] if end<len(case['states']) else []) and r['batches']==end;assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
new=C.read(O/'gate_predictions.json');old={(r['fold'],r['seed'],r['sample_id'],r['position']):r for r in C.read(N/'gate_predictions.json')};changes=[]
for r in new:
 prev=old[(r['fold'],r['seed'],r['sample_id'],r['position'])]
 for k in ['base_continue','actual_next_NET','actual_next_gain','actual_next_queries']:assert r[k]==prev[k]
 assert not r['continue_'] or prev['continue_']
 if r['continue_']!=prev['continue_']:changes.append(dict(**r,old_continue=True,new_continue=False,NET_delta=-r['actual_next_NET'],query_delta=-r['actual_next_queries'],gain_delta=-r['actual_next_gain']))
metrics=C.read(O/'search_metrics.json');oldmetrics=C.read(N/'search_metrics.json');paired=[];cells=[]
for seed in range(3):
 n=metrics[f'SHIFT_seed{seed}'];prev=oldmetrics[f'BOUNDED_seed{seed}'];delta={k:n['macro'][k]-prev['macro'][k] for k in n['macro']};nd={r['date']:r['cases'] for r in n['per_date']};rr=[r for r in changes if r['seed']==seed]
 for k,field in [('mean_net_utility','NET_delta'),('mean_queries','query_delta'),('mean_gain','gain_delta')]:assert abs(sum(r[field]/(16*nd[r['date']]) for r in rr)-delta[k])<1e-12
 paired.append(dict(seed=seed,macro_delta=delta))
 for aa,bb in zip(prev['per_date'],n['per_date']):
  assert aa['date']==bb['date'];cr=[r for r in rr if r['date']==aa['date']]
  if cr:cells.append(dict(seed=seed,date=aa['date'],changes=len(cr),NET_delta=bb['mean_net_utility']-aa['mean_net_utility'],query_delta=bb['mean_queries']-aa['mean_queries'],gain_delta=bb['mean_gain']-aa['mean_gain']))
family=C.read(O/'family_summary.json');nf=family['SHIFT'];gf=family['GATE'];of=C.read(N/'family_summary.json')['BOUNDED'];lost=[r for r in changes if r['actual_next_NET']>0];nonprof=[r for r in changes if r['actual_next_NET']<=0];counts=C.read(O/'gate_counts.json');assert len(changes)==191 and len(lost)==6 and len(nonprof)==185 and counts['profitable_retained']==49 and sum(r['actual_next_queries'] for r in changes)==723
categories=[]
for name,rr in [('profitable_losses',lost),('nonprofitable_cancellations',nonprof)]:
 out=dict(category=name,records=len(rr),raw_query_delta=sum(r['query_delta'] for r in rr),raw_gain_delta=sum(r['gain_delta'] for r in rr),calls=len({r['call'] for r in rr}),dates=len({r['date'] for r in rr}),NET_delta=0.,macro_query_delta=0.,macro_gain_delta=0.)
 for r in rr:
  nd={x['date']:x['cases'] for x in metrics[f"SHIFT_seed{r['seed']}"]['per_date']};d=3*16*nd[r['date']];out['NET_delta']+=r['NET_delta']/d;out['macro_query_delta']+=r['query_delta']/d;out['macro_gain_delta']+=r['gain_delta']/d
 categories.append(out)
for k,field in [('mean_net_utility','NET_delta'),('mean_queries','macro_query_delta'),('mean_gain','macro_gain_delta')]:assert abs(sum(r[field] for r in categories)-(nf[k]-of[k]))<1e-12
summary=dict(primary='FrozenGRID261',old_family=of,new_family=nf,family_delta={k:nf[k]-of[k] for k in nf},per_seed=paired,changes191=len(changes),profitable_lost6=len(lost),restored0=True,nonprofitable_cancelled185=len(nonprof),rawquery_saved723=723,changed_seed_date_cells=cells,category_contributions=categories,correlated_not_independent=True)
for name,val in [('matched261_summary.json',summary),('matched261_changes.json',changes),('profitable_losses.json',lost),('nonprofitable_cancellations.json',nonprof)]:C.save(name,val)
nextplan=dict(id='GRID266',scope='Readonly new191cancellations:6profitablelosses/185nonprofitable;no newfit/alpha/delta/threshold/hybridpolicy or outerstrategy',fixed='Use only published265/261scores andfixedcalinputs/model/scalar receipts,focusold margin/delta/guarding own12support andcall/date/querycontributions. Reference prior235/253ancestry limitations,do not rerun old broadancestry as newdiscovery.',analysis='Decompose positive losses vs feesavings;6losses inJan5two calls/2seeds arecorrelated. Compare published pre-shift score withguarded positive own12scores distribution/roles andownfit support. No alternative delta/quantile/mixedpolicy evaluation, no Jan5exception.',next_hypothesis='Only owntraining evidence/publicsupport mayqualify ONEfinite nextlearning scheme;no exposedouterchoice ofguardstrength/regularization/risk/seed. Preserve261mainqualitycandidate and265costtradeoff/seed1smallpositive allcounterexamples.',bounds='Branch73fits andhistorical6continuations/8285updates/604.703sactual/678.156cap;final8/physics/grid/paper/FRAME/install untouched;queryproxy notactualtime/fullweek.'))
C.save('NEXT_PROTOCOL_PLAN.json',nextplan);C.save('independent_audit.json',dict(passed=True,frozen_theta_alpha_arrays_12unchanged=True,all500protected_active464=True,11scalar_maximality_without_refitting=True,publication_specificblobs=True,all4681scores_monotone=True,new7134savedsequences_checked_oldGATEreused=True,all191changes_and2category_macroconservation=True,all6profitablelosses_kept=True,newfits_in_audit0=True,NNheadforward0=True,final8_absent=True));C.save('decision.json',dict(keep261_qualitypreserving_maincandidate=True,do_not_replace261_by265_meanNETlower=True,retain265_queryquality_tradeoff_research=True,seed1_smallpositive_not_seed_selection=True,all3querieslower=True,profitable49of55=True,newthetaNNfits0=True,branch73=True,next='GRID266readonly191change_andguard_support_diagnosis'))
shutil.copy2(C.W/'preparation_error01.json',O/'preparation_error01.json')
text=['# GRID265：回本经验保护下的停止强度校准','','冻结GRID261全部网络、41参数线性模型、alpha和tanh38输入，仅按预先固定规则校准11个向下分数偏置；1池KEEP不变。一次600秒窗口实际%.6f秒，原500条经验正例（活跃464/KEEP36）全部保留。该规则明确等价提高原分数阈值，改变停止工作点与经验校准目标，并不是继续优化原NLL。'%C.read(O/'fits.json')['wall_s'],'','## 与固定GRID261比较','','查询进一步减少，但回本记录保留55/55→49/55。191处决定变化全部是取消：新增取消185条不回本记录，同时误停6条回本记录。不能把训练侧经验保护写成外层质量保证。','', '|种子|代理净效用变化|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for r in paired:
 d=r['macro_delta'];text.append(f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|")
text+=['',f"三种子日期宏平均候选查询{of['mean_queries']:.6f}→{nf['mean_queries']:.6f}，减少{100*(of['mean_queries']-nf['mean_queries'])/of['mean_queries']:.4f}%；原始少723次。代理净效用{of['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}，选项一致率{100*of['action_match_rate']:.4f}%→{100*nf['action_match_rate']:.4f}%。其中新增补查查询均值{of['added_queries']:.6f}→{nf['added_queries']:.6f}；省查询伴随增益减少，不能当同质量提速。",'', '种子1净效用小幅增加，仍保留为研究线索，但它也取消了极小的非回本增益，不能称即时增益严格相同；不能挑种子1拼接部署。种子0略降、种子2明显下降，原GRID261继续作为质量保留的内部候选。','','## 回本损失与代价贡献','','6条回本误停全部在2037-01-05_101：种子0的call004两条；种子2的call001两条和call004两条。它们是两个物理调用的相关前缀及跨种子复用，不是6个独立场景。种子2的call001两条较大增益损失主导总体退化；完整反例在profitable_losses.json。','', '|新增取消类别|记录数|原始查询变化|宏代理净效用变化|','|---|---:|---:|---:|']
for r in categories:text.append(f"|{r['category']}|{r['records']}|{r['raw_query_delta']}|{r['NET_delta']:+.9f}|")
text+=['', '保留这次取舍结果，不追调偏置救结果。下一GRID266只读解释191处变化、185条不回本取消与6条回本损失的成本/调用集中度、原分数/偏置和自身经验保护支持；不评估其他偏置、分位数、混合策略或日期例外。','','## 核验与范围','','12份组合模型的theta/alpha逐数组等于冻结261；全部500正例保护、11标量最大可行性、4681发布预测单调性、7134条新增保存搜索序列的动作/候选ID/成本/查询、191变化及两类宏贡献守恒均核验。旧GATE轨迹直接复用。源码2040579e先发布，全部参数3fe27d2c先发后评分，全部预测8d0048f5先发后接结果，逐文件Git blob核对。准备时编码名称utf8-sig拼写错误发生在源码生成/冻结前，0拟合0评分；错误已保留并改正为utf-8-sig，研究规则未改。','','本次为独立新假设11标量，分支62→73；没有重训theta/NN或新收益头forward。6次旧续训8285更新、原NN604.703秒/用户678.156秒上限另列，不藏回原40探针或261的22拟合。own12保护和内部cal4祖先局限保留，16日期已暴露、前缀/种子/折用途相关，非全链OOF/盲测。候选查询与代理不等实际控制时延或整周质量，最终8未生成/读取，无新物理/电网/论文/FRAME或安装，原费用标签/控制器/排序别名/锁站/128/64/停滞2/.9不变。']
(O/'GRID265_report.md').write_text('\n'.join(text)+'\n',encoding='utf8');C.save('complete.json',dict(passed=True,newscalar11=True,KEEP1=True,newthetaNN0=True,wall_s=C.read(O/'fits.json')['wall_s'],scores4681=True,new_savedsequences7134=True,branch73=True,profitable49of55=True,final8_absent=True));C.archive('audited')
D=R/'论文协作资料/FRAME_停止强度学习_GRID265';D.mkdir(exist_ok=False)
for name in ['GRID265_report.md','matched261_summary.json','matched261_changes.json','profitable_losses.json','decision.json','independent_audit.json','complete.json','NEXT_PROTOCOL_PLAN.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,actual_archive=str(A),collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()}))
msg='GRID26511正例保护向下偏置+1KEEP完成(.028198s)，0theta/NN重训，原500经验正例全保留，源2040579e/参数3fe27d2c/预测8d0048f5先发。对261191决定全取消：185不回本/6回本，raw省723query，宏query13.421699→13.328710、NET.096119673→.094898621，回本55→49/55。种子NET差-.000055637/+.000108497/-.003716016，query全降；保留seed1小正与全部反例，不择种子。6回本损失全Jan5_101两调用，seed0call004两条、seed2call001/004各两条，相关而非独立；seed2大额主导退化。保留261质量主候选，265为查询/质量取舍研究，非同质量提速。全部权重不变/500保护/11极值可行/4681分数单调/新增7134路径/191变化与两类宏守恒核验，分支73。下一266只读6损失/185取消的原分数偏置及own12保护支持，不试其他偏置/混合/日期规则，原祖先/开发/整周时延界限保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;t=p.read_text(encoding='utf8');e='\n\n## 2026-10-29 — GRID265停止强度学习比较\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+e+rest
 else:t+=e
 p.write_text(t,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID265后\n\n1. '+msg+' 完整拟合/评分/审计/归档不重做。\n2. 下一266按已封存NEXT_PROTOCOL_PLAN只读191取消两类成本贡献/6回本损失的原261分数与delta/own12经验保护支持。复用235/253旧祖先结论，不重跑宽泛审计当新发现；不产替代偏置/分位数/混合策略分数或回报，不按Jan5设例外，不重训救结果。\n3. 只据自身训练与公开支持资格唯一有限下一方案，仍<=22/一次600s，不延长/扫描/拆搜索。保留261及250/230历史端点与265反例，分支73/6旧续训授权时间披露；最终8/新物理/电网/论文/FRAME不动，不安装。Githelper串行路径白名单/实际A归档协作远端核验保持。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID265停止强度学习比较\n\n[报告](protocols/'+A.name+'/audited/GRID265_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID265_AUDITED',json.dumps(dict(counts=counts,category_contributions=categories),ensure_ascii=False))
