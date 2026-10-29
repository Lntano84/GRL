"""First GRID258 saved-result audit, matched250 decomposition and delivery."""
from pathlib import Path
import sys,json,pickle,collections,shutil,hashlib
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO;OLD=R/'outputs/grid250_context_residual';Q=C.Q
assert not (O/'complete.json').exists();C.scored();assert C.read(O/'review.json')['passed'];receipts=C.read(O/'fit_receipts.json');protected=0;activeprotected=0;scores=0
for r in receipts:
 p=r['pool'];z=np.load(Q/'fit_inputs'/f'{p}.npz');cal=np.load(Q/'calibration_inputs'/f'{p}.npz');m=np.load(O/'models'/f'{p}.npz');theta=m['theta'];alpha=float(m['alpha'][0]);assert theta.shape==(41,);v,g=C.objective(theta,z['X'],z['y'],z['weight'],z['b']);assert abs(v-r['final_loss'])<1e-12
 if r['fitted']:assert r['solver_success'] and float(abs(g).max())<1e-3 and abs(float(abs(g).max())-r['gradient_inf'])<1e-12
 else:assert p=='f0_s1' and not theta.any() and alpha==0
 end=cal['b']+np.sum(cal['X']*theta[:-1],axis=1)+theta[-1];positive=cal['y']>0;assert np.all((cal['b']+alpha*(end-cal['b']))[positive]>=0);protected+=int(positive.sum())
 if r['fitted']:
  activeprotected+=int(positive.sum())
  if r['analytic_bound']<1:assert np.any((cal['b']+.5*(1+r['analytic_bound'])*(end-cal['b']))[positive]<0)
 h=np.load(C.PUBLIC/'outer_parent_public'/f'{p}.npz');pred=np.load(O/'outer_predictions'/f'{p}.npz');end=h['b']+np.sum(h['X']*theta[:-1],axis=1)+theta[-1];score=h['b']+alpha*(end-h['b']);assert np.array_equal(pred['ids'],h['ids']) and np.array_equal(pred['logit'],score) and np.array_equal(pred['continue_'],h['base_CONT']&(score>=0));assert (h['X'][:,-1]==1).all();scores+=len(score)
assert protected==500 and activeprotected==464 and scores==4681
for phase,name,pubname in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
 p=C.read(O/pubname);assert C.git('show',p['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
 for rel,h in C.read(O/name).items():assert hashlib.sha256(C.git('show',p['git_sha']+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==h
tr=C.read(O/'oof_trajectories.json');cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
for r in tr:
 case=cases[r['sample_id']];end=r['baseline_stop']+int(r['append']);assert r['gain']==float(case['gains'][:end].sum()) and r['queries']==int(case['queries'][:end].sum()) and r['chosen_pool_id']==case['chosen'][end] and r['action_match']==(case['chosen'][end]==case['chosen'][-1]);assert r['actions']==[1]*end+([0] if end<len(case['states']) else []) and r['batches']==end;assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
new=C.read(O/'gate_predictions.json');prev=C.read(OLD/'gate_predictions.json');lookup={(r['fold'],r['seed'],r['sample_id'],r['position']):r for r in prev};assert len(lookup)==len(new)==4681;metrics=C.read(O/'search_metrics.json');oldmetrics=C.read(OLD/'search_metrics.json');changes=[];paired=[];cells=[]
for r in new:
 old=lookup[(r['fold'],r['seed'],r['sample_id'],r['position'])];assert r['base_continue']==old['base_continue'] and r['actual_next_NET']==old['actual_next_NET'] and r['actual_next_queries']==old['actual_next_queries']
 if old['continue_']!=r['continue_']:
  sign=int(r['continue_'])-int(old['continue_']);changes.append(dict(fold=r['fold'],seed=r['seed'],sample_id=r['sample_id'],position=r['position'],date=r['date'],call=r['call'],old_continue=old['continue_'],new_continue=r['continue_'],actual_next_NET=r['actual_next_NET'],actual_next_gain=r['actual_next_gain'],actual_next_queries=r['actual_next_queries'],NET_delta=sign*r['actual_next_NET'],query_delta=sign*r['actual_next_queries'],gain_delta=sign*r['actual_next_gain']))
for seed in range(3):
 n=metrics[f'REFSOURCE_seed{seed}'];old=oldmetrics[f'CONTEXT_seed{seed}'];delta={k:n['macro'][k]-old['macro'][k] for k in n['macro']};nd={x['date']:x['cases'] for x in n['per_date']};rr=[r for r in changes if r['seed']==seed]
 for key,field in [('mean_net_utility','NET_delta'),('mean_queries','query_delta'),('mean_gain','gain_delta')]:assert abs(sum(r[field]/(16*nd[r['date']]) for r in rr)-delta[key])<1e-12
 paired.append(dict(seed=seed,macro_delta=delta))
 for a,b in zip(old['per_date'],n['per_date']):
  assert a['date']==b['date'];cc=[r for r in rr if r['date']==a['date']]
  if cc:cells.append(dict(seed=seed,date=a['date'],NET_delta=b['mean_net_utility']-a['mean_net_utility'],query_delta=b['mean_queries']-a['mean_queries'],changes=len(cc)))
oldfamily=C.read(OLD/'family_summary.json')['CONTEXT'];family=C.read(O/'family_summary.json')['REFSOURCE'];delta={k:family[k]-oldfamily[k] for k in family};restored=[r for r in changes if r['new_continue'] and r['actual_next_NET']>0];lost=[r for r in changes if not r['new_continue'] and r['actual_next_NET']>0];miss=[r for r in new if r['base_continue'] and not r['continue_'] and r['actual_next_NET']>0];counts=C.read(O/'gate_counts.json');assert len(changes)==211 and len(restored)==21 and len(lost)==2 and counts['profitable_retained']==48 and len(miss)==7
summary=dict(primary='FrozenGRID250',family_delta=delta,old_family=oldfamily,new_family=family,per_seed=paired,changes=len(changes),restored_profitable=len(restored),lost_profitable=len(lost),new_nonprofitable_appends=sum(r['new_continue'] and r['actual_next_NET']<=0 for r in changes),cancelled_nonprofitable=sum(not r['new_continue'] and r['actual_next_NET']<=0 for r in changes),raw_query_delta=sum(r['query_delta'] for r in changes),changed_seed_date_cells=cells,correlated_not_independent=True)
for name,value in [('matched250_summary.json',summary),('matched250_changes.json',changes),('profitable_misses.json',miss),('restored_profitable.json',restored),('new_profitable_losses.json',lost)]:C.save(name,value)
C.save('independent_audit.json',dict(passed=True,solver_gradient_11=True,positive_constraints_500_including_KEEP36=True,active_constraints464=True,step_maximality_checked=True,published_scores4681=True,saved_sequences14268=True,matched250_changes211_macro_conservation=True,allseeds_dates_adversecases=True,publicationbarriers=True,newfits_in_audit=0,new_NN_or_forest_forward=0,no_new_physics=True,final8_absent=True))
C.save('decision.json',dict(retain258_as_higher_profitable_retention_research_candidate=True,retained_profitable='48/55 vs250 29/55',restored21_lost2=True,all3_action_match_higher=True,all3_queries_higher=True,not_equal_budget_advantage=True,net_seed_delta=[r['macro_delta']['mean_net_utility'] for r in paired],do_not_replace230_main_or_discard250_lowerquery=True,next='GRID259 readonly decompose21recoveries/2newlosses/7residualJan5misses and131extra/57cancellednonprofitable; endpoint direction vs step algebra and owntraining support only. No newfit/alpha/hybridpolicy/threshold/seed selection.'))
lines=['# GRID258：参考停止点来源学习比较','','本轮固定同GRID250的40输入41参数线性修正器，使用新参考停止点数据。11线性+11原正例保护步长完成，总拟合/校准约%.6f秒，所有求解状态和原始梯度通过。另一个f0_s1按预定KEEP，未因新增正例激活。'%C.read(O/'fits.json')['wall_s'],'','## 主要变化','','固定已暴露评估中的回本记录保留29/55→48/55：恢复21条，另丢失2条。三个种子与完整局部搜索选项的一致率均提高，平均提高%.4f个百分点；但三个种子查询都增加，不能称等预算优势。'%(100*delta['action_match_rate']),'','|种子|减250代理净效用|查询变化|即时增益变化|选项一致率变化(pp)|','|---|---:|---:|---:|---:|']
for r in paired:
 d=r['macro_delta'];lines.append(f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|{100*d['action_match_rate']:+.4f}|")
lines+=['',f"三种子均值：代理净效用{oldfamily['mean_net_utility']:.9f}→{family['mean_net_utility']:.9f}；候选查询{oldfamily['mean_queries']:.6f}→{family['mean_queries']:.6f}；即时增益{oldfamily['mean_gain']:.10f}→{family['mean_gain']:.10f}。总体净效用略升，但种子0明显下降、种子1略降、种子2明显提高，不能把均值当稳定胜出。",'', '211处决定变化中，新增131条不回本补查，同时取消57条不回本补查；多恢复有效补查伴随额外查询。保留为“更高回本保留”的研究候选，并保留250低查询端点和230历史主候选，不挑种子拼接部署。','','## 反例与解释','','剩余7条回本误停全部在1月5日：种子0两条、种子2五条，分属两个物理调用；其中call001的5条是较大收益的相关前缀，不能当5个独立失败。新丢失的2条均为种子0的call001大额记录，解释了该种子代理净效用的退化。原7条全部另存，不能因为只剩一个日期就设日期/调用例外。','','新来源带来更多训练STOP正例和新风险/保护集合；10个活跃池alpha=1，另一个活跃池约0.4178，KEEP为0。数据来源、样本、风险数值和约束集合耦合，不能把本轮视为纯单一因素因果消融，也不能预先把恢复全归于步长或标签数量。','','下一259只读分解21恢复、2新增损失、131无利新增与57无利取消的收益/查询贡献，核对7残余误停、端点方向/步长代数贡献与自身训练支持。可以保留小而重复的改善，不试新alpha/阈值/混合策略，也不按外层挑特征或种子；若要下一学习方案，须再根据自身训练证据固定。','','## 核验与范围','','全部500经验正例（含KEEP36，活跃464）保护与最大步长、11个原始梯度、4681个已发布分数、14268保存序列的动作/成本/查询、211变化宏平均守恒均核对。源码输入224e33e8先发后拟合；全模型/步长122da412先发后评分；全预测2f56d329先发后连接结果。准备阶段路径白名单曾猜错归档名，0拟合时保留记录并改用冻结源码真实A=GRID258_context_residual，没有改科学代码。','','所有16日期是已暴露开发，相关前缀和种子/折复用不独立。原父NN/头own12及内部cal4祖先局限仍在，正例保护不是独立cal4验证。没有新NN/收益头/物理仿真或原FRAME比较；原部署网络、费用/标签、排序别名/锁站/128/64/停滞2/.9不变。候选查询不是实际控制时延，训练约0.075秒也不是在线性能；局部代理不是整周质量，最终8未生成/读取。']
(O/'GRID258_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');C.save('complete.json',dict(passed=True,new_linear_fits=11,new_scalar_fits=11,constantKEEP=1,newNNfits=0,wall_s=C.read(O/'fits.json')['wall_s'],scores=4681,sequences=14268,final8_absent=True));C.archive('audited')
D=R/'论文协作资料/FRAME_参考停止点学习_GRID258';D.mkdir(exist_ok=False)
for name in ['GRID258_report.md','matched250_summary.json','matched250_changes.json','profitable_misses.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,D/name)
C.save('delivery_check.json',dict(passed=True,collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()},actual_archive=str(A)))
msg=f"GRID258同250类11线性+11步长完成({C.read(O/'fits.json')['wall_s']:.6f}s)，500保护/4681分数/14268序列/211变化核验。对250回本29→48/55，恢复21损失2；三种子选项一致率均升但查询均增，均值NET{oldfamily['mean_net_utility']:.9f}→{family['mean_net_utility']:.9f}、query{oldfamily['mean_queries']:.6f}→{family['mean_queries']:.6f}。seed NET差-0.003782689/-0.000059042/+0.004013951，新增131无利/取消57。剩7误停全Jan5两调用，新丢2种子0大额相关记录。保留高回本候选及250低query/230主候选，非等预算胜出。259仅只读贡献、方向/步长和自身训练支持，不新拟合或按outer挑规则。"
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=REPO/'docs'/name;s=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID258参考停止点学习\n\n'+msg+'\n'
 if name=='RESEARCH_STATE.md':title,rest=s.split('\n',1);s=title+'\n'+entry+rest
 else:s+=entry
 p.write_text(s,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID258后\n\n1. '+msg+' 完整拟合/评分/审计/归档不重做。\n2. 下一259只读21恢复/2新增大损失/7残余Jan5误停，及131无利新增/57无利取消的日期宏贡献与raw query；匹配250同一公开输入下分解终点方向/步长代数项，不形成混合新策略评分。仅自身训练标签/公共支持提出后续唯一有限方案，不对Jan5/调用/种子例外，不扫alpha/阈值/正则/费用。\n3. 所有种子日期反例保留，原祖先/开发/整周/时延局限保持。最终8/物理/电网/论文/FRAME不动，不安装。实际源码A=GRID258_context_residual，Githelper严格串行和路径白名单，核对协作归档远端。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID258参考停止点学习\n\n[报告](protocols/'+A.name+'/audited/GRID258_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');print('GRID258_AUDITED',json.dumps(dict(counts=counts,family_delta=delta,raw_query_delta=summary['raw_query_delta']),ensure_ascii=False))
