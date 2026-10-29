"""First saved-result audit and delivery for the fixed GRID261 candidate; no fitting."""
from pathlib import Path
import sys,json,pickle,collections,shutil,hashlib
import numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
OLD=R/'outputs/grid258_reference_correction'
assert not (O/'complete.json').exists()
C.scored()
assert C.read(O/'review.json')['passed']
receipts=C.read(O/'fit_receipts.json');protected=activeprotected=scores=0
for r in receipts:
    p=r['pool'];z=np.load(C.Q/'fit_inputs'/f'{p}.npz');cal=np.load(C.Q/'calibration_inputs'/f'{p}.npz');m=np.load(O/'models'/f'{p}.npz');theta=m['theta'];alpha=float(m['alpha'][0])
    assert theta.shape==(41,)
    value,g=C.objective(theta,z['X'],z['y'],z['weight'],z['b'])
    assert abs(value-r['final_loss'])<1e-12
    if r['fitted']:
        assert r['solver_success'] and float(abs(g).max())<1e-3 and abs(float(abs(g).max())-r['gradient_inf'])<1e-12
        assert alpha==1.0
    else:assert p=='f0_s1' and not theta.any() and alpha==0
    endpoint=cal['b']+np.sum(cal['X']*theta[:-1],axis=1)+theta[-1]
    positive=cal['y']>0
    assert np.all((cal['b']+alpha*(endpoint-cal['b']))[positive]>=0)
    protected+=int(positive.sum());activeprotected+=int(positive.sum())*int(r['fitted'])
    h=np.load(C.PUBLIC/'outer_parent_public'/f'{p}.npz');pred=np.load(O/'outer_predictions'/f'{p}.npz')
    X=h['X'].copy();X[:,:38]=np.tanh(X[:,:38])
    assert np.array_equal(X[:,38:40],h['X'][:,38:40]) and (X[:,-1]==1).all()
    endpoint=h['b']+np.sum(X*theta[:-1],axis=1)+theta[-1];score=h['b']+alpha*(endpoint-h['b'])
    assert np.array_equal(pred['ids'],h['ids']) and np.array_equal(pred['logit'],score)
    assert np.array_equal(pred['base_CONT'],h['base_CONT']) and np.array_equal(pred['continue_'],h['base_CONT']&(score>=0))
    scores+=len(score)
assert protected==500 and activeprotected==464 and scores==4681
for phase,name,pubname in [('trained','trained_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
    p=C.read(O/pubname)
    assert p['passed'] and C.git('show',p['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    for rel,h in C.read(O/name).items():
        assert hashlib.sha256(C.git('show',p['git_sha']+':'+(A/phase/rel).relative_to(REPO).as_posix())).hexdigest()==h
tr=C.read(O/'oof_trajectories.json');cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())}
task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
for r in tr:
    case=cases[r['sample_id']];end=r['baseline_stop']+int(r['append'])
    assert r['gain']==float(case['gains'][:end].sum()) and r['queries']==int(case['queries'][:end].sum())
    assert r['chosen_pool_id']==case['chosen'][end] and r['action_match']==(case['chosen'][end]==case['chosen'][-1])
    assert r['actions']==[1]*end+([0] if end<len(case['states']) else []) and r['batches']==end
    assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
assert len(tr)==14268
seedof=lambda r:int(r['method'].split('seed')[-1])
base={(seedof(r),r['sample_id']):r for r in tr if r['method'].startswith('GATE_')}
bounded=[r for r in tr if r['method'].startswith('BOUNDED_')]
assert len(base)==len(bounded)==7134
assert all(r['gain']==base[seedof(r),r['sample_id']]['gain'] and r['chosen_pool_id']==base[seedof(r),r['sample_id']]['chosen_pool_id'] and r['action_match']==base[seedof(r),r['sample_id']]['action_match'] for r in bounded)
new=C.read(O/'gate_predictions.json');prev=C.read(OLD/'gate_predictions.json')
lookup={(r['fold'],r['seed'],r['sample_id'],r['position']):r for r in prev}
assert len(lookup)==len(new)==4681
metrics=C.read(O/'search_metrics.json');oldmetrics=C.read(OLD/'search_metrics.json');changes=[];paired=[];cells=[]
for r in new:
    old=lookup[(r['fold'],r['seed'],r['sample_id'],r['position'])]
    for key in ['base_continue','actual_next_NET','actual_next_queries','actual_next_gain']:assert r[key]==old[key]
    if old['continue_']!=r['continue_']:
        sign=int(r['continue_'])-int(old['continue_'])
        changes.append(dict(fold=r['fold'],seed=r['seed'],sample_id=r['sample_id'],position=r['position'],date=r['date'],call=r['call'],old_continue=old['continue_'],new_continue=r['continue_'],actual_next_NET=r['actual_next_NET'],actual_next_gain=r['actual_next_gain'],actual_next_queries=r['actual_next_queries'],NET_delta=sign*r['actual_next_NET'],query_delta=sign*r['actual_next_queries'],gain_delta=sign*r['actual_next_gain']))
for seed in range(3):
    n=metrics[f'BOUNDED_seed{seed}'];old=oldmetrics[f'REFSOURCE_seed{seed}'];delta={k:n['macro'][k]-old['macro'][k] for k in n['macro']};nd={x['date']:x['cases'] for x in n['per_date']};rr=[r for r in changes if r['seed']==seed]
    for key,field in [('mean_net_utility','NET_delta'),('mean_queries','query_delta'),('mean_gain','gain_delta')]:assert abs(sum(r[field]/(16*nd[r['date']]) for r in rr)-delta[key])<1e-12
    paired.append(dict(seed=seed,macro_delta=delta))
    for a,b in zip(old['per_date'],n['per_date']):
        assert a['date']==b['date']
        cc=[r for r in rr if r['date']==a['date']]
        if cc:cells.append(dict(seed=seed,date=a['date'],NET_delta=b['mean_net_utility']-a['mean_net_utility'],query_delta=b['mean_queries']-a['mean_queries'],changes=len(cc)))
family=C.read(O/'family_summary.json');oldfamily=C.read(OLD/'family_summary.json')['REFSOURCE'];nf=family['BOUNDED'];gf=family['GATE']
restored=[r for r in changes if r['new_continue'] and r['actual_next_NET']>0];lost=[r for r in changes if not r['new_continue'] and r['actual_next_NET']>0]
cancel=[r for r in new if r['base_continue'] and not r['continue_']];miss=[r for r in cancel if r['actual_next_NET']>0]
counts=C.read(O/'gate_counts.json')
assert len(changes)==137 and len(restored)==7 and len(lost)==0 and counts['profitable_retained']==55 and not miss
assert len(cancel)==77 and all(r['actual_next_gain']==0 and r['actual_next_NET']<=0 for r in cancel)
assert sum(r['actual_next_queries'] for r in cancel)==122
counter=collections.Counter((r['sample_id'],r['position']) for r in cancel)
gatepairs=C.read(O/'paired_summary.json')
for s in range(3):
    nd={x['date']:x['cases'] for x in metrics[f'BOUNDED_seed{s}']['per_date']};rr=[r for r in cancel if r['seed']==s]
    for key,field in [('mean_queries','actual_next_queries'),('mean_net_utility','actual_next_NET')]:
        assert abs(-sum(r[field]/(16*nd[r['date']]) for r in rr)-gatepairs[s]['macro_delta'][key])<1e-12
    assert gatepairs[s]['macro_delta']['mean_gain']==0 and gatepairs[s]['macro_delta']['mean_queries']<0 and gatepairs[s]['macro_delta']['mean_net_utility']>0
summary=dict(primary='FrozenGRID258',old_family=oldfamily,new_family=nf,family_delta={k:nf[k]-oldfamily[k] for k in nf},per_seed=paired,changes=len(changes),restored_profitable=len(restored),lost_profitable=len(lost),new_nonprofitable_appends=sum(r['new_continue'] and r['actual_next_NET']<=0 for r in changes),cancelled_nonprofitable=sum(not r['new_continue'] and r['actual_next_NET']<=0 for r in changes),raw_query_delta=sum(r['query_delta'] for r in changes),changed_seed_date_cells=cells,correlated_not_independent=True)
gate_summary=dict(reference='Shared208GATE',family_delta={k:nf[k]-gf[k] for k in nf},per_seed=gatepairs,cancelled=len(cancel),raw_queries_saved=122,all_cancelled_zero_instant_gain=True,all7134_chosen_ids_and_gains_identical=True,unique_cancel_state_records=len(counter),cancelled_calls=len({r['call'] for r in cancel}),cancelled_dates=len({r['date'] for r in cancel}),states_repeated_across_two_seeds=sum(v>1 for v in counter.values()),states_repeated_all_three=sum(v==3 for v in counter.values()),cancel_per_seed=dict(collections.Counter(r['seed'] for r in cancel)),candidate_query_reduction_fraction=(gf['mean_queries']-nf['mean_queries'])/gf['mean_queries'],no_actual_control_timing=True)
for name,val in [('matched258_summary.json',summary),('matched258_changes.json',changes),('restored_profitable.json',restored),('new_profitable_losses.json',lost),('profitable_misses.json',miss),('gate_cancellations.json',cancel),('gate_preservation_summary.json',gate_summary)]:C.save(name,val)
C.save('independent_audit.json',dict(passed=True,saved_result_audit_not_solver_rerun=True,solver_gradient_11=True,positive_constraints500_KEEP36_active464=True,active_alphas_all1=True,published_scores4681_with_same_tanh38=True,saved_sequences14268=True,all7134_gain_and_chosen_ids_equal_sharedGATE=True,all77_cancellations_gain0_raw122queries=True,matched258_changes137_macro_conservation=True,allseeds_dates_counterexamples=True,publication_blobs_verified=True,newfits_in_audit=0,new_NN_or_head_forward=0,no_new_physics=True,final8_absent=True))
nexttext='GRID262 readonly explanation of77zero-gain cancellations,122rawqueries,seed/date/call repetition,score margins and bounded-input training support. No new model/alpha/hybrid/threshold/outer policy/physics; no transformation sweep. Preserve258/250/230frontier and all adverse cells.'
C.save('decision.json',dict(retain261_as_quality_preserving_internal_candidate=True,all55_profitable_retained=True,same7134_chosen_ids_and_gains_as_sharedGATE=True,all3seeds_queries_lower_than_Gate=True,mean_query_reduction_fraction=gate_summary['candidate_query_reduction_fraction'],primary258_restored7_lost0=True,vs258_query_increased_allseeds=True,vs258_NET_seed_delta=[r['macro_delta']['mean_net_utility'] for r in paired],do_not_claim_fullweek_generalization_actual_runtime=True,retain250_lowquery_230_historical=True,next=nexttext))
lines=['# GRID261：有界公共输入学习结果','','本轮仅将全部38个既有标准化公共坐标统一映射为tanh，父先验b、停止上下文位和偏置b保持不变。标签、风险、训练/校准日期和500条经验保护记录均复用。固定11个41参数线性修正器和11个原正例保护步长，共22拟合/校准，一次600秒窗口实际%.6f秒；另1个预定KEEP未训练。全部11个求解成功、原始梯度通过。'%C.read(O/'fits.json')['wall_s'],'','## 结果','','**在现有开发路径上，公共GATE的55/55条回本记录全部保留；7134条种子×路径记录的即时增益和最终候选ID逐条相同。** 三个种子均少用少量候选查询。','',f"三种子日期宏平均候选查询：{gf['mean_queries']:.6f}→{nf['mean_queries']:.6f}，减少{100*gate_summary['candidate_query_reduction_fraction']:.4f}%。代理净效用{gf['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}。共取消77条零即时增益补查，原始节省122次候选查询；每种子取消25/26/26条。",'','|种子|减公共GATE代理净效用|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for r in gatepairs:
    d=r['macro_delta'];lines.append(f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|")
lines+=['',f"77条取消记录涉及{gate_summary['cancelled_calls']}个物理调用、{gate_summary['cancelled_dates']}个旧日期。按相同样本/位置合并后有{len(counter)}个状态，只有6个状态在两个种子重复取消，没有在全部三个种子都取消的状态。因此“三个种子总体均省查询”不等于“每个种子学到同一取消集合”，更不能把77条当独立样本。",'','## 与上一版258的匹配比较','','回本保留48/55→55/55，恢复7条、没有新增回本误停。137处决定变化包含这些恢复、更多零收益继续和少量取消调整。所有种子候选查询均增加；代理净效用两升一小降，不能称各维度全面胜出。','', '|种子|减258代理净效用|候选查询变化|即时增益变化|','|---|---:|---:|---:|']
for r in paired:
    d=r['macro_delta'];lines.append(f"|{r['seed']}|{d['mean_net_utility']:+.9f}|{d['mean_queries']:+.6f}|{d['mean_gain']:+.10f}|")
lines += ['',f"平均代理净效用{oldfamily['mean_net_utility']:.9f}→{nf['mean_net_utility']:.9f}，候选查询{oldfamily['mean_queries']:.6f}→{nf['mean_queries']:.6f}；原始查询净增{summary['raw_query_delta']}。新加无利补查{summary['new_nonprofitable_appends']}条、取消无利补查{summary['cancelled_nonprofitable']}条，完整变化与日期反例见matched258_changes.json和matched258_summary.json。相较258的质量恢复伴随更多查询，不能当成等预算优势。",'','保留261为质量保留的内部研究候选；250低查询端点和230历史候选继续保存。下一262只读分析77次取消的成本、跨种子重复、分数裕度和自身训练支持，先解释这个小改善，不再扫描变换或立即加网络。','','## 核验、计数与范围','','500条经验正例（含KEEP36、活跃464）、全部11个目标及原始梯度、4681已发布分数（部署同样tanh38）、14268保存前缀的动作/候选ID/增益/成本/查询、137变化宏守恒和全部77取消逐条核验。所有活跃alpha均为1。核验仅重算保存结果，没有重新拟合、网络forward或电网求解。源码输入44b7fe9d先发，全模型/步长19262ad5先发后评分，全预测207fd5a2先发后接结果，并逐文件核对发布Git blob。','','原253来源探针18参考NN+258的22读出=40已完成；261是明确的新有界表示假设，另加22，此数据基础分支累计62。6次旧检查点续训8285更新另列；NN累计604.703秒/用户批准678.156秒上限保留，不能说原600秒内完成。261拟合约0.045秒不等在线时延。','','虽然仍是41参数，输入映射改变函数类及单位L2的功能先验，并非严格同容量。学习参数相关的修正范数界不是固定输出±2硬截断。原父NN/头own12、内部cal4祖先局限仍在，500正例保护不是独立校准；16日期及全部开发结果已暴露，相关前缀/种子/折用途不独立。本轮没有整周闭环、真实控制计时或新电网，不能据此宣称整周质量、实际提速、全链独立OOF或盲测通过。最终8未生成/读取，费用/标签/控制器/排序别名/锁站/128/64/停滞2/.9保持，没有论文或新FRAME比较。']
(O/'GRID261_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
C.save('complete.json',dict(passed=True,new_linear_fits=11,new_scalar_fits=11,constantKEEP=1,wall_s=C.read(O/'fits.json')['wall_s'],scores=4681,sequences=14268,branch_fits62=True,final8_absent=True))
C.archive('audited')
D=R/'论文协作资料/FRAME_有界输入学习_GRID261';D.mkdir(exist_ok=False)
for name in ['GRID261_report.md','matched258_summary.json','matched258_changes.json','gate_preservation_summary.json','gate_cancellations.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,D/name)
assert all(C.sha(p)==C.sha(O/p.name) for p in D.iterdir())
C.save('delivery_check.json',dict(passed=True,collaboration=str(D),files={p.name:C.sha(p) for p in D.iterdir()},actual_archive=str(A)))
msg='GRID261固定tanh38输入完成11线性+11原步长(.045240s)、1KEEP，500保护/4681分数/14268保存路径/137变化守恒核验。相对公共GATE保留55/55回本，7134条逐条增益/最终候选ID相同；取消77条零增益补查、原始省122query，三种子query均降，宏平均13.440525→13.421699(-0.1401%)、NET.096092524→.096119673。77条只有6个相同状态跨两种子重复，不能当77独立样本。对258恢复7失0，NET种子差+.003792983/-.000033721/+.005754269，query全增，保留一小降反例，非对258等预算优势。保留261质量保留内部候选及250低query/230历史；下一262只读解释取消集合/裕度/自身训练支持，不新拟合策略或扫变换。原40探针已完，261另22，分支累计62及6续训/604.703s授权修订披露。旧开发/祖先/整周/实际时延局限保持。'
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;s=p.read_text(encoding='utf8');entry='\n\n## 2026-10-29 — GRID261有界输入学习\n\n'+msg+'\n'
    if name=='RESEARCH_STATE.md':title,rest=s.split('\n',1);s=title+'\n'+entry+rest
    else:s+=entry
    p.write_text(s,encoding='utf8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID261后\n\n1. '+msg+' 完整拟合/评分/核验/归档不重做。\n2. GRID262仅只读77次取消、122raw query和跨种子日期调用重复、分数裕度/有界修正及自身fit8支持。先封存唯一描述分析代码/输入，不产新政策/alpha/阈值/组合/拟合/外层回报；不把三个种子总体省查询说成相同状态重复。\n3. 保留全部日期/种子/反例和250/230/258端点，不按外层扫变换或择种子；原费用标签/基础控制器/排序别名/锁站/128/64/停滞2/.9不改，最终8/新物理/电网/论文/FRAME不动，不安装。公开发布屏障、Githelper串行和路径白名单、实际A归档协作远端核验保持。\n',encoding='utf8')
p=REPO/'research_archive/power-grid-2026/README.md';p.write_text('# 最新：GRID261有界输入学习\n\n[报告](protocols/'+A.name+'/audited/GRID261_report.md)；'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print('GRID261_AUDITED',json.dumps(dict(gate=gate_summary,primary258={k:summary[k] for k in ['changes','restored_profitable','lost_profitable','raw_query_delta']}),ensure_ascii=False))
