"""Audit the fixed scalar control, preserve all outcomes and publish current state."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R,O,A,REPO=C.R,C.O,C.A,C.REPO
def main():
    C.scored();assert C.read(O/'review.json')['passed'] and not (O/'complete.json').exists();rec=C.read(O/'calibration_receipts.json');known=C.read(C.Q/'known_query_mass.json');positive=0;score_count=0
    for row in rec:
        pool=row['pool'];z=np.load(C.P/'fit_inputs'/f'{pool}.npz');parent=np.load(C.S/'models'/f'{pool}.npz');new=np.load(O/'models'/f'{pool}.npz');t=new['theta'];a=float(new['alpha'][0]);cap=float(parent['alpha'][0]);mass=np.array(known[pool]['query_mass']);b=z['b'];d=z['X']@t[:-1]+t[-1]
        assert np.array_equal(t,parent['theta']) and 0<=a<=cap and np.all((b+a*d)[z['y']>0]>=0);positive+=int((z['y']>0).sum());selected=float(mass[b+a*d<0].sum());assert selected<=row['parent_STOP_budget'] and selected==row['selected_mass']
        if row['details'].get('active'):
            bound=row['details']['analytic_bound'];witness=.5*(bound+cap);assert bound<witness<=cap and float(mass[b+witness*d<0].sum())>row['parent_STOP_budget']
        else:assert a==cap
        h=np.load(C.P/'public_held_inputs'/f'{pool}.npz');saved=np.load(O/'held_predictions'/f'{pool}.npz');expected=h['b']+a*(np.sum(h['X']*t[:-1],axis=1)+t[-1]);assert np.array_equal(expected,saved['logit']) and np.array_equal(expected>=0,saved['keep']);score_count+=len(expected)
    assert positive==90 and score_count==4681
    for phase,name,pubname in [('calibrated','calibrated_freeze.json','pre_score_publication.json'),('scored','scored_freeze.json','pre_join_publication.json')]:
        pub=C.read(O/pubname);assert C.git('show',pub['git_sha']+':'+(A/phase/name).relative_to(REPO).as_posix())==(O/name).read_bytes()
    cases={r['sample_id']:r for r in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};task=C.read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=C.read(O/'oof_trajectories.json')
    for r in tr:
        c=cases[r['sample_id']];j=r['baseline_stop']+int(r['append']);assert r['gain']==float(c['gains'][:j].sum()) and r['queries']==int(c['queries'][:j].sum()) and r['chosen_pool_id']==c['chosen'][j];assert abs(r['net_utility']-(r['gain']-fee*r['queries'])/scale)<1e-12
    assert len(tr)==28536
    pred=C.read(O/'gate_predictions.json');changes=C.read(O/'changed_choices.json');family=C.read(O/'family_summary.json');paired=C.read(O/'paired_summary.json');counts=C.read(O/'gate_counts.json');assert len(changes)==12 and sum(r['actual_next_NET']>0 for r in changes)==1 and all(r['query_delta']>0 for r in changes)
    major=next(r for r in changes if r['actual_next_NET']>0);assert major['physical_state_group']=='2037-01-05_101|call001'
    repeated=[]
    for sid in ['077c228d0254d5b751b98a62','539b1f1919b80a68cb2264cb']:
        rr=[r for r in pred if r['sample_id']==sid and r['seed'] in [1,2]];assert len(rr)==2 and all(not r['budget_continue'] and r['base_continue'] for r in rr);repeated.append(dict(sample_id=sid,seeds=[1,2],still_cancelled=True))
    dates=C.read(O/'search_metrics.json')['MIXED_seed0']['per_date'];n={r['date']:r['cases'] for r in dates};decomp=[]
    for seed in range(3):
        rr=[r for r in changes if r['seed']==seed];recovered=sum(max(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);wasted=sum(min(r['NET_delta'],0)/(16*n[r['date']]) for r in rr);actual=next(r for r in paired if r['base']=='MIXED' and r['seed']==seed)['macro_delta']['mean_net_utility'];assert abs(recovered+wasted-actual)<1e-12;decomp.append(dict(seed=seed,recovered_profitable=recovered,added_unprofitable=wasted,total=actual,raw_added_queries=sum(r['query_delta'] for r in rr)))
    C.save('effect_decomposition.json',dict(per_seed=decomp,restored_major_case=major,repeated_savings_preserved=repeated,restored_profitable_rows=1,added_unprofitable_rows=11,raw_profitable_queries=sum(r['query_delta'] for r in changes if r['actual_next_NET']>0),raw_unprofitable_queries=sum(r['query_delta'] for r in changes if r['actual_next_NET']<=0),correlated_rows_not_independent_trials=True))
    C.save('decision.json',dict(retain208_main=True,retain222_tradeoff=True,retain230_as_controlled_internal_candidate=True,not_claim_superiority_to222=True,not_equal_held_budget=True,reason='Recovered dominant missed opportunity with training-only mass constraint,preserved repeated savings,but11additional unprofitable continuations and averageNET essentially tied/slightlybelow222.',next='GRID231 read-only characterize residual profitable misses and unnecessary restorations with parent and fit support;then qualify one finite learning change without held-selected thresholds,seeds or budgets.'))
    lines=['# GRID230：控制训练内补查取消强度后的结果','','这轮没有重新训练模型权重，只按已封存规则计算11个步长，另1个池保持常量继续。预算由各模型自身训练记录上的旧222取消查询代价确定；没有用外层结果选择步长。','',
    '## 有价值的变化','',
    '此前漏掉的1月5日大收益机会已经恢复，两处在种子1、2重复省查询的状态也保持了停止。相对于227，回本机会保留数从38/55回到39/55。这个结果支持继续研究修正强度与学习方向的配合，不能据此认定已经超过旧222。','',
    '新约束只改变3个池：f0_s0、f1_s1、f3_s0，其余9个池的步长不变。f0_s0从0.275001降至0.191739；这是训练内查询量约束给出的交点，没有恢复成旧0.138295，也没有针对1月5日设置例外。','',
    '## 质量和查询代价','',
    f"日期宏平均再对三种子平均：227的固定费代理净效用{family['MIXED']['mean_net_utility']:.9f}，新方案{family['BUDGET']['mean_net_utility']:.9f}；候选查询{family['MIXED']['mean_queries']:.6f}变为{family['BUDGET']['mean_queries']:.6f}，即时增益{family['MIXED']['mean_gain']:.10f}变为{family['BUDGET']['mean_gain']:.10f}。改善质量同时增加了查询，不能称等预算优势。",
    f"旧222的代理净效用为{family['STOP_ONLY']['mean_net_utility']:.9f}、查询{family['STOP_ONLY']['mean_queries']:.6f}；新方案净效用与它几乎持平、略低，查询略少。动作一致率从旧222的{100*family['STOP_ONLY']['action_match_rate']:.5f}%变为{100*family['BUDGET']['action_match_rate']:.5f}%。这些变化很小，不作显著性或总体胜出的判断。",'',
    '相对227，12处决定恢复了补查：1条回本、11条不回本；按原记录直接相加，回本记录增加11个候选查询，不回本记录增加216个。记录包含相关前缀，不能把12处当成12个独立试验。种子0净效用提高，种子1因多查无收益候选略降，种子2完全不变。','',
    '相对222，三个种子的净效用变化依次为：']
    for row in paired:
        if row['base']=='STOP_ONLY':lines.append(f"种子{row['seed']}：净效用{row['macro_delta']['mean_net_utility']:+.9f}，查询{row['macro_delta']['mean_queries']:+.6f}，即时增益{row['macro_delta']['mean_gain']:+.10f}。")
    lines+=['','## 验证与后续','',
    '原227全部权重逐项不变，90条训练STOP正例仍被保留；训练查询代价约束和交点最大性、4681个发布预测、28536条四臂保存序列及原控制结果已核对。源码、标量、预测分别发布后才进入下一步。',
    'GRID229第一次合成资格检验的一次ULP见证会被浮点舍入成零；错误与原源码保留，revision02只修复合成最大性见证，实际标量规则未变。原225/226历史失败也保持封存，均未重做。',
    '下一步只读检查剩余回本误停和新增无利补查，继续保留小而重复的节省，依据训练内支持固定一个有限学习改动。暂不扩大数据或网络，不因平均近似持平就丢掉这条线索。',
    '当前是反复暴露开发数据上的局部代理比较，不等于整周质量或控制时延；没有新增物理仿真、读取最终8、修改论文或新做FRAME比较。']
    (O/'GRID230_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
    C.scored();C.save('independent_audit.json',dict(passed=True,all_parent_theta_identical=True,all90_fitpositives_and_querymass_bounds_verified=True,active_crossing_maximality_checked_without_refitting=True,all4681_published_scores_reconstructed=True,all28536_saved_prefix_costs_choices_verified=True,publication_barriers_verified=True,all12changes_and2repeated_savings_checked=True,no_new_physical_solver=True));C.save('complete.json',dict(passed=True,new_scalar_calibrations=11,new_endpoint_fits=0,new_NN_fits=0,held_predictions=4681,saved_trajectories=28536,new_physics=0,final8_absent=True));C.archive('audited')
    dest=R/'论文协作资料/FRAME_补查强度对照_GRID230';dest.mkdir(exist_ok=False)
    for name in ['GRID230_report.md','family_summary.json','paired_summary.json','gate_counts.json','effect_decomposition.json','decision.json','independent_audit.json','complete.json']:shutil.copy2(O/name,dest/name)
    msg='GRID229仅训练内资格：10/11池alpha增大，但取消查询质量仅3池增加。GRID230固定训练结构权重×已知查询预算<=自身旧222取消代价，只算11标量，不重训端点；3池改步长。恢复Jan5大收益并保留两处重复省查询，回本38→39/55。对227净效用.095320317→.095926664，查询13.148416→13.171604，存在多查询质量折中；新增12补查含1回本/11无利，原始查询11/216。对222净效用几乎持平略低(.095926811)，查询略少(13.174668)，三种子2小正1小负，不宣称胜出。全权重/预测发布后连接，90正例/4681分数/28536序列核验。下一步231只读残余误停与无利恢复，保留局部改善。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-28 — GRID229至230训练查询强度对照\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':first,rest=t.split('\n',1);t=first+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID230后\n\n1. GRID229revision02训练内资格与GRID230固定11标量对照完成，不重做；旧229合成舍入检验失败与225/226历史失败保留。\n2. GRID231只读：核查230相对222残余回本误停、Nov16恢复与两处重复节省，及相对227的11条无利恢复。保留全部种子/日期/前缀和反例，不按外层调步长/阈值/费用。\n3. 用自身训练支持明确下一有限学习改动：当前查询强度约束已恢复大损失，不能把多查询质量提高当学习排序提升。输入限制或匹配学习目标先资格，不盲扫。\n4. 各学习阶段最多22拟合/校准、600秒。最终8/物理/电网/论文/新FRAME不动；原费用/标签/基础控制器/排序/别名/锁站/128/64/停滞2/.9保留。自主推进直至用户喊停。\n',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(('# 最新：GRID230训练查询强度对照\n\n[报告](protocols/'+A.name+'/audited/GRID230_report.md)；大收益恢复、重复节省保留，整体近旧222，额外查询和全部反例保留。\n\n').encode('utf8')+p.read_bytes());print('GRID230_AUDITED',json.dumps(dict(counts=counts,decomp=decomp)))
if __name__=='__main__':main()
