"""Independent scalar reconciliation and post-confirmation mechanism diagnosis.

No new forecasts, training labels or tuning. GRID44 remains frozen and intact.
"""
import csv,gzip,hashlib,json,math,statistics,struct,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid45';SRC=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def writecsv(p,values):
    with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(values[0]));w.writeheader();w.writerows(values)
assert not OUT.exists();OUT.mkdir();stamp=time.perf_counter()
D=read(SRC/'design.json');R=read(SRC/'review.json');assert R['passed_saved_data']
index={};paths={};calls={};checked=0
for item in D['execution_order']:
    ident=(item['week'],item['method'],item['repeat'])
    folder=SRC/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
    s=read(folder/'summary.json');a=lines(folder/'steps.jsonl.gz')
    assert s['steps']==len(a)==2017 and s['complete'] and a[-1]['complete']
    assert abs(statistics.fmean([r['raw_cost'] for r in a])*len(a)-s['cost'])<1e-6
    assert abs(sum(r['decision_s'] for r in a)-s['controller_s'])<1e-6
    assert sum(r['simulations'] for r in a)==s['public_forecasts']
    cs=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
    assert sum(c['public_queries'] for c in cs)==s['n1_forecasts']
    index[ident]=s;paths[ident]=a;calls[ident]=cs;checked+=1
family=[];summaries=[];allfirst=[]
for method in [r['method'] for r in D['methods']]:
    local=[]
    for week in D['weeks']:
        p=[];cost=[]
        for repeat in [0,1]:
            s=index[week,method,repeat];f=index[week,'FULL',repeat]
            p.append((f['controller_s']-s['controller_s'])/f['controller_s'])
            cost.append((s['cost']-f['cost'])/max(1,abs(f['cost'])))
        row={'week':week,'method':method,'mean_paired_control_reduction':statistics.mean(p),
             'mean_complete_cost_regret':statistics.mean(cost),'quality_screen':max(cost)<=.005}
        family.append(row);local.append(row)
    summary={'method':method,'macro_speed_reduction':statistics.mean(r['mean_paired_control_reduction'] for r in local),
             'macro_cost_regret':statistics.mean(r['mean_complete_cost_regret'] for r in local),
             'max_cost_regret':max(r['mean_complete_cost_regret'] for r in local),
             'families_quality_screen_passed':sum(r['quality_screen'] for r in local)}
    same=next(s for s in R['summaries'] if s['method']==method)
    if method!='FULL':assert abs(same['macro_time_reduction_full']-summary['macro_speed_reduction'])<1e-12
    assert same['quality_qualified_pairs']==2*summary['families_quality_screen_passed']
    summaries.append(summary)
    if method=='FULL':continue
    for week in D['weeks']:
        a=paths[week,method,0];b=paths[week,'FULL',0]
        # Physical action divergence, not merely rounding in a scalar cost.
        different=next((i for i,(u,v) in enumerate(zip(a,b)) if u['action_hash']!=v['action_hash']),None)
        if different is None:
            allfirst.append({'week':week,'method':method,'first_action_divergence':None,'matched_search_input':None,
                 'full_choice':None,'method_choice':None,'full_rank_in_method_order':None,'method_queries':None,
                 'stop_reason':None,'method_selected_rho':None,'full_selected_rho':None,'fp32_reward_gap':None,
                 'full_choice_already_observed':None,'pure_reward_tie':None});continue
        step=a[different]['step'];mc=next((c for c in calls[week,method,0] if c['step']==step),None)
        fc=next((c for c in calls[week,'FULL',0] if c['step']==step),None)
        matched=mc is not None and fc is not None and mc['state_hash']==fc['state_hash']
        # A same-observation check is only part of matching: base vectors must agree.
        if matched:
            import numpy as np
            def basehash(method,cs,c):
                j=cs.index(c);p=SRC/'runs'/f'r0__{week}__{method}'/f'call{j:03d}_state.npz'
                with np.load(p) as z:return hashlib.sha256(z['base_action'].tobytes()).hexdigest()
            matched=basehash(method,calls[week,method,0],mc)==basehash('FULL',calls[week,'FULL',0],fc)
        x=y=None
        if matched:
            x=next((r for r in mc['outcomes'] if r['pool_id']==mc['chosen_pool_id']),None)
            y=next((r for r in fc['outcomes'] if r['pool_id']==fc['chosen_pool_id']),None)
        allfirst.append({'week':week,'method':method,'first_action_divergence':step,'matched_search_input':matched,
             'full_choice':None if fc is None else fc['chosen_pool_id'],'method_choice':None if mc is None else mc['chosen_pool_id'],
             'full_rank_in_method_order':mc['order'].index(fc['chosen_pool_id'])+1 if matched and fc['chosen_pool_id'] in mc['order'] else None,
             'method_queries':None if mc is None else mc['public_queries'],'stop_reason':None if mc is None else mc.get('stop_reason'),
             'method_selected_rho':None if x is None else x['rho'],'full_selected_rho':None if y is None else y['rho'],
             'fp32_reward_gap':None if x is None or y is None else key(y)[0]-key(x)[0],
             'full_choice_already_observed':None if not matched else fc['chosen_pool_id'] in {r['pool_id'] for r in mc['outcomes']},
             'pure_reward_tie':None if x is None or y is None else key(x)[0]==key(y)[0]})
critical=[r for r in allfirst if r['week']=='2035-04-23_7' and r['method'] in ['LOCAL_GNN0','LOCAL_GNN1','LOCAL_PPO']]
assert len(critical)==3 and all(r['matched_search_input'] and r['first_action_divergence']==540 and r['method_choice']==81 and r['full_choice']==844 and r['fp32_reward_gap']>0 and not r['full_choice_already_observed'] for r in critical)
result={'passed_independent_scalar_audit':True,'runs_checked':checked,'family_macro':summaries,'first_divergences':allfirst,
  'critical_mechanism':'At April23 step540 both original GNNseeds and PPO choose81 after a plateau while rho remains above author safe. FULL chooses844 at another substation. Same physical/base inputs;true reward gap,not tie. This locates the first differing decision;does not prove that one decision caused the full-week cost gap or that ranking has no role.',
  'new_forecasts':0,'new_fits':0,'test_based_training_labels':0,
  'next_scope':'Keep original models and frozen confirmation verdict. Study any proposed common guard on existing development feedback/episodes only;never refit on these8families or rename later reuse as independent confirmation.',
  'wall_s':time.perf_counter()-stamp}
(OUT/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');writecsv(OUT/'family_summary.csv',family);writecsv(OUT/'first_divergences.csv',allfirst)
report=['# GRID45 确认结果的独立复算与缺口定位','',f'独立复算{checked}条交付；新增预测/训练均为0。','',
 '| 方法 | 平均控制提速 | 平均完成周成本差 | 最坏成本差 | 质量筛查日期 |','|---|---:|---:|---:|---:|']
for s in summaries:report.append(f"| {s['method']} | {100*s['macro_speed_reduction']:.3f}% | {100*s['macro_cost_regret']:.5f}% | {100*s['max_cost_regret']:.5f}% | {s['families_quality_screen_passed']}/8 |")
report+=['','4月23日540步：原GNN两个种子和PPO输入相同，均选81；FULL选844。候选844未被检查，真实FP32奖励差约0.00686，不是并列。当前最佳rho=0.92207、FULL=0.91521，两者都未低于作者0.9安全阈值。',
 '现有跨变电站停查规则在这个状态留下了未检查的更优动作。排序影响检查顺序，因此不能简单说“只怪停止，与模型无关”；也不能用首次分歧证明整周代价的单动作因果。',
 '', '联合质量门槛未通过，但全部存活、八日期平均提速约14.6%可保留；平均约0.079%成本差不代表最坏日期符合0.5%，更不是等价证据。',
 '对PPO增量有限、SoftGNN也有强日期表现，不宣称原模型全面领先；它的更高最坏成本同时报告。',
 '', '下一步不调确认集模型：先在既有开发数据检验一条由作者安全阈值限定的共同保护，不新增规则菜单；原版结果和新版本分开保留。']
(OUT/'GRID45_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
files=[Path(__file__).resolve(),SRC/'design.json',SRC/'review.json']
(OUT/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'runs':checked,'summaries':summaries,'critical':critical},indent=2))
