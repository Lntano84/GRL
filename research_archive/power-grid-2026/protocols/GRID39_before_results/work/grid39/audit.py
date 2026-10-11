"""Saved vectors, cost, strict feedback and paired delivered-controller comparison."""
import csv,gzip,hashlib,json,math,os,struct
from collections import Counter
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid39'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
F=read(OUT/'finished.json');D=read(OUT/'design.json');assert F['passed_engineering'] and len(F['summaries'])==12
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
index={};paths={};callsets={};physical=forecasts=static_checks=0
for s in F['summaries']:
    ident=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f'r{ident[0]}__{ident[1]}__{ident[2]}';actual=rows(folder/'steps.jsonl.gz')
    assert len(actual)==s['steps'] and s['illegal']==s['ambiguous']==0
    assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
    assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
    assert sum(r['simulations'] for r in actual)==s['public_forecasts']
    with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
    for i,r in enumerate(actual):
        assert digest(actions[i])==r['action_hash'] and digest(observations[i])==r['after_hash']
        if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
    reference=rows(ROOT/'outputs/grid26/runs'/f"r0__{s['week']}__FULL"/'steps.jsonl.gz');ks=['step','before_hash','action_hash','after_hash','raw_cost','done','complete']
    same=len(actual)==len(reference) and all(all(a[k]==b[k] for k in ks) for a,b in zip(actual,reference));assert same==s['exact_full_trajectory']
    if s['method']=='FULL':assert same
    calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
    byinput={}
    for i,c in enumerate(calls):
        assert len({r['pool_id'] for r in c['outcomes']})==c['public_queries']
        if s['method']=='STATIC_ID':assert c['order']==sorted(c['representatives']) and c['prior_s']==0;static_checks+=1
        valid=[]
        for r in c['outcomes']:
            good=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous']);assert good==r['strict_admissible']
            if good:valid.append(r)
        assert (max(valid,key=key)['pool_id'] if valid else None)==c['chosen_pool_id']
        with np.load(folder/f'call{i:03d}_state.npz') as z:input_key=(digest(z['observation']),digest(z['base_action']))
        assert input_key[0]==c['state_hash'] and input_key not in byinput;byinput[input_key]=c
    profile=read(folder/'components.json');active=[c for c in profile['calls'] if c['in_control']]
    times={name:math.fsum(c['wall_s'] for c in active if c['phase']==name) for name in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
    remainder=s['controller_s']-sum(times.values())-s['all_forecast_s'];assert remainder>=0
    index[ident]={**s,**times,'remaining_s':remainder,'cache_counts':profile['cache_counts'],'safe_cache_counts':profile['safe_cache_counts']}
    paths[ident]=[(r['before_hash'],r['action_hash'],r['after_hash'],r['raw_cost']) for r in actual];callsets[ident]=byinput;physical+=s['steps'];forecasts+=s['public_forecasts']
quality=[];quality_index={}
for ident,s in index.items():
    full=index[(ident[0],ident[1],'FULL')];survival=s['complete'] if full['complete'] else s['steps']>=full['steps']
    comparable=s['steps']==full['steps'];regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if comparable else None
    q={'repeat':ident[0],'week':ident[1],'method':ident[2],'steps':s['steps'],'complete':s['complete'],'no_earlier_failure':bool(survival),'cost_regret':regret,
       'quality_qualified':bool(survival and (regret is not None and regret<=.005))}
    quality.append(q);quality_index[ident]=q
repeats=[]
# One timing repeat; no nonexistent repeat comparison.
summaries=[]
for m in D['methods']:
    g=[r for r in index.values() if r['method']==m['method']];q=[r for r in quality if r['method']==m['method']]
    summaries.append({'method':m['method'],'runs':len(g),'control_s':math.fsum(r['controller_s'] for r in g),'source_s':math.fsum(r['n1_source_s'] for r in g),'queries':sum(r['n1_forecasts'] for r in g),
       'quality_qualified_pairs':sum(r['quality_qualified'] for r in q),'completed_runs':sum(r['complete'] for r in g),
       'max_comparable_cost_regret':max((r['cost_regret'] for r in q if r['cost_regret'] is not None),default=None)})
comparisons=[]
for method,opponent in [('REVISED_GNN0','FULL'),('REVISED_GNN1','FULL')]:
    pairs=[];matched=[]
    for repeat in [0]:
        for week in D['weeks']:
            a=(repeat,week,method);b=(repeat,week,opponent);x,y=index[a],index[b];comparable=x['steps']==y['steps']
            pairs.append({'repeat':repeat,'week':week,'comparable_steps':comparable,'same_path':paths[a]==paths[b],
               'method_control_s':x['controller_s'],'opponent_control_s':y['controller_s'],'method_queries':x['n1_forecasts'],'opponent_queries':y['n1_forecasts'],
               'time_reduction':1-x['controller_s']/y['controller_s'] if comparable else None,
               'cost_difference_of_full':(x['cost']-y['cost'])/max(1,abs(index[(repeat,week,'FULL')]['cost'])) if comparable else None,
               'method_quality_qualified':quality_index[a]['quality_qualified'],'opponent_quality_qualified':quality_index[b]['quality_qualified']})
            for state in sorted(set(callsets[a])&set(callsets[b])):
                u,v=callsets[a][state],callsets[b][state];assert u['representatives']==v['representatives'] and u['aliases']==v['aliases']
                ux={r['pool_id']:r for r in u['outcomes']};vx={r['pool_id']:r for r in v['outcomes']}
                for i in set(ux)&set(vx):
                    for k in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']:assert ux[i][k]==vx[i][k]
                matched.append({'repeat':repeat,'week':week,'step':u['step'],'method_queries':u['public_queries'],'opponent_queries':v['public_queries'],'same_choice':u['returned_action_hash']==v['returned_action_hash']})
    comparable=all(p['comparable_steps'] for p in pairs)
    comparisons.append({'method':method,'opponent':opponent,'all_survival_exposures_comparable':comparable,
       'pooled_control_reduction':1-math.fsum(p['method_control_s'] for p in pairs)/math.fsum(p['opponent_control_s'] for p in pairs) if comparable else None,
       'pooled_query_reduction':1-sum(p['method_queries'] for p in pairs)/sum(p['opponent_queries'] for p in pairs) if comparable else None,
       'positive_time_pairs':sum(p['time_reduction'] is not None and p['time_reduction']>0 for p in pairs),'pairs':pairs,'matched_calls':matched})
R={'passed_saved_data':True,'runs':12,'physical_rows':physical,'public_forecasts':forecasts,'static_order_checks':static_checks,'summaries':summaries,'comparisons':comparisons,'quality':quality,'repeat_checks':repeats,
   'new_fits':0,'reserved_evaluations':0,'warning':'4 exposed development dates and one timing repeat. Original GRID37 timings are historical, not paired to new models. Both seeds retained; .5% is a quality investment tolerance, not statistical equivalence. Common author PPO/caches remain in all arms. Saved-data audit is not independent AC feasibility solving. Incomparable earlier failure never counts as speed win.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
with (OUT/'method_summary.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,list(summaries[0]));w.writeheader();w.writerows(summaries)
lines=['# GRID39 一版目标修订的实际闭环评价','',f'12条；{physical}物理步；{forecasts}公开预测。', '',
 '| 方法 | 控制秒总计 | 查询 | 质量筛查通过 | 完成 | 最大可比成本差 |','|---|---:|---:|---:|---:|---:|']
for r in summaries:
    cost='不可比' if r['max_comparable_cost_regret'] is None else f"{100*r['max_comparable_cost_regret']:.5f}%"
    lines.append(f"| {r['method']} | {r['control_s']:.3f} | {r['queries']} | {r['quality_qualified_pairs']}/4 | {r['completed_runs']}/4 | {cost} |")
lines+=['','| 方法 | 对照 | 控制时间减少 | 查询减少 | 时间正向配对 |','|---|---|---:|---:|---:|']
for c in comparisons:
    speed='不可比' if c['pooled_control_reduction'] is None else f"{100*c['pooled_control_reduction']:.3f}%"
    query='不可比' if c['pooled_query_reduction'] is None else f"{100*c['pooled_query_reduction']:.3f}%"
    lines.append(f"| {c['method']} | {c['opponent']} | {speed} | {query} | {c['positive_time_pairs']}/4 |")
lines+=['',R['warning'],'','质量、时间和选择变化一起看，不把通过容差理解为逐步等同FULL；失败/较慢/较差模型均保留。']
(OUT/'GRID39_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:12 revised-model development comparisons. No new fits/final dates.\n',encoding='utf-8')
print(json.dumps({'passed':True,'summaries':summaries,'comparisons':[{k:c[k] for k in ['method','opponent','all_survival_exposures_comparable','pooled_control_reduction','pooled_query_reduction','positive_time_pairs']} for c in comparisons]},indent=2))
