"""Frozen saved-data audit: date-family quality, matched timing, all seeds kept.

No AC re-solve or RF/GNN refit. Bootstrap repeats date families, never runs.
"""
import csv,gzip,hashlib,json,math,os,struct
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid44'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def mean(values):return math.fsum(values)/len(values) if values else None
def interval(values):
    if not values:return None
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(20261011)
    sampled=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    return list(map(float,np.quantile(sampled,[.025,.975])))
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,data):
    if not data:return
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)

def main():
    D=read(OUT/'design.json');F=read(OUT/'finished.json')
    assert F['passed_engineering'] and len(F['summaries'])==len(D['execution_order'])==96
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    index={};paths={};callsets={};physical=forecasts=exhaustive=0
    expected={(r['repeat'],r['week'],r['method']) for r in D['execution_order']}
    for s in F['summaries']:
        ident=(s['repeat'],s['week'],s['method']);assert ident in expected and ident not in index
        folder=OUT/'runs'/f'r{ident[0]}__{ident[1]}__{ident[2]}'
        actual=rows(folder/'steps.jsonl.gz');assert len(actual)==s['steps'] and s['illegal']==s['ambiguous']==0
        assert s['exact_full_trajectory'] is None and s['first_full_divergence'] is None
        assert abs(math.fsum(r['raw_cost'] for r in actual)-s['cost'])<1e-6
        assert abs(math.fsum(r['decision_s'] for r in actual)-s['controller_s'])<1e-6
        assert sum(r['simulations'] for r in actual)==s['public_forecasts']
        assert [r['step'] for r in actual]==list(range(1,len(actual)+1))
        with np.load(folder/'vectors.npz') as z:actions=z['action'];observations=z['observation']
        assert len(actions)==len(observations)==len(actual)
        for i,r in enumerate(actual):
            assert digest(actions[i])==r['action_hash'] and digest(observations[i])==r['after_hash']
            assert r['full_reference_match'] is None
            if r['ledger'] is not None:assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
        calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
        assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
        byinput={}
        for i,c in enumerate(calls):
            ids=[r['pool_id'] for r in c['outcomes']]
            assert len(set(ids))==c['public_queries'] and set(ids)<=set(c['representatives'])
            if s['method']=='FULL':assert set(ids)==set(c['representatives']);exhaustive+=1
            valid=[]
            for r in c['outcomes']:
                good=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
                assert good==r['strict_admissible']
                if good:valid.append(r)
            chosen=max(valid,key=key)['pool_id'] if valid else None
            assert chosen==c['chosen_pool_id']
            if chosen is None:assert c['returned_action_hash'] is None
            with np.load(folder/f'call{i:03d}_state.npz') as z:ik=(digest(z['observation']),digest(z['base_action']))
            assert ik[0]==c['state_hash'] and ik not in byinput;byinput[ik]=c
        profile=read(folder/'components.json');active=[c for c in profile['calls'] if c['in_control']]
        times={name:math.fsum(c['wall_s'] for c in active if c['phase']==name) for name in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
        remainder=s['controller_s']-sum(times.values())-s['all_forecast_s'];assert remainder>=0
        index[ident]={**s,**times,'remaining_s':remainder,'p95_decision_s':float(np.quantile([r['decision_s'] for r in actual],.95))}
        paths[ident]=[(r['step'],r['before_hash'],r['action_hash'],r['after_hash'],r['raw_cost'],r['done'],r['complete']) for r in actual]
        callsets[ident]=byinput;physical+=s['steps'];forecasts+=s['public_forecasts']
    assert set(index)==expected
    quality=[];qi={};repeat_checks=[]
    for ident,s in index.items():
        full=index[(ident[0],ident[1],'FULL')]
        survival=bool(s['complete'] if full['complete'] else s['steps']>=full['steps'])
        sameh=s['steps']==full['steps'];costeligible=bool(full['complete'] and s['complete'])
        regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if costeligible else None
        q={'repeat':ident[0],'week':ident[1],'method':ident[2],'steps':s['steps'],'complete':s['complete'],
           'full_steps':full['steps'],'full_complete':full['complete'],'no_earlier_failure':survival,
           'equal_horizon_full':sameh,'complete_week_cost_regret':regret,
           'failed_equal_horizon_cost_difference_diagnostic':(s['cost']-full['cost'])/max(1,abs(full['cost'])) if sameh and not full['complete'] else None,
           'quality_qualified':bool(survival and (not full['complete'] or (regret is not None and regret<=.005))),
           'exact_full_path':paths[ident]==paths[(ident[0],ident[1],'FULL')],
           'first_full_divergence':next((a[0] for a,b in zip(paths[ident],paths[(ident[0],ident[1],'FULL')]) if a!=b),None)}
        quality.append(q);qi[ident]=q
    for week in D['weeks']:
        for m in D['methods']:
            a=(0,week,m['method']);b=(1,week,m['method'])
            # Report reproducibility rather than discard a nondeterministic run.
            repeat_checks.append({'week':week,'method':m['method'],'same_path':paths[a]==paths[b],
                                  'repeat0_steps':index[a]['steps'],'repeat1_steps':index[b]['steps']})
    pairs_to_compare=[(m['method'],'FULL') for m in D['methods'] if m['method']!='FULL']
    pairs_to_compare += [(g,o) for g in ['LOCAL_GNN0','LOCAL_GNN1'] for o in ['LOCAL_PPO','SOFTGNN0','SOFTGNN1']]
    comparisons=[];familytable=[];matched_checks=0
    for method,opponent in pairs_to_compare:
        pairs=[];familyrows=[]
        for week in D['weeks']:
            local=[]
            for repeat in [0,1]:
                a=(repeat,week,method);b=(repeat,week,opponent);x,y=index[a],index[b]
                comparable=x['steps']==y['steps']
                p={'repeat':repeat,'week':week,'method':method,'opponent':opponent,'equal_horizon':comparable,
                   'same_path':paths[a]==paths[b],'method_control_s':x['controller_s'],'opponent_control_s':y['controller_s'],
                   'method_queries':x['n1_forecasts'],'opponent_queries':y['n1_forecasts'],
                   'method_source_s':x['n1_source_s'],'opponent_source_s':y['n1_source_s'],
                   'time_reduction':1-x['controller_s']/y['controller_s'] if comparable else None,
                   'cost_difference_of_full':(x['cost']-y['cost'])/max(1,abs(index[(repeat,week,'FULL')]['cost'])) if x['complete'] and y['complete'] and index[(repeat,week,'FULL')]['complete'] else None,
                   'method_quality_qualified':qi[a]['quality_qualified'],'opponent_quality_qualified':qi[b]['quality_qualified']}
                pairs.append(p);local.append(p)
                for state in sorted(set(callsets[a])&set(callsets[b])):
                    u,v=callsets[a][state],callsets[b][state]
                    assert u['representatives']==v['representatives'] and u['aliases']==v['aliases']
                    ux={r['pool_id']:r for r in u['outcomes']};vx={r['pool_id']:r for r in v['outcomes']}
                    for i in set(ux)&set(vx):
                        for k in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']:assert ux[i][k]==vx[i][k]
                        matched_checks+=1
            both=all(p['equal_horizon'] for p in local)
            fr={'week':week,'method':method,'opponent':opponent,'both_repeats_equal_horizon':both,
                'mean_time_reduction':mean([p['time_reduction'] for p in local]) if both else None,
                'mean_complete_cost_difference_of_full':mean([p['cost_difference_of_full'] for p in local]) if all(p['cost_difference_of_full'] is not None for p in local) else None,
                'method_quality_both_repeats':all(p['method_quality_qualified'] for p in local),
                'opponent_quality_both_repeats':all(p['opponent_quality_qualified'] for p in local)}
            familyrows.append(fr);familytable.append(fr)
        comparable=[p for p in pairs if p['equal_horizon']];values=[r['mean_time_reduction'] for r in familyrows if r['both_repeats_equal_horizon']]
        comparisons.append({'method':method,'opponent':opponent,'comparable_families':len(values),
          'macro_control_reduction':mean(values),'family_bootstrap95':interval(values),
          'positive_families':sum(v>0 for v in values),
          'pooled_equal_horizon_control_reduction':1-math.fsum(p['method_control_s'] for p in comparable)/math.fsum(p['opponent_control_s'] for p in comparable) if comparable else None,
          'pooled_equal_horizon_query_reduction':1-sum(p['method_queries'] for p in comparable)/sum(p['opponent_queries'] for p in comparable) if comparable and sum(p['opponent_queries'] for p in comparable) else None,
          'pooled_equal_horizon_source_reduction':1-math.fsum(p['method_source_s'] for p in comparable)/math.fsum(p['opponent_source_s'] for p in comparable) if comparable and math.fsum(p['opponent_source_s'] for p in comparable)>0 else None,
          'families':familyrows,'pairs':pairs})
    summaries=[]
    for m in D['methods']:
        method=m['method'];g=[r for r in index.values() if r['method']==method];q=[r for r in quality if r['method']==method]
        comp=next((c for c in comparisons if c['method']==method and c['opponent']=='FULL'),None)
        costs=[r['complete_week_cost_regret'] for r in q if r['complete_week_cost_regret'] is not None]
        passed=all(r['quality_qualified'] for r in q) and comp is not None and comp['macro_control_reduction'] is not None and comp['macro_control_reduction']>=.05 and comp['positive_families']>=math.ceil(.75*comp['comparable_families'])
        summaries.append({'method':method,'runs':len(g),'completed_runs':sum(r['complete'] for r in g),
           'no_earlier_failure_pairs':sum(r['no_earlier_failure'] for r in q),'quality_qualified_pairs':sum(r['quality_qualified'] for r in q),
           'complete_cost_comparable_pairs':len(costs),'max_complete_week_cost_regret':max(costs) if costs else None,
           'control_s':math.fsum(r['controller_s'] for r in g),'source_s':math.fsum(r['n1_source_s'] for r in g),'queries':sum(r['n1_forecasts'] for r in g),
           'mean_family_p95_decision_s':mean([r['p95_decision_s'] for r in g]),
           'macro_time_reduction_full':None if comp is None else comp['macro_control_reduction'],
           'investment_screen_passed':None if method=='FULL' else passed})
    own=[c for c in comparisons if c['method'] in ['LOCAL_GNN0','LOCAL_GNN1'] and c['opponent']=='FULL']
    ownfamily=[]
    for week in D['weeks']:
        vals=[r['mean_time_reduction'] for c in own for r in c['families'] if r['week']==week]
        if all(v is not None for v in vals):ownfamily.append({'week':week,'seed_average_time_reduction':mean(vals)})
    ownsummary={'comparable_families':len(ownfamily),'macro_control_reduction':mean([r['seed_average_time_reduction'] for r in ownfamily]),
                'family_bootstrap95':interval([r['seed_average_time_reduction'] for r in ownfamily]),'families':ownfamily,
                'both_model_seeds_screen_passed':all(r['investment_screen_passed'] for r in summaries if r['method'] in ['LOCAL_GNN0','LOCAL_GNN1'])}
    R={'passed_saved_data':True,'runs':96,'date_families':8,'physical_rows':physical,'public_forecasts':forecasts,
       'exhaustive_full_calls_checked':exhaustive,'matched_feedback_entries_checked':matched_checks,
       'new_fits':0,'reserved_evaluations':96,'summaries':summaries,'comparisons':comparisons,'quality':quality,
       'repeat_checks':repeat_checks,'own_seed_average':ownsummary,
       'warning':'First reserved date-family confirmation of frozen controller/model;8families from one IDF2023 grid. Timing repeats/model seeds are not independent families. FULL failed weeks retained; no speed ratio for unequal horizon; cost primary only on completed weeks. .5%screen is not equivalence. SoftGNN is resource-matched adaptation; authorPPO training exposure unknown. Audit uses saved vectors/feedback,not independent ACsolve or refit. No graph-only novelty established.'}
    save('review.json',R);csvsave('method_summary.csv',summaries);csvsave('family_comparisons.csv',familytable);csvsave('quality.csv',quality);csvsave('repeat_checks.csv',repeat_checks)
    lines=['# GRID44 八个预留日期家族确认','',f'96条串行回放；{physical}物理步；{forecasts}公开预测。','',
      '| 方法 | 完成 | 无提前失败 | 质量筛查 | 完成周最大成本差 | 对FULL宏平均控制时间减少 |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        cost='无可比完成周' if r['max_complete_week_cost_regret'] is None else f"{100*r['max_complete_week_cost_regret']:.5f}%"
        speed='参照' if r['macro_time_reduction_full'] is None else f"{100*r['macro_time_reduction_full']:.3f}%"
        lines.append(f"| {r['method']} | {r['completed_runs']}/16 | {r['no_earlier_failure_pairs']}/16 | {r['quality_qualified_pairs']}/16 | {cost} | {speed} |")
    lines+=['','| 方法 | 对照 | 可比家族 | 宏平均提速 | 家族bootstrap95% | 时间正向家族 |','|---|---|---:|---:|---|---:|']
    for c in comparisons:
        speed='不可比' if c['macro_control_reduction'] is None else f"{100*c['macro_control_reduction']:.3f}%"
        ci='无' if c['family_bootstrap95'] is None else '['+', '.join(f'{100*v:.3f}%' for v in c['family_bootstrap95'])+']'
        lines.append(f"| {c['method']} | {c['opponent']} | {c['comparable_families']} | {speed} | {ci} | {c['positive_families']} |")
    lines+=['',f"两原模型种子均通过投资筛查：{ownsummary['both_model_seeds_screen_passed']}。",'',R['warning'],
            '',f"两次计时物理路径一致：{sum(r['same_path'] for r in repeat_checks)}/{len(repeat_checks)}。",'所有早失败/成本较差/较慢对照均保留；预留日期使用后不再宣称未暴露。']
    (OUT/'GRID44_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:96reserved-family runs,all seeds/failures retained. No pending GRID44 worker.\n',encoding='utf-8')
    print(json.dumps({'passed':True,'runs':96,'physical_rows':physical,'summaries':summaries,'own_seed_average':ownsummary},indent=2))

if __name__=='__main__':main()
