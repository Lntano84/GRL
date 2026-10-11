"""All delivered costs, vectors, dispatched actions, paired repeat and component accounting."""
import csv,gzip,hashlib,json,math,os,struct
from pathlib import Path
from collections import Counter,defaultdict
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid34'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(x):return struct.unpack('f',struct.pack('f',x['reward']))[0],-x['pool_id']
F=load(OUT/'finished.json');D=load(OUT/'design.json');assert F['passed_engineering'] and len(F['summaries'])==48
for p,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
records=[];index={};paths={};calls_by_key={};physical=forecasts=0
for s in F['summaries']:
    item=(s['repeat'],s['week'],s['method']);folder=OUT/'runs'/f'r{item[0]}__{item[1]}__{item[2]}'
    actual=rows(folder/'steps.jsonl.gz');meta=load(folder/'metadata.json')
    assert len(actual)==s['steps'] and s['illegal']==0 and s['ambiguous']==0
    assert abs(math.fsum(x['raw_cost'] for x in actual)-s['cost'])<1e-6
    assert abs(math.fsum(x['decision_s'] for x in actual)-s['controller_s'])<1e-6
    assert sum(x['simulations'] for x in actual)==s['public_forecasts']
    with np.load(folder/'vectors.npz') as z:
        for i,row in enumerate(actual):
            assert digest(z['action'][i])==row['action_hash'] and digest(z['observation'][i])==row['after_hash']
            if row['ledger'] is not None:assert abs(row['ledger']['recomputed_raw_cost']-row['raw_cost'])<=row['ledger_tolerance']
    reference=rows(ROOT/'outputs/grid26/runs'/f"r0__{s['week']}__FULL"/'steps.jsonl.gz')
    keys=['step','before_hash','action_hash','after_hash','raw_cost','done','complete']
    same=len(actual)==len(reference) and all(all(a[k]==b[k] for k in keys) for a,b in zip(actual,reference))
    assert same==s['exact_full_trajectory']
    if s['method']=='FULL':assert same
    calls=[load(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
    assert len(calls)==s['n1_calls'] and sum(c['public_queries'] for c in calls)==s['n1_forecasts']
    queries={}
    for i,c in enumerate(calls):
        assert len({x['pool_id'] for x in c['outcomes']})==c['public_queries']
        valid=[x for x in c['outcomes'] if x['strict_admissible']]
        assert (max(valid,key=key)['pool_id'] if valid else None)==c['chosen_pool_id']
        for x in c['outcomes']:
            assert x['strict_admissible']==bool(0<x['rho']<c['rho_limit'] and not x['done'] and not x['exceptions'] and not x['illegal'] and not x['ambiguous'])
        with np.load(folder/f'call{i:03d}_state.npz') as z:base_hash=digest(z['base_action'])
        identifier=(c['state_hash'],base_hash);assert identifier not in queries
        queries[identifier]=c
    profile=load(folder/'components.json');active=[x for x in profile['calls'] if x['in_control']]
    times={name:math.fsum(x['wall_s'] for x in active if x['phase']==name) for name in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
    other=s['controller_s']-sum(times.values())-s['all_forecast_s'];assert other>=0
    groups=defaultdict(list)
    for x in profile['solver_records']:
        if x['in_control'] and x['solver'] is not None:groups[(x['phase'],x['owner'],x['solver'])].append(x)
    stats=[]
    for (phase,owner,solver),g in sorted(groups.items()):
        value={'phase':phase,'owner':owner,'solver':solver,'calls':len(g),'statuses':dict(Counter(x['status'] for x in g))}
        for k in ['wall_s','compilation_s','backend_s','iterations']:
            present=[x[k] for x in g if x[k] is not None];value[k]=math.fsum(present) if present else None;value[k+'_missing']=len(g)-len(present)
        stats.append(value)
    records.append({**s,**times,'remaining_s':other,'component_calls':dict(Counter(x['phase'] for x in active)),
        'solver_statistics':stats,'cache_counts':profile['cache_counts'],'safe_cache_counts':profile['safe_cache_counts']})
    index[item]=records[-1];paths[item]=[(r['before_hash'],r['action_hash'],r['after_hash'],r['raw_cost']) for r in actual];calls_by_key[item]=queries
    physical+=s['steps'];forecasts+=s['public_forecasts']
quality=[]
for item,s in index.items():
    full=index[(item[0],item[1],'FULL')]
    completion_ok=s['complete'] if full['complete'] else s['steps']>=full['steps']
    regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if s['steps']==full['steps'] else None
    quality.append({'repeat':item[0],'week':item[1],'method':item[2],'complete':s['complete'],'steps':s['steps'],
        'no_new_earlier_failure':bool(completion_ok),'comparable_cost_regret':regret,
        'cost_within_pilot_scale':None if regret is None else regret<=.005})
repeat_checks=[]
for week in D['weeks']:
    for m in D['methods']:
        a,b=index[(0,week,m['method'])],index[(1,week,m['method'])]
        repeat_checks.append({'week':week,'method':m['method'],'same_delivered_path':paths[(0,week,m['method'])]==paths[(1,week,m['method'])],
            'same_queries':a['n1_forecasts']==b['n1_forecasts'],'same_cost':a['cost']==b['cost']})
comparisons=[]
for g in ['LOCAL_GNN0','LOCAL_GNN1']:
    for opponent in ['FULL','LOCAL_PPO','LOCAL_MLP0','LOCAL_MLP1']:
        pairs=[];matched=[]
        for repeat in [0,1]:
            for week in D['weeks']:
                ik=(repeat,week,g);jk=(repeat,week,opponent);a,b=index[ik],index[jk]
                pairs.append({'repeat':repeat,'week':week,'gnn_control_s':a['controller_s'],'opponent_control_s':b['controller_s'],
                    'control_reduction':1-a['controller_s']/b['controller_s'],'gnn_queries':a['n1_forecasts'],'opponent_queries':b['n1_forecasts'],
                    'query_reduction':1-a['n1_forecasts']/b['n1_forecasts'],
                    'gnn_source_s':a['n1_source_s'],'opponent_source_s':b['n1_source_s'],
                    'same_delivered_path':paths[ik]==paths[jk],
                    'cost_difference_of_full':(a['cost']-b['cost'])/max(1,abs(index[(repeat,week,'FULL')]['cost'])) if a['steps']==b['steps'] else None})
                ga,gb=calls_by_key[ik],calls_by_key[jk]
                for state in sorted(set(ga)&set(gb)):
                    x,y=ga[state],gb[state];assert x['representatives']==y['representatives'] and x['aliases']==y['aliases']
                    ox={r['pool_id']:r for r in x['outcomes']};oy={r['pool_id']:r for r in y['outcomes']}
                    for i in set(ox)&set(oy):
                        for k in ['rho','reward','done','strict_admissible','exceptions','illegal','ambiguous']:assert ox[i][k]==oy[i][k]
                    matched.append({'repeat':repeat,'week':week,'step':x['step'],'gnn_queries':x['public_queries'],
                        'opponent_queries':y['public_queries'],'same_choice':x['returned_action_hash']==y['returned_action_hash']})
        gn=math.fsum(p['gnn_control_s'] for p in pairs);op=math.fsum(p['opponent_control_s'] for p in pairs)
        identical=[p for p in pairs if p['same_delivered_path']]
        mq=sum(p['gnn_queries'] for p in matched);oq=sum(p['opponent_queries'] for p in matched)
        comparisons.append({'gnn':g,'opponent':opponent,'pooled_control_reduction':1-gn/op,
            'pooled_query_reduction':1-sum(p['gnn_queries'] for p in pairs)/sum(p['opponent_queries'] for p in pairs),
            'pooled_source_reduction':1-math.fsum(p['gnn_source_s'] for p in pairs)/math.fsum(p['opponent_source_s'] for p in pairs),
            'control_positive_pairs':sum(p['control_reduction']>0 for p in pairs),'pairs':pairs,'matched_calls':matched,
            'matched_states':len(matched),'matched_same_choices':sum(p['same_choice'] for p in matched),
            'matched_query_reduction':None if not oq else 1-mq/oq,
            'matched_query_directions':{'gnn_fewer':sum(p['gnn_queries']<p['opponent_queries'] for p in matched),
                'gnn_more':sum(p['gnn_queries']>p['opponent_queries'] for p in matched),
                'equal':sum(p['gnn_queries']==p['opponent_queries'] for p in matched)},
            'identical_delivered_path_pairs':len(identical),
            'identical_path_control_reduction':None if not identical else 1-math.fsum(p['gnn_control_s'] for p in identical)/math.fsum(p['opponent_control_s'] for p in identical),
            'identical_path_query_reduction':None if not identical else 1-sum(p['gnn_queries'] for p in identical)/sum(p['opponent_queries'] for p in identical)})
summaries=[]
for m in D['methods']:
    g=[r for r in records if r['method']==m['method']]
    summaries.append({'method':m['method'],'runs':len(g),'control_s':math.fsum(x['controller_s'] for x in g),
        'n1_source_s':math.fsum(x['n1_source_s'] for x in g),'n1_queries':sum(x['n1_forecasts'] for x in g),
        'complete_runs':sum(x['complete'] for x in g),
        'completed_date_cost_regret_mean':math.fsum(q['comparable_cost_regret'] for q in quality if q['method']==m['method'] and q['complete'])/sum(x['complete'] for x in g),
        'max_comparable_cost_regret':max(q['comparable_cost_regret'] for q in quality if q['method']==m['method'] and q['comparable_cost_regret'] is not None)})
R={'passed':True,'runs':48,'physical_rows':physical,'public_forecasts':forecasts,'quality':quality,'repeat_checks':repeat_checks,
   'summaries':summaries,'comparisons':comparisons,'components':records,
   'all_rankers_within_quality_gate':all(q['no_new_earlier_failure'] and q['cost_within_pilot_scale'] is not False for q in quality),
   'new_fits':0,'reserved_evaluations':0,
   'process_wall_scope':'finished.wall_s covers remaining-phase invocation only, not previous12pilot workers. remaining_phase_timing.json records wrapper scope explicitly; controller_s per-step accounting covers all48runs.',
   'warning':'4exposed development dates(two overlapoldtrain),2timingrepeats each; repetitions/modelseeds not independentfamilies.0.5%qualityinvestmentthreshold not equivalence. Shared compilation gains not GNN novelty; framework supervised scorer uses author PPO prior,not new RL training.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
with (OUT/'method_summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
flat=[{'gnn':c['gnn'],'opponent':c['opponent'],**p} for c in comparisons for p in c['pairs']]
with (OUT/'paired_results.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
lines=['# GRID34 所有强对照共享优化后的实际控制比较','',
       f'48条回放、{physical}物理步、{forecasts}次公开预测，完整保存数据核验。新拟合/RL/预留日期评价为0。', '',
       '| 方法 | 控制秒总计 | N1查询 | N1搜索秒 | 完成回放 | 共同完成周平均成本差 |', '|---|---:|---:|---:|---:|---:|']
for s in summaries:lines.append(f"| {s['method']} | {s['control_s']:.3f} | {s['n1_queries']} | {s['n1_source_s']:.3f} | {s['complete_runs']}/8 | {100*s['completed_date_cost_regret_mean']:.5f}% |")
lines+=['', '| GNN | 相同优化的对照 | 控制时间减少 | 查询减少 | 搜索时间减少 | 控制正向配对 |', '|---|---|---:|---:|---:|---:|']
for c in comparisons:lines.append(f"| {c['gnn']} | {c['opponent']} | {100*c['pooled_control_reduction']:.3f}% | {100*c['pooled_query_reduction']:.3f}% | {100*c['pooled_source_reduction']:.3f}% | {c['control_positive_pairs']}/8 |")
lines+=['',R['warning'], '',
        R['process_wall_scope'], '',
        '六月共同失败不作零成本成功；不同轨迹的源搜索差含状态分布变化，另列同状态/同输入匹配比较。两个模型种子和每周两个计时重复全部展示，不选择最好种子。所有原始缓存/模型/停止配置不变。']
(OUT/'GRID34_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:48fairshared-engineeringruns,originalmodelsandfinal8datesunchanged.\n',encoding='utf-8')
print(json.dumps({k:R[k] for k in ['passed','runs','physical_rows','public_forecasts','all_rankers_within_quality_gate','summaries']},indent=2))
print(json.dumps([{k:c[k] for k in ['gnn','opponent','pooled_control_reduction','pooled_query_reduction','pooled_source_reduction','control_positive_pairs','matched_states','matched_same_choices']} for c in comparisons],indent=2))
