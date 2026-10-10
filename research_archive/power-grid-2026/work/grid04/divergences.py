"""Only saved traces: locate same-prior-state first action divergences."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid04'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
design=load(OUT/'design_freeze.json')
pool=load(OUT/'pool_metadata.json')
by_hash={h:i for i,h in enumerate(pool['action_hashes'])}
assert len(by_hash)==352
results=[]
for scene in design['selected_development_weeks']:
    for left,right in [('NN20','NN352'),('NN352','FULL'),('NN20','LOCAL20'),('NN20','RANDOM20')]:
        a_dir=OUT/'runs_v3'/f'{scene}__{left}';b_dir=OUT/'runs_v3'/f'{scene}__{right}'
        a=lines(a_dir/'steps.jsonl');b=lines(b_dir/'steps.jsonl')
        first=next((i for i,(x,y) in enumerate(zip(a,b),1) if x['action_hash']!=y['action_hash']),None)
        row={'scenario':scene,'left':left,'right':right,'left_steps':len(a),'right_steps':len(b),'first_different_action':first,
            'common_prefix_observations_identical':all(x['observation_hash']==y['observation_hash'] for x,y in zip(a[:first-1] if first else a,b[:first-1] if first else b)),
            'interpretation':'No claim about optimal actions or causal attribution of later blackouts.'}
        if first is not None:
            x,y=a[first-1],b[first-1]
            ae=[e for e in lines(a_dir/'events.jsonl') if e['step']==first]
            be=[e for e in lines(b_dir/'events.jsonl') if e['step']==first]
            row['same_prior_observation']=x['before_observation_hash']==y['before_observation_hash']
            row['left_after_rho']=x['after_rho_max'];row['right_after_rho']=y['after_rho_max']
            for side,events,record in [('left',ae,x),('right',be,y)]:
                sl=[e for e in events if e['kind']=='shortlist']
                modules=[e for e in events if e['kind']=='module' and e['module']=='topo_12_unsafe']
                row[side+'_shortlist_ids']=sl[0]['ids'] if sl else None
                row[side+'_topo12_return_hash']=modules[0]['returned_action_hash'] if modules else None
                row[side+'_topo12_pool_row']=by_hash.get(row[side+'_topo12_return_hash'])
                row[side+'_solver_statuses']=[e['problem_status'] for e in events if e['kind']=='solve' and not e['root_dispatch']]
                simulations=[e for e in events if e['kind']=='simulate' and e['module']=='topo_12_unsafe']
                eligible=[e for e in simulations if e['rho_max'] is not None and 0<e['rho_max']<record['before_rho_max'] and not e['done'] and not e['exceptions']]
                row[side+'_topo12_simulated_count']=len(simulations)
                row[side+'_topo12_illegal_count']=sum(e['illegal'] for e in simulations)
                row[side+'_topo12_eligible_count']=len(eligible)
                row[side+'_topo12_best_eligible_forecast_rho']=min((e['rho_max'] for e in eligible),default=None)
                row[side+'_delivered_action_forecast_rho']=record['selected_forecast_rho_max']
            returned=row['right_topo12_pool_row'];short=row['left_shortlist_ids']
            row['right_topo12_return_was_tested_by_left']=returned in short if returned is not None and short is not None else None
            row['same_state_pool_omission_witness']=bool(row['same_prior_observation'] and right=='NN352' and left=='NN20' and returned is not None and short is not None and returned not in short)
            full_order=row['right_shortlist_ids']
            row['right_topo12_rank_in_right_shortlist']=full_order.index(returned)+1 if full_order is not None and returned in full_order else None
        results.append(row)
(OUT/'first_divergences.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps([{k:r.get(k) for k in ['scenario','left','right','first_different_action','same_prior_observation','same_state_pool_omission_witness']} for r in results],indent=2))
