"""Read saved vectors/feedback independently; no policy, power flow or fitting."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid04'
started=time.perf_counter()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def stats(values):
    return {'sum':float(np.sum(values)),'p50':float(np.percentile(values,50)),
        'p95':float(np.percentile(values,95)),'max':float(np.max(values))} if values else None
design=load(OUT/'design_freeze.json')
assert sha(OUT/'design_freeze.json')==(OUT/'design_freeze.sha256').read_text()
manifest=load(OUT/'run_manifest_v3.json')
assert manifest==[{**r,'exit':m['exit'],'wall_s':m['wall_s']} for r,m in zip(design['plan'],manifest)]
assert len(manifest)==20 and all(r['exit']==0 for r in manifest),'Incomplete matrix: do not issue a quality verdict'
for path,h in load(OUT/'execution_code_v3.json').items(): assert sha(ROOT/path)==h
reused=load(OUT/'reused_assets_frozen.json')
for path,h in reused.items(): assert sha(ROOT/path)==h
pre=load(OUT/'preflight_v3.json')
assert pre['passed'] and pre['physical_steps']==0
pool=load(OUT/'pool_metadata.json')
assert sha(OUT/'pool_metadata.json')==pre['pool_metadata_sha256']
incident=[np.asarray(x,dtype=int) for x in pool['incident_lines']]
records_by_run={}
metadata_by_run={}
summaries=[]
vectors_checked=0
ledger_checked=0
shortlists_checked=0
maximum_residual=0.0
for run in manifest:
    folder=OUT/'runs_v3'/f"{run['scenario']}__{run['policy']}"
    meta=load(folder/'metadata.json');end=load(folder/'completion.json');execution=load(folder/'execution_count.json')
    metadata_by_run[(run['scenario'],run['policy'])]=meta
    rows=lines(folder/'steps.jsonl');events=lines(folder/'events.jsonl')
    assert len(rows)==end['steps']==execution['physical_calls']==execution['saved_steps']
    assert meta['scenario']==run['scenario'] and meta['policy']==run['policy']
    assert meta['max_steps']==2017 and len(rows)<=2017
    by_step=defaultdict(list)
    for event in events: by_step[event['step']].append(event)
    with np.load(folder/'initial.npz',allow_pickle=False) as z:
        previous=z['observation'].copy();previous_rho=z['rho'].copy()
        previous_curtailed=float(z['curtailment_mw'].sum(dtype=np.float64))
    initial_hash=digest(previous)
    prices=np.asarray(meta['gen_cost_per_MW'],dtype=np.float64)
    dt=meta['delta_time_seconds']/3600
    assert dt==1/12
    offsets={};cursor=0
    for f in meta['action_vector_layout']:
        offsets[f['name']]=slice(cursor,cursor+f['size']);cursor+=f['size']
    assert cursor==1605
    for index,row in enumerate(rows,1):
        assert row['step']==index and row['before_observation_hash']==digest(previous)
        assert row['physical_observation_unchanged_during_act']
        file=folder/row['file'];assert sha(file)==row['file_sha256']
        with np.load(file,allow_pickle=False) as z:
            assert digest(z['observation'])==row['observation_hash'] and digest(z['action'])==row['action_hash']
            assert float(z['rho'].max())==row['after_rho_max']
            assert np.isfinite(z['action']).all() and np.isfinite(z['rho']).all()
            assert int(np.count_nonzero(z['action'][offsets['_set_topo_vect']]))==row['changed_topology_entries']
            sim=[e for e in by_step[index] if e['kind']=='simulate']
            assert len(sim)==row['simulate_calls'] and all(e['horizon']==1 for e in sim)
            assert sum(e['wall_s'] for e in sim)<=row['act_wall_s']+1e-6
            for entry in [e for e in by_step[index] if e['kind']=='shortlist']:
                ids=entry['ids'];assert len(ids)==len(set(ids))==(352 if run['policy']=='NN352' else 20)
                assert all(0<=i<352 for i in ids)
                if run['policy']=='RANDOM20':
                    # Reimplement hash permutation, without importing the controller.
                    expected=sorted(range(352),key=lambda i:hashlib.sha256(f"GRID04-cheap-v1:{index-1}:{i}".encode()).digest())[:20]
                    assert ids==expected
                if run['policy']=='LOCAL20':
                    keys=[]
                    for i in range(352):
                        v=np.asarray(previous_rho[incident[i]],dtype=np.float64)
                        keys.append((-float(max(v)) if len(v) else 0.0,
                            -sum(max(float(x)-0.9,0.0)**2 for x in v),
                            hashlib.sha256(f'GRID04-cheap-v1:{i}'.encode()).digest()))
                    expected=sorted(range(352),key=lambda i:keys[i])[:20]
                    assert ids==expected
                shortlists_checked+=1
            if row['ledger_eligible']:
                margin=float(max(prices[z['gen_p']>0]))
                loss=(z['gen_p'].astype(np.float64).sum()-z['load_p'].astype(np.float64).sum())*dt
                redisp=np.abs(z['actual_dispatch'].astype(np.float64)).sum()*dt
                current_curtailed=float(z['curtailment_mw'].astype(np.float64).sum())
                curtailed=current_curtailed*dt
                delta=(current_curtailed-previous_curtailed)*dt
                storage=np.abs(z['storage_power'].astype(np.float64)).sum()*dt
                calculated=margin*(loss+redisp+delta+storage)
                x=row['ledger']
                for key,value in [('marginal_cost',margin),('losses_mwh',loss),('redispatch_mwh',redisp),
                    ('curtailed_mwh',curtailed),('curtailment_delta_mwh',delta),('storage_throughput_mwh',storage),('recomputed_raw_cost',calculated)]:
                    assert abs(x[key]-value)<1e-8,(run,index,key,x[key],value)
                assert abs(calculated-row['raw_operational_cost'])<=row['cost_rounding_tolerance']
                maximum_residual=max(maximum_residual,abs(calculated-row['raw_operational_cost']))
                ledger_checked+=1
            previous_curtailed=float(z['curtailment_mw'].astype(np.float64).sum())
            previous=z['observation'].copy();previous_rho=z['rho'].copy();vectors_checked+=1
    final=rows[-1]
    complete=bool(final['current_step']>=meta['max_steps'] or final['chronics_done'])
    assert complete==end['reached_native_horizon']
    module=defaultdict(lambda:{'calls':0,'simulate_calls':0,'wall_s':0.0})
    for e in events:
        if e['kind']=='module':
            d=module[e['module']];d['calls']+=1;d['simulate_calls']+=e['simulate_calls'];d['wall_s']+=e['wall_s']
    solves=[e for e in events if e['kind']=='solve' and not e['root_dispatch']]
    valid=[r for r in rows if r['ledger_eligible']]
    summary={**run,'steps':len(rows),'native_horizon':meta['max_steps'],'completed_native_week':complete,
        'premature_end':not complete,'initial_observation_hash':initial_hash,'final_observation_hash':final['observation_hash'],
        'illegal_steps':sum(r['illegal'] for r in rows),'ambiguous_steps':sum(r['ambiguous'] for r in rows),
        'exception_steps':sum(bool(r['exceptions']) for r in rows),'terminal_exception_types':final['exception_types'],
        'native_raw_cost_sum':sum(r['raw_operational_cost'] for r in rows),
        'verified_physical_cost_sum':sum(r['raw_operational_cost'] for r in valid),
        'unverified_or_error_terminal_count':len(rows)-len(valid),
        'unverified_or_error_terminal_cost_sum':sum(r['raw_operational_cost'] for r in rows if not r['ledger_eligible']),
        'losses_mwh':sum(r['ledger']['losses_mwh'] for r in valid),
        'redispatch_mwh':sum(r['ledger']['redispatch_mwh'] for r in valid),
        'physical_curtailed_mwh':sum(r['ledger']['curtailed_mwh'] for r in valid),
        'storage_throughput_mwh':sum(r['ledger']['storage_throughput_mwh'] for r in valid),
        'terminal_renewable_component':final['renewable_component'],'terminal_assistant_component':final['assistant_component'],
        'act_time':stats([r['act_wall_s'] for r in rows]),'simulate_calls':sum(r['simulate_calls'] for r in rows),
        'max_after_rho':max(r['after_rho_max'] for r in rows),'overloaded_after_steps':sum(r['after_rho_max']>1 for r in rows),
        'modules':dict(module),'solver_statuses':dict(Counter(e['problem_status'] for e in solves)),
        'shortlist_calls':sum(e['kind']=='shortlist' for e in events),'construction_s':meta['agent_construct_s']}
    summaries.append(summary);records_by_run[(run['scenario'],run['policy'])]=rows
    print('AUDITED',run['scenario'],run['policy'],len(rows),flush=True)
assert sum(r['steps'] for r in summaries)+1<=design['caps']['physical_steps']
startup=[]
for name,base in [('run_manifest.json','runs'),('run_manifest_main.json','runs_main')]:
    for r in load(OUT/name):
        count=load(OUT/base/f"{r['scenario']}__{r['policy']}"/'execution_count.json')['physical_calls']
        startup.append({**r,'physical_calls':count})
assert sum(x['physical_calls'] for x in startup)==1
with np.load(OUT/'runs_main/2035-01-29_4__FULL/steps/0001.npz',allow_pickle=False) as failed, np.load(OUT/'runs_v3/2035-01-29_4__FULL/steps/0001.npz',allow_pickle=False) as valid:
    assert all(np.array_equal(failed[k],valid[k],equal_nan=True) for k in failed.files)
assert sum(r['wall_s'] for r in summaries+startup)<=design['caps']['total_runner_s']
matrix_log=(OUT/'matrix_v3.log').read_text(encoding='utf-8')
matrix_wall=float(matrix_log.strip().splitlines()[-1].split('MATRIX_COMPLETE ')[1])
assert matrix_wall+sum(r['wall_s'] for r in startup)<=design['caps']['total_runner_s']
paired=[]
for scene in design['selected_development_weeks']:
    initial=[r['initial_observation_hash'] for r in summaries if r['scenario']==scene]
    assert len(set(initial))==1
    comparable_fields=['seed','parameters','rho_danger','rho_safe','optim_config','versions','action_vector_layout','gen_cost_per_MW','delta_time_seconds']
    first_meta=metadata_by_run[(scene,'FULL')]
    for policy in design['policies']:
        other_meta=metadata_by_run[(scene,policy)]
        assert all(first_meta[f]==other_meta[f] for f in comparable_fields),(scene,policy,'native configuration mismatch')
    reference=records_by_run[(scene,'NN20')]
    for policy in ['NN352','LOCAL20','RANDOM20','FULL']:
        other=records_by_run[(scene,policy)]
        action_diff=[i for i,(a,b) in enumerate(zip(reference,other),1) if a['action_hash']!=b['action_hash']]
        state_diff=[i for i,(a,b) in enumerate(zip(reference,other),1) if a['observation_hash']!=b['observation_hash']]
        shared_prior=[]
        for a,b in zip(reference,other):
            if a['before_observation_hash']!=b['before_observation_hash']:break
            shared_prior.append((a,b))
        paired.append({'scenario':scene,'reference':'NN20','other':policy,'common_steps':min(len(reference),len(other)),
            'identical_prior_observation_prefix_steps':len(shared_prior),
            'same_prior_prefix_NN20_act_s':sum(a['act_wall_s'] for a,b in shared_prior),
            'same_prior_prefix_other_act_s':sum(b['act_wall_s'] for a,b in shared_prior),
            'same_prior_prefix_NN20_simulates':sum(a['simulate_calls'] for a,b in shared_prior),
            'same_prior_prefix_other_simulates':sum(b['simulate_calls'] for a,b in shared_prior),
            'identical_complete_actions':not action_diff and len(reference)==len(other),
            'identical_complete_observations':not state_diff and len(reference)==len(other),
            'first_action_difference':action_diff[0] if action_diff else None,
            'first_observation_difference':state_diff[0] if state_diff else None,
            'first_different_action_same_prior_observation':reference[action_diff[0]-1]['before_observation_hash']==other[action_diff[0]-1]['before_observation_hash'] if action_diff else None})
(OUT/'per_run.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
(OUT/'paired_trajectories.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
columns=['scenario','policy','steps','completed_native_week','native_raw_cost_sum','verified_physical_cost_sum','simulate_calls',
    'max_after_rho','overloaded_after_steps','physical_curtailed_mwh','redispatch_mwh','storage_throughput_mwh',
    'terminal_renewable_component','terminal_assistant_component','illegal_steps','ambiguous_steps','exception_steps','wall_s']
with (OUT/'GRID04_per_run.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
    for row in summaries:w.writerow({c:row[c] for c in columns})
audit={'passed':True,'main_runs':len(summaries),'physical_steps_main':sum(r['steps'] for r in summaries),
    'physical_steps_failed_startup':1,'total_physical_steps':sum(r['steps'] for r in summaries)+1,
    'failed_first_physical_step_exactly_reproduced_after_io_fix':True,
    'vectors_rehashed_and_rechecked':vectors_checked,'public_ledgers_recomputed':ledger_checked,
    'shortlists_independently_recomputed_or_membership_checked':shortlists_checked,'max_cost_abs_residual':maximum_residual,
    'same_seed_initial_observations_match':True,'reused_assets_unchanged':len(reused),
    'native_parameters_and_common_controller_configuration_match':True,
    'main_runner_process_wall_s':sum(r['wall_s'] for r in summaries),
    'main_runner_including_orchestration_wall_s':matrix_wall,
    'startup_process_wall_s':sum(r['wall_s'] for r in startup),'audit_wall_s':time.perf_counter()-started,
    'normalized_competition_score_certified':False,
    'scope':'Independent saved-vector/ledger/hash/ranking arithmetic and original native feedback; not an independent AC solver, model retraining, QP constraint audit or full published competition evaluation.'}
audit['main_run_persisted_bytes']=sum(p.stat().st_size for p in (OUT/'runs_v3').rglob('*') if p.is_file())
assert audit['main_run_persisted_bytes']<=design['caps']['output_bytes']
(OUT/'GRID04_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
