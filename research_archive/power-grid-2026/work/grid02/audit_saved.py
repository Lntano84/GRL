"""Saved-artifact consistency audit; no environment, policy inference or power flow."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid02'
def load(p):
    return json.loads(p.read_text(encoding='utf-8'))
def jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a):
    return {'mean':float(np.mean(a)), 'p50':float(np.percentile(a,50)),
            'p95':float(np.percentile(a,95)), 'max':float(np.max(a)), 'sum':float(np.sum(a))} if a else None

qualification=load(OUT/'preflight.json')
assert qualification['status']=='PASS_NO_PHYSICAL_STEPS'
assert qualification['mapping_count']==qualification['action_count']==352
assert qualification['input_exactly_observed_rho']
assert qualification['legacy_mapping_migration_checked']
migration=load(OUT/'mapping_migration.json')
assert len(migration['rows'])==352 and all(x['stored_fields_and_cache_equal_vector'] and x['restored_vector_equal'] for x in migration['rows'])
with np.load(OUT/'preflight_vectors.npz',allow_pickle=False) as z:
    assert digest(z['mapping_vectors'])==qualification['mapping_vector_sha256']
    assert z['mapping_vectors'].shape==(352,1605)
source=load(OUT/'source_manifest.json')
for x in source['files']:
    assert sha(ROOT/'work/grid02/original_nn'/x['path'])==x['sha256']
adapt=load(OUT/'adaptation_manifest.json')
for name,expected in adapt['hashes'].items():
    assert sha(ROOT/'work/grid02/ljn_nn'/name)==expected, name
for name in ['modules/topology_nn_policy.py','gym_assets/action_space.py','models/RL_training_PPO.zip']:
    assert sha(ROOT/'work/grid02/ljn_nn'/name)==sha(ROOT/'work/grid02/original_nn'/name)
for name in ['modules/convex_optim.py','modules/base_module.py','modules/topology_heuristic.py']:
    assert sha(ROOT/'work/grid02/ljn_nn'/name)==sha(ROOT/'work/grid01/ljn_heuristic'/name)
original=ast.parse((ROOT/'work/grid01/original_ljn/LJNAgent.py').read_text(encoding='utf-8'))
adapted=ast.parse((ROOT/'work/grid02/ljn_nn/LJNAgent.py').read_text(encoding='utf-8'))
for name in ['LJNAgent','LJNAgentTopoNN']:
    assert ast.dump(next(x for x in original.body if isinstance(x,ast.ClassDef) and x.name==name))==ast.dump(next(x for x in adapted.body if isinstance(x,ast.ClassDef) and x.name==name))
frozen=load(OUT/'run_code_freeze_full.json')
for name,expected in frozen.items():
    assert sha(ROOT/name)==expected, name

manifest=load(OUT/'run_manifest_smoke.json')+load(OUT/'run_manifest_full.json')
assert len(manifest)==8 and all(r['exit']==0 for r in manifest)
summaries=[]
main_records=[]
record_store={}
for run in manifest:
    folder=OUT/f"{run['policy']}_scenario{run['scenario']}_{run['mode']}"
    records=jsonl(folder/'steps.jsonl')
    events=jsonl(folder/'events.jsonl')
    finish=load(folder/'completion.json')
    meta=load(folder/'metadata.json')
    assert len(records)==finish['steps']<=meta['max_steps']
    assert meta['policy']==run['policy']
    assert meta['NN_topk']==({'FULL':None,'NN20':20,'NN352':352}[run['policy']])
    with np.load(folder/'initial.npz',allow_pickle=False) as z:
        previous=z['observation'].copy()
        previous_rho=z['rho'].copy()
        initial_hash=digest(previous)
    offsets={}
    cursor=0
    for field in meta['action_vector_layout']:
        offsets[field['name']]=slice(cursor,cursor+field['size'])
        cursor+=field['size']
    assert cursor==1605
    by_step=defaultdict(list)
    for e in events:
        by_step[e['step']].append(e)
    for index,row in enumerate(records,1):
        assert row['step']==index
        assert row['before_observation_hash']==digest(previous)
        path=folder/row['file']
        assert sha(path)==row['file_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert digest(z['observation'])==row['observation_hash']
            assert digest(z['action'])==row['action_hash']
            assert np.isfinite(z['action']).all() and np.isfinite(z['rho']).all()
            assert float(z['rho'].max())==row['after_rho_max']
            assert row['physical_observation_unchanged_during_act']
            for field,key in [('_redispatch','redispatch_L1'),('_storage_power','storage_L1')]:
                assert abs(float(np.abs(z['action'][offsets[field]]).sum())-row[key])<1e-9
            assert int(np.count_nonzero(z['action'][offsets['_curtail']]!=-1))==row['curtailment_modified']
            sims=[x for x in by_step[index] if x['kind']=='simulate']
            assert len(sims)==row['simulate_calls']
            assert sum(x['wall_s'] for x in sims)<=row['act_wall_s']+1e-6
            assert all(x['horizon']==1 for x in sims)
            nn=[x for x in by_step[index] if x['kind']=='nn_topk']
            if run['policy']=='FULL':
                assert not nn
            else:
                for entry in nn:
                    assert entry['input_hash']==digest(previous_rho)
                    assert len(entry['ids'])==len(set(entry['ids']))==meta['NN_topk']
                    assert all(0<=i<352 for i in entry['ids'])
            if row['selected_action_forecast_available']:
                assert any(x['action_hash']==row['action_hash'] and x['horizon']==1 for x in sims)
                if row['selected_forecast_rho_max'] is not None:
                    assert float(np.max(z['forecast_rho']))==row['selected_forecast_rho_max']
            else:
                assert np.isnan(z['forecast_rho']).all() and np.isnan(z['forecast_p_or']).all()
            previous=z['observation'].copy()
            previous_rho=z['rho'].copy()
    modules=[x for x in events if x['kind']=='module']
    solves=[x for x in events if x['kind']=='solve' and not x['root_dispatch']]
    nn=[x for x in events if x['kind']=='nn_topk']
    module_summary={name:{'calls':sum(x['module']==name for x in modules),
                          'simulate_calls':sum(x['simulate_calls'] for x in modules if x['module']==name),
                          'wall_s':sum(x['wall_s'] for x in modules if x['module']==name)}
                    for name in ['reconnect','recover_topo','topo_12_unsafe','topo_n1_unsafe','optim']}
    end=records[-1]
    summary={**run,'steps':len(records),'native_max_episode_duration':meta['native_max_episode_duration'],
             'native_done':finish['done'],'chronics_done_at_end':end['chronics_done'],
             'reached_recorded_native_horizon':end['current_step']>=meta['native_max_episode_duration'],
             'illegal':sum(x['illegal'] for x in records),'ambiguous':sum(x['ambiguous'] for x in records),
             'exception_steps':sum(bool(x['exceptions']) for x in records),
             'exception_types':dict(Counter(t for x in records for t in x['exception_types'])),
             'initial_observation_hash':initial_hash,'final_observation_hash':end['observation_hash'],
             'nonzero_actions':sum(x['nonzero_action'] for x in records),
             'simulate_calls':sum(x['simulate_calls'] for x in records),
             'act_wall_s':stats([x['act_wall_s'] for x in records]),
             'max_after_rho':max(x['after_rho_max'] for x in records),
             'overloaded_after_steps':sum(x['after_rho_max']>1 for x in records),
             'reward_sum':sum(x['reward'] for x in records),
             'module_summary':module_summary,'solver_statuses':dict(Counter(x['problem_status'] for x in solves)),
             'nn_calls':len(nn),'nn_wall_s':stats([x['wall_s'] for x in nn]),
             'selected_forecast_available':sum(x['selected_action_forecast_available'] for x in records)}
    summaries.append(summary)
    record_store[(run['policy'],run['scenario'],run['mode'])]=records
    if run['mode']=='full':
        main_records.extend(records)

repeat=[]
baseline_repeat=[]
paired=[]
keys=['before_observation_hash','observation_hash','action_hash','reward','done','illegal','ambiguous','exceptions','simulate_calls']
for scenario in [0,1]:
    a=record_store[('NN20',scenario,'full')]
    b=record_store[('NN20',scenario,'smoke')]
    assert len(b)==24
    for index,(first,second) in enumerate(zip(a,b),1):
        assert all(first[k]==second[k] for k in keys),(scenario,index)
    repeat.append({'scenario':scenario,'NN20_smoke_matches_main_first24':True})
    a=record_store[('FULL',scenario,'full')]
    b=jsonl(ROOT/f'outputs/grid01/scenario{scenario}_full/steps.jsonl')
    same=all(all(first[k]==second[k] for k in keys) for first,second in zip(a,b))
    baseline_repeat.append({'scenario':scenario,'common_steps':min(len(a),len(b)),
                            'matches_GRID01_actions_observations_rewards_flags_calls':same})
    assert same,('GRID01 full baseline changed',scenario)
    left=record_store[('NN20',scenario,'full')]
    right=record_store[('NN352',scenario,'full')]
    assert left[0]['before_observation_hash']==right[0]['before_observation_hash']==a[0]['before_observation_hash']
    diffs=[i for i,(x,y) in enumerate(zip(left,right),1) if x['action_hash']!=y['action_hash']]
    state_keys=['before_observation_hash','observation_hash','action_hash','reward','done','illegal','ambiguous','exceptions']
    same_states=len(left)==len(right) and all(all(x[k]==y[k] for k in state_keys) for x,y in zip(left,right))
    assert same_states,('NN20 and NN352 states differ despite observed matching actions',scenario)
    paired.append({'scenario':scenario,'common_observed_steps':min(len(left),len(right)),
                   'first_action_difference':diffs[0] if diffs else None,
                   'different_action_steps_in_common_prefix':len(diffs),
                   'all_observations_actions_rewards_native_flags_identical':same_states})

total_steps=sum(x['steps'] for x in summaries)
assert total_steps<=3498<=8112
assert sum(x['wall_s'] for x in summaries)<3600
audit={'status':'PASS_SAVED_ARTIFACT_CONSISTENCY','actual_steps':total_steps,
       'main_steps':len(main_records),'main_act_wall_s':stats([x['act_wall_s'] for x in main_records]),
       'runs':summaries,'repeat_checks':repeat,'GRID01_repeat_checks':baseline_repeat,
       'NN20_NN352_comparison':paired,'source_hashes_and_class_AST_checked':True,
       'model_and_input_mapping_checked':True,'qualification_physical_steps':0,
       'note':'Offline consistency audit, native flags and prefix replay. Not independent AC-physics validation or formal challenge scoring.'}
(OUT/'GRID02_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
fields=['policy','scenario','mode','steps','native_done','chronics_done_at_end','reached_recorded_native_horizon',
        'illegal','ambiguous','exception_steps','nonzero_actions','simulate_calls','act_mean_s','act_p95_s','act_max_s',
        'max_after_rho','overloaded_after_steps','nn_calls','wall_s']
with (OUT/'GRID02_per_run.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    for summary in summaries:
        row={k:summary[k] for k in fields if k in summary}
        row.update({'act_mean_s':summary['act_wall_s']['mean'],'act_p95_s':summary['act_wall_s']['p95'],
                    'act_max_s':summary['act_wall_s']['max']})
        writer.writerow(row)
print(audit['status'],'actual_steps',total_steps,flush=True)
