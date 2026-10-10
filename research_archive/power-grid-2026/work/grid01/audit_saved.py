"""Read-only simulation-artifact audit; does not call agent or AC power flow."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid01'
def load(p):
    return json.loads(p.read_text(encoding='utf-8'))
def jsonl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def stats(a):
    if not a:
        return None
    return {'mean':float(np.mean(a)),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),
            'max':float(np.max(a)),'sum':float(np.sum(a))}
manifest=load(OUT/'run_manifest.json')
assert len(manifest)==4 and all(x['exit']==0 for x in manifest),manifest
runs=[]
mainrecords=[]
mainevents=[]
residuals=[]
for run in manifest:
    folder=OUT/f"scenario{run['scenario']}_{run['mode']}"
    records=jsonl(folder/'steps.jsonl')
    events=jsonl(folder/'events.jsonl')
    finish=load(folder/'completion.json')
    assert len(records)==finish['steps']
    assert len(records)<= (258 if run['mode']=='full' else 24)
    previous=np.load(folder/'initial.npz',allow_pickle=False)['observation']
    meta=load(folder/'metadata.json')
    offsets={}
    cursor=0
    for entry in meta['action_vector_layout']:
        offsets[entry['name']]=slice(cursor,cursor+entry['size'])
        cursor+=entry['size']
    assert cursor==1605
    by_step=defaultdict(list)
    for event in events:
        by_step[event['step']].append(event)
    for index,rec in enumerate(records,1):
        assert rec['step']==index and rec['before_observation_hash']==digest(previous)
        p=folder/rec['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['file_sha256']
        with np.load(p,allow_pickle=False) as v:
            assert digest(v['observation'])==rec['observation_hash']
            assert digest(v['action'])==rec['action_hash']
            assert np.isfinite(v['action']).all()
            for field,key in [('_redispatch','redispatch_L1'),('_storage_power','storage_L1')]:
                assert abs(float(np.abs(v['action'][offsets[field]]).sum())-rec[key])<1e-9,(index,field)
            assert int(np.count_nonzero(v['action'][offsets['_curtail'] ]!=-1))==rec['curtailment_modified']
            assert np.isfinite(v['rho']).all()
            assert float(v['rho'].max())==rec['after_rho_max']
            assert rec['physical_observation_unchanged_during_act']
            sims=[x for x in by_step[index] if x['kind']=='simulate']
            assert len(sims)==rec['simulate_calls']
            assert sum(x['wall_s'] for x in sims)<=rec['act_wall_s']+1e-6
            if rec['selected_action_forecast_available']:
                assert np.isfinite(v['forecast_rho']).all()
                assert v['forecast_rho'].shape==v['rho'].shape==(186,)
                assert float(v['forecast_rho'].max())==rec['selected_forecast_rho_max']
                matched=[x for x in sims if x['action_hash']==rec['action_hash'] and x['horizon']==1]
                assert matched
                if run['mode']=='full' and not(rec['illegal'] or rec['ambiguous'] or rec['done'] or rec['exceptions'] or rec['selected_forecast_done']):
                    f=rec['selected_forecast_flags']
                    if not(f['illegal'] or f['ambiguous'] or f['exceptions']):
                        residuals.append({'scenario':run['scenario'],'step':index,
                            'rho_abs_mean':float(np.abs(v['forecast_rho']-v['rho']).mean()),
                            'rho_abs_max':float(np.abs(v['forecast_rho']-v['rho']).max()),
                            'forecast_rho_max':rec['selected_forecast_rho_max'],'actual_rho_max':rec['after_rho_max'],
                            'p_or_abs_mean':float(np.abs(v['forecast_p_or']-v['p_or']).mean())})
            else:
                assert np.isnan(v['forecast_rho']).all() and np.isnan(v['forecast_p_or']).all()
            previous=v['observation'].copy()
    sims=[x for x in events if x['kind']=='simulate']
    modules=[x for x in events if x['kind']=='module']
    solves=[x for x in events if x['kind']=='solve' and not x['root_dispatch']]
    assert all(x['horizon']==1 for x in sims)
    summary={'scenario':run['scenario'],'mode':run['mode'],'steps':len(records),'native_done':finish['done'],
        'process_wall_s':run['wall_s'],'after_import_wall_s':finish['wall_after_import_s'],
        'act_wall_s':stats([x['act_wall_s'] for x in records]),'simulate_calls':len(sims),
        'simulate_wall_s':float(sum(x['wall_s'] for x in sims)),
        'nonzero_actions':sum(x['nonzero_action'] for x in records),
        'illegal':sum(x['illegal'] for x in records),'ambiguous':sum(x['ambiguous'] for x in records),
        'exception_steps':sum(bool(x['exceptions']) for x in records),
        'max_after_rho':max(x['after_rho_max'] for x in records),
        'overloaded_after_steps':sum(x['after_rho_max']>1 for x in records),
        'module_calls':dict(Counter(x['module'] for x in modules)),
        'solver_attempts':len(solves),'solver_statuses':dict(Counter(x['problem_status'] for x in solves)),
        'selected_forecast_available':sum(x['selected_action_forecast_available'] for x in records)}
    runs.append(summary)
    if run['mode']=='full':
        mainrecords.extend(records)
        mainevents.extend(events)

replays=[]
for scenario in [0,1]:
    full=OUT/f'scenario{scenario}_full'
    repeat=OUT/f'scenario{scenario}_replay'
    a=jsonl(full/'steps.jsonl')
    b=jsonl(repeat/'steps.jsonl')
    assert len(b)==min(24,len(a))
    for index,(first,second) in enumerate(zip(a,b),1):
        for key in ['before_observation_hash','observation_hash','action_hash','reward','done','illegal','ambiguous',
                    'exceptions','simulate_calls']:
            assert first[key]==second[key],(scenario,index,key)
        with np.load(full/first['file'],allow_pickle=False) as va,np.load(repeat/second['file'],allow_pickle=False) as vb:
            for key in ['observation','action','rho','p_or','forecast_rho','forecast_p_or']:
                assert np.array_equal(va[key],vb[key],equal_nan=True),(scenario,index,key)
    replays.append({'scenario':scenario,'steps':len(b),'arrays_actions_rewards_flags_equal':True})

old=OUT/'integration_failure_cvxpy_index'
failed_records=jsonl(old/'steps.jsonl')
assert len(failed_records)==60
new_first=jsonl(OUT/'scenario0_full/steps.jsonl')[:60]
for a,b in zip(failed_records,new_first):
    for key in ['observation_hash','action_hash','reward','done','illegal','ambiguous','exceptions']:
        assert a[key]==b[key],('failed-prefix-change',a['step'],key)
old_cache=OUT/'INVALID_action_vector_cache'
cache_steps=sum(len(jsonl(p)) for p in old_cache.glob('scenario*/steps.jsonl'))
assert cache_steps==564
cached_batch_observation_changes=0
cached_batch_action_vector_changes=0
for scenario in [0,1]:
    for mode in ['full','replay']:
        prior=jsonl(old_cache/f'scenario{scenario}_{mode}/steps.jsonl')
        current=jsonl(OUT/f'scenario{scenario}_{mode}/steps.jsonl')
        assert len(prior)==len(current)
        for a,b in zip(prior,current):
            cached_batch_observation_changes+=int(a['observation_hash']!=b['observation_hash'])
            cached_batch_action_vector_changes+=int(a['action_hash']!=b['action_hash'])
total_steps=len(failed_records)+cache_steps+sum(x['steps'] for x in runs)
assert total_steps<=1188
adapt=load(OUT/'adaptation_manifest.json')
assert adapt['selected_class_AST_identical'] and adapt['factory_function_AST_identical']
generated_ignored=[]
for name,expected in adapt['adapted_hashes'].items():
    if '__pycache__' in Path(name).parts:
        generated_ignored.append(name)
        continue
    assert hashlib.sha256((ROOT/'work/grid01/ljn_heuristic'/name).read_bytes()).hexdigest()==expected
sources=load(OUT/'source_manifest.json')
for rec in sources['files']:
    if not rec.get('absent'):
        p=ROOT/'work/grid01/original_ljn'/rec['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
module_summary={}
for name in ['reconnect','recover_topo','topo_12_unsafe','topo_n1_unsafe','optim']:
    matches=[x for x in mainevents if x['kind']=='module' and x['module']==name]
    module_summary[name]={'calls':len(matches),'returned_non_none':sum(not x['returned_none'] for x in matches),
        'simulate_calls':sum(x['simulate_calls'] for x in matches),'wall_s':stats([x['wall_s'] for x in matches])}
main_sim=[x for x in mainevents if x['kind']=='simulate']
leaf_solves=[x for x in mainevents if x['kind']=='solve' and not x['root_dispatch']]
root_solves=[x for x in mainevents if x['kind']=='solve' and x['root_dispatch']]
summary={'status':'PASS_BOUNDED_INTEGRATION_AND_SAVED_ARTIFACT_CHECKS','main_steps':len(mainrecords),
    'replay_steps':sum(x['steps'] for x in runs if x['mode']=='replay'),'failed_integration_steps':60,
    'invalid_action_cache_batch_steps':cache_steps,'all_executed_steps':total_steps,
    'runs':runs,'replay_checks':replays,'failed_prefix_matches_replacement_first60':True,
    'source_manifest_generated_pycache_entries_ignored':generated_ignored,
    'invalid_vs_corrected_observation_hash_changes':cached_batch_observation_changes,
    'invalid_vs_corrected_action_vector_hash_changes':cached_batch_action_vector_changes,
    'main_act_wall_s':stats([x['act_wall_s'] for x in mainrecords]),'main_simulate_calls':len(main_sim),
    'main_simulate_wall_s':float(sum(x['wall_s'] for x in main_sim)),
    'main_actual_illegal':sum(x['illegal'] for x in mainrecords),'main_actual_ambiguous':sum(x['ambiguous'] for x in mainrecords),
    'main_actual_exception_steps':sum(bool(x['exceptions']) for x in mainrecords),
    'main_module_summary':module_summary,
    'main_solver_attempts':len(leaf_solves),'main_solver_statuses':dict(Counter(x['problem_status'] for x in leaf_solves)),
    'main_solver_types':dict(Counter(x['solver_argument'] for x in leaf_solves)),
    'main_solver_root_dispatches':len(root_solves),'main_solver_root_wall_s':float(sum(x['wall_s'] for x in root_solves)),
    'main_forecast_sim_done_count':sum(x['done'] for x in main_sim),
    'main_forecast_sim_illegal_count':sum(x['illegal'] for x in main_sim),
    'main_forecast_sim_ambiguous_count':sum(x['ambiguous'] for x in main_sim),
    'main_forecast_sim_exception_count':sum(bool(x['exceptions']) for x in main_sim),
    'selected_forecast_pairs':len(residuals),
    'selected_rho_abs_max_stats':stats([x['rho_abs_max'] for x in residuals]),
    'selected_p_or_abs_mean_stats':stats([x['p_or_abs_mean'] for x in residuals]),
    'false_safe_at_threshold1':sum(x['forecast_rho_max']<1 and x['actual_rho_max']>1 for x in residuals),
    'scope':'Two public bundled prefixes, single Grid2Op seed, no independent AC solver, no untouched research test set.',
    'warning':'Passive telemetry contributes overhead; timings are qualification measurements, not production comparisons. No learned method evaluated.'}
(OUT/'GRID01_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
(OUT/'selected_forecast_residuals.json').write_text(json.dumps(residuals,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['status','main_steps','all_executed_steps','main_simulate_calls','main_actual_illegal',
    'main_solver_attempts','main_solver_statuses','false_safe_at_threshold1']},indent=2))
