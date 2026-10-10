"""Offline artifact consistency audit; no new simulation or independent AC solver."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid00'
def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

runs=read('trajectories.json')
v=np.load(OUT/'vectors.npz',allow_pickle=False)
sim=read('simulation_checks.json')
assert len(runs)==4
assert sum(r['steps'] for r in runs)==96
per_run=[]
for r in runs:
    name=f"{r['agent']}_{r['repeat']}"
    obs=v[name+'_obs']
    act=v[name+'_actions']
    assert len(r['records'])==r['steps']==len(act)==len(obs)-1
    assert r['steps']<=24
    for i,rec in enumerate(r['records']):
        assert digest(obs[i+1])==rec['observation_hash']
        assert digest(act[i])==rec['action_hash']
        assert np.isfinite(obs[i+1]).all() and np.isfinite(act[i]).all()
    assert sum(rec['highres_calls'] for rec in r['records'])==r['highres_calls']
    per_run.append({'agent':r['agent'],'repeat':r['repeat'],'steps':r['steps'],
        'wall_s':r['wall_s'],'terminated':r['terminal'],'highres_calls':r['highres_calls'],
        'rho_max_over_steps':max(x['rho_max'] for x in r['records']),
        'sum_rewards':sum(x['reward'] for x in r['records'])})
for cls in ['DoNothingAgent','RecoPowerlineAgent']:
    for field in ['obs','actions']:
        assert np.array_equal(v[f'{cls}_0_{field}'],v[f'{cls}_1_{field}'])
    records=[r['records'] for r in runs if r['agent']==cls]
    for a,b in zip(*records):
        for key in ['rho_max','reward','done','is_illegal','is_ambiguous','exceptions','highres_calls']:
            assert a[key]==b[key]
assert all(x['physical_obs_unchanged'] and x['backend_origin_flows_unchanged'] and x['step_unchanged'] for x in sim)
assert all(x['counter_after']-x['counter_before']==1 for x in sim)
assets=read('bundled_assets.json')
for f in assets['files']:
    p=Path(assets['root'])/f['path']
    assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for f in read('wheel_manifest.json')['wheels']:
    p=ROOT/'work/grid00/wheels'/f['url'].rsplit('/',1)[1]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
result={'status':'PASS_INTERFACE_CHECKS_WITH_DISCLOSED_POSTPROCESSING_FAILURE',
    'actual_steps':96,'trajectory_count':4,'scenario_count_executed':1,
    'per_run':per_run,'illegal_count':sum(x['is_illegal'] for r in runs for x in r['records']),
    'ambiguous_count':sum(x['is_ambiguous'] for r in runs for x in r['records']),
    'exception_steps':sum(bool(x['exceptions']) for r in runs for x in r['records']),
    'replay_observation_and_action_arrays_equal':True,'replay_rewards_and_flags_equal':True,
    'forecast_does_not_advance_observed_physical_state':True,
    'forecast_actual_vector_shapes':{'forecast':list(v['forecast_h1'].shape),'actual':list(v['DoNothingAgent_0_obs'][1].shape)},
    'original_worker_exit':1,'original_failure_stage':'postprocessing after all smoke steps and replay checks',
    'original_failure':read('failure.json'),
    'smoke_elapsed_to_failure_s':read('failure.json')['elapsed_s'],
    'first_import_s':read('import_ready.json')['import_s'],
    'audit_scope':'Saved array/log/hash consistency; no new simulation, no independent AC feasibility solver.',
    'strong_baseline_executed':False,'training_runs':0,
    'warning':'One bundled test chronic and short prefixes; no comparative performance or learning-value conclusion.'}
save('GRID00_audit.json',result)
print(json.dumps({k:result[k] for k in ['status','actual_steps','illegal_count','ambiguous_count','exception_steps']},indent=2))
