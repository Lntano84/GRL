"""Audit one intervention from persisted vectors and source hashes, no simulator."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid07'
SCENE='2035-01-08_5';FOLDER=OUT/'runs'/f'{SCENE}__SINGLE_EXPAND_221'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

run=load(OUT/'counterfactual_run_manifest.json');design=load(OUT/'counterfactual_design.json')
assert run['completed'] and run['engineering_gate_passed'] and run['resource_caps_passed'],run
for name in ['counterfactual_execution_frozen.json','counterfactual_reused_assets_frozen.json','offline_inputs_frozen.json']:
    for rel,h in load(OUT/name).items(): assert sha(ROOT/rel)==h,rel
assert sha(OUT/'counterfactual_design.json')==load(OUT/'counterfactual_design.sha256')['sha256']
meta=load(FOLDER/'metadata.json');end=load(FOLDER/'completion.json');count=load(FOLDER/'execution_count.json')
rows=lines(FOLDER/'steps.jsonl');events=lines(FOLDER/'events.jsonl')
assert len(rows)==count['physical_calls']==count['saved_steps']==end['physical_calls']
assert len(rows)<=2017 and count['total_native_simulations']<=60000
assert sum(e['kind']=='simulate' for e in events)==sum(r['simulate_calls'] for r in rows)==count['total_native_simulations']
assert count['prefix_checks']==221
ref_root=ROOT/'outputs/grid06/runs'
ref={p:lines(ref_root/f'{SCENE}__{p}'/'steps.jsonl') for p in ['NN20','FALLBACK90']}
baseline_meta=load(ref_root/f'{SCENE}__NN20'/'metadata.json')
for key in ['parameters','rho_danger','rho_safe','optim_config','versions','gen_cost_per_MW','delta_time_seconds','action_vector_layout']:
    assert meta[key]==baseline_meta[key],key
for i in range(221):
    target=ref['NN20' if i<220 else 'FALLBACK90'][i]
    for key in ['before_observation_hash','action_hash','observation_hash']:
        assert rows[i][key]==target[key],(i+1,key)
assert meta['top_k']==20 and meta['fixed_fallback']['only_at_step']==221
fb=[e for e in events if e['kind']=='fallback']
assert len(fb)==1 and fb[0]['step']==221 and fb[0]['expanded'] and not fb[0]['forced_test']
assert fb[0]['returned_pool_row']==255
by_step=defaultdict(list)
for e in events: by_step[e['step']].append(e)
pool=load(ROOT/'outputs/grid04/pool_metadata.json')['action_hashes']
modules_checked=0
for r in rows:
    es=by_step[r['step']]
    tm=[e for e in es if e['kind']=='module' and e['module']=='topo_12_unsafe']
    if not tm: continue
    assert len(tm)==1
    shortlist=[e for e in es if e['kind']=='shortlist']
    sims=[e for e in es if e['kind']=='simulate' and e['module']=='topo_12_unsafe']
    if r['step']==221:
        assert [e['top_k'] for e in shortlist]==[20,352]
        rank=shortlist[1]['ids']
        ids=shortlist[0]['ids']+fb[0]['additional_ids']
    else:
        assert len(shortlist)==1 and shortlist[0]['top_k']==20
        ids=rank=shortlist[0]['ids']
    assert len(sims)==len(ids) and len(set(ids))==len(ids)
    scores={}
    for idx,e in zip(ids,sims):
        assert e['action_hash']==pool[idx]
        ok=e['rho_max'] is not None and 0<e['rho_max']<r['before_rho_max'] and not e['exceptions'] and not e['done']
        scores[idx]=np.float32(e['reward'] if ok else -100)
    values=np.array([scores[i] for i in rank],dtype=np.float32)
    chosen=pool[rank[int(np.argmax(values))]] if values.max()>-100 else None
    assert tm[0]['returned_action_hash']==chosen
    modules_checked+=1

with np.load(FOLDER/'initial.npz',allow_pickle=False) as z:
    previous_hash=digest(z['observation']);previous_curtailed=float(z['curtailment_mw'].sum(dtype=np.float64))
price=np.array(meta['gen_cost_per_MW']);dt=meta['delta_time_seconds']/3600
ledgers=0;largest=0.
component={k:0. for k in ['loss','redispatch','curtailment_change','storage']}
for r in rows:
    path=FOLDER/r['file'];assert sha(path)==r['file_sha256']
    with np.load(path,allow_pickle=False) as z:
        assert r['before_observation_hash']==previous_hash
        assert digest(z['observation'])==r['observation_hash'] and digest(z['action'])==r['action_hash']
        if r['ledger_eligible']:
            marginal=float(price[z['gen_p']>0].max())
            level=float(z['curtailment_mw'].sum(dtype=np.float64))
            amounts=dict(loss=float(z['gen_p'].sum(dtype=np.float64)-z['load_p'].sum(dtype=np.float64))*dt,
                redispatch=float(np.abs(z['actual_dispatch']).sum(dtype=np.float64))*dt,
                curtailment_change=(level-previous_curtailed)*dt,
                storage=float(np.abs(z['storage_power']).sum(dtype=np.float64))*dt)
            calculated=marginal*sum(amounts.values())
            residual=abs(calculated-r['raw_operational_cost'])
            assert residual<=r['cost_rounding_tolerance']
            largest=max(largest,residual);ledgers+=1
            for k,x in amounts.items():component[k]+=marginal*x
        previous_hash=r['observation_hash'];previous_curtailed=float(z['curtailment_mw'].sum(dtype=np.float64))
complete=bool(rows[-1]['current_step']>=meta['max_steps'] or rows[-1]['chronics_done'])
assert complete==end['reached_native_horizon']
cf=sum(r['raw_operational_cost'] for r in rows)
baseline=sum(r['raw_operational_cost'] for r in ref['NN20'])
fallback=sum(r['raw_operational_cost'] for r in ref['FALLBACK90'])
summary=dict(passed=True,engineering_gate_passed=True,rows_and_vectors_checked=len(rows),
    physical_steps=len(rows),native_simulations=count['total_native_simulations'],ledgers_recomputed=ledgers,
    max_ledger_abs_residual=largest,topology_calls_reconstructed=modules_checked,
    expansion_events=1,intervention_step=221,forced_production_expansions=0,
    illegal_actual_steps=sum(r['illegal'] for r in rows),ambiguous_actual_steps=sum(r['ambiguous'] for r in rows),
    exception_actual_steps=sum(bool(r['exceptions']) for r in rows),
    exception_simulation_calls=sum(e['kind']=='simulate' and bool(e['exceptions']) for e in events),
    completed_native_week=complete,counterfactual_raw_cost=cf,
    archived_nn20_raw_cost=baseline,archived_fallback_raw_cost=fallback,
    relative_cost_change_vs_archived_nn20=(cf/baseline-1) if complete else None,
    relative_cost_change_vs_archived_fallback=(cf/fallback-1) if complete else None,
    operational_cost_components=component,process_wall_s=run['process_wall_s'],
    scope='Single post-hoc intervention on an exposed week, archived baseline; not a deployable policy or generalization test. Exact engineering gate reproduction reduces but does not quantify later cross-run solver noise. No training.')
save('GRID07_counterfactual_audit.json',summary)
print(json.dumps(summary,indent=2),flush=True)
