"""Final saved-data arithmetic and seal. Do not rerun the agent or simulator."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid07'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

s=load(OUT/'GRID07_summary.json');a=load(OUT/'GRID07_counterfactual_audit.json')
cf=lines(OUT/'runs/2035-01-08_5__SINGLE_EXPAND_221/steps.jsonl')
old={p:lines(ROOT/'outputs/grid06/runs'/f'2035-01-08_5__{p}'/'steps.jsonl') for p in ['NN20','FALLBACK90']}
assert all(len(x)==2017 for x in [cf,*old.values()])
cost={p:sum(r['raw_operational_cost'] for r in rows) for p,rows in {'CF':cf,**old}.items()}
assert cost['CF']==a['counterfactual_raw_cost']
assert cost['NN20']==a['archived_nn20_raw_cost']
assert cost['FALLBACK90']==a['archived_fallback_raw_cost']
assert abs(s['single_intervention_week_relative_cost_change_vs_nn20']-(cost['CF']-cost['NN20'])/cost['NN20'])<1e-14
assert s['first_intervention_extra_cost']==cost['CF']-cost['NN20']
assert s['full_rule_extra_cost']==cost['FALLBACK90']-cost['NN20']
assert not s['training_readiness'] and s['model_fits']==0
assert a['illegal_actual_steps']==a['ambiguous_actual_steps']==a['exception_actual_steps']==0
assert len([p for p in (OUT/'runs').iterdir() if p.is_dir()])==1
assert a['expansion_events']==1 and a['intervention_step']==221
for i in range(220):
    assert cf[i]['action_hash']==old['NN20'][i]['action_hash']
    assert cf[i]['observation_hash']==old['NN20'][i]['observation_hash']
assert cf[220]['action_hash']==old['FALLBACK90'][220]['action_hash']
assert cf[220]['observation_hash']==old['FALLBACK90'][220]['observation_hash']
for rel,h in load(ROOT/'outputs/grid06/delivery_manifest.json').items():assert sha(ROOT/rel)==h,rel
for name in ['counterfactual_execution_frozen.json','counterfactual_reused_assets_frozen.json','offline_inputs_frozen.json']:
    for rel,h in load(OUT/name).items():assert sha(ROOT/rel)==h,rel
review=dict(saved_data_arithmetic_passed=True,old_grid06_artifacts_unchanged=True,
    engineering_gate_and_counterfactual_audit_passed=True,one_native_attempt=True,
    physical_steps=2017,native_simulates=24185,models_trained=0,
    engineering_failure=False,training_readiness=False,
    scope='Separate arithmetic on persisted feedback, not a second simulation, independent baseline rollout or model fit.',
    scientific_figure_visually_inspected=True)
(OUT/'GRID07_postrun_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.suffix!='.log' and p.name!='delivery_manifest.json'}
files.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'work/grid07').glob('*.py'))})
(OUT/'delivery_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
for rel,h in load(OUT/'delivery_manifest.json').items():assert sha(ROOT/rel)==h,rel
print(json.dumps(review,indent=2),flush=True)
