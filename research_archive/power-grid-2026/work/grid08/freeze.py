import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid08'; OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'design.json').exists()
split=json.loads((ROOT/'outputs/grid03/scenario_split_frozen.json').read_text())
dev=json.loads((ROOT/'outputs/grid04/design_freeze.json').read_text())['selected_development_weeks']
validation=split['all_splits']['validation']
evals=[]
for m in [2,5,8,11]:
    pool=[s for s in validation if s.startswith(f'2035-{m:02}-')]
    evals.append(min(pool,key=lambda s:hashlib.sha256(('GRID08-residual-v1:'+s).encode()).hexdigest()))
design={'stage':'GRID08','task':'Binary learned residual safe-continuous restoration over immutable NN20',
 'training_weeks':dev,'evaluation_weeks':evals,'test_split_unopened':True,'environment_seed':0,
 'cadence':6,'proposal_current_rho_lt':.9,'proposal_forecast_rho_le':.93,
 'baselines':['NN20','ALWAYS_RESTORE','IMMEDIATE_COST'],
 'models':['mlp','gnn'],'training_seed':20261009,'training_epochs':2,
 'ppo':{'rollout':256,'epochs':4,'minibatch':32,'learning_rate':.0003,'gamma':.995,'gae_lambda':.95,
        'clip':.2,'entropy':.01,'cost_scale':10000.,'blackout_penalty':100.},
 'screen':'Completion first, no cost ranking incomplete pairs. No preset large improvement threshold; assess paired directions and safety. One seed is a development pilot, no statistical generalization claim.',
 'adaptation':'Preserve v0 checkpoints, logs and hashes; v1 may use training-only imitation warm start or reward/credit diagnosis. All tried versions reported. No selection on sealed final test.',
 'caps':{'process_wall_s':7200,'physical_steps':80000,'output_bytes':1073741824},
 'limits':['Separate restoration helper does not change original optimizer memory.',
 'One-step forecast acceptance is not guaranteed future safety.',
 'Existing author model train identity unknown; our fine-tune weeks disjoint from our evaluation.',
 'This auxiliary binary task does not yet learn topology-action ranking or validate original IM framework.']}
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
original={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'work/grid02/ljn_nn').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
original['outputs/grid07/delivery_manifest.json']=hashlib.sha256((ROOT/'outputs/grid07/delivery_manifest.json').read_bytes()).hexdigest()
(OUT/'original_assets.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('# GRID08\n\nDESIGN_FROZEN; safe restoration qualification pending. No training yet.\n',encoding='utf-8')
print(json.dumps(design,indent=2))
