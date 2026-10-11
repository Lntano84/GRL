"""Freeze existing GNN/MLP/PPO matched-state controls; no refitting or new simulator calls."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid33'
assert json.loads((ROOT/'outputs/grid32/review.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
D={'stage':'GRID33_EXISTING_RANKERS_MATCHED_STATE_SCREEN',
   'weeks':json.loads((ROOT/'outputs/grid26/design.json').read_text())['weeks'],
   'rankers':[{'method':f'{kind.upper()}{seed}','kind':kind,'seed':seed,
               'checkpoint':f'outputs/grid21/models/{kind}_seed{seed}/final.pt'}
              for kind in ['gnn','mlp'] for seed in [0,1]]+[{'method':'PPO','checkpoint':None}],
   'policy':'PATIENCE64','initial':128,'batch':64,'minimum':256,'patience':2,'safe':.9,
   'input':'Same26 public feature/prior snapshots from original exact-path guarded GNN0, with eligible representative/alias sets shared. Privileged FULL outcomes exposed only by the diagnostic dispatcher after an action is selected.',
   'goal':'Measure existing graph ranker increment over same-information MLP and author PPO, after common optimizer engineering. No shared compilation saving is credited to GNN.',
   'gate':'A ranker missing the known March step1733 FULL action cannot enter this unchanged local-stop closed-loop pilot. Passing is numeric qualification only, not proof of cost/survival, architecture necessity or unseen generalization.',
   'rule':'Report all4existing supervised checkpoints, never select a new seed or fit. Matched26states are exposed development diagnostics, not independent observations.',
   'caps':{'wall_s':180,'new_fits':0,'new_forecasts':0,'reserved_evaluations':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid33/offline.py',ROOT/'work/grid28/policy.py',
       ROOT/'work/grid27/policy.py',ROOT/'work/grid21/prior.py',ROOT/'work/grid20/components.py']
files += [ROOT/r['checkpoint'] for r in D['rankers'] if r['checkpoint']]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:existing5rankers matched26states,no fitting/new simulator/final dates.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'rankers':5,'matched_states':26}))
