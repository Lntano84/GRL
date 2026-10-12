"""Freeze equivalent-metadata qualification before any new timing."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid54'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists();assert read(ROOT/'outputs/grid53/timing_review.json')['passed']
cases=[]
for stage,methods in [('grid51',['LOCAL_GNN0','LOCAL_GNN1','COMPRESSED_GNN0','COMPRESSED_GNN1']),('grid34',['LOCAL_PPO'])]:
    D=read(ROOT/f'outputs/{stage}/design.json')
    for item in D['execution_order']:
        if item['repeat']!=0 or item['method'] not in methods:continue
        folder=ROOT/f'outputs/{stage}/runs'/f"r0__{item['week']}__{item['method']}"
        for p in sorted(folder.glob('call[0-9][0-9][0-9].json')):
            c=read(p);assert c['prior_hash'] is not None
            cases.append({'item':item,'call':p.relative_to(ROOT).as_posix(),
                          'sha256':sha(p),'state_sha256':sha(p.with_name(p.stem+'_state.npz'))})
assert len(cases)==130,len(cases)
D={'stage':'GRID54_EQUIVALENT_PRIOR_METADATA_QUALIFICATION','cases':cases,
   'change':'Only input-only action dictionary metadata and mapping cache. Validate fresh canonical action fields every call; rebuild on replacement or in-place change; never use stale to_vect cache. Lazy first build is charged inside prior/control time.',
   'qualification':'130 saved old-development priors must equal original scores/order and saved hashes exactly. Reconstruct final model ranking/feature hashes. Check all old/new/PPO alias hooks. Mutation invalidation must reproduce original behavior. No candidate simulate, new training, stopping/threshold change or hidden feedback input.',
   'timing':'Each saved case recomputes original and cached prior; alternate execution order. First build reported separately and included; local timings are not closed-loop speed.',
   'fairness':'Provide identical cache to old/new/PPO and any loaded user of public_prior. FULL and SoftGNN do not call it. Do not subtract local savings from historical results or claim shared cache as GNN contribution.'}
OUT.mkdir();(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list(HERE.glob('*.py'))+[ROOT/'work/grid21/prior.py',ROOT/'work/grid24/guard_adapter.py',
        ROOT/'work/grid50/online_adapter.py',ROOT/'work/grid50/compressed_residual.py',ROOT/'work/grid20/components.py',
        ROOT/'work/grid01/trace_agent.py',OUT/'design.json']
for seed in [0,1]:
    files += [ROOT/f'outputs/grid21/models/gnn_seed{seed}/final.pt',ROOT/f'outputs/grid50/models/gnn_seed{seed}/final.pt']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:130saved old-development qualification cases;no new forecasts/training.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'cases':len(cases),'new_training':0,'new_candidate_forecasts':0}))
