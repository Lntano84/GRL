"""Freeze actual controller comparisons; every opponent receives identical shared caches."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid34';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'design.json').exists()
assert json.loads((ROOT/'outputs/grid33/audit.json').read_text())['passed']
methods=[{'method':'FULL','rule':'FULL','budget':None,'checkpoint':None,'policy':'PROTECTED'}]
for kind,seed in [('gnn',0),('gnn',1),('ppo',None),('mlp',0),('mlp',1)]:
    methods.append({'method':'LOCAL_PPO' if kind=='ppo' else f'LOCAL_{kind.upper()}{seed}',
       'rule':'OLDNN' if kind=='ppo' else f'{kind.upper()}_seed{seed}','budget':128,
       'checkpoint':None if kind=='ppo' else f'outputs/grid21/models/{kind}_seed{seed}/final.pt','policy':'PATIENCE64'})
for m in methods:m.update(cache_dc=True,cache_safe=True)
weeks=json.loads((ROOT/'outputs/grid26/design.json').read_text())['weeks'];order=[]
for w,week in enumerate(weeks):
    cycle=methods[w:]+methods[:w]
    for repeat in [0,1]:
        for m in cycle if repeat==0 else cycle[::-1]:order.append({**m,'week':week,'repeat':repeat,'stage_phase':'pilot' if w==0 else 'remaining'})
D={'stage':'GRID34_SHARED_CACHE_FAIR_RANKER_CONTROL','weeks':weeks,'methods':methods,'execution_order':order,'repeats':2,
   'serial_workers':1,'timing':'Fresh processes, single thread; all arms shareDC+safeQP cache and identical component instrumentation. Counterbalanced reversed6arm order on each date. Include model/features/all forecast/solve/guard/caching in controller time; initialization, physical environment and export separate.',
   'changes':'Only existing ranked topology search and frozen semantic stop; no optimizer/control math/solver/cadence/model weights changes. FULL exhaustive as strongest quality reference, MLP receives same public graph/plan/base/prior information.',
   'quality_gate':'March pilot all5rankers must finish whenFULL finishes, no new/earlier failure, cost increase<=0.5%vsFULL. Investment threshold,not equivalence. If a pilot fails retain every arm and diagnose before widening; do not exclude/tune failing seeds.',
   'gate':'All existing rankers passed known-critical-action offline screen. Shared caches are common engineering, never GNN architecture contribution.',
   'priorities':['Quality/survival first','N1queries and source time vsPPO/bothMLPs','Whole controller time vs optimizedFULL and optimizedrankers','All per-date/model-seed/repeat outcomes, no bestseed selection'],
   'caps':{'episode_s':1500,'wall_s':11000,'physical_steps':100000,'public_forecasts':750000,'new_fits':0,'reserved_evaluations':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid34/run.py',ROOT/'work/grid34/derive.py',ROOT/'work/grid28/run.py',
       ROOT/'work/grid28/engine.py',ROOT/'work/grid28/policy.py',ROOT/'work/grid27/policy.py',ROOT/'work/grid26/engine.py',
       ROOT/'work/grid24/guard_adapter.py',ROOT/'work/grid21/prior.py',ROOT/'work/grid20/components.py',
       ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid30/cache_safe.py',ROOT/'work/grid08/common.py']
files += [ROOT/m['checkpoint'] for m in methods if m['checkpoint']]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:12Marchpilot,36remainingonlyafterqualitygate;existingweights,no finaldates.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':len(order),'pilot_runs':12,'models':5,'fresh_fits':0}))
