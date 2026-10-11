"""Freeze fair serial static-order comparison after March quality qualification."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid37'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
R=read(ROOT/'outputs/grid36/review.json');assert R['passed_saved_data']
assert next(r for r in R['rows'] if r['method']=='STATIC_ID')['quality_qualified']
old=read(ROOT/'outputs/grid34/design.json')
methods=[dict(m) for m in old['methods'] if m['method'] in ['FULL','LOCAL_GNN0','LOCAL_GNN1']]
methods.append({'method':'STATIC_ID','rule':'STATIC_ID','budget':128,'checkpoint':None,'policy':'PATIENCE64','cache_dc':True,'cache_safe':True})
order=[]
for w,week in enumerate(old['weeks']):
    cycle=methods[w:]+methods[:w]
    for repeat in [0,1]:
        for m in cycle if repeat==0 else cycle[::-1]:order.append({**m,'week':week,'repeat':repeat,'stage_phase':'pilot'})
D={'stage':'GRID37_FAIR_STATIC_ORDER_COMPARISON','weeks':old['weeks'],'methods':methods,'execution_order':order,'repeats':2,'serial_workers':1,
   'timing':old['timing'].replace('reversed6arm','reversed4arm'),
   'changes':'STATIC_ID replaces ranking only. Both original GNN seeds and FULL are freshly rerun; shared caches, stopping, semantic completion, FP32 tie, solver, cadence and control objective unchanged.',
   'scope':'Four previously exposed development weeks, not final confirmation. March STATIC_ID pilot passed; PHYSICS_TAIL/LOCAL_STRESS failures retained and not widened unchanged. Original author basecontroller PPO branches remain in every arm.',
   'priorities':['No new or earlier failure versus paired FULL','Comparable delivered cost <=1.005*paired FULL','Query/source and whole-controller time all paired and pooled, no best GNN seed','If static meets quality gate, retain its time-quality tradeoff as strong baseline; failure does not by itself prove graph necessity'],
   'quality_gate':'Per date/repeat require completion whenever FULL completes, no earlier failure otherwise, comparable cost increase <=0.5%. Investment screen not equivalence; incomplete exposures never yield favorable cost comparison.',
   'caps':{'episode_s':1500,'wall_s':8500,'physical_steps':70000,'public_forecasts':500000,'new_fits':0,'reserved_evaluations':0}}
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid37/run.py',ROOT/'work/grid37/derive.py',ROOT/'work/grid36/cheap_rank.py',ROOT/'work/grid35b/physics_order.py']
files += [ROOT/p for p in read(ROOT/'outputs/grid34/code_freeze.json')]
files=list(dict.fromkeys(files))
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:32 fair4arm runs on exposed dates; no new training/final evaluation.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':len(order),'new_fits':0}))
