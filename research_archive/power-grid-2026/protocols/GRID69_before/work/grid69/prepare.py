"""Freeze a single repair before any new physical run; no parameter sweep."""
from pathlib import Path
import datetime as dt, hashlib, json
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid69'; OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (OUT/'design.json').exists()
pilot=['2035-07-09_2','2035-12-10_6','2035-03-05_0']
weeks=list(dict.fromkeys(read(ROOT/'outputs/grid65/design.json')['weeks']+read(ROOT/'outputs/grid68/design.json')['weeks']))
base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
methods=[dict(base,method=p,policy=p) for p in ['FULL','FRAME_VERIFY']]
order=[]
for j,w in enumerate(pilot):
    for m in (methods if j%2==0 else methods[::-1]):
        order.append(dict(m,week=w,repeat=0,stage_phase='pilot'))
for w in weeks:
    if w not in pilot:order.append(dict(methods[1],week=w,repeat=0,stage_phase='remaining'))
design=dict(stage='GRID69',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    weeks=weeks,pilot=pilot,methods=methods,execution_order=order,repeats=1,
    environment_seed=0,quality_threshold=.005,
    scope='Known-date repair development. All13 exposed dates retained. Three contemporaneous FULL pilots; other FULL references reused for quality and queries only. No new independent confirmation, no model fitting.',
    timing='Same GRID55 worker and GRID28 controller. One timing repeat only; pilot paired times descriptive, no stable timing superiority claim. Historical times not a matched timing experiment.',
    repair='Initial128 then64 batches; every compressed stop requires >=256 queries and two consecutive non-improving batches, then unchanged alias-preserving incumbent-substation completion. Exhaust on no admissible action. No immediate rho-safe return. No date/step-specific rules.',
    decision='Pilot: July must complete with FULL, December and March must complete and satisfy old0.5% cost screen; query saving on equal-horizon pilots >=50% macro. If unmet, stop before remaining; retain all partial results. If met, evaluate all remaining10 dates without tuning. All13 quality reported, even if pilot succeeds. Do not revise old GRID68 verdict.',
    caps=dict(episode_s=1200,wall_s=10800,physical_steps=35000,public_forecasts=800000,new_fits=0))
save(OUT/'design.json',design)
frozen=read(ROOT/'outputs/grid68/code_freeze.json')
for p,h in frozen.items():assert sha(ROOT/p)==h,p
for p in [*sorted((ROOT/'work/grid69').glob('*.py')),OUT/'design.json']:
    frozen[p.relative_to(ROOT).as_posix()]=sha(p)
save(OUT/'code_freeze.json',frozen)
(OUT/'RUN_STATE.md').write_text('FROZEN: one verification-floor repair, pilot pending. Original FRAME preserved.\n',encoding='utf-8')
print(json.dumps({'runs':len(order),'pilot_runs':6,'dates':len(weeks),'frozen':len(frozen)}))
