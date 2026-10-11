"""Freeze actual quality of all three cheap orders,not a cross-batch speed claim."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid36'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid35/audit.json')['passed'] and read(ROOT/'outputs/grid35b/audit.json')['passed']
assert all(x['critical_choice_match'] for x in read(ROOT/'outputs/grid35/review.json')['summaries'])
assert read(ROOT/'outputs/grid35b/review.json')['summary']['critical_choice_match']
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
week='2035-03-05_0';methods=[{'method':rule,'rule':rule,'checkpoint':None,'budget':128,'policy':'PATIENCE64','cache_dc':True,'cache_safe':True} for rule in ['STATIC_ID','PHYSICS_TAIL','LOCAL_STRESS']]
D={'stage':'GRID36_CHEAP_ORDER_CLOSED_LOOP_QUALITY','weeks':[week],'methods':methods,
   'execution_order':[{**m,'week':week,'repeat':0,'stage_phase':'pilot'} for m in methods],
   'timing':'Fresh single-thread serial processes,same GRID34component timers/commonDC+safeQPcache. All features/ranking/queries/control accounted. Cross-batch FULL/GNN timing comparison is descriptive only,not certified speed.',
   'reference':'GRID34 r0 March cachedFULL and bothGNNs: only deterministic trajectory/cost/queries used for qualification,not selecting speed winners.',
   'quality_gate':'All3rules retained; per-rule closed-loop eligibility requires2017complete,zeroillegal/ambiguity,and cost<=1.005*FULL. Failingrule not expanded unchanged; passingrule only qualifies further controlled comparison. No automatic changingorders/thresholds or modeltraining.',
   'scope':'One exposed March pilot percheaprule,not independent validation. Previous GNN/MLP/PPO results retained. No learning weights/no author PPO ranking within these cheap orders; author basecontrol otherwise unchanged.',
   'caps':{'episode_s':1500,'wall_s':5000,'physical_steps':7000,'public_forecasts':90000,'new_fits':0,'reserved_evaluations':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid36/derive.py',ROOT/'work/grid36/run.py',ROOT/'work/grid36/cheap_rank.py',
   ROOT/'work/grid35b/physics_order.py',ROOT/'work/grid28/run.py',ROOT/'work/grid28/engine.py',ROOT/'work/grid28/policy.py',
   ROOT/'work/grid27/policy.py',ROOT/'work/grid26/engine.py',ROOT/'work/grid24/guard_adapter.py',ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid30/cache_safe.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:3cheapMarchqualitypilots,no newfits/final dates.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':3}))
