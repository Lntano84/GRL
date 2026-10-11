"""Safe-QP structural and actual-state qualification before any deployment."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid30'
assert json.loads((ROOT/'outputs/grid29c/review.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=False)
D={'stage':'GRID30_SAFE_QP_CACHE_QUALIFICATION','week':'2035-03-05_0',
   'motivation':'GRID29C safe restoration QP compilation34-36s versus backend2.3s on153calls. Test same mathematical problem parameterization; no tolerance/solver/cadence relaxation.',
   'change':'Bounded32entry topology/reactance/scalar-coefficient/delta-time cache. Original parameters are still refreshed by original caller, with per-call shared current parameter values.',
   'manual_gate':'Real Parameter two-bus tests: unchanged solution under current input/target/bound changes, invalidation for coefficients/reactance/topology/delta_time, original failure semantics.',
   'live_gate':'Actual delivered original controller stays unchanged. Before each actual safe QP, copy all current CVXPY Parameter values and pre-solve flow into two isolated replicas, alternate order. Compare numerical outputs <=1e-6MW, statuses/qualification, post-solve storage memory, and resulting Grid2Op action; never deploy probes.',
   'no_speed_claim':'Instrumented control time includes shadow computation and cannot be used as whole-controller acceleration. If numerical qualification fails, preserve failure and do not deploy.',
   'caps':{'episode_s':1500,'physical_steps':2500,'public_forecasts':15000},
   'new_fits':0,'reserved_evaluations':0}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
paths=['work/grid30/cache_safe.py','work/grid30/qualify.py','work/grid30/prepare.py',
       'work/grid29/cache_dc.py','work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED:manual and live-memory qualification only; no cached safe QP deployed.\n',encoding='utf-8')
print(json.dumps(D,indent=2))
