"""Freeze live-memory DC paired diagnostic, not another whole-time sweep."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29b'
assert json.loads((ROOT/'outputs/grid29/comparison.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=False)
D={'stage':'GRID29B_LIVE_DC_PAIR_DIAGNOSTIC','week':'2035-03-05_0',
   'change':'Actual delivered controller remains original protectedGNN0. At each real DC call, two isolated optimizer replicas receive the same observation, actual reactance and storage accumulator. Alternate shadow ORIGINAL/CACHED order; never deploy shadow results.',
   'qualification':'Both replicas must match actual original DC convergence and flows <=1e-6MW, and closed delivery must match sealed FULL exactly.',
   'scope':'Instrumented one March trajectory, not speed comparison of full control. Measure actual-state paired DC component plus all explicit solve compilation/backend statistics; include positional solver_type.',
   'missing_data':'None solver statistics remain missing, not zero. Recursive generic fallback records are distinguished from explicit solver calls.',
   'caps':{'episode_s':1200,'physical_steps':2500,'public_forecasts':15000},
   'new_fits':0,'reserved_evaluations':0}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
paths=[ROOT/'work/grid29b/probe_dc.py',Path(__file__).resolve(),ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED: no live DC diagnostic yet; instrumented control time is not a speed metric.\n',encoding='utf-8')
print(json.dumps(D,indent=2))
