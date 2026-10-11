"""Freeze four quality-first serial engineering comparisons after shadow passes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid29/pilot'
S=json.loads((ROOT/'outputs/grid29/shadow.json').read_text());assert S['passed']
OUT.mkdir(parents=True,exist_ok=False)
base=json.loads((ROOT/'outputs/grid28/runtime_design.json').read_text())
methods=[{'method':v,'rule':'GNN_seed0','budget':128,'checkpoint':'outputs/grid21/models/gnn_seed0/final.pt',
          'policy':'PROTECTED','cache_dc':v=='CACHED'} for v in ['ORIGINAL','CACHED']]
items=[]
for repeat in range(2):
    for method in (methods if repeat==0 else list(reversed(methods))):
        items.append({'repeat':repeat,'week':base['weeks'][0],**method})
design={'stage':'GRID29_SHARED_DC_PROGRAM_CACHE_PILOT','weeks':[base['weeks'][0]],'methods':methods,
        'execution_order':items,'repeats':2,'serial_workers':1,
        'timing':'Fresh processes, single thread; perf_counter full control; archive/AC physical step/setup separate. Balanced paired orders ORIGINAL,CACHED/CACHED,ORIGINAL. No new learned component.',
        'gate':'Require exact historical delivered observation/action/cost path. Any difference stops this four-run pilot before widening. Local same-state shadow is not sufficient.',
        'caps':{'episode_s':1200,'wall_s':1800,'physical_steps':9000,'public_forecasts':60000},
        'new_fits':0,'reserved_evaluations':0}
(OUT/'runtime_design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
audit=(ROOT/'work/grid28/audit.py').read_text(encoding='utf-8')
audit=audit.replace('OUT = ROOT / "outputs/grid28"','OUT = ROOT / "outputs/grid29/pilot"')
audit=audit.replace('default="pilot"','default="all"')
(ROOT/'work/grid29/audit_pilot.py').write_text(audit,encoding='utf-8')
freeze=json.loads((ROOT/'outputs/grid28/code_freeze.json').read_text())
for path in [ROOT/'work/grid29/cache_dc.py',ROOT/'work/grid29/pilot.py',ROOT/'work/grid29/audit_pilot.py',Path(__file__).resolve()]:
    freeze[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED: four-run same-controller cache pilot, no run started yet.\n',encoding='utf-8')
print(json.dumps(design,indent=2))
