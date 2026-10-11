"""Freeze balanced component measurement before running either arm."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29c'
assert json.loads((ROOT/'outputs/grid29b/audit.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=False)
base=json.loads((ROOT/'outputs/grid29/pilot/runtime_design.json').read_text())
base.update({'stage':'GRID29C_COMPLETE_COMPONENT_PAIRS',
    'purpose':'Explain successful10.93% and weak0.39% whole-control results using full delivered-arm DC/continuous QP timing, not probe subtraction.',
    'change':'No mathematical/model/control change: shared DC cache versus original. Add identical low-overhead component/statistics records to both arms.',
    'analysis':'Report all2balanced pairs; sum disjoint DC versus nonDC explicit solves, full run_dc/unsafe/safe call times, forecast time and remaining control time. Retain null stats and distinguish recursive solver=None.',
    'quality_gate':'Every action/observation/cost/forecast count must equal sealed protected FULL. Any mismatch blocks speed claim.',
    'scope':'Exposed March only, same GNN0/K128/guard/cadence, no fitting, no final date families. Instrumentation included in control; no independent speed/generalization claim.',
    'caps':{**base['caps'],'wall_s':2000,'physical_steps':8068,'public_forecasts':45300}})
for item in base['execution_order']:
    item['week']='2035-03-05_0'
    item.pop('phase',None)
assert [(i['repeat'],i['method']) for i in base['execution_order']]==[(0,'ORIGINAL'),(0,'CACHED'),(1,'CACHED'),(1,'ORIGINAL')]
(OUT/'design.json').write_text(json.dumps(base,indent=2),encoding='utf-8')
paths=['work/grid29c/run.py','work/grid29c/prepare.py','work/grid29/cache_dc.py','work/grid28/run.py']
(OUT/'code_freeze.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('PREPARED:4balanced full-arm component profiles, original/cache; no outcomes.\n',encoding='utf-8')
print(json.dumps({'prepared':True,'runs':4,'new_fits':0,'reserved_evaluations':0}))
