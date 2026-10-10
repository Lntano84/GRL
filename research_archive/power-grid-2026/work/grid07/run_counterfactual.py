"""One subprocess with explicit import/total watchdog; preserve failures, no retry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid07'
SCENE='2035-01-08_5'
FOLDER=OUT/'runs'/f'{SCENE}__SINGLE_EXPAND_221'

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

assert not FOLDER.exists(), 'Never retry or overwrite'
design=load(OUT/'counterfactual_design.json')
assert sha(OUT/'counterfactual_design.json')==load(OUT/'counterfactual_design.sha256')['sha256']
for name in ['counterfactual_execution_frozen.json','counterfactual_reused_assets_frozen.json']:
    for rel,h in load(OUT/name).items(): assert sha(ROOT/rel)==h,rel
FOLDER.mkdir(parents=True)
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: env[k]='1'
env['MPLCONFIGDIR']=str(ROOT/'work/grid02/mplconfig')
start=time.perf_counter()
reason=None
with (FOLDER/'console.log').open('w',encoding='utf-8') as f:
    child=subprocess.Popen([sys.executable,str(ROOT/'work/grid07/run_intervention.py'),
        '--scenario',SCENE,'--policy','SINGLE_EXPAND_221'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    while child.poll() is None:
        elapsed=time.perf_counter()-start
        ready=(FOLDER/'import_ready.json').exists()
        if (not ready and elapsed>design['caps']['import_s']) or elapsed>design['caps']['total_runner_s']:
            reason='import_cap' if not ready else 'total_cap'
            child.kill();child.wait()
            break
        time.sleep(.5)
    code=child.wait()
elapsed=time.perf_counter()-start
gate=load(FOLDER/'engineering_gate.json') if (FOLDER/'engineering_gate.json').exists() else None
size=sum(p.stat().st_size for p in FOLDER.rglob('*') if p.is_file())
resource=(reason is None and elapsed<=design['caps']['total_runner_s'] and size<=design['caps']['output_bytes'])
record=dict(exit_code=code,process_wall_s=elapsed,watchdog_reason=reason,import_ready=(FOLDER/'import_ready.json').exists(),
    engineering_gate_passed=bool(gate and gate['passed']),output_bytes=size,resource_caps_passed=resource,
    simulations_and_physical_counts='See durable execution_count.json; worker enforces both counts.',
    completed=(code==0),one_attempt_only=True)
save('counterfactual_run_manifest.json',record)
(OUT/'RUN_STATE.md').write_text('# GRID07\n\n'+('COUNTERFACTUAL_RECORDED_AWAITING_AUDIT' if code==0 and resource else 'STOPPED_ENGINEERING_OR_RESOURCE_FAILURE')+'\nOne attempt only; no retry, other data, training or automation.\n',encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
sys.exit(0 if code==0 and resource and record['engineering_gate_passed'] else 1)
