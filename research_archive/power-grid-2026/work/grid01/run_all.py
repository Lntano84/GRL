"""Serial bounded processes; no silent retries after action execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid01'
started=time.monotonic()
records=[]
for scenario,mode in [(0,'full'),(1,'full'),(0,'replay'),(1,'replay')]:
    if time.monotonic()-started>=1800:
        raise TimeoutError('Frozen whole simulation cap reached')
    dst=OUT/f'scenario{scenario}_{mode}'
    dst.mkdir(exist_ok=True)
    assert not (dst/'import_ready.json').exists(),'Existing run: do not overwrite'
    ready=dst/'import_ready.json'
    run_start=time.monotonic()
    deadline=run_start+180
    seen=False
    with (dst/'console.log').open('w',encoding='utf-8') as log:
        environ=dict(os.environ)
        environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        process=subprocess.Popen([sys.executable,str(ROOT/'work/grid01/run_one.py'),'--scenario',str(scenario),'--mode',mode],
                                 stdout=log,stderr=subprocess.STDOUT,env=environ)
        while process.poll() is None:
            if ready.exists() and not seen:
                seen=True
                deadline=time.monotonic()+600
            if time.monotonic()>deadline or time.monotonic()-started>1800:
                process.kill()
                process.wait()
                (dst/'watchdog_timeout.json').write_text(json.dumps({'import_seen':seen,'wall_s':time.monotonic()-run_start}),encoding='utf-8')
                raise TimeoutError('Trajectory or whole allocation exceeded')
            time.sleep(.1)
    records.append({'scenario':scenario,'mode':mode,'exit':process.returncode,'wall_s':time.monotonic()-run_start})
    (OUT/'run_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(records[-1],flush=True)
    if process.returncode:
        sys.exit(process.returncode)
print('ALL COMPLETE',time.monotonic()-started,flush=True)
