"""Serial matrix with per-run checkpoint, no automatic retry or substitution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid04'
design=json.loads((OUT/'design_freeze.json').read_text())
assert json.loads((OUT/'preflight_v3.json').read_text())['passed']
manifest_path=OUT/'run_manifest_v3.json'
assert not manifest_path.exists(),'Never repeat the matrix implicitly'
snapshot={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'work/grid04').glob('*.py'))}
(OUT/'execution_code_v3.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
prior_wall=sum(r['wall_s'] for name in ['run_manifest.json','run_manifest_main.json'] if (OUT/name).exists() for r in json.loads((OUT/name).read_text()))
started=time.monotonic()
records=[]
for run in design['plan']:
    assert time.monotonic()-started+prior_wall<design['caps']['total_runner_s']
    folder=OUT/'runs_v3'/f"{run['scenario']}__{run['policy']}"
    assert not folder.exists()
    folder.mkdir(parents=True)
    ready=folder/'import_ready.json'
    t=time.monotonic();deadline=t+180;seen=False
    environ=dict(os.environ)
    environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'MPLCONFIGDIR':str(ROOT/'work/grid02/mplconfig')})
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'work/grid04/run_one.py'),
            '--scenario',run['scenario'],'--policy',run['policy']],stdout=log,stderr=subprocess.STDOUT,env=environ)
        while process.poll() is None:
            if ready.exists() and not seen:
                seen=True;deadline=time.monotonic()+design['caps']['per_run_after_import_s']
            if time.monotonic()>deadline or time.monotonic()-started+prior_wall>design['caps']['total_runner_s']:
                process.kill();process.wait()
                (folder/'watchdog_timeout.json').write_text(json.dumps({'import_seen':seen,'wall_s':time.monotonic()-t}),encoding='utf-8')
                break
            time.sleep(.2)
    row={**run,'exit':process.returncode,'wall_s':time.monotonic()-t}
    records.append(row)
    manifest_path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text(f"# GRID04\n\nRUNNING: {len(records)}/20 trajectories ended. Last: {row}.\nNo training or automation.\n",encoding='utf-8')
    print(json.dumps(row),flush=True)
    if process.returncode: sys.exit(process.returncode)
    # Bound total persisted output before starting another trajectory.
    size=sum(p.stat().st_size for p in (OUT/'runs_v3').rglob('*') if p.is_file())
    assert size<=design['caps']['output_bytes'],'Output ceiling reached'
(OUT/'RUN_STATE.md').write_text('# GRID04\n\n20/20 trajectories completed. Saved-file audit pending. No training/automation.\n',encoding='utf-8')
print('MATRIX_COMPLETE',time.monotonic()-started,flush=True)
