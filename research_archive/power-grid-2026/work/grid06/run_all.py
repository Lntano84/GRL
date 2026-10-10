"""Twelve serial sealed-validation runs. Preserve and stop on errors/limits."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
design=load(OUT/'design_freeze.json');qualification=load(OUT/'qualification.json')
assert qualification['passed'] and qualification['physical_steps']==qualification['native_simulates']==0
assert not (OUT/'run_manifest.json').exists()
for filename in ['reused_assets_frozen.json','data_files_frozen.json']:
    for p,h in load(OUT/filename).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
snapshot={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'work/grid06').glob('*.py'))}
(OUT/'execution_code_frozen.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
started=time.monotonic();records=[];prior_wall=qualification['wall_s']
for job in design['plan']:
    assert time.monotonic()-started+prior_wall<design['caps']['total_runner_including_qualification_s']
    folder=OUT/'runs'/f"{job['scenario']}__{job['policy']}"
    assert not folder.exists();folder.mkdir(parents=True)
    environ=dict(os.environ);environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'MPLCONFIGDIR':str(ROOT/'work/grid02/mplconfig')})
    t=time.monotonic();deadline=t+design['caps']['import_s'];seen=False
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'work/grid06/run_one.py'),'--scenario',job['scenario'],'--policy',job['policy']],stdout=log,stderr=subprocess.STDOUT,env=environ)
        while proc.poll() is None:
            if (folder/'import_ready.json').exists() and not seen:
                seen=True;deadline=time.monotonic()+design['caps']['per_run_after_import_s']
            if time.monotonic()>deadline or time.monotonic()-started+prior_wall>design['caps']['total_runner_including_qualification_s']:
                proc.kill();proc.wait();(folder/'watchdog_timeout.json').write_text(json.dumps({'wall_s':time.monotonic()-t,'import_seen':seen}));break
            time.sleep(.2)
    record={**job,'exit':proc.returncode,'wall_s':time.monotonic()-t};records.append(record)
    (OUT/'run_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text(f"# GRID06\n\nRUNNING: {len(records)}/12. Last: {record}. No training or automation.\n",encoding='utf-8')
    print(json.dumps(record),flush=True)
    if proc.returncode:sys.exit(proc.returncode)
    assert sum(p.stat().st_size for p in (OUT/'runs').rglob('*') if p.is_file())<=design['caps']['output_bytes']
    assert sum(load(OUT/'runs'/f"{r['scenario']}__{r['policy']}"/'execution_count.json')['physical_calls'] for r in records)<=design['caps']['physical_steps']
(OUT/'RUN_STATE.md').write_text('# GRID06\n\n12/12 jobs ended; saved-data audit pending. No training or automation.\n',encoding='utf-8')
print('MATRIX_COMPLETE',time.monotonic()-started,flush=True)
