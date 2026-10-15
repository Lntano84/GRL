"""Preserve interrupted child; resume unchanged frozen GRID63 and audit full interval."""
from pathlib import Path
import datetime as dt, hashlib, json, shutil, subprocess, sys, traceback
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid63'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    manifest=read(OUT/'manifest.json');assert len(manifest['summaries'])==44
    design=read(OUT/'design.json');old_runtime=read(OUT/'runtime.json')
    for i,s in enumerate(manifest['summaries']):
        folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
        assert s==read(folder/'summary.json')==read(OUT/f'worker_result_{i:02d}.json')
        assert (folder/'components.json').exists() and (folder/'vectors.npz').exists()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=OUT/'interruptions'/stamp;target.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','runtime.json','RUN_STATE.md']:
        shutil.copy2(OUT/name,target/name)
    item=design['execution_order'][44]
    partial=OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
    assert partial.resolve().is_relative_to((OUT/'runs').resolve())
    assert not (partial/'summary.json').exists() and not (OUT/'worker_result_44.json').exists()
    assert partial.exists() and (OUT/'RUN.lock').exists()
    partial_receipts=[{'path':p.relative_to(partial).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in partial.rglob('*') if p.is_file()]
    shutil.move(str(partial),str(target/'partial_child_44'))
    shutil.move(str(OUT/'worker_44.log'),str(target/'worker_44_interrupted.log'))
    shutil.move(str(OUT/'RUN.lock'),str(target/'stale_RUN.lock'))
    receipt={'reason':'External execution session ended with code 1073807364; no GRID63 process remained. Not a policy failure.',
        'verified_completed_runs':44,'partial_worker_index':44,'partial_files':partial_receipts,
        'code_freeze_unchanged':True,'completed_runs_not_repeated':True,
        'original_started_utc':old_runtime['started_utc'],'recovery_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'partial_progress':read(target/'partial_child_44/progress.json'),
        'partial_costs_are_not_zero':'Recorded progress covers only a prefix. Exact total interrupted computation is unknown.'}
    save(target/'recovery_receipt.json',receipt)
    save(OUT/'recovery_receipt.json',receipt)
    rc=subprocess.run([sys.executable,'-u',str(ROOT/'work/grid63/run.py')],cwd=ROOT).returncode
    if rc:raise RuntimeError(f'unchanged resumed runner exited {rc}')
    finished=read(OUT/'finished.json');shutil.copy2(OUT/'finished.json',target/'finished_resume_segment.json')
    finished['resume_segment_started_utc']=finished['started_utc']
    finished['resume_segment_wall_s']=finished['wall_s']
    finished['started_utc']=old_runtime['started_utc']
    finished['wall_s']=(dt.datetime.fromisoformat(finished['ended_utc'])-dt.datetime.fromisoformat(finished['started_utc'])).total_seconds()
    finished['wall_scope']='Elapsed original start to final end including interruption gap; successful-run wall sums and partial prefix separately preserved.'
    finished['recovery_receipt']='recovery_receipt.json'
    save(OUT/'finished.json',finished)
    rc=subprocess.run([sys.executable,'-u',str(ROOT/'work/grid63/audit.py')],cwd=ROOT).returncode
    if rc:raise RuntimeError(f'saved-data audit exited {rc}')
    save(OUT/'recovery_finished.json',{'passed':True,'completed_runs':63,'original_code_unchanged':True,'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    print('GRID63_RESUMED_AND_AUDITED',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        save(OUT/'recovery_error.json',{'traceback':traceback.format_exc()});raise
