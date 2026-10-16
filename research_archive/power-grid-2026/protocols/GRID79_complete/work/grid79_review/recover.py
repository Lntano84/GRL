"""Preserve an interrupted incomplete worker; never rerun a completed entry."""
from pathlib import Path
import datetime as dt,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=(ROOT/'outputs/grid79').resolve()
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'finished.json').exists() and not (OUT/'recovery_01.json').exists()
    old=read(OUT/'runtime.json');D=read(OUT/'design.json');M=read(OUT/'manifest.json');frozen=read(OUT/'code_freeze.json')
    for name,h in frozen.items():assert sha(ROOT/name)==h,name
    completed={};evidence=[]
    for s in M['summaries']:
        folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";assert read(folder/'summary.json')==s
        completed[s['repeat'],s['week'],s['method']]=s
        for name in ['metadata.json','summary.json','components.json','steps.jsonl.gz','vectors.npz']:
            p=folder/name;assert p.exists();evidence.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    pending=[(j,x) for j,x in enumerate(D['execution_order']) if (x['repeat'],x['week'],x['method']) not in completed]
    partial=[];pids={int(old['pid'])}
    for j,x in pending:
        folder=OUT/'runs'/f"r{x['repeat']}__{x['week']}__{x['method']}"
        if folder.exists():
            assert not (folder/'summary.json').exists() and not (OUT/f'worker_result_{j:02d}.json').exists()
            if (folder/'metadata.json').exists():pids.add(int(read(folder/'metadata.json')['process_id']))
            partial.append((j,folder))
    assert len(M['summaries'])==51 and [x[0] for x in partial]==[51]
    query=' OR '.join(f'ProcessId={p}' for p in sorted(pids))
    command=f"@(Get-CimInstance Win32_Process -Filter '{query}').Count"
    got=subprocess.check_output(['powershell','-NoProfile','-Command',command],creationflags=subprocess.CREATE_NO_WINDOW,text=True).strip();assert got=='0',got
    dest=(OUT/'interrupted_attempts'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    assert dest.is_relative_to(OUT) and not dest.exists();dest.mkdir(parents=True)
    before=sha(OUT/'manifest.json');moved=[]
    for j,folder in partial:
        for p in [folder,OUT/f'worker_{j:02d}.log']:
            source=p.resolve();target=(dest/p.name).resolve();assert source.is_relative_to(OUT) and target.is_relative_to(OUT)
            source.rename(target);moved.append(dict(source=str(source),preserved=str(target)))
    for name in ['RUN.lock','runtime.json']:
        p=OUT/name
        if p.exists():p.rename(dest/name)
    assert sha(OUT/'manifest.json')==before
    result=dict(passed=True,frozen_files=len(frozen),completed_preserved=len(completed),completed_manifest_sha256=before,completed_files=evidence,
        partial_worker_indices=[x[0] for x in partial],pending_indices=[j for j,x in pending],old_pids_verified_stopped=sorted(pids),moved=moved,
        cause='Previous parent/worker not running; incomplete worker has no final result or Python traceback. External interruption cause not established.',
        verification_note='Initial read-only verification print reused a variable and raised TypeError after assertions; no files modified by that check.',
        timing_note='An interruption separates part of repeat1. Keep all completed timing; rerun incomplete worker from start. Disclose boundary rather than adding or dropping timing repeats.')
    (OUT/'recovery_01.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('RECOVERY VERIFIED:51 complete preserved; one incomplete attempt archived; continue unchanged indices51-55.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['completed_files','moved']},ensure_ascii=False))
if __name__=='__main__':main()
