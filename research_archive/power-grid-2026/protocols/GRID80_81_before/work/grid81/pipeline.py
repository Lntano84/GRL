"""Persistent serial workflow: wait existing generator, qualify, archive, benchmark, audit, archive."""
from pathlib import Path
import argparse,ctypes,datetime as dt,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid81';PY=ROOT/'work/grid02/.venv/Scripts/python.exe'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp');os.environ['PYTHONIOENCODING']='utf-8'
os.environ.pop('PYTHONPATH',None)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
def save(name,x):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def task(script,*args):
    log=OUT/(Path(script).stem+('_'+args[-1] if args else '')+'_pipeline.log')
    with log.open('a',encoding='utf-8') as f:
        subprocess.run([str(PY),'-X','utf8',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def push(message):
    env=dict(os.environ);env['PATH']='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd;'+env['PATH']
    with (OUT/'publication.log').open('a',encoding='utf-8') as f:
        subprocess.run([os.environ['ComSpec'],'/d','/c','scripts\\git_push.cmd',message],cwd=ROOT/'work/GRL-results-integration',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--generator-pid',type=int,required=True);a=ap.parse_args()
    assert not (OUT/'pipeline.lock').exists();(OUT/'pipeline.lock').write_text(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    save('pipeline_state.json',dict(phase='waiting_existing_generator',pid=os.getpid(),generator_pid=a.generator_pid,started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    try:
        # SYNCHRONIZE process handle prevents PID reuse from being mistaken for the original generator.
        handle=ctypes.windll.kernel32.OpenProcess(0x00100000,False,a.generator_pid)
        if handle:
            status=ctypes.windll.kernel32.WaitForSingleObject(handle,7200*1000);ctypes.windll.kernel32.CloseHandle(handle)
            assert status==0,('generator timeout',status)
        assert (ROOT/'outputs/grid80/generation.json').exists(),'Generator did not produce complete six-scenario receipt; inspect preserved errors, no automatic resampling.'
        save('pipeline_state.json',dict(phase='qualifying_generated_data',pid=os.getpid()))
        task('work/grid80/qualify.py')
        receipt=json.loads((ROOT/'outputs/grid80/generation.json').read_text(encoding='utf-8'))
        hashes=dict(receipt['files'])
        for p in list((ROOT/'work/grid80/fresh_env').iterdir())+[ROOT/'outputs/grid80/generation.json',ROOT/'outputs/grid80/qualification.json']:
            if p.is_file():hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        save('data_freeze.json',hashes)
        task('work/grid_research_audit/archive_grid80_81.py','--phase','qualified')
        push('Qualify frozen GRID80 fresh chronics before GRID81 closed loop')
        save('pipeline_state.json',dict(phase='serial_benchmark',pid=os.getpid(),planned=60))
        task('work/grid81/run.py')
        save('pipeline_state.json',dict(phase='saved_data_audit',pid=os.getpid()))
        task('work/grid81/audit.py');task('work/grid81/finalize.py')
        task('work/grid_research_audit/archive_grid80_81.py','--phase','complete')
        push('Archive GRID81 fresh scenario and random gate comparison results')
        save('pipeline_state.json',dict(phase='complete',pid=os.getpid(),finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),report='outputs/grid81/GRID81_report.md'))
    except BaseException:
        save('pipeline_error.json',dict(traceback=traceback.format_exc(),time_utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);(OUT/'pipeline.lock').unlink()
if __name__=='__main__':main()
