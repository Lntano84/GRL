"""One persistent serial benchmark, saved-data audit, report and publication."""
from pathlib import Path
import ctypes,datetime as dt,json,os,subprocess,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid87';PY=ROOT/'work/grid02/.venv/Scripts/python.exe'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp');os.environ['PYTHONIOENCODING']='utf-8';os.environ.pop('PYTHONPATH',None)
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
def save(n,x):
    p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def task(script,*args):
    with (OUT/(Path(script).stem+'_pipeline.log')).open('a',encoding='utf-8') as f:
        subprocess.run([str(PY),'-X','utf8',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def main():
    assert not (OUT/'pipeline.lock').exists() and not (OUT/'decision.json').exists()
    (OUT/'pipeline.lock').write_text(str(os.getpid()));ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save('pipeline_state.json',dict(phase='serial_benchmark_13',pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        if not (OUT/'finished.json').exists():task('work/grid87/run.py')
        save('pipeline_state.json',dict(phase='saved_data_audit',pid=os.getpid()))
        task('work/grid87/audit.py');task('work/grid87/finalize.py')
        target=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID87_complete'
        if not target.exists():task('work/grid87/archive.py','--phase','complete')
        env=dict(os.environ);env['PATH']='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd;'+env['PATH']
        with (OUT/'publication.log').open('a',encoding='utf-8') as f:
            subprocess.run([os.environ['ComSpec'],'/d','/c','scripts\\git_push.cmd','Archive GRID87 fixed256 original13 quality regression'],cwd=ROOT/'work/GRL-results-integration',env=env,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        save('pipeline_state.json',dict(phase='complete',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),report='outputs/grid87/GRID87_report.md'))
    except BaseException:
        save('pipeline_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);(OUT/'pipeline.lock').unlink()
if __name__=='__main__':main()
