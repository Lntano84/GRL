"""Persistent serial training, qualification, benchmark, audit and publication."""
from pathlib import Path
import ctypes,datetime as dt,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84';PY=ROOT/'work/grid02/.venv/Scripts/python.exe'
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp');os.environ['PYTHONIOENCODING']='utf-8';os.environ.pop('PYTHONPATH',None)
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
def save(name,x):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def task(script,*args):
    with (OUT/(Path(script).stem+'_pipeline.log')).open('a',encoding='utf-8') as log:
        subprocess.run([str(PY),'-X','utf8',str(ROOT/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def publish(message):
    env=dict(os.environ);env['PATH']='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd;'+env['PATH']
    with (OUT/'publication.log').open('a',encoding='utf-8') as log:
        subprocess.run([os.environ['ComSpec'],'/d','/c','scripts\\git_push.cmd',message],cwd=ROOT/'work/GRL-results-integration',env=env,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
def phase(name):save('pipeline_state.json',dict(phase=name,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
def main():
    assert not (OUT/'pipeline.lock').exists() and not (OUT/'decision.json').exists()
    (OUT/'pipeline.lock').write_text(str(os.getpid()));ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if not (OUT/'training_finished.json').exists():phase('three_fixed_cost_fits');task('work/grid84/train.py')
        if not (OUT/'preflight.json').exists():phase('training_feedback_qualification');task('work/grid84/preflight.py')
        dest=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID84_models'
        if not dest.exists():phase('model_archive');task('work/grid84/archive.py','--phase','models');publish('Archive GRID84 three additional fixed-cost baseline seeds')
        if not (OUT/'finished.json').exists():phase('serial_benchmark_72');task('work/grid84/run.py')
        phase('saved_data_audit');task('work/grid84/audit.py');task('work/grid84/finalize.py')
        dest=ROOT/'work/GRL-results-integration/research_archive/power-grid-2026/protocols/GRID84_complete'
        if not dest.exists():task('work/grid84/archive.py','--phase','complete')
        publish('Archive GRID84 five-seed matched comparison and full distribution')
        phase('complete')
    except BaseException:
        save('pipeline_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);(OUT/'pipeline.lock').unlink()
if __name__=='__main__':main()
