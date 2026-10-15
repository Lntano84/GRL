"""Serial finite repair pilot, then gated exposed-date regression."""
from pathlib import Path
import argparse,ctypes,datetime as dt,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid76'
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--phase',choices=['pilot','remaining'],default='pilot');a=ap.parse_args()
    D=read(OUT/'design.json')
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    if a.worker is not None:
        src=ROOT/'work/grid55/run.py';text=src.read_text(encoding='utf-8')
        text=text.replace("OUT=ROOT/'outputs/grid55'","OUT=ROOT/'outputs/grid76'")
        marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert text.count(marker)==1
        patch="    import importlib.util\n    ms=importlib.util.spec_from_file_location('grid76_mechanism',ROOT/'work/grid76/mechanism_adapter.py')\n    ma=importlib.util.module_from_spec(ms);ms.loader.exec_module(ma)\n    ps=importlib.util.spec_from_file_location('grid76_policy',ROOT/'work/grid76/policies.py')\n    pm=importlib.util.module_from_spec(ps);ps.loader.exec_module(pm)\n    ma.P=pm;ma.install_mechanism_adapter(engine.Search)\n"
        text=text.replace(marker,patch+marker)
        sys.argv=[str(src),'--worker',str(a.worker)]
        exec(compile(text,str(src),'exec'),{'__file__':str(src),'__name__':'__main__'});return
    assert not (OUT/'RUN.lock').exists() and not (OUT/'finished.json').exists()
    if a.phase=='remaining':assert read(OUT/'pilot_review.json')['continue_regression']
    summaries=read(OUT/'manifest.json')['summaries'] if (OUT/'manifest.json').exists() else []
    existing={(x['repeat'],x['week'],x['method']) for x in summaries}
    (OUT/'RUN.lock').write_text(str(os.getpid()));t0=time.perf_counter();started=dt.datetime.now(dt.timezone.utc).isoformat()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    save(OUT/'runtime.json',dict(pid=os.getpid(),started_utc=started,phase=a.phase))
    (OUT/'RUN_STATE.md').write_text('RUNNING '+a.phase+': original preserved, one serial repair.\n',encoding='utf-8')
    try:
        for i,item in enumerate(D['execution_order']):
            if item['stage_phase']!=a.phase or (item['repeat'],item['week'],item['method']) in existing:continue
            folder=OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}";result=OUT/f'worker_result_{i:02d}.json'
            assert not folder.exists() and not result.exists(),'Partial outputs require explicit recovery review'
            assert time.perf_counter()-t0<D['caps']['wall_s']
            begin=dt.datetime.now(dt.timezone.utc).isoformat()
            with (OUT/f'worker_{i:02d}.log').open('x',encoding='utf-8') as f:
                proc=subprocess.run([sys.executable,str(Path(__file__)), '--worker',str(i)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=D['caps']['episode_s']+60,creationflags=subprocess.CREATE_NO_WINDOW)
            assert proc.returncode==0,(i,proc.returncode)
            save(OUT/f'interval_{i:02d}.json',dict(index=i,started_utc=begin,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),item=item))
            s=read(result);summaries.append(s);save(OUT/'manifest.json',dict(planned=len(D['execution_order']),summaries=summaries))
            print(json.dumps(dict(completed=len(summaries),method=s['method'],week=s['week'],steps=s['steps'],cost=s['cost'],queries=s['n1_forecasts'])),flush=True)
        save(OUT/(a.phase+'_finished.json'),dict(passed_engineering=True,started_utc=started,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),wall_s=time.perf_counter()-t0,runs=len(summaries)))
        if len(summaries)==len(D['execution_order']):save(OUT/'finished.json',dict(passed_engineering=True,summaries=summaries,new_fits=0))
        (OUT/'RUN_STATE.md').write_text(a.phase.upper()+' RUNS COMPLETE: saved-data review pending.\n',encoding='utf-8')
    except BaseException:
        import traceback;save(OUT/'runner_error.json',dict(traceback=traceback.format_exc(),runs=len(summaries)));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);(OUT/'RUN.lock').unlink()
if __name__=='__main__':main()
