"""Serial immutable workers for GRID67/68; no training, no repeated completed items."""
from pathlib import Path
import argparse,ctypes,datetime as dt,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[67,68],required=True);ap.add_argument('--worker',type=int);a=ap.parse_args()
 OUT=ROOT/f'outputs/grid{a.stage}';D=read(OUT/'design.json')
 for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 if a.worker is not None:
  source_path=ROOT/'work/grid55/run.py';source=source_path.read_text(encoding='utf-8')
  assert source.count("OUT=ROOT/'outputs/grid55'")==1
  source=source.replace("OUT=ROOT/'outputs/grid55'",f"OUT=ROOT/'outputs/grid{a.stage}'")
  marker='    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose';assert source.count(marker)==1
  patch="    sys.path.insert(0,str(ROOT/'work/grid67'))\n    from mechanism_adapter import install_mechanism_adapter\n    install_mechanism_adapter(engine.Search)\n"
  source=source.replace(marker,patch+marker)
  sys.argv=[str(source_path),'--worker',str(a.worker)]
  exec(compile(source,str(source_path),'exec'),{'__file__':str(source_path),'__name__':'__main__'});return
 assert not (OUT/'finished.json').exists() and not (OUT/'RUN.lock').exists()
 (OUT/'RUN.lock').write_text(str(os.getpid()))
 stamp=time.perf_counter();started=dt.datetime.now(dt.timezone.utc).isoformat()
 lease=ctypes.windll.kernel32.SetThreadExecutionState;lease.argtypes=[ctypes.c_uint32];lease.restype=ctypes.c_uint32;assert lease(0x80000001)
 summaries=read(OUT/'manifest.json')['summaries'] if (OUT/'manifest.json').exists() else []
 existing={(s['repeat'],s['week'],s['method']) for s in summaries}
 save(OUT/'runtime.json',{'pid':os.getpid(),'started_utc':started,'permanent_power_plan_changed':False})
 (OUT/'RUN_STATE.md').write_text('RUNNING: frozen serial workers; no training.\n',encoding='utf-8')
 try:
  for i,item in enumerate(D['execution_order']):
   identity=(item['repeat'],item['week'],item['method'])
   if identity in existing:continue
   result=OUT/f'worker_result_{i:02d}.json';folder=OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
   if result.exists() and (folder/'components.json').exists():
    s=read(result);assert s==read(folder/'summary.json')
   else:
    assert not folder.exists() and not result.exists(),'Partial output needs explicit recovery review'
    assert time.perf_counter()-stamp<D['caps']['wall_s']
    begin=dt.datetime.now(dt.timezone.utc).isoformat()
    with (OUT/f'worker_{i:02d}.log').open('x',encoding='utf-8') as f:
     p=subprocess.run([sys.executable,str(Path(__file__)),'--stage',str(a.stage),'--worker',str(i)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=D['caps']['episode_s']+60,creationflags=subprocess.CREATE_NO_WINDOW)
    assert p.returncode==0,(a.stage,i,p.returncode)
    save(OUT/f'interval_{i:02d}.json',{'index':i,'started_utc':begin,'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'item':item});s=read(result)
   summaries.append(s);existing.add(identity);save(OUT/'manifest.json',{'planned':len(D['execution_order']),'summaries':summaries})
   print(json.dumps({'stage':a.stage,'completed':len(summaries),'planned':len(D['execution_order']),'method':item['method'],'week':item['week'],'steps':s['steps']}),flush=True)
  assert len(summaries)==len(D['execution_order'])
  assert sum(s['steps'] for s in summaries)<=D['caps']['physical_steps']
  assert sum(s['public_forecasts'] for s in summaries)<=D['caps']['public_forecasts']
  save(OUT/'finished.json',{'passed_engineering':True,'summaries':summaries,'started_utc':started,'ended_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'wall_s':time.perf_counter()-stamp,'new_fits':0})
  (OUT/'RUN_STATE.md').write_text('RUNS COMPLETE: saved-data audit pending.\n',encoding='utf-8')
 except BaseException:
  import traceback;save(OUT/'runner_error.json',{'traceback':traceback.format_exc(),'completed':len(summaries)});raise
 finally:
  lease(0x80000000);(OUT/'RUN.lock').unlink()
if __name__=='__main__':main()
