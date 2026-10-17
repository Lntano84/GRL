"""One checkpointed serial benchmark; no outcome-based decisions."""
from pathlib import Path
import ctypes,datetime as dt,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review';W=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def validate_complete(folder,result):
 s=read(folder/'summary.json');assert s==read(result)
 assert (folder/'components.json').exists() and (folder/'vectors.npz').exists()
 with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:n=sum(1 for _ in f)
 assert n==s['steps'];return s
def main():
 mutex=ctypes.windll.kernel32.CreateMutexW(None,True,'GRL_GRID89_SERIAL_BENCHMARK')
 assert ctypes.windll.kernel32.GetLastError()!=183,'Another suite owns benchmark mutex'
 if (OUT/'active_worker.json').exists() and not (OUT/'benchmark_finished.json').exists():
  old=read(OUT/'active_worker.json');handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,old['pid'])
  if handle:
   exitcode=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(exitcode));ctypes.windll.kernel32.CloseHandle(handle)
   assert exitcode.value!=259,'Recorded worker still alive; refuse a second runner'
 assert read(OUT/'preflight.json')['passed'] and read(OUT/'generation.json')['passed']
 for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 for p,h in read(OUT/'data_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:
  for phase in ['order','parameters','meta','sealed']:
   base=OUT/phase;D=read(base/'design.json');summaries=read(base/'manifest.json')['summaries'] if (base/'manifest.json').exists() else []
   existing={(s['week'],s['method'],s['repeat']) for s in summaries}
   for index,item in enumerate(D['execution_order']):
    elapsed=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(read(OUT/'design.json')['frozen_utc'])).total_seconds()
    assert elapsed<43200,'Frozen twelve-hour suite limit reached; preserve partial results'
    key=(item['week'],item['method'],item['repeat'])
    if key in existing:continue
    folder=base/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}";result=base/f'worker_result_{index:02d}.json'
    if folder.exists():
     if result.exists():
      summaries.append(validate_complete(folder,result));existing.add(key);save(base/'manifest.json',dict(planned=len(D['execution_order']),summaries=summaries));continue
     # Only a stopped child can leave this branch. Keep its attempt and rerun the incomplete item, never a completed one.
     raise RuntimeError(f'Partial output requires recovery audit: {folder}')
    stamp=dt.datetime.now(dt.timezone.utc).isoformat();save(OUT/'pipeline_state.json',dict(phase='benchmark',suite=phase,completed_total=sum(read(OUT/x/'manifest.json')['summaries'].__len__() if (OUT/x/'manifest.json').exists() else 0 for x in ['order','parameters','meta','sealed']),planned_total=246,index=index,item=item,pid=os.getpid(),utc=stamp))
    (OUT/'RUN_STATE.md').write_text(f'RUNNING {phase} {index+1}/{len(D["execution_order"])}; one serial worker; all outcomes retained.\n',encoding='utf-8')
    with (base/f'worker_{index:03d}.log').open('a',encoding='utf-8') as f:
     child=subprocess.Popen([sys.executable,'-u',str(W/'worker.py'),'--phase',phase,'--index',str(index)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     save(OUT/'active_worker.json',dict(pid=child.pid,parent=os.getpid(),phase=phase,index=index,item=item))
     try:code=child.wait(timeout=D['caps']['episode_s']+120)
     except subprocess.TimeoutExpired:
      subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT);child.wait(timeout=60);raise
    assert code==0,(phase,index,code)
    s=validate_complete(folder,result);summaries.append(s);existing.add(key)
    save(base/'manifest.json',dict(planned=len(D['execution_order']),summaries=summaries));save(base/f'interval_{index:03d}.json',dict(started_utc=stamp,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),item=item))
    print(json.dumps(dict(phase=phase,completed=len(summaries),planned=len(D['execution_order']))),flush=True)
   assert len(summaries)==len(D['execution_order']);save(base/'finished.json',dict(passed_engineering=True,summaries=summaries))
  save(OUT/'benchmark_finished.json',dict(passed=True,runs=246,all_phases=['order','parameters','meta','sealed'],utc=dt.datetime.now(dt.timezone.utc).isoformat()))
 finally:
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);ctypes.windll.kernel32.ReleaseMutex(mutex);ctypes.windll.kernel32.CloseHandle(mutex)
if __name__=='__main__':
 try:main()
 except BaseException:
  save(OUT/'benchmark_error.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),traceback=traceback.format_exc()));raise
