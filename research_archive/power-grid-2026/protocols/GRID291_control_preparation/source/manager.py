"""One serial finite manager. No automatic retry, resume, or budget extension."""
from pathlib import Path
import sys,os,time,json,subprocess,uuid,traceback,ctypes
from guard import R,O,read,sha,require_authorization,verify_frozen
W=Path(__file__).parent
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def main():
 start=time.perf_counter();require_authorization(remote=True);verify_frozen(data=True)
 if not read(O/'preparation_review.json')['passed']:raise PermissionError('preparation review not passed')
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];kernel.CreateMutexW.restype=ctypes.c_void_p
 handle=kernel.CreateMutexW(None,True,'GRL_GRID291_SINGLE_MANAGER')
 if not handle or ctypes.get_last_error()==183:raise RuntimeError('manager already exists')
 kernel.ReleaseMutex.argtypes=[ctypes.c_void_p];kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.SetThreadExecutionState(0x80000001)
 try:
  with (O/'benchmark_started.json').open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),wall_cap_s=14400,monotonic_start=start),f)
  token=uuid.uuid4().hex;save(O/'run_context.json',dict(manager_pid=os.getpid(),token=token,start_monotonic=start,deadline_monotonic=start+14400))
  results=[];D=read(O/'design.json')
  for index,item in enumerate(D['execution_order']):
   remaining=14400-(time.perf_counter()-start)
   if remaining<=0:raise TimeoutError('original total window exhausted')
   folder=O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
   if folder.exists():raise RuntimeError('partial/complete item exists; do not rerun')
   claim=O/'claims'/f'{index:03d}.json';claim.parent.mkdir(exist_ok=True)
   with claim.open('x',encoding='utf8') as f:json.dump(dict(manager_pid=os.getpid(),token=token,index=index),f)
   save(O/'pipeline_state.json',dict(phase='benchmark',index=index,completed=len(results),planned=192,pid=os.getpid()))
   with (O/f'worker_{index:03d}.log').open('x',encoding='utf8') as log:
    env=os.environ.copy();env['GRL_GRID291_TOKEN']=token
    jobstart=time.perf_counter();child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(W/'worker.py'),'--index',str(index)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    save(O/'active_worker.json',dict(pid=child.pid,parent=os.getpid(),index=index))
    try:ret=child.wait(timeout=max(0.,min(1200-(time.perf_counter()-jobstart),start+14400-time.perf_counter())))
    except subprocess.TimeoutExpired:
     subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();save(O/'timeout.json',dict(index=index,partial_folder=str(folder),original_total_cap_s=14400,original_worker_cap_s=1200));raise
   if ret!=0:raise RuntimeError(('worker engineering failure',index,ret))
   s=read(folder/'summary.json');assert s==read(O/f'worker_result_{index:02d}.json')
   for n in ['steps.jsonl.gz','vectors.npz','metadata.json','components.json']:assert (folder/n).exists()
   results.append(s);assert sum(x['steps'] for x in results)<=D['caps']['physical_steps'] and sum(x['public_forecasts'] for x in results)<=D['caps']['public_forecasts']
   save(O/'manifest.json',dict(planned=192,summaries=results));save(O/f'manager_receipt_{index:03d}.json',dict(index=index,worker_process_wall_s=time.perf_counter()-jobstart,remaining_total_s=14400-(time.perf_counter()-start)))
  if time.perf_counter()-start>14400:raise TimeoutError('original window exhausted while saving receipts')
  save(O/'finished.json',dict(passed_engineering=True,runs=192,summaries=results,benchmark_wall_s=time.perf_counter()-start));save(O/'pipeline_state.json',dict(phase='saved_review_pending',completed=192))
 finally:
  kernel.SetThreadExecutionState(0x80000000);kernel.ReleaseMutex(handle);kernel.CloseHandle(handle)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if (O/'benchmark_started.json').exists():save(O/'pipeline_error.json',dict(error=repr(e),traceback=traceback.format_exc(),preserve_all_partials=True,no_automatic_retry=True))
  raise
