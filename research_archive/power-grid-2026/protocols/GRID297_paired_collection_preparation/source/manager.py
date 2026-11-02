"""One irreversible-claim, serial 8h window. No resume/retry or label generation."""
import os,sys,time,json,uuid,subprocess,traceback
from pathlib import Path
from guard import R,O,read,require_authorization,verify_frozen
W=Path(__file__).parent
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def folder(item):return O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
def main():
 start=time.perf_counter();require_authorization();verify_frozen();D=read(O/'design.json')
 with (O/'collection_started.json').open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),monotonic_start=start,deadline=start+28800),f)
 token=uuid.uuid4().hex;results=[];total_steps=total_forecasts=0
 try:
  for index,item in enumerate(D['execution_order']):
   remaining=start+28800-time.perf_counter()
   if remaining<=0 or total_steps>=580896 or total_forecasts>=9000000:raise TimeoutError('total collection resource cap')
   if folder(item).exists():raise RuntimeError('existing full/partial item; no retry')
   ctx=dict(manager_pid=os.getpid(),token=token,start_monotonic=start,deadline_monotonic=start+28800,worker_deadline_monotonic=min(start+28800,time.perf_counter()+1200),forecast_remaining=9000000-total_forecasts,physical_remaining=580896-total_steps)
   save(O/'run_context.json',ctx);p=O/'claims'/f'{index:03d}.json';p.parent.mkdir(exist_ok=True)
   with p.open('x',encoding='utf8') as f:json.dump(dict(index=index,manager_pid=os.getpid(),token=token),f)
   save(O/'pipeline_state.json',dict(phase='collection',completed=len(results),index=index,planned=288))
   with (O/f'worker_{index:03d}.log').open('x',encoding='utf8') as log:
    env=os.environ.copy();env['GRL_GRID297_TOKEN']=token;env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(R/'tmp/grid100')
    child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'worker.py'),'--index',str(index)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(O/'active_worker.json',dict(index=index,pid=child.pid,parent=os.getpid()))
    try:ret=child.wait(timeout=max(0.,ctx['worker_deadline_monotonic']-time.perf_counter()))
    except subprocess.TimeoutExpired:
     subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();raise
   if ret:raise RuntimeError(('worker failure; preserve partial, no retry',index,ret))
   summary=read(folder(item)/'summary.json');event=read(folder(item)/'intervention.json')
   assert summary==read(O/f'worker_result_{index:02d}.json')
   total_steps+=summary['steps'];total_forecasts+=event['forecasts'];assert total_steps<=580896 and total_forecasts<=9000000
   results.append(dict(index=index,summary=summary,intervention=event));save(O/'manifest.json',dict(planned=288,records=results,total_steps=total_steps,total_forecasts=total_forecasts))
   if item['within_pair']==1:
    first=results[-2]['intervention'];second=event
    if bool(first['event'])!=bool(second['event']):raise RuntimeError('matched episode trigger missing in one arm')
    if first['event']:assert first['event']['common']==second['event']['common']
    else:assert first['complete_no_event_signature']==second['complete_no_event_signature']
    save(O/f'pair_{item["pair_index"]:03d}.json',dict(pair_index=item['pair_index'],prefix_matched=True,event_present=bool(first['event']),no_labels_derived=True))
  if time.perf_counter()>start+28800:raise TimeoutError('total window during final save')
  save(O/'finished.json',dict(runs=288,pairs=144,wall_s=time.perf_counter()-start,new_fits=0,labels_derived=0));save(O/'pipeline_state.json',dict(phase='source_collection_complete_label_join_pending',completed=288))
 except BaseException as e:
  save(O/'collection_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=len(results),preserve_all_partials=True,no_retry=True));raise
if __name__=='__main__':main()
