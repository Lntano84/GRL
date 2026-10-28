"""Fixed six-slot process pool with a single immutable global deadline."""
from pathlib import Path
import os,sys,time,subprocess,traceback
import runtime254 as C
PY=sys.executable

def launch(args,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);f=(out/'stdout.log').open('wb');e=(out/'stderr.log').open('wb');env=dict(os.environ,TEMP=str(C.R/'tmp/grid254'),TMP=str(C.R/'tmp/grid254'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1');script=C.W/('fake_worker.py' if str(args[0])=='fake' else 'worker.py');worker_args=args[1:] if str(args[0])=='fake' else args;p=subprocess.Popen([PY,'-X','utf8','-u',str(script),*map(str,worker_args)],stdout=f,stderr=e,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);f.close();e.close();return p

def stop_processes(active):
 for p,*_ in active:
  if p.poll() is None:p.terminate()
 for p,*_ in active:
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)

def run_pool(specs,deadline,stop_path,mode):
 assert mode in ['train','fake'];active=[];pending=list(specs);done=[];launched=[]
 try:
  while pending or active:
   C.before(deadline)
   if Path(stop_path).exists():raise InterruptedError('Fixed stop flag observed; no new jobs.')
   for p,item in list(active):
    code=p.poll()
    if code is None:continue
    active.remove((p,item))
    if code!=0:raise RuntimeError(f'Job failed without retry: {item["node"]}, exit={code}')
    done.append(item['node'])
   while pending and len(active)<6:
    C.before(deadline);item=pending.pop(0);p=launch(item['args'],item['logdir']);active.append((p,item));launched.append(dict(node=item['node'],pid=p.pid))
   time.sleep(.05)
  return dict(passed=True,completed=done,launched=launched,all_workers_exited=True)
 except BaseException as e:
  stop_processes(active);return dict(passed=False,error=repr(e),completed=done,launched=launched,pending=[v['node'] for v in pending],all_workers_exited=all(p.poll() is not None for p,*_ in active))

def train_run():
 root=C.TRAIN_OUT;jobs=C.read(C.O/'jobs.json');assert len(jobs)==18;assert not (root/'run_context.json').exists() and not (root/'training_started.json').exists(),'Existing run must be inspected, never restarted.'
 q=C.read(C.O/'qualification.json');auth=C.read(root/'authorization.json');assert q['passed'] and q['execution_qualified'] and auth['passed'] and auth['all_sources_inputs_published'] and auth['qualification_sha256']==C.sha(C.O/'qualification.json') and auth['qualified_freeze_sha256']==C.sha(C.O/'qualified_freeze.json') and auth['jobs_sha256']==C.sha(C.O/'jobs.json') and auth['models']==18 and auth['fit_cap_s']==600
 C.verify();C.claim(root/'training_started.json',dict(stage='referenceNN18',optimizer_fits_permitted=18,max_workers=6,window_s=600));start=time.monotonic();C.save(root/'run_context.json',dict(started_monotonic=start,deadline=start+600,max_workers=6,nodes=[j['node'] for j in jobs]));specs=[dict(node=j['node'],args=['train',j['node'],root],logdir=root/'jobs'/j['node']) for j in jobs];result=run_pool(specs,start+600,root/'STOP_REQUESTED','train');result['wall_s']=time.monotonic()-start
 if result['passed']:
  records=[C.read(root/'jobs'/j['node']/'complete.json') for j in jobs];assert all(r['passed'] and r['updates']==7250 and r['global_elapsed_s']<600 for r in records);result['records']=records;C.save(root/'reference_finished.json',result)
 else:C.save(root/'reference_error.json',result)
 return result
if __name__=='__main__':
 assert sys.argv[1]=='train';r=train_run();print(r['passed']);sys.exit(0 if r['passed'] else 1)
