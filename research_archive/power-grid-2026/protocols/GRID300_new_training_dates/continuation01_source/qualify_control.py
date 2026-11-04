"""One targeted control-only check; old numerical tests/models are not rerun."""
import ast,hashlib,os,subprocess,sys,time
from guard import R,W,O,I,A,read,save,sha,git,validate_receipt,SCOPE,check
import auto_publish
def main():
 assert not (O/'continuation01_preparation_review.json').exists() and not (O/'continuation01_authorization.json').exists() and not (O/'continuation01_started.json').exists()
 assert not git('status','--porcelain').strip();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'continuation01_source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 copied=['state.py','adapter.py','source_builder.py','worker.py','stages.py'];assert all(sha(W/n)==sha(W.parent/n) for n in copied)
 text=(W/'manager.py').read_text(encoding='utf8');assert 'start=time.perf_counter()' not in text and "start=original['monotonic_start']" in text and "O/'continuation01_started.json'" in text
 tree=ast.parse(text)
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run'];assert calls and all(n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='replay' for n in calls)
 assert not (W/'generate_worker.py').exists()
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 original=read(O/'collection_started.json');auth={'scope_id':SCOPE,'generation_new':0,'generation_reused':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'original_start_monotonic':original['monotonic_start'],'original_deadline_monotonic':original['deadline'],'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'design_sha256':'D','source_freeze_sha256':'F','original_authorization_sha256':sha(O/'physical_authorization.json'),'failed_generation_gate_preserved':True,'approved_by_user':True,'user_instruction':'synthetic only, never physical approval','user_message_reference':'synthetic'}
 validate_receipt(auth,'D','F');rejected=0
 for k in auth:
  bad=dict(auth);bad[k]=None
  try:validate_receipt(bad,'D','F')
  except PermissionError:rejected+=1
  else:raise AssertionError('changed receipt accepted: '+k)
 scratch=R/'tmp/grid300_continuation01_control';scratch.mkdir(exist_ok=False);oldO=auto_publish.O;oldP=auto_publish.subprocess.Popen;oldV=auto_publish.verify_stage;control=[]
 try:
  for case,active,returncode in [('success',False,0),('active_worker',True,0),('publish_failure',False,3)]:
   folder=scratch/case;folder.mkdir();save(folder/'active_worker.json',{'active':active});auto_publish.O=folder;seen=[]
   class Fake:
    def __init__(self,args,**kwargs):assert args[-2:]==['--stage','baseline'];seen.append('spawn');self.returncode=returncode
    def poll(self):return self.returncode
   auto_publish.subprocess.Popen=Fake;auto_publish.verify_stage=lambda stage:seen.append('verified')
   try:auto_publish.release_baseline({'deadline_monotonic':original['deadline']},original['deadline'])
   except AssertionError:assert case=='active_worker' and not seen
   except RuntimeError:assert case=='publish_failure' and seen==['spawn']
   else:assert case=='success' and seen==['spawn','verified']
   control.append({'case':case,'sequence':seen,'passed':True})
 finally:auto_publish.O=oldO;auto_publish.subprocess.Popen=oldP;auto_publish.verify_stage=oldV
 for ctx,kind in [({'deadline_monotonic':original['deadline'],'worker_deadline_monotonic':time.perf_counter()-1},TimeoutError),({'deadline_monotonic':original['deadline']+1},PermissionError)]:
  try:check(ctx)
  except kind:control.append({'case':kind.__name__,'passed':True})
  else:raise AssertionError('bad deadline accepted')
 denials=[]
 for name,args in [('manager.py',[]),('worker.py',['--index','0']),('publish_stage.py',['--stage','baseline'])]:
  result=subprocess.run([sys.executable,'-B','-X','utf8',str(W/name),*args],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=20);out=result.stdout.decode('utf8',errors='replace');assert result.returncode!=0 and 'continuation01 authorization absent' in out;denials.append(name);(O/(name+'.continuation01_denied.txt')).write_text(out,encoding='utf8')
 assert not (O/'runs').exists() and not (O/'claims_continuation01').exists() and not (O/'continuation01_started.json').exists()
 review={'passed':True,'source_sha':head,'exact_source_blobs':blobs,'copied_numerical_modules_byte_identical':copied,'old_model_or_numerical_qualification_not_repeated':True,'no_generation_entrypoint_or_call':True,'invalid_new_auth_fields_rejected':rejected,'synchronous_publication_and_deadline_cases':control,'missing_new_auth_entrypoints_denied':denials,'original_start_monotonic':original['monotonic_start'],'original_deadline_monotonic':original['deadline'],'new_model_loads':0,'new_predictions':0,'new_physics':0,'new_generation':0,'new_labels':0,'new_fits':0,'live_continuation_not_started':True}
 save(O/'continuation01_preparation_review.json',review);print('GRID300_CONTINUATION_CONTROL_CHECK_PASSED',len(copied),rejected,len(control),len(denials))
if __name__=='__main__':main()
