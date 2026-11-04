"""One finite stdlib-only prep check. No real models/environment smoke."""
import copy,json,os,subprocess,sys,time
from guard import R,W,O,I,A,read,sha,save,git,validate_receipt,SCOPE
from state import State
from source_builder import build,instrument_worker
def main():
 start=time.perf_counter();assert not (O/'preparation_review.json').exists() and not (O/'physical_authorization.json').exists();assert not git('status','--porcelain').strip()
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h
 for p,h in read(O/'static_freeze.json').items():assert sha(R/p)==h
 for p,h in read(O/'model_freeze.json').items():assert sha(R/p)==h
 modelblobs={}
 for local,ref in read(O/'published_model_bindings.json').items():
  body=git('show',head+':'+ref['archive_path']);assert __import__('hashlib').sha256(body).hexdigest()==ref['sha256']==sha(R/local);modelblobs[local]=ref
 compile(build(),'generated_runtime.py','exec');compile(instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),'instrumented_worker.py','exec')
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 d=read(O/'design.json');calendar=read(O/'calendar.json');roles={r['pool']:r for r in read(O/'pool_roles.json')};items=d['baseline_runtime']['execution_order'];assert len(items)==144 and len(calendar['dates'])==16 and len({(x['pool'],x['week']) for x in items})==144
 assert sum(i['KEEP'] for i in items)==12 and sum(i['partition']=='own_training_fit8' for i in items)==96
 for i in items:assert i['date'] in roles[i['pool']]['fit_dates' if i['partition']=='own_training_fit8' else 'cal_dates'] and i['date'] not in roles[i['pool']]['outer_dates_excluded']
 auth={'scope_id':SCOPE,'generation_scenarios':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'wall_cap_s':86400,'generation_cap_s':7200,'generation_worker_cap_s':1800,'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'approved_by_user':True,'user_instruction':'synthetic fixture only, never written as physical authorization','user_message_reference':'synthetic','design_sha256':'D','source_freeze_sha256':'F','calendar_sha256':'C'}
 validate_receipt(auth,'D','F','C');rejected=0
 for key in auth:
  bad=dict(auth);bad[key]=None
  try:validate_receipt(bad,'D','F','C')
  except PermissionError:rejected+=1
  else:raise AssertionError('bad scope accepted: '+key)
 def row(n):return {'step':n,'before_hash':'s'+str(n-1),'action_hash':'a'+str(n),'after_hash':'s'+str(n),'raw_cost':float(n),'done':False,'complete':False,'restore':False}
 def offer(s):s.begin({'step':2,'state_hash':'s1','base_action_hash':'base','order_hash':'order','aliases_hash':'aliases','rho_limit':1.1});s.observe([],[],{},[],1.1,set(),[7,8],2,0)
 native={'native_q':[1.,0.],'NN_input':{'public':[1.]}};synthetic=0
 for ordinary in [False,True]:
  original={'continue_':ordinary,'log':{}};base=State({'arm':'BASELINE261'});base.accept(row(1));offer(base);assert base.force([0.],native,original) is original;target=base.candidates[0];assert base.event is None;synthetic+=1
  opposite=State({'arm':'STOP_NEXT' if ordinary else 'CONT_NEXT','target':target},[row(1)]);opposite.before(1,'s0','a1');opposite.accept(row(1));offer(opposite);got=opposite.force([0.],native,original);assert got['continue_']==(not ordinary);opposite.before(2,'s1','DIFFERENT_FULL_JOINT_ACTION');opposite.accept(row(2));assert opposite.force([0.],native,original) is original;assert opposite.record()['prefix_matched'];synthetic+=1
  bad=State({'arm':'STOP_NEXT' if ordinary else 'CONT_NEXT','target':target},[row(1)]);bad.accept(row(1));offer(bad)
  try:bad.force([999.],native,original)
  except RuntimeError:pass
  else:raise AssertionError('changed public marker was allowed')
  assert bad.event is None;synthetic+=1
 bad=State({'arm':'CONT_NEXT','target':target},[row(1)])
 try:bad.before(1,'s0','WRONG_JOINT_ACTION')
 except RuntimeError:synthetic+=1
 else:raise AssertionError('pre-step action mismatch allowed')
 try:bad.before(2,'s1','a2')
 except RuntimeError:synthetic+=1
 else:raise AssertionError('missing target accepted')
 # The selected event can be later than the first nativeSTOP; earlier ones must
 # keep the original261 action, including a STOP, without consuming the target.
 base=State({'arm':'BASELINE261'});base.accept(row(1));offer(base);ordinary={'continue_':False,'log':{}};base.force([0.],native,ordinary);base.accept(row(2))
 later={'step':3,'state_hash':'s2','base_action_hash':'base3','order_hash':'order3','aliases_hash':'aliases3','rho_limit':1.1};base.begin(later);base.observe([],[],{},[],1.1,set(),[9],1,0);base.force([1.],native,ordinary);target=base.candidates[-1]
 follow=State({'arm':'CONT_NEXT','target':target},[row(1),row(2)]);follow.accept(row(1));offer(follow);assert follow.force([0.],native,ordinary) is ordinary and follow.event is None;follow.before(2,'s1','a2');follow.accept(row(2));follow.begin(later);follow.observe([],[],{},[],1.1,set(),[9],1,0);assert follow.force([1.],native,ordinary)['continue_'] is True;follow.before(3,'s2','different');synthetic+=1
 denials=[]
 for script,args in [('manager.py',[]),('worker.py',['--index','0']),('generate_worker.py',['--index','0'])]:
  result=subprocess.run([sys.executable,'-B','-X','utf8',str(W/script),*args],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=20)
  output=result.stdout.decode('utf8',errors='replace');assert result.returncode!=0 and 'new physical authorization absent' in output;denials.append({'entrypoint':script,'returncode':result.returncode,'denied_before_live_import':True});(O/(script+'.missing_auth.txt')).write_text(output,encoding='utf8')
 assert not (O/'collection_started.json').exists() and not (O/'claims').exists() and not (O/'runs').exists() and not any((R/d['environment']/'chronics').iterdir())
 review={'passed':True,'source_sha':head,'exact_source_blobs':blobs,'exact_prior_payload_blobs':modelblobs,'model_payload_files84_hashed_not_deserialized':len(modelblobs),'baseline_members':144,'new_dates':16,'synthetic_state_cases':synthetic,'synthetic_invalid_auth_fields_rejected':rejected,'missing_auth_entrypoint_denials':denials,'generated_source_compilation_passed':True,'new_model_loads':0,'new_predictions':0,'new_generation':0,'new_physics':0,'new_labels':0,'new_fits':0,'resource_passed':False,'new_year_live_compatibility_tested':False,'wall_s':time.perf_counter()-start,'note':'Finite synthetic checks and byte/interface bindings only. No live Windows manager-worker or2038 generator/NN/physical smoke; first actual failure consumes original scope, no automatic retry.'}
 save(O/'preparation_review.json',review);print('GRID300_OFFLINE_PREPARATION_PASSED',synthetic,rejected,len(modelblobs),review['wall_s'])
if __name__=='__main__':main()
