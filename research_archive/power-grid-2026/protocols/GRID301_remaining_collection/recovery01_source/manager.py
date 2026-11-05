"""New108 baselines then at most572 opposites; never replay index35 or old35."""
import json,os,subprocess,sys,time,traceback,uuid
from guard import R,W,O,OLD,read,save,require_authorization,verify_frozen,check
from stages import folder,finish_baseline
def main():
 require_authorization(); launch=read(O/'launch_context.json');check(launch)
 assert not any((O/n).exists() for n in ['started_recovery01.json','claims_recovery01','runs'])
 with (O/'started_recovery01.json').open('x',encoding='utf8') as f: json.dump({**launch,'manager_pid':os.getpid(),'no_retry':True},f)
 # Integrity work and all subsequent startup/loading count in the new window.
 verify_frozen();check(launch)
 D=read(O/'design.json'); reused=read(OLD/'manifest.json')['records']; assert [x['index'] for x in reused]==list(range(35))
 records=[]; steps=0; forecasts=0; ctx={**launch,'manager_pid':os.getpid(),'token':uuid.uuid4().hex}
 def replay(item):
  nonlocal steps,forecasts
  index=item['index']; assert index!=35 and index not in range(35) and not folder(item).exists()
  if steps>=1371560 or forecasts>=22500000: raise TimeoutError('new approved resource cap')
  current={**ctx,'kind':'replay','index':index,'worker_deadline_monotonic':min(ctx['deadline_monotonic'],time.perf_counter()+1200),'physical_remaining':1371560-steps,'forecast_remaining':22500000-forecasts}
  check(current);save(O/'run_context.json',current);save(O/'pipeline_state.json',{'phase':'baseline' if index<144 else 'opposite','index':index,'new_accepted':len(records),'reused_baselines':35,'skipped_partial_index':35})
  path=O/'claims_recovery01'/f'replay_{index:03d}.json';path.parent.mkdir(exist_ok=True)
  with path.open('x',encoding='utf8') as f: json.dump({k:current[k] for k in ['manager_pid','token','kind','index']},f)
  with (O/f'replay_recovery01_{index:03d}.log').open('x',encoding='utf8') as log:
   env={**os.environ,'GRL_GRID300_TOKEN':ctx['token'],'PYTHONDONTWRITEBYTECODE':'1','TEMP':str(R/'tmp/grid100'),'TMP':str(R/'tmp/grid100')}
   child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'worker.py'),'--index',str(index)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   save(O/'active_worker.json',{'active':True,'pid':child.pid,'manager_pid':os.getpid(),'index':index})
   try:
    while child.poll() is None:check(current);time.sleep(.5)
   except BaseException:
    if child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait()
    raise
   finally:save(O/'active_worker.json',{'active':False,'index':index,'manager_pid':os.getpid()})
   if child.returncode:raise RuntimeError('worker failed; no retry: '+str(index))
  check(current);summary=read(folder(item)/'summary.json');source=read(folder(item)/'intervention.json')
  assert summary==read(O/f'worker_result_{index:02d}.json') and summary['steps']==source['steps'] and summary['new_model_fits']==0 and source['new_cost_labels']==0
  if index>=144:assert source['prefix_matched'] and source['event']['common']==item['target']['common']
  steps+=summary['steps'];forecasts+=source['forecasts'];assert steps<=1371560 and forecasts<=22500000
  records.append({'index':index,'summary':summary,'source':source})
  save(O/'manifest.json',{'records':records,'new_accepted':len(records),'reused_baselines':35,'physical_steps_new':steps,'forecasts_new':forecasts,'labels_derived':0})
 try:
  for item in D['baseline_runtime']['execution_order'][36:]:replay(item)
  opposite=finish_baseline(reused+records.copy());check(ctx)
  from auto_publish import release_baseline
  release_baseline(ctx,ctx['deadline_monotonic'])
  for item in opposite:replay(item)
  check(ctx);save(O/'finished.json',{'new_runs':len(records),'new_baselines':108,'reused_baselines':35,'skipped_partial_index':35,'opposite_runs':len(opposite),'physical_steps_new':steps,'forecasts_new':forecasts,'wall_s':time.perf_counter()-ctx['start_monotonic'],'new_fits':0,'labels_derived':0})
  save(O/'pipeline_state.json',{'phase':'collection_complete_saved_review_pending','new_runs':len(records),'reused_baselines':35,'skipped_partial_index':35})
 except BaseException:
  save(O/'collection_error_recovery01.json',{'traceback':traceback.format_exc(),'new_accepted':len(records),'physical_steps_new':steps,'forecasts_new':forecasts,'wall_s':time.perf_counter()-ctx['start_monotonic'],'no_retry_or_extension':True,'preserve_all_partial':True});raise
if __name__=='__main__':main()
