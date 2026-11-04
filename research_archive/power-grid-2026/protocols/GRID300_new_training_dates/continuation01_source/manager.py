"""One24h irreversible serial acquisition. Publication waits retain original clock."""
import json,os,subprocess,sys,time,traceback,uuid
from guard import R,W,O,read,save,require_authorization,verify_frozen,check,verify_stage
from stages import folder,finish_baseline
def main():
 require_authorization();verify_frozen();original=read(O/'collection_started.json');start=original['monotonic_start'];D=read(O/'design.json');check({'deadline_monotonic':original['deadline']})
 if (O/'claims_continuation01').exists() or (O/'runs').exists():raise RuntimeError('old full/partial work exists; no automatic retry/resume')
 with (O/'continuation01_started.json').open('x',encoding='utf8') as f:json.dump({'pid':os.getpid(),'monotonic_start':start,'deadline':start+86400,'generation_deadline':start+7200,'no_retry':True},f)
 token=uuid.uuid4().hex;ctx={'manager_pid':os.getpid(),'token':token,'start_monotonic':start,'deadline_monotonic':start+86400};steps=0;records=[];generation=read(O/'generation_manifest.json')['records'];forecasts=sum(x['schema_forecast_calls'] for x in generation)
 def run(kind,index,limit):
  nonlocal forecasts
  if forecasts>=22500000 or (kind=='replay' and steps>=1452240):raise TimeoutError('global forecast/physical cap reached before starting another worker')
  current={**ctx,'kind':kind,'index':index,'worker_deadline_monotonic':min(ctx['deadline_monotonic'],limit),'forecast_remaining':22500000-forecasts,'physical_remaining':1452240-steps}
  check(current);save(O/'run_context.json',current);claim=O/'claims_continuation01'/f'{kind}_{index:03d}.json';claim.parent.mkdir(exist_ok=True)
  with claim.open('x',encoding='utf8') as f:json.dump({'token':token,'manager_pid':os.getpid(),'kind':kind,'index':index},f)
  script=W/('generate_worker.py' if kind=='generation' else 'worker.py')
  with (O/f'{kind}_{index:03d}.log').open('x',encoding='utf8') as log:
   env=os.environ.copy();env['GRL_GRID300_TOKEN']=token;env['PYTHONDONTWRITEBYTECODE']='1';env['TEMP']=env['TMP']=str(R/'tmp/grid100')
   child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(script),'--index',str(index)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);save(O/'active_worker.json',{'kind':kind,'index':index,'pid':child.pid,'manager_pid':os.getpid()})
   try:
    while child.poll() is None:check(current);time.sleep(.5)
   except BaseException:
    if child.poll() is None:
     if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
     else:child.terminate()
     child.wait()
    raise
   finally:save(O/'active_worker.json',{'active':False,'kind':kind,'index':index,'manager_pid':os.getpid()})
   if child.returncode:raise RuntimeError((kind,index,child.returncode,'preserve partial, no retry'))
  check(current)
 def publication(name,deadline):
  assert name=='baseline'
  from auto_publish import release_baseline
  release_baseline(ctx,deadline)
 def replay(item):
  nonlocal steps,forecasts
  assert item['index']==len(records) and not folder(item).exists();save(O/'pipeline_state.json',{'phase':'baseline' if item['index']<144 else 'opposite','index':item['index'],'completed':len(records),'max_runs':720,'original_deadline':ctx['deadline_monotonic']})
  run('replay',item['index'],time.perf_counter()+1200);summary=read(folder(item)/'summary.json');source=read(folder(item)/'intervention.json');assert summary==read(O/f"worker_result_{item['index']:02d}.json")
  assert summary['steps']==source['steps'] and summary['new_model_fits']==0 and source['new_cost_labels']==0
  if item.get('target') is not None:assert source['prefix_matched'] and source['event']['common']==item['target']['common']
  steps+=summary['steps'];forecasts+=source['forecasts'];assert steps<=1452240 and forecasts<=22500000
  record={'index':item['index'],'summary':summary,'source':source};records.append(record);save(O/'manifest.json',{'records':records,'completed':len(records),'physical_steps':steps,'forecasts':forecasts,'labels_derived':0});return record
 try:
  verify_stage('generation')
  for item in D['baseline_runtime']['execution_order']:replay(item)
  opposite=finish_baseline(records.copy());publication('baseline',ctx['deadline_monotonic'])
  for item in opposite:replay(item)
  check(ctx);save(O/'finished.json',{'runs':len(records),'baseline_runs':144,'opposite_runs':len(opposite),'wall_s':time.perf_counter()-start,'physical_steps':steps,'forecasts':forecasts,'generation_scenarios':16,'new_fits':0,'labels_derived':0});save(O/'pipeline_state.json',{'phase':'source_collection_complete_label_review_pending','completed':len(records),'planned_actual':144+len(opposite),'max_runs':720})
 except BaseException:
  save(O/'collection_error_continuation01.json',{'traceback':traceback.format_exc(),'completed_runs':len(records),'completed_generation':len(generation),'wall_s':time.perf_counter()-start,'physical_steps_accepted':steps,'forecasts_accepted':forecasts,'partial_work_not_zero':True,'preserve_all_partials':True,'no_retry_or_extension':True});raise
if __name__=='__main__':main()
