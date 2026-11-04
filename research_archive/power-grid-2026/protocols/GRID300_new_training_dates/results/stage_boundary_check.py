"""Supplement only unexercised queue extremes and deadline/stop boundaries."""
from pathlib import Path
import copy,json,time
import guard,stages
from guard import R,O,read,save,sha
def main():
 assert read(O/'preparation_review.json')['passed'] and not (O/'stage_boundary_review.json').exists()
 scratch=R/'tmp/grid300_prep_boundaries';scratch.mkdir(exist_ok=False);actual_O=guard.O;checks=[]
 try:
  for kind,count in [('empty',0),('maximum',576)]:
   fake=scratch/kind;fake.mkdir();stages.O=fake;design=copy.deepcopy(read(O/'design.json'));save(fake/'design.json',design);records=[]
   for item in design['baseline_runtime']['execution_order']:
    folder=stages.folder(item);folder.mkdir(parents=True);(folder/'synthetic_only.txt').write_text('Synthetic fixture, not a physical episode.\n',encoding='utf8')
    markers=[] if kind=='empty' else [{'common':{'step':t,'call_index':q,'batch':0,'state_hash':'SYNTHETIC_ONLY'},'ordinary261_continue':bool(q%2)} for q,t in enumerate([1,505,1009,1513])]
    records.append({'index':item['index'],'source':{'candidates':markers}})
   got=stages.finish_baseline(records);runtime=read(fake/'runtime_design.json');missing=read(fake/'selected_markers.json')['empty_bins'];assert len(got)==count and len(runtime['execution_order'])==144+count and len(missing)==576-count
   assert [i['index'] for i in runtime['execution_order']]==list(range(144+count))
   assert all(i['arm']==('STOP_NEXT' if i['target']['ordinary261_continue'] else 'CONT_NEXT') and i['within_pair']==0 for i in got)
   checks.append({'fixture':kind,'opposite_runs':count,'missing_bins':len(missing),'passed':True})
  guard.O=scratch
  try:guard.check({'deadline_monotonic':time.perf_counter()-1})
  except TimeoutError:checks.append({'fixture':'expired_total_window','passed':True})
  else:raise AssertionError('expired clock accepted')
  guard.check({'deadline_monotonic':time.perf_counter()+60});checks.append({'fixture':'original_clock_not_reset','passed':True})
  save(scratch/'STOP_REQUEST.json',{'synthetic':True})
  try:guard.check({'deadline_monotonic':time.perf_counter()+60})
  except RuntimeError:checks.append({'fixture':'user_stop_before_next_step','passed':True})
  else:raise AssertionError('stop request ignored')
 finally:guard.O=actual_O;stages.O=actual_O
 assert not (O/'collection_started.json').exists() and not (O/'physical_authorization.json').exists()
 review={'passed':True,'checks':checks,'new_models_loaded':0,'new_predictions':0,'new_generation':0,'new_physics':0,'original_qualify_not_repeated':True,'synthetic_fixture_directory':str(scratch)}
 save(O/'stage_boundary_review.json',review)
 # Preserve the initial first-pass receipt, then link this targeted supplement.
 initial=read(O/'preparation_review.json');save(O/'preparation_review_initial.json',initial);initial['queue_and_clock_boundary_review_sha256']=sha(O/'stage_boundary_review.json');initial['queue_and_clock_boundaries_passed']=True;save(O/'preparation_review.json',initial)
 print('GRID300_QUEUE_AND_CLOCK_BOUNDARIES_PASSED',len(checks))
if __name__=='__main__':main()
