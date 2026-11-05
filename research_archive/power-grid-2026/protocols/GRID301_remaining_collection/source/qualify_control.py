"""One bounded source-first qualification; never load real models or environments."""
import ast,copy,os,subprocess,sys,time
from guard import R,W,O,I,A,read,save,sha,git,expected_fields,validate_receipt,validate_clock
from stages import fixed_selection
def main():
 start=time.perf_counter();assert not (O/'preparation_review.json').exists() and not (O/'physical_authorization.json').exists() and not (O/'launch_context.json').exists()
 assert not git('status','--porcelain').strip(); head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 old=R/'work/grid300_new_training_dates/continuation01'; copied=['state.py','adapter.py','worker.py','auto_publish.py']
 assert all(sha(W/n)==sha(old/n) for n in copied)
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 import source_builder
 compile(source_builder.build(),'generated_worker','exec')
 assert 'grid301_remaining_collection' in source_builder.build() and not (W/'generate_worker.py').exists()
 auth={**expected_fields('D','F'),'approved_by_user':True,'user_instruction':'synthetic qualification only','user_message_reference':'synthetic'};validate_receipt(auth,'D','F');rejected=[]
 for k in auth:
  bad=dict(auth);bad[k]=None
  try:validate_receipt(bad,'D','F')
  except PermissionError:rejected.append(k)
  else:raise AssertionError(k)
 launch=dict(start_monotonic=100.,deadline_monotonic=86500.,start_utc_epoch=1000.,deadline_utc_epoch=87400.,boot_id='BOOT_A');ctx=dict(launch);validate_clock(ctx,launch,101.,1001.,'BOOT_A');cases=[]
 for name,change,mono,utc,boot,kind in [
  ('utc_expired',{},101.,87400.,'BOOT_A',TimeoutError),('mono_expired',{},86500.,1001.,'BOOT_A',TimeoutError),
  ('boot_changed',{},101.,1001.,'BOOT_B',PermissionError),('mono_reset',{},5.,1001.,'BOOT_A',PermissionError),
  ('utc_backwards',{},101.,900.,'BOOT_A',PermissionError),('deadline_extension',{'deadline_monotonic':86501.},101.,1001.,'BOOT_A',PermissionError),
  ('worker_timeout',{'worker_deadline_monotonic':101.},101.,1001.,'BOOT_A',TimeoutError)]:
  try:validate_clock({**ctx,**change},launch,mono,utc,boot)
  except kind:cases.append(name)
  else:raise AssertionError(name)
 items=[x for x in read(O/'design.json')['baseline_runtime']['execution_order'] if x['index']!=35];synthetic=[]
 for item in items:
  candidates=[]
  for band in range(4):
   for offset in [3,1]:candidates.append({'common':dict(step=band*504+offset,call_index=offset,batch=0),'ordinary261_continue':bool(band%2)})
  synthetic.append({'index':item['index'],'source':{'candidates':candidates}})
 selected,empty=fixed_selection(items,synthetic);assert len(selected)==572 and not empty and len({s['index'] for s in selected})==572
 assert all(s['baseline_index']!=35 and s['target']['common']['step']%504==1 for s in selected)
 assert sum('outputs/grid300_' in x['baseline_folder'] for x in selected)==140
 assert sum('outputs/grid301_' in x['baseline_folder'] for x in selected)==432
 synthetic[0]['source']['candidates']=[]; fewer,empty=fixed_selection(items,synthetic);assert len(fewer)==568 and len(empty)==4
 import auto_publish
 scratch=R/'tmp/grid301_control';scratch.mkdir(exist_ok=False)
 originals=(auto_publish.O,auto_publish.check,auto_publish.subprocess.Popen,auto_publish.verify_stage);publication_cases=[]
 try:
  for case,active,rc in [('published',False,0),('active_worker',True,0),('helper_failed',False,3)]:
   f=scratch/case;f.mkdir();save(f/'active_worker.json',{'active':active});auto_publish.O=f;seen=[]
   auto_publish.check=lambda c:validate_clock(c,launch,101.,1001.,'BOOT_A')
   class Fake:
    def __init__(self,args,**kw):assert args[-2:]==['--stage','baseline'];seen.append('spawn');self.returncode=rc
    def poll(self):return self.returncode
   auto_publish.subprocess.Popen=Fake;auto_publish.verify_stage=lambda stage:seen.append('verified')
   try:auto_publish.release_baseline(dict(launch),launch['deadline_monotonic'])
   except AssertionError:assert case=='active_worker' and not seen
   except RuntimeError:assert case=='helper_failed' and seen==['spawn']
   else:assert case=='published' and seen==['spawn','verified']
   publication_cases.append({'case':case,'sequence':seen})
 finally:auto_publish.O,auto_publish.check,auto_publish.subprocess.Popen,auto_publish.verify_stage=originals
 denials=[]
 for name,args in [('launch.py',[]),('manager.py',[]),('worker.py',['--index','36']),('publish_stage.py',['--stage','baseline'])]:
  p=subprocess.run([sys.executable,'-B','-X','utf8',str(W/name),*args],cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
  out=p.stdout.decode('utf8',errors='replace');assert p.returncode!=0 and 'GRID301 authorization absent' in out;denials.append(name);(O/(name+'.denied.txt')).write_text(out,encoding='utf8')
 assert not any((O/n).exists() for n in ['launch_context.json','started.json','claims','runs']) and time.perf_counter()-start<600
 save(O/'preparation_review.json',{'passed':True,'source_commit':head,'exact_source_blobs':blobs,'copied_modules_byte_identical':copied,'old_model_or_numerical_checks_not_repeated':True,'invalid_authorization_fields_rejected':rejected,'clock_rejection_cases':cases,'synchronous_publication_cases':publication_cases,'synthetic_routing':{'baseline_count':143,'old_reused_opposite_references':140,'new_opposite_references':432,'missing_index35_preserved':True,'max572_and_empty_no_replacement':True},'missing_approval_entrypoints_denied':denials,'new_generation':0,'new_model_loads':0,'new_predictions':0,'new_physics':0,'new_labels':0,'new_fits':0,'wall_s':time.perf_counter()-start,'static_pass_not_resource_guarantee':True})
 print('PREPARATION_PASSED',len(blobs),len(rejected),len(cases),len(denials))
if __name__=='__main__':main()
