"""Public selection and hash catalogues only; no suffix targets or predictions."""
import collections,shutil,time
from guard import R,O,read,sha,save
def folder(item):return O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
def stage(name,files):
 p=O/'stage_payload'/name;p.mkdir(parents=True,exist_ok=False)
 for filename in files:shutil.copy2(O/filename,p/filename)
 save(O/f'{name}_ready.json',{'stage':name,'files':{n:sha(p/n) for n in files},'labels_derived':0,'wait_for_publication':True})
def finish_generation(receipts):
 freeze={p:h for r in receipts for p,h in r['files'].items()};assert len(receipts)==16
 for p,h in freeze.items():assert sha(R/p)==h,p
 save(O/'generated_freeze.json',freeze);save(O/'generation_manifest.json',{'records':receipts,'completed':16,'new_fits':0,'new_policy_steps':0})
 stage('generation',['generated_freeze.json','generation_manifest.json'])
def finish_baseline(records):
 start=time.perf_counter();assert len(records)==144
 runtime=read(O/'design.json')['baseline_runtime'];runtime={'environment':runtime['environment'],'caps':runtime['caps'],'timing':runtime['timing'],'execution_order':[dict(i) for i in runtime['execution_order']]}
 selected=[];empty=[];raw={}
 for item,record in zip(runtime['execution_order'],records):
  assert item['index']==record['index'];candidates=record['source']['candidates']
  for band in range(4):
   group=[m for m in candidates if min(3,(m['common']['step']-1)//504)==band]
   if not group:empty.append({'baseline_index':item['index'],'pool':item['pool'],'date':item['date'],'quarter':band,'status':'NO_EVENT_NO_REPLACEMENT'});continue
   marker=min(group,key=lambda m:(m['common']['step'],m['common']['call_index'],m['common']['batch']));index=144+len(selected);other=dict(item)
   other.update(index=index,arm='STOP_NEXT' if marker['ordinary261_continue'] else 'CONT_NEXT',method=f"OPPOSITE_Q{band}_B{item['index']:03d}__{item['pool']}",target=marker,baseline_index=item['index'],baseline_folder=folder(item).relative_to(R).as_posix(),quarter=band)
   selected.append(other)
  for p in folder(item).rglob('*'):
   if p.is_file():raw[p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
  if time.perf_counter()-start>600:raise TimeoutError('baseline source sealing600s cap; no new model/physics, do not retry')
 assert len(selected)<=576;runtime['execution_order'].extend(selected)
 save(O/'baseline_manifest.json',{'records':records,'completed':144,'labels_derived':0});save(O/'baseline_raw_manifest.json',raw);save(O/'selected_markers.json',{'new_opposite_items':selected,'empty_bins':empty,'selection':'earliest nativeSTOP in each fixed quarter, no cost/completion selection','labels_derived':0});save(O/'runtime_design.json',runtime)
 stage('baseline',['baseline_manifest.json','baseline_raw_manifest.json','selected_markers.json','runtime_design.json'])
 return selected
