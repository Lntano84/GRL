"""Keep all143 available baseline receipts, including early terminations."""
import shutil,time
from guard import R,O,OLD,read,save,sha
def folder(item): return O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
def baseline_folder(item):
 root=OLD if item['index']<35 else O
 return root/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
def fixed_selection(items,records):
 selected=[]; empty=[]
 assert [i['index'] for i in items]==[j for j in range(144) if j!=35]
 assert [r['index'] for r in records]==[i['index'] for i in items]
 for item,record in zip(items,records):
  for band in range(4):
   candidates=[m for m in record['source']['candidates'] if min(3,(m['common']['step']-1)//504)==band]
   if not candidates: empty.append(dict(baseline_index=item['index'],quarter=band,status='NO_EVENT_NO_REPLACEMENT')); continue
   marker=min(candidates,key=lambda m:(m['common']['step'],m['common']['call_index'],m['common']['batch']))
   x=dict(item); x.update(index=144+len(selected),arm='STOP_NEXT' if marker['ordinary261_continue'] else 'CONT_NEXT',method=f"OPPOSITE_Q{band}_B{item['index']:03d}__{item['pool']}",target=marker,baseline_index=item['index'],baseline_folder=baseline_folder(item).relative_to(R).as_posix(),quarter=band)
   selected.append(x)
 assert len(selected)<=572
 return selected,empty
def finish_baseline(records):
 start=time.perf_counter(); D=read(O/'design.json'); original=D['baseline_runtime']; items=[x for x in original['execution_order'] if x['index']!=35]
 selected,empty=fixed_selection(items,records); raw={}
 for item in items:
  for p in baseline_folder(item).rglob('*'):
   if p.is_file(): raw[p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
  if time.perf_counter()-start>600: raise TimeoutError('one source sealing window; no retry')
 runtime={**original,'execution_order':list(original['execution_order'])+selected}
 save(O/'baseline_manifest.json',{'records':records,'completed':143,'missing_partial_baseline_index':35,'labels_derived':0})
 save(O/'baseline_raw_manifest.json',raw); save(O/'runtime_design.json',runtime)
 save(O/'selected_markers.json',{'new_opposite_items':selected,'empty_bins':empty,'missing_baseline_index':35,'no_outcome_selection':True,'labels_derived':0})
 names=['baseline_manifest.json','baseline_raw_manifest.json','selected_markers.json','runtime_design.json']; S=O/'stage_payload/baseline'; S.mkdir(parents=True,exist_ok=False)
 for n in names: shutil.copy2(O/n,S/n)
 save(O/'baseline_ready.json',{'stage':'baseline','files':{n:sha(S/n) for n in names},'labels_derived':0})
 return selected
