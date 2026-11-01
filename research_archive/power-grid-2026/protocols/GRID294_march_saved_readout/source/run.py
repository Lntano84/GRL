"""One600s saved ledger/common-prefix readout. Standard library only; no live imports."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,shutil,subprocess,sys,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid294_march_saved_readout';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID294_march_saved_readout';P=R/'outputs/grid291_control_preparation';PLAN=R/'outputs/grid293_march_evidence_inventory/NEXT_PROTOCOL_PLAN.json';WEEK='2037-03-02_100'
COMP={'losses':'losses_mwh','redispatch':'redispatch_mwh','curtailment_delta':'curtailment_delta_mwh','storage_throughput':'storage_throughput_mwh'}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def stage(phase,paths):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in paths:
  rel=p.relative_to(O) if p.is_relative_to(O) else Path(p.name);q=d/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})
def prepare():
 assert not O.exists();plan=read(PLAN);assert plan['id']=='GRID294_MARCH_SAVED_COST_AND_PREFIX_READOUT';review=read(P/'saved_partial_review.json');assert review['passed_saved_data'];jobs=[];paths=[W/'run.py',PLAN,P/'design.json',P/'saved_partial_review.json',P/'raw_records_manifest.json',R/'outputs/grid292_activation_support/public_native_rows.json',R/'work/grid03/score_adapter.py',R/'work/grid08/common.py',R/'work/grid55/worker.py']
 manifest=read(P/'raw_records_manifest.json')['files']
 for row in review['records']:
  if row['week']!=WEEK:continue
  folder=P/'runs'/f"r{row['repeat']}__{row['week']}__{row['arm']}__{row['pool']}"
  assert row['complete'] and row['steps']==2017
  job=dict(index=row['index'],pool=row['pool'],arm=row['arm'],week=row['week'],repeat=row['repeat'],folder=folder.relative_to(R).as_posix(),trajectory_signature=row['trajectory_signature'])
  jobs.append(job)
  for n in ['summary.json','metadata.json','steps.jsonl.gz']:
   p=folder/n;assert sha(p)==manifest[p.relative_to(P).as_posix()]['sha256'];paths.append(p)
 assert len(jobs)==42 and len({j['pool'] for j in jobs})==12
 fullsummary=R/read(P/'design.json')['full_references'][WEEK];full=fullsummary.parent
 original=next(x for x in read(R/'outputs/grid293_march_evidence_inventory/design.json')['documents'] if x['id']=='March_FULL_summary');assert sha(fullsummary)==original['sha256']
 paths +=[full/n for n in ['summary.json','metadata.json','steps.jsonl.gz']]
 part=Path(read(P/'timeout.json')['partial_folder']);assert part.resolve().is_relative_to(R.resolve());assert read(P/'timeout.json')['index']==186
 for n in ['partial_steps.jsonl','failure.json','metadata.json']:
  p=part/n;assert sha(p)==manifest[p.relative_to(P).as_posix()]['sha256'];paths.append(p)
 index=read(R/'outputs/grid292_activation_support/public_native_rows.json');allowed={(j['pool'],j['arm'],j['repeat'],j['week']) for j in jobs}
 calls=sorted({R/r['source_call'] for r in index if not r['partial'] and (r['pool'],r['arm'],r['repeat'],r['week']) in allowed})
 for p in calls:assert sha(p)==manifest[p.relative_to(P).as_posix()]['sha256']
 paths+=calls
 design=dict(id='GRID294',plan=plan,jobs=jobs,reference_folder=full.relative_to(R).as_posix(),partial_folder=part.relative_to(R).as_posix(),ledger_schema_source='work/grid03/score_adapter.py:public_cost_ledger',cost_components={k:'per-step marginal_cost * '+v for k,v in COMP.items()},physical_curtailment_level='curtailed_mwh is physical energy only, NOT a substitute for the original reward curtailment_delta_mwh.',call_lookup='Use only published292 native decision source_call for exact first differing delivered-step lookup. No scan of unrelated calls, noFULL call lookup assumed, missing/altered observation association recorded.',readout_cap_s=600,prohibited=['NPZ/model arrays','live common/torch/grid2op','predict/simulate/prepare/decoder','new policy return or mixed strategy','FULL timing comparison','new or repeated physical item'])
 O.mkdir();save(O/'design.json',design);save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths+[O/'design.json']))});stage('source',[W/'run.py',O/'design.json',O/'source_freeze.json']);print('GRID294_SOURCE_FROZEN',len(set(paths)),len(jobs),len(calls),flush=True)
def rows(p,deadline):
 result=[];op=gzip.open if p.name.endswith('.gz') else open
 with op(p,'rt',encoding='utf8') as f:
  for j,t in enumerate(f):
   if j%128==0:assert time.perf_counter()<deadline,'original600sreadoutcap'
   result.append(json.loads(t))
 return result
def aggregate(rr,summary,dt):
 raw=math.fsum(r['raw_cost'] for r in rr);required=set(COMP.values())|{'marginal_cost','recomputed_raw_cost','curtailed_mwh'};bad=[r['step'] for r in rr if r.get('ledger') is None or not required<=r['ledger'].keys()]
 if summary is not None:assert len(rr)==summary['steps'] and abs(raw-summary['cost'])<1e-6
 result=dict(steps=len(rr),raw_cost=raw,ledger_missing_or_schema_steps=bad,cost_parts=None,physical_energy=None,reconstruction_residual=None,step_blocks=[])
 if bad:return result
 terms={k:[r['ledger']['marginal_cost']*r['ledger'][v] for r in rr] for k,v in COMP.items()}
 for j,r in enumerate(rr):assert abs(math.fsum(terms[k][j] for k in COMP)-r['ledger']['recomputed_raw_cost'])<1e-6
 result['cost_parts']={k:math.fsum(v) for k,v in terms.items()};result['physical_energy']={k:math.fsum(r['ledger'][k] for r in rr) for k in [*COMP.values(),'curtailed_mwh']};result['reconstruction_residual']=raw-math.fsum(result['cost_parts'].values());block=int(round(24/dt));assert block>0
 for start in range(0,len(rr),block):
  end=min(start+block,len(rr));result['step_blocks'].append(dict(first_step=rr[start]['step'],last_step=rr[end-1]['step'],raw_cost=math.fsum(r['raw_cost'] for r in rr[start:end]),cost_parts={k:math.fsum(v[start:end]) for k,v in terms.items()}))
 return result
def prefix(rr,fr,meta,fmeta,job,index,deadline):
 checks={k:meta.get(k)==fmeta.get(k) for k in ['library_hashes','dt_hours','cost_per_MW']};initial=rr[0]['before_hash']==fr[0]['before_hash'];result=dict(initial_before_equal=initial,metadata_checks=checks,first_difference=None,status='SAME_RECORDED_ACTION_PREFIX')
 if not initial or not all(checks.values()):result['status']='CONTEXT_OR_LIBRARY_VERSION_MISMATCH';return result
 for a,b in zip(rr,fr):
  if a['before_hash']!=b['before_hash']:
   result.update(status='STATE_DRIFT_BEFORE_DELIVERED_ACTION_DIFFERENCE',first_difference=dict(step=a['step']));return result
  if a['action_hash']==b['action_hash']:continue
  point=dict(step=a['step'],common_before_hash=a['before_hash'],current_delivered_action_hash=a['action_hash'],FULL_delivered_action_hash=b['action_hash'],after_state_equal=a['after_hash']==b['after_hash'],current_restore=a['restore'],FULL_restore=b['restore'],current_raw_cost=a['raw_cost'],FULL_raw_cost=b['raw_cost'],common_prefix_raw_cost_delta=math.fsum(r['raw_cost'] for r in rr[:a['step']-1])-math.fsum(r['raw_cost'] for r in fr[:b['step']-1]),saved_current_calls=[],FULL_call_lookup='No published292FULL index; no ordinal guess or unrelated call scan.')
  matches=sorted({x['source_call'] for x in index if not x['partial'] and (x['pool'],x['arm'],x['repeat'],x['week'],x['step'])==(job['pool'],job['arm'],job['repeat'],job['week'],a['step'])})
  for rel in matches:
   assert time.perf_counter()<deadline; c=read(R/rel);assert c['step']==a['step'];same=c['state_hash']==a['before_hash'];selected=next((x for x in c['outcomes'] if x['pool_id']==c['chosen_pool_id']),None)
   point['saved_current_calls'].append(dict(source_call=rel,call_state_equals_step_before=same,chosen_pool_id=c['chosen_pool_id'],shortlist_choice=c['shortlist_choice'],returned_action_hash=c['returned_action_hash'],chosen_combined_hash=None if selected is None else selected['combined_hash'],chosen_combined_equals_delivered=None if selected is None else selected['combined_hash']==a['action_hash'],guard_reason=c['guard_reason'],public_queries=c['public_queries'],local_queries=c['local_queries'],model_decisions=[{k:d[k] for k in ['batch','action','optional_extension','native_q','parent_b','parent_CONT','residual_logit','engineering_fallback'] if k in d} for d in c['learned_decisions']],association_status='Exact recorded step and before-state' if same else 'Saved step only; state mismatch, no common-state model attribution'))
  if not matches:point['saved_current_call_status']='No exact step member in published292 index; no additional call scan or prediction.'
  result.update(status='FIRST_DELIVERED_ACTION_DIFFERENCE_ON_COMMON_BEFORE',first_difference=point);return result
 return result
def readout():
 start=time.perf_counter();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(I).as_posix())==(O/'source_freeze.json').read_bytes()
 with (O/'readout_started.json').open('x',encoding='utf8') as f:json.dump(dict(source_published=head,pid=__import__('os').getpid(),monotonic_start=start,deadline=start+600,cap_s=600),f)
 deadline=start+600
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 D=read(O/'design.json');F=R/D['reference_folder'];fs=read(F/'summary.json');fm=read(F/'metadata.json');fr=rows(F/'steps.jsonl.gz',deadline);assert fs['complete'] and fs['week']==WEEK and fs['steps']==2017
 fa=aggregate(fr,fs,fm['dt_hours']);save(O/'reference_cost.json',dict(summary={k:fs[k] for k in ['method','week','steps','complete','cost']},aggregate=fa));index=read(R/'outputs/grid292_activation_support/public_native_rows.json');done=[]
 for job in D['jobs']:
  assert time.perf_counter()<deadline,'original600sreadoutcap';receipt=O/'run_receipts'/f"{job['index']:03d}.json";assert not receipt.exists(),'Do not repeat a completed readout unit'
  folder=R/job['folder'];s=read(folder/'summary.json');meta=read(folder/'metadata.json');rr=rows(folder/'steps.jsonl.gz',deadline);aa=aggregate(rr,s,meta['dt_hours']);p=prefix(rr,fr,meta,fm,job,index,deadline)
  delta=dict(raw_cost=aa['raw_cost']-fa['raw_cost'],cost_parts=None,reconstruction_residual=None,step_blocks=[])
  if aa['cost_parts'] is not None and fa['cost_parts'] is not None:
   delta['cost_parts']={k:aa['cost_parts'][k]-fa['cost_parts'][k] for k in COMP};delta['reconstruction_residual']=aa['reconstruction_residual']-fa['reconstruction_residual'];assert abs(delta['raw_cost']-math.fsum([*delta['cost_parts'].values(),delta['reconstruction_residual']]))<1e-6
   for a,b in zip(aa['step_blocks'],fa['step_blocks']):delta['step_blocks'].append(dict(first_step=a['first_step'],last_step=a['last_step'],raw_cost_delta=a['raw_cost']-b['raw_cost'],cost_parts_delta={k:a['cost_parts'][k]-b['cost_parts'][k] for k in COMP}))
  result=dict(**job,recorded_cost=aa,cost_delta_vs_FULL=delta,prefix=p,complete=True);save(receipt,result);done.append(result);print('READOUT_UNIT',len(done),42,'index',job['index'],'firststep',None if p['first_difference'] is None else p['first_difference']['step'],flush=True)
 part=R/D['partial_folder'];pr=rows(part/'partial_steps.jsonl',deadline);failure=read(part/'failure.json');assert len(pr)==failure['rows']==192;pm=read(part/'metadata.json');pa=aggregate(pr,None,pm['dt_hours']);save(O/'partial_separate.json',dict(index=186,recorded_cost=pa,not_compared_to_full2017=True,no_quality_or_speed_claim=True))
 assert time.perf_counter()<deadline and len(done)==42
 groups=[]
 for sig in sorted({r['trajectory_signature'] for r in done}):
  members=[r for r in done if r['trajectory_signature']==sig];a=members[0];assert all(r['cost_delta_vs_FULL']==a['cost_delta_vs_FULL'] for r in members)
  groups.append(dict(signature=sig,indices=[r['index'] for r in members],pools=sorted({r['pool'] for r in members}),arms=sorted({r['arm'] for r in members}),repeats=sorted({r['repeat'] for r in members}),cost_delta_vs_FULL=a['cost_delta_vs_FULL'],prefix_statuses=dict(collections.Counter(r['prefix']['status'] for r in members)),first_difference_steps=sorted({r['prefix']['first_difference']['step'] for r in members if r['prefix']['first_difference'] is not None})))
 save(O/'trajectory_groups.json',groups);save(O/'readout_results.json',dict(reference=fa,runs=done,groups=groups,partial_separate=read(O/'partial_separate.json')))
 save(O/'review.json',dict(passed_saved_readout=True,complete_current_runs=42,reference_runs=1,partial_runs=1,signature_groups=len(groups),per_run_receipts=42,cost_component_units='Original score units from marginal_cost times stored MWh fields; physical curtailment level separately retained.',original_reward_delta_curtailment_preserved=True,FULL_time_not_compared=True,new_fits=0,new_predictions=0,new_physics=0,new_policy_scores_or_returns=0,NPZ_or_model_arrays_read=0,live_modules_imported=sorted(set(sys.modules)&{'torch','grid2op','common','numpy'}),wall_s=time.perf_counter()-start,original_readout_cap_s=600,all_pools_retained=True))
 print('GRID294_READOUT_PASSED',json.dumps(read(O/'review.json')),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
