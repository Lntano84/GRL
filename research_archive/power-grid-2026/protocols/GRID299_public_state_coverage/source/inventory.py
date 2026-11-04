"""One public-only saved-call inventory. Does not derive outcomes or run a model."""
from pathlib import Path
import collections,hashlib,json,math,subprocess,time,traceback
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid299_public_state_coverage';P=R/'outputs/grid297_paired_collection_preparation';A=I/'research_archive/power-grid-2026/protocols/GRID299_public_state_coverage'
START=None;REF={}
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def fixed(rel):
 if START is not None and time.perf_counter()-START>600:raise TimeoutError('single600s, no automatic retry')
 b=(R/rel).read_bytes();assert sha(b)==REF[rel]['sha256'] and len(b)==REF[rel]['bytes'],rel;return json.loads(b)
def folder(item):return (P/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}").relative_to(R).as_posix()
def main():
 global START,REF
 assert not (O/'started.json').exists();assert not git('status','--porcelain').strip()
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p.read_bytes())
 for rel,h in read(O/'source_freeze.json').items():assert sha((R/rel).read_bytes())==h
 save(O/'launch_verification.json',{'passed':True,'source_sha':head,'exact_source_blobs':blobs,'remote_verified':True})
 START=time.perf_counter();save(O/'started.json',{'start_monotonic':START,'wall_cap_s':600,'no_retry':True});REF=read(O/'input_references.json');d=fixed('outputs/grid297_paired_collection_preparation/design.json');roles={r['pool']:r for r in fixed('outputs/grid297_paired_collection_preparation/pool_roles.json')}
 candidates=[];selected=[];baseline_members=[];calls_read=0;empty_bins=[];reuse=[]
 for pair in d['pairs']:
  j=pair['pair_index'];items=[i for i in d['execution_order'] if i['pair_index']==j];assert len(items)==2
  ee={i['arm']:fixed(folder(i)+'/intervention.json') for i in items};event=ee['STOP_NEXT']['event'];assert bool(event)==bool(ee['CONT_NEXT']['event'])
  if event:
   assert event['common']==ee['CONT_NEXT']['event']['common'];ordinary=bool(event['ordinary261']['continue_']);assert ordinary==bool(ee['CONT_NEXT']['event']['ordinary261']['continue_']);basearm='CONT_NEXT' if ordinary else 'STOP_NEXT'
  else:basearm='STOP_NEXT'
  base=next(i for i in items if i['arm']==basearm);f=folder(base);role=roles[pair['pool']];assert pair['date'] not in role['outer_dates_excluded'] and pair['date'] in role['fit_dates' if pair['role']=='fit8' else 'cal_dates']
  baseline_members.append({'pair_index':j,'pool':pair['pool'],'date':pair['date'],'role':pair['role'],'KEEP':pair['KEEP'],'baseline_arm':basearm,'folder':f,'selection_reason':'matches pre-intervention original261 action; no future outcome selection'})
  files=sorted(rel for rel in REF if rel.startswith(f+'/call') and rel.endswith('.json'));own=[]
  for rel in files:
   call=fixed(rel);calls_read+=1;assert call['model_pool']==pair['pool'];ordinal=int(Path(rel).stem[4:]);step=call['step'];assert 1<=step<=2017
   for di,dec in enumerate(call['learned_decisions']):
    if not dec.get('optional_extension',False):continue
    q=dec['native_q'];ids=dec['next_ids'];assert len(q)==2 and all(math.isfinite(v) for v in q) and q[1]<=q[0] and 1<=len(ids)<=64 and len(ids)==len(set(ids)) and dec['owner_pool']==pair['pool']
    tensorhash=digest(dec['NN_input']);old=False
    if event:
     e=event['common'];old=(step==e['step'] and call['state_hash']==e['state_hash'] and dec['batch']==e['batch'] and ids==e['next_ids'] and tensorhash==e['public_NN_input_hash'])
    marker={'pool':pair['pool'],'date':pair['date'],'week':pair['week'],'role':pair['role'],'KEEP':pair['KEEP'],'source_pair_index':j,'baseline_arm':basearm,'baseline_folder':f,'call_ordinal':ordinal,'decision_index':di,'step':step,'quarter':min(3,(step-1)//504),'batch':dec['batch'],'next_ids':ids,'known_batch_queries':len(ids),'state_hash':call['state_hash'],'public_NN_input_hash':tensorhash,'raw41_hash':digest(dec['raw_features']),'ordinary261_continue':bool(dec['action']),'existing_first_event_pair':j if old else None,'source_call':rel}
    own.append(marker)
  own.sort(key=lambda x:(x['step'],x['call_ordinal'],x['decision_index']));candidates.extend(own)
  for band in range(4):
   group=[x for x in own if x['quarter']==band]
   if group:
    m=dict(group[0]);m['selection_id']=len(selected);m['new_opposite_arm']=None if m['existing_first_event_pair'] is not None else ('STOP_NEXT' if m['ordinary261_continue'] else 'CONT_NEXT');selected.append(m)
    if m['existing_first_event_pair'] is not None:reuse.append(m)
   else:empty_bins.append({'source_pair_index':j,'pool':pair['pool'],'date':pair['date'],'role':pair['role'],'quarter':band,'status':'NO_ELIGIBLE_STATE_NO_REPLACEMENT'})
  save(O/'member_receipts'/f'{j:03d}.json',{'baseline':baseline_members[-1],'calls_read':len(files),'public_nativeSTOP_candidates':len(own),'selected':sum(x['source_pair_index']==j for x in selected),'no_new_labels':True})
 new=[s for s in selected if s['existing_first_event_pair'] is None]
 for n,m in enumerate(new):m['proposed_run_index']=n
 def stats(ss):return {'records':len(ss),'unique_date_state_keys':len({(s['date'],s['state_hash']) for s in ss}),'unique_raw41_hashes':len({s['raw41_hash'] for s in ss}),'dates':len({s['date'] for s in ss}),'pools':len({s['pool'] for s in ss})}
 resources={'new_episodes':len(new),'reused_pairs':len(reuse),'reused_original261_baseline_episodes':len({s['baseline_folder'] for s in new}),'maximum_physical_steps':2017*len(new),'simple_wall_s':25309.4708868/288*len(new),'simple_hours':25309.4708868/288*len(new)/3600,'simple_forecasts':1717785/288*len(new),'proposed_total_cap_s':28800,'proposed_episode_cap_s':1200,'proposed_forecast_cap':9000000,'resource_passed':False,'qualification':'Historical original297 averages only; all paths, later interventions, shared reference and prefix guards may differ. No new timing experiment; not authorization or guarantee.'}
 summary={'pairs_baselines':144,'calls_read':calls_read,'all_candidates':stats(candidates),'selected4quarters':stats(selected),'reused_first_pairs':stats(reuse),'proposed_new_states':stats(new),'missing_bins':len(empty_bins),'by_pool_role':{pool:{role:stats([s for s in selected if s['pool']==pool and s['role']==role]) for role in ['fit8','cal4']} for pool in roles},'new_fits':0,'new_predictions':0,'new_physics':0,'new_cost_labels':0,'wall_s':time.perf_counter()-START}
 save(O/'baseline_members.json',baseline_members);save(O/'public_candidates.json',candidates);save(O/'selected_states.json',selected);save(O/'proposed_new_runs.json',new);save(O/'empty_bins.json',empty_bins);save(O/'resource_estimates.json',resources);save(O/'summary.json',summary)
 save(O/'complete.json',{'passed':True,'wall_s':time.perf_counter()-START,'saved_calls_read':calls_read,'new_fits':0,'new_predictions':0,'new_physics':0,'new_cost_labels':0,'new_physical_authorization':False})
 print('GRID299_PUBLIC_COVERAGE_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True);print('RESOURCE_PROPOSAL',json.dumps(resources),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save(O/'inventory_error.json',{'traceback':traceback.format_exc(),'wall_s':None if START is None else time.perf_counter()-START,'do_not_retry_complete_or_partial':True});raise
