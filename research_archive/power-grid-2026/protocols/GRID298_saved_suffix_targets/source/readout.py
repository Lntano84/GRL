"""One600s saved JSON/ledger readout. Stdlib only. No live model or grid calls."""
from pathlib import Path
import collections,gzip,hashlib,json,math,subprocess,sys,time,traceback
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';W=Path(__file__).parent
O=R/'outputs/grid298_saved_suffix_targets';A=I/'research_archive/power-grid-2026/protocols/GRID298_saved_suffix_targets'
START=None;REFS=None
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):return json.loads(p.read_text(encoding='utf8'))
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def checktime():
 if START is not None and time.perf_counter()-START>600:raise TimeoutError('Original600s window; retain receipts, no retry/extension')
def frozen(rel):
 checktime();p=R/rel;b=p.read_bytes();ref=REFS[rel];assert len(b)==ref['bytes'] and sha(b)==ref['sha256'],rel;return b
def js(rel):return json.loads(frozen(rel))
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
KEYS=['step','before_hash','action_hash','after_hash','raw_cost','done','complete']
TERMS=['losses_mwh','redispatch_mwh','curtailment_delta_mwh','storage_throughput_mwh']
def tolerance(rows):return 64*sys.float_info.epsilon*max(1.,math.fsum(abs(r['raw_cost']) for r in rows))*max(1,len(rows))
def audit_episode(item):
 folder=item['folder'];s=js(folder+'/summary.json');m=js(folder+'/metadata.json');e=js(folder+'/intervention.json')
 for k,v in item.items():
  if k!='folder':assert s[k]==v and m[k]==v,(item['index'],k)
 data=gzip.decompress(frozen(folder+'/steps.jsonl.gz'));rows=[json.loads(line) for line in data.splitlines() if line];del data
 assert len(rows)==s['steps']==e['steps'] and rows
 total=0.;missing=0;maxerr=0.;parts={k:0. for k in TERMS}
 for n,r in enumerate(rows,1):
  if n%128==0:checktime()
  assert r['step']==n and math.isfinite(r['raw_cost'])
  if n>1:assert rows[n-2]['after_hash']==r['before_hash'] and not rows[n-2]['done']
  total+=r['raw_cost'];ledger=r['ledger']
  if ledger is None:missing+=1
  else:
   assert all(math.isfinite(v) for v in ledger.values());err=abs(ledger['recomputed_raw_cost']-r['raw_cost']);maxerr=max(maxerr,err)
   assert err<=r['ledger_tolerance']
   computed=ledger['marginal_cost']*sum(ledger[k] for k in TERMS)
   assert abs(computed-ledger['recomputed_raw_cost'])<=max(1e-9,abs(computed)*1e-12)
   for k in TERMS:parts[k]+=ledger['marginal_cost']*ledger[k]
 assert total==s['cost'] and rows[-1]['complete']==s['complete']
 assert sum(r['simulations'] for r in rows)==s['public_forecasts']
 assert sum(r['illegal'] for r in rows)==s['illegal'] and sum(r['ambiguous'] for r in rows)==s['ambiguous'] and sum(bool(r['exceptions']) for r in rows)==s['exception_steps']
 assert s['new_model_fits']==0
 if s['complete']:assert len(rows)==2017
 result={'index':item['index'],'pool':item['pool'],'date':item['date'],'role':item['partition'],'KEEP':item['KEEP'],'arm':item['arm'],'steps':s['steps'],'complete':s['complete'],'cost':total,'public_forecasts':s['public_forecasts'],'forecasts_including_setup':e['forecasts'],'illegal':s['illegal'],'ambiguous':s['ambiguous'],'exception_steps':s['exception_steps'],'terminal_done':rows[-1]['done'],'terminal_exceptions':rows[-1]['exceptions'],'missing_ledger_rows':missing,'max_ledger_error':maxerr,'component_costs':parts,'component_raw_residual':total-sum(parts.values()),'event_present':bool(e['event']),'source_bytes_verified':True}
 save(O/'episode_receipts'/f"{item['index']:03d}.json",result)
 return s,m,e,rows,result
def main():
 global START,REFS
 assert not (O/'started.json').exists(),'No duplicate/partial retry'
 assert not git('status','--porcelain').strip(),'Source publication must finish first'
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 source_blobs={}
 for p in (A/'source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();source_blobs[rel]=sha(p.read_bytes())
 for rel,h in load(O/'source_freeze.json').items():assert sha((R/rel).read_bytes())==h
 save(O/'launch_verification.json',{'passed':True,'source_sha':head,'remote_verified':True,'exact_source_blobs':source_blobs,'new_predictions':0,'new_physics':0})
 START=time.perf_counter();save(O/'started.json',{'start_monotonic':START,'wall_cap_s':600,'source_sha':head,'no_retry':True})
 REFS=load(O/'input_references.json');members=load(O/'members.json');roles=js('outputs/grid297_paired_collection_preparation/pool_roles.json');byrole={r['pool']:r for r in roles}
 pairs=[];eps=[]
 for j in range(144):
  items=members[2*j:2*j+2];assert len(items)==2 and {i['arm'] for i in items}=={'STOP_NEXT','CONT_NEXT'} and all(i['pair_index']==j for i in items)
  receipt=js(f'outputs/grid297_paired_collection_preparation/pair_{j:03d}.json');assert receipt['prefix_matched'] and receipt['no_labels_derived']
  arm={i['arm']:audit_episode(i) for i in items};s,sm,se,sr,er=arm['STOP_NEXT'];c,cm,ce,cr,ec=arm['CONT_NEXT'];eps.extend([er,ec])
  assert s['pool']==c['pool'] and s['date']==c['date'] and s['partition']==c['partition']
  role=byrole[s['pool']];rname='fit_dates' if s['partition']=='own_training_fit8' else 'cal_dates';assert s['date'] in role[rname] and s['date'] not in role['outer_dates_excluded'] and s['KEEP']==role['KEEP']
  for key in ['library_hashes','checkpoint_sha256','cost_per_MW','dt_hours','cadence']:assert sm[key]==cm[key],(j,key)
  assert bool(se['event'])==bool(ce['event'])==receipt['event_present']
  out={'pair_index':j,'pool':s['pool'],'date':s['date'],'role':s['partition'],'KEEP':s['KEEP'],'STOP':er,'CONT':ec,'target_cost_CONT_minus_STOP':None,'target_valid':False,'new_training_qualified':False}
  if not se['event']:
   assert se['complete_no_event_signature']==ce['complete_no_event_signature'] and len(sr)==len(cr)
   assert all(all(x[k]==y[k] for k in KEYS) for x,y in zip(sr,cr));out['status']='NO_ELIGIBLE_EVENT'
  else:
   a=se['event'];b=ce['event'];assert a['common']==b['common'];common=a['common'];t=common['step'];assert 1<=t<=min(len(sr),len(cr))
   assert common['next_ids'] and len(common['next_ids'])<=64 and a['arm']=='STOP_NEXT' and b['arm']=='CONT_NEXT' and not a['forced_continue'] and b['forced_continue']
   assert all(all(x[k]==y[k] for k in KEYS) for x,y in zip(sr[:t-1],cr[:t-1]))
   prefix=[{k:r[k] for k in KEYS} for r in sr[:t-1]];assert digest(prefix)==common['prefix_signature'];assert sum(r['raw_cost'] for r in sr[:t-1])==common['prefix_cost']
   assert sr[t-1]['before_hash']==cr[t-1]['before_hash']==common['state_hash']
   for event,rows in [(a,sr),(b,cr)]:assert event['delivered']=={k:rows[t-1][k] for k in ['step','before_hash','action_hash','after_hash','restore','raw_cost','done','complete']}
   suffix_s=math.fsum(r['raw_cost'] for r in sr[t-1:]);suffix_c=math.fsum(r['raw_cost'] for r in cr[t-1:]);delta=suffix_c-suffix_s;whole_delta=c['cost']-s['cost'];tol=tolerance(sr)+tolerance(cr)
   assert abs(delta-whole_delta)<=tol
   clean=all(not r['illegal'] and not r['ambiguous'] and not r['exceptions'] for r in sr+cr)
   same=all(all(x[k]==y[k] for k in KEYS) for x,y in zip(sr,cr)) and len(sr)==len(cr)
   out.update(trigger_step=t,known_batch_queries=len(common['next_ids']),common_before_hash=common['state_hash'],common_public_input_hash=common['public_NN_input_hash'],common_prefix_signature=common['prefix_signature'],prefix_cost=common['prefix_cost'],STOP_suffix_cost=suffix_s,CONT_suffix_cost=suffix_c,whole_total_delta=whole_delta,arithmetic_residual=delta-whole_delta,arithmetic_tolerance=tol,delivered_action_changed=a['delivered']['action_hash']!=b['delivered']['action_hash'],trigger_after_state_changed=a['delivered']['after_hash']!=b['delivered']['after_hash'],same_saved_trajectory=same,clean_both=clean,ordinary261_continue=a['ordinary261']['continue_'],suffix_forecast_delta=sum(r['simulations'] for r in cr[t-1:])-sum(r['simulations'] for r in sr[t-1:]))
   if s['complete'] and c['complete'] and s['steps']==c['steps']==2017:
    out.update(target_valid=True,target_cost_CONT_minus_STOP=delta,status='COMPLETE_MATCHED_COST_TARGET')
   else:out['status']='INCOMPLETE_OR_UNEQUAL_HORIZON_NO_TARGET'
  pairs.append(out);save(O/'pair_receipts'/f'{j:03d}.json',out)
  if j%12==11:print('SAVED_PAIRS',j+1,144,round(time.perf_counter()-START,3),flush=True)
 checktime();save(O/'pairs.json',pairs);save(O/'episodes.json',eps)
 def aggregate(pp):
  valid=[x for x in pp if x['target_valid']];nz=[x for x in valid if x['target_cost_CONT_minus_STOP']!=0];benefit=[x for x in valid if x['target_cost_CONT_minus_STOP']<0];harm=[x for x in valid if x['target_cost_CONT_minus_STOP']>0]
  return {'pairs':len(pp),'events':sum(x['status']!='NO_ELIGIBLE_EVENT' for x in pp),'valid_complete_targets':len(valid),'clean_valid_targets':sum(x.get('clean_both',False) for x in valid),'CONT_lower_cost':len(benefit),'CONT_higher_cost':len(harm),'zero_cost_delta':len(valid)-len(nz),'delivered_changed':sum(x.get('delivered_action_changed',False) for x in pp),'all_actions_states_cost_same':sum(x.get('same_saved_trajectory',False) for x in pp),'net_cost_delta_descriptive':math.fsum(x['target_cost_CONT_minus_STOP'] for x in valid),'benefit_abs_cost':-math.fsum(x['target_cost_CONT_minus_STOP'] for x in benefit),'harm_abs_cost':math.fsum(x['target_cost_CONT_minus_STOP'] for x in harm),'unique_dates':len({x['date'] for x in pp}),'unique_event_state_hashes':len({x['common_before_hash'] for x in pp if 'common_before_hash' in x})}
 allstats=aggregate(pairs);poolstats={pool:{role:aggregate([p for p in pairs if p['pool']==pool and p['role']==role]) for role in ['own_training_fit8','own_training_cal4']} for pool in byrole}
 summary={'all':allstats,'by_pool_role':poolstats,'KEEP':aggregate([p for p in pairs if p['KEEP']]),'active_fit8':aggregate([p for p in pairs if not p['KEEP'] and p['role']=='own_training_fit8']),'active_cal4':aggregate([p for p in pairs if not p['KEEP'] and p['role']=='own_training_cal4']),'status_counts':dict(collections.Counter(p['status'] for p in pairs)),'completed_episodes':sum(e['complete'] for e in eps),'physical_rows_checked':sum(e['steps'] for e in eps),'wall_s':time.perf_counter()-START,'new_fits':0,'new_predictions':0,'new_physics':0,'cost_targets_derived':allstats['valid_complete_targets'],'training_qualified':False}
 save(O/'summary.json',summary);save(O/'complete.json',{'passed':True,'episodes':288,'pairs':144,'wall_s':time.perf_counter()-START,'new_fits':0,'new_predictions':0,'new_physics':0,'target_count':allstats['valid_complete_targets'],'training_qualified':False})
 print('GRID298_SAVED_REVIEW_COMPLETE',json.dumps(summary['all']),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save(O/'readout_error.json',{'traceback':traceback.format_exc(),'wall_s':None if START is None else time.perf_counter()-START,'do_not_retry_completed_or_partial':True});raise
