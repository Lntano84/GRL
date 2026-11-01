"""One saved-only review after the original window expired. Never imports live models/envs."""
from pathlib import Path
import collections, csv, gzip, hashlib, json, math, struct, time
import numpy as np
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid291_control_preparation'
W=Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()
def save(p,v):
 assert not p.exists(),p
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def key(r): return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def main():
 started=time.perf_counter(); assert not (O/'saved_partial_review.json').exists()
 assert not (O/'finished.json').exists() and read(O/'timeout.json')['index']==186
 D=read(O/'design.json'); summaries=read(O/'manifest.json')['summaries']; assert len(summaries)==186
 freeze=read(O/'source_freeze.json')
 for p in W.glob('*.py'):
  rel=p.relative_to(R).as_posix()
  if rel in freeze: assert sha(p)==freeze[rel],rel
 counters=collections.Counter(); records=[]; signatures={}; decisions=[]; limits=[]; latencies={}
 full={k:read(R/v) for k,v in D['full_references'].items()}
 for j,s in enumerate(summaries):
  assert s['index']==j and all(s[k]==D['execution_order'][j][k] for k in D['execution_order'][j])
  folder=O/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
  assert s==read(folder/'summary.json')==read(O/f'worker_result_{j:02d}.json')
  rows=[json.loads(t) for t in gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8')]
  assert len(rows)==s['steps'] and [r['step'] for r in rows]==list(range(1,len(rows)+1))
  assert abs(math.fsum(r['raw_cost'] for r in rows)-s['cost'])<1e-6
  assert sum(r['simulations'] for r in rows)==s['public_forecasts']
  assert abs(math.fsum(r['decision_s'] for r in rows)-s['controller_s'])<1e-6
  with np.load(folder/'vectors.npz') as z:
   assert z['action'].shape[0]==z['observation'].shape[0]==len(rows)
   for k,r in enumerate(rows):
    assert digest(z['action'][k])==r['action_hash'] and digest(z['observation'][k])==r['after_hash']
    assert k==0 or r['before_hash']==rows[k-1]['after_hash']
    if r['ledger'] is not None: assert abs(r['ledger']['recomputed_raw_cost']-r['raw_cost'])<=r['ledger_tolerance']
    assert r['all_forecast_calls']==r['simulations']
    counters['illegal_steps']+=int(bool(r['illegal'])); counters['ambiguous_steps']+=int(bool(r['ambiguous'])); counters['exception_steps']+=int(bool(r['exceptions']))
   counters['saved_vector_rows']+=2*len(rows)
  signature=digest(np.array([[r['step']] for r in rows],dtype='int64'))+hashlib.sha256(json.dumps([[r[k] for k in ['before_hash','action_hash','after_hash','raw_cost','done','complete']] for r in rows],separators=(',',':')).encode()).hexdigest()
  signatures[s['repeat'],s['week'],s['method']]=signature
  calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]; assert len(calls)==s['n1_calls']
  fallbacks=collections.Counter(); nativestops=0; cancellations=0
  for c in calls:
   out=c['outcomes']; ids=[r['pool_id'] for r in out]; assert len(ids)==len(set(ids))==c['public_queries']
   assert set(ids)<=set(c['representatives']) and all(int(v) in c['representatives'] for v in c['aliases'].values())
   valid=[]
   for r in out:
    expected=bool(0<r['rho']<c['rho_limit'] and not r['done'] and not r['exceptions'] and not r['illegal'] and not r['ambiguous'])
    assert expected==r['strict_admissible']
    if expected:valid.append(r)
   best=max(valid,key=key,default=None); assert c['chosen_pool_id']==(None if best is None else best['pool_id'])
   if best is not None:
    assert c['returned_action_hash'] is not None
    # returned action excludes the base action; combined_hash includes it.
    counters['selected_return_equals_combined_hash']+=int(c['returned_action_hash']==best['combined_hash'])
   ds=c['learned_decisions']; ext=[r for r in ds if r['optional_extension']];assert len(ext)<=1
   if ext: assert ext[0] is ds[-1] and ext[0]['native_q'][1]<=ext[0]['native_q'][0]
   for d in ds:
    if 'native_q' in d:
     assert d['owner_pool']==s['pool'] and len(d['raw83'])==83 and len(d['raw_features'])==41
     counters['native_decisions']+=1
     if not d['optional_extension']:assert bool(d['action'])==(d['native_q'][1]>d['native_q'][0])
    if d['optional_extension']:
     nativestops+=1; gain=d['parent_p']*d['parent_mu']; b=(gain-d['known_charge'])/(gain+d['known_charge'])
     assert b==d['parent_b'] and d['parent_CONT']==(b>0) and len(d['parent_input124'])==124
     cont=d['parent_CONT']
     if s['arm']=='BOUNDED261':
      assert len(d['residual_input40'])==40 and d['residual_input40'][38]==b and d['residual_input40'][39]==1
      cont=cont and d['residual_logit']>=0
     assert bool(d['action'])==bool(cont)
     cancellations+=int(d['parent_CONT'] and not cont)
    decisions.append(dict(index=j,pool=s['pool'],week=s['week'],repeat=s['repeat'],arm=s['arm'],step=c['step'],batch=d['batch'],action=d['action'],optional_extension=d['optional_extension'],parent_b=d.get('parent_b'),parent_CONT=d.get('parent_CONT'),residual_logit=d.get('residual_logit'),engineering_fallback=d.get('engineering_fallback'),queries=len(d['next_ids'])))
   if c['engineering_fallback'] is not None:fallbacks[c['engineering_fallback']]+=1
   assert c['local_feature_s']+c['local_inference_s']+c['prepare_s']+c['forecast_s']<=c['source_s']+1e-5
   counters['candidate_feedback_rows']+=len(out)
  parts=read(folder/'components.json'); opt=math.fsum(r['wall_s'] for r in parts['calls'] if r['in_control'])
  feature=math.fsum(c['feature_s'] for c in calls); inference=math.fsum(c['local_inference_s'] for c in calls)
  searchother=s['n1_source_s']-s['n1_forecast_s']-feature-inference
  other=s['controller_s']-opt-s['all_forecast_s']-feature-inference-searchother
  assert min(searchother,other)>=-1e-5
  f=full[s['week']];quality='REFERENCE_LIMITED' if not f['complete'] else 'FAIL_COMPLETION' if not s['complete'] else 'PASS_COMPLETE' if s['cost']<=1.005*f['cost'] else 'FAIL_COST'
  active={c['step'] for c in calls}; all_l=[r['decision_s'] for r in rows]; search_l=[r['decision_s'] for r in rows if r['step'] in active]
  latencies[j]=(all_l,search_l)
  rec=dict(index=j,arm=s['arm'],pool=s['pool'],repeat=s['repeat'],week=s['week'],steps=s['steps'],complete=s['complete'],cost=s['cost'],quality=quality,full_cost_ratio=s['cost']/f['cost'] if s['complete'] and f['complete'] else None,controller_s=s['controller_s'],candidate_queries=s['n1_forecasts'],total_queries=s['public_forecasts'],native_stops=nativestops,residual_cancellations=cancellations,engineering_fallbacks=dict(fallbacks),time_parts=dict(optimizer=opt,all_forecast=s['all_forecast_s'],features=feature,inference=inference,searchother=searchother,other=other),all_p95=float(np.percentile(all_l,95)),all_p99=float(np.percentile(all_l,99)),search_p95=float(np.percentile(search_l,95)) if search_l else None,search_p99=float(np.percentile(search_l,99)) if search_l else None,trajectory_signature=signature)
  records.append(rec)
  if j%25==0: print('SAVED_RUN',j+1,186,flush=True)
 lookup={(r['pool'],r['repeat'],r['week'],r['arm']):r for r in records}; pairs=[]
 for a in records:
  if a['arm']!='BOUNDED261':continue
  b=lookup.get((a['pool'],a['repeat'],a['week'],'PARENT208'))
  if b is None:continue
  same=a['steps']==b['steps'] and a['complete']==b['complete']
  def ratio(k):return a[k]/b[k]-1 if same and a[k] is not None and b[k] else None
  pairs.append(dict(pool=a['pool'],repeat=a['repeat'],week=a['week'],same_horizon=same,complete_pair=a['complete'] and b['complete'],same_trajectory=a['trajectory_signature']==b['trajectory_signature'],controller_change=ratio('controller_s'),candidate_query_change=ratio('candidate_queries'),total_query_change=ratio('total_queries'),all_p95_change=ratio('all_p95'),all_p99_change=ratio('all_p99'),search_p95_change=ratio('search_p95'),search_p99_change=ratio('search_p99'),cost_change=a['cost']/b['cost']-1 if a['complete'] and b['complete'] else None,cost_delta=a['cost']-b['cost'] if a['complete'] and b['complete'] else None,candidate_query_delta=a['candidate_queries']-b['candidate_queries'],total_query_delta=a['total_queries']-b['total_queries'],parent_quality=b['quality'],candidate_quality=a['quality']))
 repetitions=[]
 for r in records:
  if r['repeat']!=1:continue
  old=lookup[r['pool'],0,r['week'],r['arm']]
  repetitions.append(dict(pool=r['pool'],week=r['week'],arm=r['arm'],same_trajectory=r['trajectory_signature']==old['trajectory_signature']))
 partial=O/'runs/r1__2037-03-02_100__BOUNDED261__f0_s2';pr=[json.loads(t) for t in (partial/'partial_steps.jsonl').read_text().splitlines()]
 failure=read(partial/'failure.json');assert len(pr)==failure['rows']==192 and sum(r['simulations'] for r in pr)==failure['public_forecasts']
 with np.load(partial/'partial_vectors.npz') as z:
  assert len(z['action'])==len(z['observation'])==len(pr)
  for k,r in enumerate(pr):assert digest(z['action'][k])==r['action_hash'] and digest(z['observation'][k])==r['after_hash']
 assert math.fsum(r['raw_cost'] for r in pr)>0
 partial_receipt=dict(index=186,steps=len(pr),recorded_cost=math.fsum(r['raw_cost'] for r in pr),controller_s=failure['controller_s'],total_queries=failure['public_forecasts'],reason='ORIGINAL_TOTAL_WINDOW_EXHAUSTED',not_compared_to_full_horizon=True)
 macros={}
 for label,selected in [('all_available_pairs',pairs),('repeat0_complete_design',[p for p in pairs if p['repeat']==0]),('complete_week_pairs',[p for p in pairs if p['complete_pair']])]:
  macros[label]=dict(pairs=len(selected),**{k:float(np.mean([p[k] for p in selected if p[k] is not None])) for k in ['controller_change','candidate_query_change','total_query_change','all_p95_change','all_p99_change','search_p95_change','search_p99_change','cost_change']})
 perseed={str(s):{k:float(np.mean([p[k] for p in pairs if p['pool'].endswith('_s'+str(s)) and p[k] is not None])) for k in ['controller_change','candidate_query_change','total_query_change','cost_change']} for s in [0,1,2]}
 save(O/'saved_partial_review.json',dict(passed_saved_data=True,benchmark_scope_complete=False,planned_runs=192,accepted_runs=len(records),partial_run=partial_receipt,unstarted_indices=list(range(187,192)),unaccepted_items=D['execution_order'][186:],records=records,matched_pairs=pairs,repeat_checks=repetitions,macros=macros,perseed=perseed,checks=dict(counters),full_reference_status={w:dict(complete=f['complete'],cost=f['cost']) for w,f in full.items()},new_predictions=0,new_physics=0,new_fits=0,scope='Budget-limited saved-data review; no original full-192 audit or replay executed.',wall_s=time.perf_counter()-started))
 save(O/'saved_decisions.json',decisions)
 with (O/'run_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
  fields=[k for k in records[0] if k not in ['time_parts','engineering_fallbacks']];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
 print('PARTIAL_REVIEW_PASSED',json.dumps(dict(macros=macros,perseed=perseed,checks=dict(counters),repeated_trajectories=sum(x['same_trajectory'] for x in repetitions),repeat_checks=len(repetitions)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
