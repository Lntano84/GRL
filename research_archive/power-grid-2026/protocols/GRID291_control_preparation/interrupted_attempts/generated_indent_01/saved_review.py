"""Saved-only readout; never invokes model forecasts or a physical solver."""
from pathlib import Path
import json,gzip,math
import numpy as np
from guard import R,O,read,sha
def main():
 D=read(O/'design.json');done=read(O/'finished.json');assert done['runs']==192
 records=[]
 for s in done['summaries']:
  folder=O/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}";rows=[json.loads(t) for t in gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8')];assert len(rows)==s['steps']
  assert abs(math.fsum(r['raw_cost'] for r in rows)-s['cost'])<1e-6
  calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))];assert len(calls)==s['n1_calls']
  for call in calls:
   ds=call['learned_decisions'];extensions=[r for r in ds if r['optional_extension']];assert len(extensions)<=1
   if extensions:
    ext=extensions[0];assert ext is ds[-1] and ext['native_q'][1]<=ext['native_q'][0]
   assert call['local_feature_s']+call['local_inference_s']+call['prepare_s']+call['forecast_s']<=call['source_s']+1e-5
  parts=read(folder/'components.json');optimizer=math.fsum(r['wall_s'] for r in parts['calls'] if r['in_control']);features=math.fsum(c['feature_s'] for c in calls);inference=math.fsum(c['local_inference_s'] for c in calls);searchother=s['n1_source_s']-s['n1_forecast_s']-features-inference;other=s['controller_s']-optimizer-s['all_forecast_s']-features-inference-searchother;assert min(searchother,other)>=-1e-5
  f=read(R/D['full_references'][s['week']]);quality='PASS_COMPLETE' if f['complete'] and s['complete'] and s['cost']<=1.005*f['cost'] else 'FAIL_COST' if f['complete'] and s['complete'] else 'FAIL_COMPLETION' if f['complete'] else 'REFERENCE_LIMITED'
  active={c['step'] for c in calls};alllat=[r['decision_s'] for r in rows];slat=[r['decision_s'] for r in rows if r['step'] in active]
  records.append(dict(arm=s['arm'],pool=s['pool'],repeat=s['repeat'],week=s['week'],steps=s['steps'],complete=s['complete'],cost=s['cost'],quality=quality,controller_s=s['controller_s'],candidate_queries=s['n1_forecasts'],total_queries=s['public_forecasts'],fallback_calls=sum(c['engineering_fallback'] is not None for c in calls),time_parts=dict(optimizer=optimizer,all_forecast=s['all_forecast_s'],features=features,inference=inference,searchother=searchother,other=other),all_p95=float(np.percentile(alllat,95)),all_p99=float(np.percentile(alllat,99)),search_p95=float(np.percentile(slat,95)) if slat else None,search_p99=float(np.percentile(slat,99)) if slat else None))
 pairs=[]
 for r in records:
  if r['arm']!='BOUNDED261':continue
  p=next(x for x in records if x['arm']=='PARENT208' and (x['pool'],x['repeat'],x['week'])==(r['pool'],r['repeat'],r['week']));same=r['steps']==p['steps'] and r['complete']==p['complete']
  pairs.append(dict(pool=r['pool'],repeat=r['repeat'],week=r['week'],same_horizon=same,controller_delta_s=r['controller_s']-p['controller_s'] if same else None,query_delta=r['candidate_queries']-p['candidate_queries'],full_week_cost_delta=r['cost']-p['cost'] if r['complete'] and p['complete'] else None,candidate_quality=r['quality'],parent_quality=p['quality']))
 (O/'saved_review.json').write_text(json.dumps(dict(passed=True,runs=records,matched_pairs=pairs,new_predictions=0,not_independent_pool_repeats=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
