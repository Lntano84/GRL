"""Prespecified descriptive neighbor comparison; no learned classifier or policies."""
import collections,gzip,json,math
import run as D
np=D.np;O=D.O;R=D.R

def normalize(raw,nfit):
 raw=np.asarray(raw,float);fit=raw[:nfit];med=[]
 for j in range(raw.shape[1]):
  finite=fit[np.isfinite(fit[:,j]),j];med.append(float(np.median(finite)) if len(finite) else 0.)
 missing=~np.isfinite(raw);imputed=np.where(missing,np.asarray(med),raw)
 scale=np.quantile(imputed[:nfit],.75,axis=0)-np.quantile(imputed[:nfit],.25,axis=0)
 scale=np.where(scale>0,scale,np.ptp(imputed[:nfit],axis=0));scale=np.where(scale>0,scale,1.)
 return np.concatenate([(imputed-np.asarray(med))/scale,missing.astype(float)],axis=1),dict(medians=med,scales=scale.tolist(),append_missing=True,fit_rows=nfit)

def run():
 D.verify();assert D.read(O/'state.json')['stage']=='extracted_pending_comparison'
 rows=D.G.rowfile(D.B.O/'fit_rows.json.gz')+D.G.rowfile(D.B.O/'calibration_rows.json.gz')+D.B.eligible_data('validation')
 parts=['fit']*386+['calibration']*198+['validation']*142
 with gzip.open(O/'public_state_descriptors.jsonl.gz','rt',encoding='utf8') as f:physical=list(map(json.loads,f))
 assert [r['meta']['sample_id'] for r in rows]==[r['sample_id'] for r in physical]
 oldnames=D.read(D.B.O/'preprocessing.json')['features']
 def raw(values,names):return [[np.nan if v[k] is None else float(v[k]) for k in names] for v in values]
 original,on=normalize(raw([r['features'] for r in rows],oldnames),386)
 state,sn=normalize(raw([r['features'] for r in physical],D.FIELDS),386)
 assert on['medians']==D.read(D.B.O/'preprocessing.json')['medians']
 views=dict(ORIGINAL18=original,STATE12=state,COMBINED30=np.concatenate([original,state],axis=1))
 D.save('normalization.json',dict(original=on,state=sn,views={k:v.shape[1] for k,v in views.items()},distance='RMS normalized channels including missing flags. Same original depth/entry-safe branch;exclude calendar date. Not learned representation.'))
 profile=[]
 for k in D.FIELDS:
  rr=[r['features'][k] for r in physical]
  profile.append(dict(feature=k,fit_distinct_values=len({v for v in rr[:386]}),fit_missing=sum(v is None for v in rr[:386]),
   validation_distinct_values=len({v for v in rr[584:]}),validation_missing=sum(v is None for v in rr[584:])))
 D.save('descriptor_profile.json',profile)
 records=[];audit=collections.Counter()
 for name,v in views.items():
  for i,r in enumerate(rows):
   eligible=[j for j,s in enumerate(rows[:386]) if s['meta']['date']!=r['meta']['date'] and s['meta']['visible_count']==r['meta']['visible_count'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
   distance={j:float(np.sqrt(np.mean((v[i]-v[j])**2))) for j in eligible}
   ordered=sorted(eligible,key=lambda j:(distance[j],rows[j]['meta']['sample_id']))
   selected=[];calls=set()
   for j in ordered:
    if rows[j]['meta']['call_id'] in calls:continue
    selected.append(j);calls.add(rows[j]['meta']['call_id'])
    if len(selected)==5:break
   assert len(calls)==len(selected)
   for j in selected:
    scalar=(math.fsum(float(v[i,k]-v[j,k])**2 for k in range(v.shape[1]))/v.shape[1])**.5
    assert abs(scalar-distance[j])<1e-12;audit['independent_scalar_distances']+=1
   if len(selected)==5:
    worst=distance[selected[-1]]
    for j in eligible:
     if j not in selected:assert distance[j]>=worst-1e-12
   records.append(dict(view=name,sample_id=r['meta']['sample_id'],call_id=r['meta']['call_id'],partition=parts[i],date=r['meta']['date'],
    depth=r['meta']['visible_count'],stop_label=r['target']['stop_label'],available_distinct_calls=len({rows[j]['meta']['call_id'] for j in eligible}),
    k=len(selected),continue_neighbors=sum(not rows[j]['target']['stop_label'] for j in selected),
    neighbors=[dict(sample_id=rows[j]['meta']['sample_id'],call_id=rows[j]['meta']['call_id'],date=rows[j]['meta']['date'],stop_label=rows[j]['target']['stop_label'],distance=distance[j]) for j in selected]))
 D.save('all_neighbor_records.json',records)
 summary=[];dates=[]
 for name in views:
  for part in ['fit','calibration','validation']:
   rr=[r for r in records if r['view']==name and r['partition']==part];cont=[r for r in rr if not r['stop_label']];stop=[r for r in rr if r['stop_label']]
   summary.append(dict(view=name,partition=part,rows=len(rr),continue_rows=len(cont),continue_full5=sum(r['k']==5 for r in cont),
    continue_full5_zero_continue=sum(r['k']==5 and r['continue_neighbors']==0 for r in cont),
    stop_rows=len(stop),stop_full5=sum(r['k']==5 for r in stop),stop_full5_has_continue=sum(r['k']==5 and r['continue_neighbors']>0 for r in stop),
    insufficient5=sum(r['k']<5 for r in rr)))
   for date in sorted({r['date'] for r in rr}):
    dr=[r for r in cont if r['date']==date];sr=[r for r in stop if r['date']==date]
    dates.append(dict(view=name,partition=part,date=date,continue_rows=len(dr),continue_full5=sum(r['k']==5 for r in dr),
     continue_full5_zero_continue=sum(r['k']==5 and r['continue_neighbors']==0 for r in dr),
     stop_full5=sum(r['k']==5 for r in sr),stop_full5_has_continue=sum(r['k']==5 and r['continue_neighbors']>0 for r in sr)))
 D.save('neighbor_summary.json',summary);D.save('neighbor_per_date.json',dates)
 critical=D.read(D.P.O/'case_comparison_summary.json');byid={p['sample_id']:p['features'] for p in physical};index={r['meta']['sample_id']:r for r in rows}
 cases=[]
 for c in critical:
  identity=c['case']['sample_id'];nn=c['neighbors']['neighbors']
  ids=[identity]+[r['sample_id'] for r in nn.values() if r]
  cases.append(dict(call_id=c['case']['call_id'],sample_id=identity,stop_label=index[identity]['target']['stop_label'],
   contexts=[dict(sample_id=k,date=index[k]['meta']['date'],stop_label=index[k]['target']['stop_label'],original_features=index[k]['features'],state_features=byid[k]) for k in ids],
   nearest5=[r for r in records if r['sample_id']==identity]))
 D.save('case_context_comparisons.json',cases)
 bycall={s['call_id']:s for s in D.read(O/'state_context.json')}
 pressure=[]
 for part in ['fit','calibration','validation']:
  rr=[(r,p) for r,p,q in zip(rows,physical,parts) if q==part and not r['target']['stop_label']]
  local=[(r,p) for r,p in rr if p['features']['incumbent_overload_endpoint_share']==0]
  pressure.append(dict(partition=part,continue_prefixes=len(rr),continue_no_current_incident_overload=len(local),
   affected_calls=len({r['meta']['call_id'] for r,p in local}),affected_dates=len({r['meta']['date'] for r,p in local})))
 D.save('local_pressure_counterexamples.json',pressure)
 assert not D.read(O/'extraction_review.json')['environment_or_physical_calls']
 review=dict(passed=True,rows=726,views=3,all_neighbor_records=len(records),checks=dict(audit),fit_only_normalization=True,
  original_targets_unchanged=True,all_adverse_cases_and_low_support_rows_retained=True,
  new_fits=0,new_predicts=0,new_policy_replays=0,new_physical=0,final8_absent=True)
 D.save('comparison_review.json',review);D.verify();D.save('state.json',dict(stage='compared_pending_report'))
 print(json.dumps(dict(summary=summary,pressure=pressure),ensure_ascii=False))

if __name__=='__main__':run()
