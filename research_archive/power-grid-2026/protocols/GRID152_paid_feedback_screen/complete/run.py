"""Fixed-tail dominance diagnosis and fit/cal-only screen of eight legacy paid-feedback fields."""
from pathlib import Path
import collections,gzip,hashlib,importlib.util,json,math,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid152_paid_feedback_screen'
sp=importlib.util.spec_from_file_location('grid152_prior',R/'work/grid151_fixed_score_mean/run.py');Q=importlib.util.module_from_spec(sp);sp.loader.exec_module(Q)
G=Q.G;B=Q.B;REPO=Q.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID152_paid_feedback_screen'
read=Q.read;sha=Q.sha
EXTRA=['terminal_fraction','illegal_fraction','ambiguous_fraction','exception_fraction','admissible_reward_mean','admissible_reward_std','station_admissible_fraction','incumbent_relative_position']
fs=importlib.util.spec_from_file_location('grid152_legacy_projection',R/'work/grid101_gain_learning/features.py');F=importlib.util.module_from_spec(fs);fs.loader.exec_module(F)
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 Q.verify();assert read(Q.O/'complete.json')['passed_audit']
 for n in ['source_hashes.json','code_freeze.json']:
  for p,h in read(O/n).items():assert sha(R/p)==h
def archive():
 d=A/'complete';d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir();Q.verify();assert read(Q.O/'review.json')['passed']
 save('design.json',dict(id='GRID152_PAID_FEEDBACK_INFORMATION_SCREEN',
  scope='Read-only. No new fit,predict,policy/threshold replay or new validation-feature extraction.',
  tail='On original386fit/198cal only,count STOP rows componentwise dominated by CONT rows in bothRF140/LR150scores. Also match depth/entry-safe branch. This bounds two-score monotone separability of individual prefixes only,not whole first-hit policy impossibility.',
  fixed_fields=EXTRA,provenance='Exact existingGRID10137 projection,not invented/tuned fields;current global target unchanged. Original18 and prior sixGRID142batch fields retained,not replaced or declared useless.',
  data='Fit386/cal198 only,new8development features remain unextracted. Never touch final8.',
  extraction='Redacted paid prefix,known order/aliases/substation membership only;None/zero future fields rejected. All8independent scalar recomputations and suffix checks.',
  geometry='Compare only original18 versus18+fixed8,append missing flags,fit386medians/IQR with range fallback. Nearest5 distinct fitcalls,same original querydepth/initialsafe branch,exclude same date. No feature/weight/distance search.',
  gate='All45calCONT have5neighbors;strictly fewer calCONT empty neighborhoods than original18;calSTOP contaminated-neighborhood count no greater;each of4caldates CONT empty count no greater. If fail,no fit of this8-field variant. This is only a descriptive screening rule,not predicted gain or independent validation.',
  prohibited=['fit','validation_feature_selection/extraction','additionalfeaturevariants','model/C/threshold/weightgrid','new label or target','physics/newdates/test8','paper/scale']))
 paths=[O/'design.json',R/'work/grid101_gain_learning/features.py',R/'outputs/grid142_batch_representation/design.json',R/'outputs/grid148_state_context/neighbor_summary.json',B.O/'fit_rows.json.gz',B.O/'calibration_rows.json.gz',B.O/'train_scores.npz',G.O/'train_scores.npz',Q.O/'calibration_history.json',Q.O/'cutoff.json']
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));verify();print('GRID152_FROZEN')
def project(c):
 assert set(c)=={'order','aliases','rho_limit','visible','subids'}
 visible=c['visible'];assert all(set(r)==set(FIELDS) for r in visible);b=F.best(visible);assert b
 aliases={int(k):v for k,v in c['aliases'].items()};subids=c['subids'];station=subids[b['pool_id']];targets={int(v) for k,v in aliases.items() if subids[k]==station};seen={r['pool_id'] for r in visible};pending=[i for i in c['order'][len(visible):] if i in targets]
 x=dict(visible=visible,public_order=c['order'],public_aliases=aliases,public_subids=subids,visible_count=len(visible),incumbent_id=b['pool_id'],remaining_same_station_ids=pending,known_completion_queries=len(pending),rho_limit=c['rho_limit'])
 values=F.project(x);out={k:float(values[F.NAMES.index(k)]) for k in EXTRA}
 valid=[float(np.float32(r['reward'])) for r in visible if r['strict_admissible']];local=[r for r in visible if r['pool_id'] in targets]
 manual=dict(terminal_fraction=sum(bool(r['done']) for r in visible)/len(visible),illegal_fraction=sum(bool(r['illegal']) for r in visible)/len(visible),
  ambiguous_fraction=sum(bool(r['ambiguous']) for r in visible)/len(visible),exception_fraction=sum(bool(r['exceptions']) for r in visible)/len(visible),
  admissible_reward_mean=math.fsum(valid)/len(valid),admissible_reward_std=math.sqrt(math.fsum((v-math.fsum(valid)/len(valid))**2 for v in valid)/len(valid)),
  station_admissible_fraction=sum(bool(r['strict_admissible']) for r in local)/len(local),incumbent_relative_position=[r['pool_id'] for r in visible].index(b['pool_id'])/len(visible))
 assert all(abs(out[k]-manual[k])<1e-12 for k in EXTRA);return out
FIELDS=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
def normal(raw):
 raw=np.asarray(raw,float);fit=raw[:386];med=[float(np.median(fit[np.isfinite(fit[:,j]),j])) if np.isfinite(fit[:,j]).any() else 0. for j in range(raw.shape[1])]
 missing=~np.isfinite(raw);x=np.where(missing,med,raw);scale=np.quantile(x[:386],.75,axis=0)-np.quantile(x[:386],.25,axis=0);scale=np.where(scale>0,scale,np.ptp(x[:386],axis=0));scale=np.where(scale>0,scale,1.)
 return np.concatenate([(x-np.asarray(med))/scale,missing.astype(float)],axis=1)
def run():
 verify();assert not (O/'started.json').exists();save('started.json',dict(fits=0))
 fit=G.S.rowfile(B.O/'fit_rows.json.gz');cal=G.S.rowfile(B.O/'calibration_rows.json.gz');rows=fit+cal;parts=['fit']*386+['calibration']*198
 rz=np.load(B.O/'train_scores.npz');lz=np.load(G.O/'train_scores.npz');rf=np.concatenate([rz['fit'],rz['calibration']]);lr=np.concatenate([lz['fit'],lz['calibration']]);dom=[]
 for part in ['fit','calibration']:
  ii=[i for i,p in enumerate(parts) if p==part];pos=[i for i in ii if rows[i]['target']['stop_label']];neg=[i for i in ii if not rows[i]['target']['stop_label']]
  for mode in ['all_two_scores','same_depth_entry_branch']:
   detail=[]
   for i in pos:
    candidates=[j for j in neg if mode=='all_two_scores' or (rows[j]['meta']['visible_count']==rows[i]['meta']['visible_count'] and rows[j]['features']['prefix_safe']==rows[i]['features']['prefix_safe'])]
    above=[j for j in candidates if rf[j]>=rf[i] and lr[j]>=lr[i]]
    detail.append(dict(sample_id=rows[i]['meta']['sample_id'],dominated_by=[rows[j]['meta']['sample_id'] for j in above]))
   dom.append(dict(partition=part,scope=mode,stop_rows=len(pos),dominated_stop_rows=sum(bool(x['dominated_by']) for x in detail),detail=detail))
 save('two_score_dominance.json',dom)
 extra=[]
 for r in rows:
  c=B.call_at(r['meta']['source']);n=r['meta']['visible_count'];visible=[{k:v[k] for k in FIELDS} for v in c['outcomes'][:n]];contract=dict(order=c['order'],aliases=c['aliases'],rho_limit=c['rho_limit'],visible=visible,subids=B.subids());x=project(contract)
  poison=dict(c,outcomes=c['outcomes'][:n]+[dict(v,reward=-999.,rho=999.,pool_id=-1) for v in c['outcomes'][n:]])
  assert project(dict(contract,visible=[{k:v[k] for k in FIELDS} for v in poison['outcomes'][:n]]))==x
  try:project(dict(contract,future_label=r['target']['stop_label']))
  except AssertionError:pass
  else:raise AssertionError('Future label accepted')
  extra.append(x)
 with gzip.open(O/'fit_cal_public_extra.jsonl.gz','wt',encoding='utf8') as f:
  for r,x in zip(rows,extra):f.write(json.dumps(dict(sample_id=r['meta']['sample_id'],features=x))+'\n')
 profile=[dict(feature=k,fit_distinct_values=len({r[k] for r in extra[:386]}),cal_distinct_values=len({r[k] for r in extra[386:]})) for k in EXTRA];save('extra_profile.json',profile)
 names=read(B.O/'design.json')['features'];raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in rows]);new=np.asarray([[r[k] for k in EXTRA] for r in extra])
 views=dict(ORIGINAL18=normal(raw),LEGACY26=normal(np.concatenate([raw,new],axis=1)));records=[];sums=[];dates=[]
 for view,x in views.items():
  for i,r in enumerate(rows):
   jj=[j for j,s in enumerate(fit) if s['meta']['date']!=r['meta']['date'] and s['meta']['visible_count']==r['meta']['visible_count'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
   dd={j:float(np.sqrt(np.mean((x[i]-x[j])**2))) for j in jj};ordered=sorted(jj,key=lambda j:(dd[j],fit[j]['meta']['sample_id']));chosen=[];seen=set()
   for j in ordered:
    if fit[j]['meta']['call_id'] in seen:continue
    chosen.append(j);seen.add(fit[j]['meta']['call_id'])
    if len(chosen)==5:break
   for j in chosen:assert abs(dd[j]-math.sqrt(math.fsum((float(x[i,k])-float(x[j,k]))**2 for k in range(x.shape[1]))/x.shape[1]))<1e-12
   records.append(dict(view=view,partition=parts[i],sample_id=r['meta']['sample_id'],date=r['meta']['date'],stop_label=r['target']['stop_label'],k=len(chosen),continue_neighbors=sum(not fit[j]['target']['stop_label'] for j in chosen),neighbors=[fit[j]['meta']['sample_id'] for j in chosen]))
  for part in ['fit','calibration']:
   rr=[r for r in records if r['view']==view and r['partition']==part];neg=[r for r in rr if not r['stop_label']];pos=[r for r in rr if r['stop_label']]
   sums.append(dict(view=view,partition=part,continue_rows=len(neg),continue_full5=sum(r['k']==5 for r in neg),continue_empty=sum(r['k']==5 and not r['continue_neighbors'] for r in neg),stop_full5=sum(r['k']==5 for r in pos),stop_contaminated=sum(r['k']==5 and r['continue_neighbors']>0 for r in pos)))
   for date in sorted({r['date'] for r in rr}):dates.append(dict(view=view,partition=part,date=date,continue_empty=sum(r['date']==date and not r['stop_label'] and r['k']==5 and not r['continue_neighbors'] for r in rr)))
 save('all_neighbor_records.json',records);save('neighbor_summary.json',sums);save('neighbor_per_date.json',dates)
 get=lambda v:next(s for s in sums if s['partition']=='calibration' and s['view']==v)
 a=get('ORIGINAL18');b=get('LEGACY26');old={r['date']:r['continue_empty'] for r in dates if r['partition']=='calibration' and r['view']=='ORIGINAL18'};newdates={r['date']:r['continue_empty'] for r in dates if r['partition']=='calibration' and r['view']=='LEGACY26'}
 passed=a['continue_full5']==b['continue_full5']==45 and b['continue_empty']<a['continue_empty'] and b['stop_contaminated']<=a['stop_contaminated'] and all(newdates[d]<=old[d] for d in old)
 save('decision.json',dict(passed_prespecified_information_screen=passed,new_fit_permitted_by_this_screen=passed,new_fit_executed=False,
  original_cal=a,new_cal=b,date_no_harm=all(newdates[d]<=old[d] for d in old),cal_dates_original=old,cal_dates_new=newdates,
  if_failed='Do not fit this8-field variant or try a different subset/metric to rescue it.',final8_absent=True))
 verify();save('review.json',dict(passed=True,new_fits=0,new_predicts=0,new_policy_replays=0,new_physical=0,
  public_fit_cal_prefixes=584,independent_scalar_values=584*8,suffix_checks=584,future_key_rejections=584,
  validation_extra_features_extracted=False,old_sources_models_labels_cutoffs_unchanged=True,final8_absent=True))
 save('state.json',dict(stage='analysed'));print(json.dumps(dict(decision=read(O/'decision.json'),dominance=[{k:v for k,v in d.items() if k!='detail'} for d in dom],profile=profile),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
