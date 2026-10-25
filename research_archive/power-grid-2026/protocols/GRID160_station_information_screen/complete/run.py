"""Fixed public station-summary information screen; no classifier fitting."""
from pathlib import Path
import collections,gzip,hashlib,importlib.util,json,math,shutil,sys
import numpy as np
spec=importlib.util.spec_from_file_location('g160_station',Path('D:/Codex/GRL研究/work/grid159_station_bridge/run.py'));G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
spec=importlib.util.spec_from_file_location('g160_fixed_geometry',Path('D:/Codex/GRL研究/work/grid152_paid_feedback_screen/run.py'));N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
B=G.B;R=B.R;W=Path(__file__).parent;O=R/'outputs/grid160_station_information_screen';REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID160_station_information_screen'
EXTRA=['incumbent_station_seen_fraction','queried_station_fraction','unseen_station_mass_fraction','remaining_mass_outside_incumbent_fraction','admissible_station_fraction','station_best_reward_mean','station_best_reward_std','top_two_station_reward_gap']
read=G.read;sha=G.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 G.verify()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);G.verify();assert read(G.O/'complete.json')['passed']
 save('design.json',dict(id='GRID160_STATION_INFORMATION_SCREEN',fields=EXTRA,scope='Original386 fit/198 calibration prefixes only,new development station features not extracted. No fitting/new model scores/physical or test8.',
  definitions='Public station membership through original aliases. Incumbent station seen/total;seen stations/all;remaining membership mass of entirely unseen stations/all membership mass;remaining membership mass outside incumbent station/all remaining;admissible observed stations/all;mean/std/top-two gap of per-station best observed float32 rewards. No future station labels or unqueried feedback.',
  geometry='Exactly inheritedGRID152: fit-only median/IQR(range fallback),append missingflags;Euclidean RMS;5 distinct fitcalls,exclude same date,match querydepth and original prefix-safe branch. Compare18 vs18+fixed8 only.',
  gate='All45 calCONT full5;strictly fewer CONTzero-CONT-neighbor rows;STOP contaminated count notgreater;each4caldate CONTempty no greater. If fail,no fit this26 variant,no subset/distance/weight search.',
  limits='Descriptive qualification only,not proof of predictiveness or incapacity. Same calibration dates repeatedly used,not independent test. Raw station profile remains separate from future mechanism audit.',
  prohibited=['new fits/scores','new dev features','field/metric/weight scans','new labels/thresholds','new physics/data/test8','paper','scale']))
 paths=[G.O/'complete.json',R/'work/grid159_station_bridge/run.py',R/'work/grid152_paid_feedback_screen/run.py',B.O/'fit_rows.json.gz',B.O/'calibration_rows.json.gz',R/'outputs/grid152_paid_feedback_screen/neighbor_summary.json',O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID160_FROZEN')
def project(public):
 assert set(public)=={'order','aliases','prefix'};pp=G.profile(public);b=B.P.best(public['prefix']);assert b;current=B.subids()[b['pool_id']];st=next(x for x in pp if x['station']==current);mass=sum(x['representatives'] for x in pp);remaining=sum(x['remaining'] for x in pp);best=sorted([x['best_observed_reward'] for x in pp if x['best_observed_reward'] is not None],reverse=True);assert best
 out=dict(incumbent_station_seen_fraction=st['seen']/st['representatives'],queried_station_fraction=sum(x['seen']>0 for x in pp)/len(pp),unseen_station_mass_fraction=sum(x['remaining'] for x in pp if not x['seen'])/mass,remaining_mass_outside_incumbent_fraction=(remaining-st['remaining'])/remaining if remaining else 0.,admissible_station_fraction=len(best)/len(pp),station_best_reward_mean=math.fsum(best)/len(best),station_best_reward_std=math.sqrt(math.fsum((v-math.fsum(best)/len(best))**2 for v in best)/len(best)),top_two_station_reward_gap=best[0]-best[1] if len(best)>1 else None)
 # Independent scalar calculation from per-station sets, without using serialized mechanism targets.
 subs=B.subids();ids=collections.defaultdict(set);seen={x['pool_id']:x for x in public['prefix']}
 for a,rep in public['aliases'].items():
  if int(rep) in public['order']:ids[subs[int(a)]].add(int(rep))
 seen_n={s:len(v&seen.keys()) for s,v in ids.items()};rem={s:len(v)-seen_n[s] for s,v in ids.items()};rw=[]
 for s,v in ids.items():
  vv=[float(np.float32(seen[i]['reward'])) for i in v&seen.keys() if seen[i]['strict_admissible']]
  if vv:rw.append(max(vv))
 rw.sort(reverse=True);total=sum(map(len,ids.values()));remtotal=sum(rem.values());manual=[seen_n[current]/len(ids[current]),sum(v>0 for v in seen_n.values())/len(ids),sum(rem[s] for s in ids if not seen_n[s])/total,(remtotal-rem[current])/remtotal if remtotal else 0.,len(rw)/len(ids),float(np.mean(rw)),float(np.std(rw)),rw[0]-rw[1] if len(rw)>1 else None]
 for k,v in zip(EXTRA,manual):assert (out[k] is None and v is None) or abs(out[k]-v)<1e-12
 return out
def analyse():
 verify();assert not (O/'review.json').exists();fit=G.rowfile(B.O/'fit_rows.json.gz');cal=G.rowfile(B.O/'calibration_rows.json.gz');assert len(fit)==386 and len(cal)==198;rows=fit+cal;extra=[]
 for r in rows:
  c=B.call_at(r['meta']['source']);n=r['meta']['visible_count'];public=dict(order=c['order'],aliases=c['aliases'],prefix=[{k:x[k] for k in ['pool_id','reward','rho','strict_admissible']} for x in c['outcomes'][:n]]);x=project(public);poison=dict(c,outcomes=c['outcomes'][:n]+[dict(a,reward=-999,rho=999,strict_admissible=False) for a in c['outcomes'][n:]]);assert project(dict(public,prefix=[{k:a[k] for k in ['pool_id','reward','rho','strict_admissible']} for a in poison['outcomes'][:n]]))==x
  try:project(dict(public,future_label=r['target']['stop_label']))
  except AssertionError:pass
  else:raise AssertionError('Future field accepted')
  extra.append(x)
 with gzip.open(O/'fit_cal_station_summary.jsonl.gz','wt',encoding='utf8') as f:
  for r,x in zip(rows,extra):f.write(json.dumps(dict(sample_id=r['meta']['sample_id'],features=x))+'\n')
 names=read(B.O/'preprocessing.json')['features'];raw=np.array([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in rows]);new=np.array([[np.nan if x[k] is None else x[k] for k in EXTRA] for x in extra]);records=[];sums=[];dates=[]
 for view,x in [('ORIGINAL18',N.normal(raw)),('STATION26',N.normal(np.concatenate([raw,new],axis=1)))]:
  for i,r in enumerate(rows):
   jj=[j for j,s in enumerate(fit) if s['meta']['date']!=r['meta']['date'] and s['meta']['visible_count']==r['meta']['visible_count'] and s['features']['prefix_safe']==r['features']['prefix_safe']];dd={j:float(np.sqrt(np.mean((x[i]-x[j])**2))) for j in jj};ordered=sorted(jj,key=lambda j:(dd[j],fit[j]['meta']['sample_id']));chosen=[];seen=set()
   for j in ordered:
    if fit[j]['meta']['call_id'] in seen:continue
    chosen.append(j);seen.add(fit[j]['meta']['call_id'])
    if len(chosen)==5:break
   for j in chosen:assert abs(dd[j]-math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(x[i],x[j]))/x.shape[1]))<1e-12
   records.append(dict(view=view,partition='fit' if i<386 else 'calibration',sample_id=r['meta']['sample_id'],date=r['meta']['date'],stop_label=r['target']['stop_label'],k=len(chosen),continue_neighbors=sum(not fit[j]['target']['stop_label'] for j in chosen),neighbors=[fit[j]['meta']['sample_id'] for j in chosen]))
  for part in ['fit','calibration']:
   rr=[r for r in records if r['view']==view and r['partition']==part];neg=[r for r in rr if not r['stop_label']];pos=[r for r in rr if r['stop_label']];sums.append(dict(view=view,partition=part,continue_rows=len(neg),continue_full5=sum(r['k']==5 for r in neg),continue_empty=sum(r['k']==5 and not r['continue_neighbors'] for r in neg),stop_full5=sum(r['k']==5 for r in pos),stop_contaminated=sum(r['k']==5 and r['continue_neighbors']>0 for r in pos)))
   for date in sorted({r['date'] for r in rr}):dates.append(dict(view=view,partition=part,date=date,continue_empty=sum(r['date']==date and not r['stop_label'] and r['k']==5 and not r['continue_neighbors'] for r in rr)))
 get=lambda v:next(s for s in sums if s['partition']=='calibration' and s['view']==v);a=get('ORIGINAL18');b=get('STATION26');old={r['date']:r['continue_empty'] for r in dates if r['partition']=='calibration' and r['view']=='ORIGINAL18'};newd={r['date']:r['continue_empty'] for r in dates if r['partition']=='calibration' and r['view']=='STATION26'};passed=a['continue_full5']==b['continue_full5']==45 and b['continue_empty']<a['continue_empty'] and b['stop_contaminated']<=a['stop_contaminated'] and all(newd[k]<=old[k] for k in old)
 baseline=[s for s in read(R/'outputs/grid152_paid_feedback_screen/neighbor_summary.json') if s['view']=='ORIGINAL18'];assert baseline==[s for s in sums if s['view']=='ORIGINAL18']
 verify();save('neighbor_records.json',records);save('neighbor_summary.json',sums);save('per_date.json',dates);save('decision.json',dict(passed_prespecified_information_screen=passed,fit_executed=False,new_fit_permitted_by_this_screen=passed,no_subset_rescue=True,old_cal_dates=old,new_cal_dates=newd));save('review.json',dict(passed=True,public_rows=584,independent_scalar_checks=4672,suffix_checks=584,future_key_rejections=584,baseline_reproduced=True,new_fits=0,new_scores=0,new_dev_features=False,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(summaries=sums,decision=read(O/'decision.json'))))
if __name__=='__main__':globals()[sys.argv[1]]()
