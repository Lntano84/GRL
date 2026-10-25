"""Fixed RF140 leave-one-fit-date-out diagnostic, not candidate selection."""
from pathlib import Path
import collections,datetime,gzip,hashlib,importlib.util,json,pickle,shutil,sys,time
import numpy as np
spec=importlib.util.spec_from_file_location('g156_reference',Path('D:/Codex/GRL研究/work/grid140_global_compression/common.py'));B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
R=B.R;W=Path(__file__).parent;O=R/'outputs/grid156_date_holdout';REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID156_date_holdout'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def rowfile(p):
 with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for f in sorted((O/'folds').glob('*')) if (O/'folds').exists() else []:
  if f.is_file():q=d/'folds';q.mkdir(exist_ok=True);shutil.copy2(f,q/f.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
 B.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 B.P.sealed()
def prepare():
 assert not O.exists();O.mkdir(parents=True);B.verify_trained();d=read(B.O/'design.json')
 save('design.json',dict(id='GRID156_DATE_HOLDOUT_DIAGNOSTIC',authorization='User autonomous continuation; stage update notified before fitting. No new data or scaling.',
  objective='Determine whether original18 RF140 learns stable cross-date discrimination, rather than selecting another model using exposed development results.',
  population='Original12 fit dates/386 eligible prefixes only; all replicates and calls of held date withheld together. Original161 fit complete search calls replayed once as held-out calls, plus old in-sample reference and fixedCAP256.',
  parameters=d['model_parameters'],dates=d['fit_dates'],fits_permitted=12,total_fit_wall_cap_s=120,
  preprocessing='Per-fold18 medians fit using11 training dates only;missing median0,append18 flags. No full12 median leakage into held date.',
  weights='Original equal-date/equal-call/equal-prefix rule recomputed using11 training dates only,mean1. Same seed/parameters all folds. No class reweighting.',
  threshold=read(B.O/'cutoff.json')['primary_threshold'],threshold_note='Old fixed calibrated cutoff reused as a separate operational readout;no recalibration,threshold grid,or new calibration/development scoring. Fold scores may have different calibration;within-date ranking metrics also reported.',
  metrics='Continue-positive AP/AUC and class prevalence,per-date and128-only;pooled cross-fold score metrics descriptive. Complete-search original-action preservation and query counts;in-sample contrast is not an independent baseline.',
  status='Diagnostic12 fits,not12 candidate choices;all folds retained. No bestfold/seed/modelselection or deployment.',
  prohibited=['new data','physics','new dev/cal scoring','test8','different features/labels/parameters/seed','refit after failure','paper','automatic scale']))
 paths=[R/'outputs/grid150_linear_control/original_fit_rows.json.gz',R/'outputs/grid140_global_compression'/ 'fit_sample_weights.json',B.O/'design.json',B.O/'cutoff.json',B.O/'train_scores.npz',O/'design.json']
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID156_FROZEN')
def medians(rows,names):
 out=[]
 for k in names:
  a=[float(r['features'][k]) for r in rows if r['features'][k] is not None];out.append(float(np.median(a)) if a else 0.)
 return out
def weights(rows):
 dates={r['meta']['date'] for r in rows};calls=collections.defaultdict(set);cp=collections.Counter(r['meta']['call_id'] for r in rows)
 for r in rows:calls[r['meta']['date']].add(r['meta']['call_id'])
 w=np.array([1/(len(dates)*len(calls[r['meta']['date']])*cp[r['meta']['call_id']]) for r in rows]);w*=len(rows)/w.sum();return w
def metrics(rows,p):
 from sklearn.metrics import average_precision_score,roc_auc_score
 y=np.asarray([not r['target']['stop_label'] for r in rows],int);q=1-np.asarray(p,float);both=len(set(y))==2
 return dict(rows=len(rows),continue_rows=int(y.sum()),continue_prevalence=float(y.mean()),continue_AP=float(average_precision_score(y,q)) if y.sum() else None,continue_AUC=float(roc_auc_score(y,q)) if both else None)
def train():
 verify();assert not (O/'fit_started.json').exists();d=read(O/'design.json');rows=rowfile(R/'outputs/grid150_linear_control/original_fit_rows.json.gz');assert len(rows)==386;names=read(B.O/'preprocessing.json')['features'];assert sorted({r['meta']['date'] for r in rows})==d['dates']
 save('fit_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits_permitted=12));f=O/'folds';f.mkdir();allp=np.full(len(rows),np.nan);receipts=[];elapsed_total=0.
 from sklearn.ensemble import RandomForestClassifier
 for date in d['dates']:
  ti=[i for i,r in enumerate(rows) if r['meta']['date']!=date];vi=[i for i,r in enumerate(rows) if r['meta']['date']==date];tr=[rows[i] for i in ti];va=[rows[i] for i in vi];assert len({r['meta']['date'] for r in tr})==11
  assert not ({r['meta']['call_id'] for r in tr}&{r['meta']['call_id'] for r in va});med=medians(tr,names);xx=B.transform(tr,names,med);xv=B.transform(va,names,med);w=weights(tr);y=np.array([r['target']['stop_label'] for r in tr],int)
  m=RandomForestClassifier(**d['parameters']);started=time.perf_counter();m.fit(xx,y,sample_weight=w);elapsed=time.perf_counter()-started;elapsed_total+=elapsed;assert elapsed_total<=120
  p=m.predict_proba(xv)[:,list(m.classes_).index(1)];export=B.export_model(m);assert max(abs(B.exported_scores(export,xv)-p))<1e-10;allp[vi]=p
  with (f/(date+'.pkl')).open('wb') as h:pickle.dump(m,h,pickle.HIGHEST_PROTOCOL)
  (f/(date+'_export.json')).write_text(json.dumps(export)+'\n',encoding='utf8',newline='\n')
  receipt=dict(date=date,fit_rows=len(tr),heldout_rows=len(va),fit_dates=sorted({r['meta']['date'] for r in tr}),train_sample_ids=[r['meta']['sample_id'] for r in tr],heldout_sample_ids=[r['meta']['sample_id'] for r in va],weights=w.tolist(),medians=med,fit_wall_s=elapsed,parameters=m.get_params(),model_sha256=sha(f/(date+'.pkl')),export_sha256=sha(f/(date+'_export.json')))
  (f/(date+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n');receipts.append(receipt);save('fit_progress.json',dict(completed=len(receipts),total=12,total_fit_wall_s=elapsed_total))
 assert np.isfinite(allp).all();np.savez_compressed(O/'heldout_scores.npz',scores=allp);save('fit_receipt.json',dict(completed=12,total_fit_wall_s=elapsed_total,rows_scored=386,no_refits=True));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*f.iterdir(),O/'heldout_scores.npz',O/'fit_receipt.json']});save('state.json',dict(stage='trained'));archive('trained');print('GRID156_TRAINED',elapsed_total)
def analyse():
 verify();assert not (O/'review.json').exists();d=read(O/'design.json');rows=rowfile(R/'outputs/grid150_linear_control/original_fit_rows.json.gz');p=np.load(O/'heldout_scores.npz')['scores'];old=np.load(B.O/'train_scores.npz')['fit'];checks=collections.Counter();perdate=[]
 for path,h in read(O/'trained_freeze.json').items():assert sha(O/path)==h,path
 names=read(B.O/'preprocessing.json')['features']
 for date in d['dates']:
  tr=[r for r in rows if r['meta']['date']!=date];ix=[i for i,r in enumerate(rows) if r['meta']['date']==date];rr=[rows[i] for i in ix];receipt=read(O/'folds'/(date+'_receipt.json'));assert receipt['medians']==medians(tr,names) and np.allclose(receipt['weights'],weights(tr),rtol=0,atol=0)
  assert set(receipt['train_sample_ids']).isdisjoint(receipt['heldout_sample_ids']);assert date not in receipt['fit_dates'];xx=B.transform(rr,names,receipt['medians']);export=read(O/'folds'/(date+'_export.json'));assert max(abs(B.exported_scores(export,xx)-p[ix]))<1e-10;checks['heldout_export_scores']+=len(ix)
  perdate.append(dict(date=date,heldout=metrics(rr,p[ix]),in_sample=metrics(rr,old[ix]),heldout128=metrics([r for r in rr if r['meta']['visible_count']==128],[p[i] for i in ix if rows[i]['meta']['visible_count']==128])))
 sm={(r['meta']['call_id'],r['meta']['visible_count']):float(v) for r,v in zip(rows,p)};om={(r['meta']['call_id'],r['meta']['visible_count']):float(v) for r,v in zip(rows,old)};replays=[]
 for ref in B.refs(dates=d['dates']):
  assert ref['partition']=='train'
  for method,scoremap in [('RF156_DATE_HELDOUT',sm),('RF140_IN_SAMPLE_REFERENCE',om)]:replays.append(dict(method=method,**B.replay(ref,'MODEL',d['threshold'],scoremap)))
  for method in ['CAP256','FRAME']:replays.append(dict(method=method,**B.replay(ref,method)))
 summaries={m:B.summarize([r for r in replays if r['method']==m]) for m in sorted({r['method'] for r in replays})};assert summaries['FRAME']['calls']==161
 pooled=metrics(rows,p);ins=metrics(rows,old);ranked=[x['heldout'] for x in perdate if x['heldout']['continue_AUC'] is not None]
 macro=dict(dates_with_both_classes=len(ranked),mean_within_date_continue_AUC=float(np.mean([x['continue_AUC'] for x in ranked])),mean_within_date_continue_AP=float(np.mean([x['continue_AP'] for x in ranked])),mean_within_date_continue_prevalence=float(np.mean([x['continue_prevalence'] for x in ranked])))
 verify();save('per_date_metrics.json',perdate);save('ranking_metrics.json',dict(pooled_heldout=pooled,pooled_in_sample=ins,within_date_macro=macro,pooled128=metrics([r for r in rows if r['meta']['visible_count']==128],[p[i] for i,r in enumerate(rows) if r['meta']['visible_count']==128])))
 save('replays.json',replays);save('search_summary.json',summaries);save('all_bad_heldout_stops.json',[r for r in replays if r['method']=='RF156_DATE_HELDOUT' and r['bad_learned_stop']]);save('decision.json',dict(diagnostic_only=True,new_policy_adopted=False,ready_to_scale=False,no_fold_selection=True,no_threshold_change=True,all_folds_retained=True));save('review.json',dict(passed=True,checks=dict(checks),fits=12,whole_date_split=True,heldout_dates_not_used_for_preprocessing_or_weights=True,replays=len(replays),new_dev_or_cal_scores=0,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(ranking=read(O/'ranking_metrics.json'),heldout=summaries['RF156_DATE_HELDOUT']),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
