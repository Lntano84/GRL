"""One same-parameter fit at192; unchanged action-preserving compression label."""
from pathlib import Path
import collections,datetime,gzip,hashlib,importlib.util,json,pickle,shutil,subprocess,sys,time
import numpy as np
spec=importlib.util.spec_from_file_location('g158_reference',Path('D:/Codex/GRL研究/work/grid140_global_compression/common.py'));B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
R=B.R;W=Path(__file__).parent;O=R/'outputs/grid158_single_192';REPO=B.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID158_single_192'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def rowfile(p):
 with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def gzsave(n,x):
 with gzip.open(O/n,'wt',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False)
def verify():
 B.verify_trained()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
 B.P.sealed()
def verify_trained():
 verify()
 for n,h in read(O/'trained_freeze.json').items():assert sha(O/n)==h,n
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def scoremap(rows,p):return {(r['meta']['call_id'],r['meta']['visible_count']):float(v) for r,v in zip(rows,p)}
def replay(ref,tau,sm):
 # Original loop; only the new decision location is192. Global initial prefix stays128.
 c=B.call_at(ref['source']);order=c['order'];n=min(128,len(order));initial=B.P.best(c['outcomes'][:n]);safe=initial is not None and initial['rho']<.9;b=initial;stale=0;visited=[];kind='exhaustion';stopped=None
 while n<len(order):
  forced=(not safe and b is not None and b['rho']<.9) or (b is not None and n>=256 and stale>=2)
  if forced:kind='original_forced_stop';break
  if n==192 and b is not None:
   p=sm[(ref['call_id'],192)];visited.append(dict(visible_count=192,score=p,stop=p>tau))
   if p>tau:kind='learned_early_stop';stopped=p;break
  old=b;n=min(n+64,len(order));b=B.P.best(c['outcomes'][:n]);stale=0 if B.P.improves(old,b) else stale+1
 extra=[]
 if b is not None and n<len(order) and (not safe or B.P.bid(b)!=B.P.bid(initial)):
  subs=B.subids();station=subs[b['pool_id']];ids={int(rep) for a,rep in c['aliases'].items() if subs[int(a)]==station};extra=[x for x in c['outcomes'][n:] if x['pool_id'] in ids];b=B.P.best(c['outcomes'][:n]+extra)
 q=n+len(extra);original=ref['global_n']+ref['local_queries'];assert len(visited)<=1
 return dict(call_id=ref['call_id'],partition=ref['partition'],date=ref['date'],scenario=ref['scenario'],global_queries=n,local_queries=len(extra),queries=q,original_queries=original,selected=B.P.bid(b),original_selected=ref['frame_choice'],action_match=B.P.bid(b)==ref['frame_choice'],stop_kind=kind,stopped_score=stopped,model_decisions=visited,model_decision_count=len(visited),bad_learned_stop=kind=='learned_early_stop' and (B.P.bid(b)!=ref['frame_choice'] or q>=original))
def shadow(ref,tau,sm):
 c=B.call_at(ref['source']);masked={(ref['call_id'],n):(sm.get((ref['call_id'],192),-1.) if n==192 else -1.) for n in range(128,len(c['order']),64)};a=replay(ref,tau,sm);b=B.replay(ref,'MODEL',tau,masked)
 for k in ['selected','global_queries','local_queries','queries','action_match','bad_learned_stop','stop_kind']:assert a[k]==b[k]
 assert a['model_decisions']==[x for x in b['model_decisions'] if x['visible_count']==192];return a
def calibrate(refs,sm):
 tau=.5;history=[]
 for _ in range(3):
  rr=[replay(r,tau,sm) for r in refs];bad=[r for r in rr if r['bad_learned_stop']];history.append(dict(threshold=tau,summary=B.summarize(rr),bad=[dict(call_id=r['call_id'],score=r['stopped_score']) for r in bad]))
  if not bad:return tau,history
  tau=max(r['stopped_score'] for r in bad)
 raise AssertionError('Preserve failed calibration;no refit')
def prepare():
 assert not O.exists();O.mkdir(parents=True);B.verify_trained();assert read(R/'outputs/grid157_checkpoint_information/complete.json')['passed'];base=read(B.O/'design.json');fit=[r for r in rowfile(R/'outputs/grid150_linear_control/original_fit_rows.json.gz') if r['meta']['visible_count']==192];cal=[r for r in rowfile(R/'outputs/grid150_linear_control/original_calibration_rows.json.gz') if r['meta']['visible_count']==192];assert len(fit)==153 and sum(not r['target']['stop_label'] for r in fit)==10
 d=dict(id='GRID158_SINGLE192_FIXED_MODEL',authorization='Autonomous user continuation;explicit pretraining notice before this one fit.',model_parameters=base['model_parameters'],features=base['features'],fit_dates=base['fit_dates'],calibration_dates=base['calibration_dates'],validation_dates=base['validation_dates'],fit_rows=len(fit),fit_continue=10,calibration_rows=len(cal),fit_limit=1,fit_wall_cap_s=120,
  task='Exactly one learned stop at192 if originally eligible;otherwise original global loop/forced stops and same completion unchanged. No128 learned exit,no later learned exit.',
  hypothesis='Remove mixture of entry-static and later-dynamic fitting populations;fixed192 original-model controls distinguish refitting from placement. Both learning population and exit structure change,not a single-factor intervention.',
  preprocessing='Re-estimate18 medians on fit192 only,append18missing flags. Equal date mass N/(12*n_date),one192 row per physical call. No class weighting.',
  labels='Original exact original-FRAME chosen pool_id plus strictly fewer total queries;no operational-label replacement.',
  calibration='Same4 calibration dates. Empirical zero joint-failure increasing threshold from.5,strict>. OldRF192-only separately calibrated before new dev scoring;old original cutoff also retained.',
  primary='RF158_192_CALIBRATED',controls=['FRAME','CAP192','CAP256','RF140_GLOBAL','RF140_192_ORIGINAL_CUTOFF','RF140_192_CALIBRATED','RF158_192_FIXED_0.5'],
  screen='Original62 dev calls all action identity,no non-saving learned stops;date macro queries reduction greater thanCAP256,each date queries no greater with at leastone strictly fewer.',
  limits='Chosen after exposed development diagnosis,not independent confirmation. All prior failures retained. No bestthreshold/checkpoint/model choice,physics,final8,data,paper or automatic scale. No other checkpoint scan after this.')
 save('design.json',d);gzsave('fit_rows.json.gz',fit);gzsave('calibration_rows.json.gz',cal);paths=[O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz']];paths +=[B.O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','train_scores.npz','validation_scores.npz','cutoff.json','model_export.json']];paths +=[R/'outputs/grid157_checkpoint_information/complete.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
 dummy={(r['meta']['call_id'],192):.75 for r in fit+cal};checks=0
 for ref in B.refs('train',dates=d['fit_dates']+d['calibration_dates']):
  for tau in [.5,1.]:
   a=shadow(ref,tau,dummy);checks+=1
   if tau==1.:assert a['action_match'] and a['queries']==a['original_queries']
 save('preflight.json',dict(passed=True,shadow_replays=checks,no_stop_equals_original=True,fit_rows=len(fit),cal_rows=len(cal)));verify();save('state.json',dict(stage='frozen'));archive('frozen');print('GRID158_FROZEN',len(cal))
def train():
 verify();assert not (O/'fit_started.json').exists();d=read(O/'design.json');fit=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');names=d['features'];med=[]
 for k in names:
  v=[float(r['features'][k]) for r in fit if r['features'][k] is not None];med.append(float(np.median(v)) if v else 0.)
 x=B.transform(fit,names,med);xc=B.transform(cal,names,med);counts=collections.Counter(r['meta']['date'] for r in fit);assert len(counts)==12;weights=np.array([len(fit)/(12*counts[r['meta']['date']]) for r in fit]);y=np.array([r['target']['stop_label'] for r in fit],int)
 save('preprocessing.json',dict(features=names,medians=med,append_missing_flags=True,dimensions=36));save('fit_weights.json',weights.tolist());save('fit_started.json',dict(fits_permitted=1,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 from sklearn.ensemble import RandomForestClassifier
 m=RandomForestClassifier(**d['model_parameters']);t=time.perf_counter();m.fit(x,y,sample_weight=weights);elapsed=time.perf_counter()-t;assert elapsed<=120
 with (O/'model.pkl').open('wb') as f:pickle.dump(m,f,pickle.HIGHEST_PROTOCOL)
 export=B.export_model(m);save('model_export.json',export);pf=m.predict_proba(x)[:,1];pc=m.predict_proba(xc)[:,1];assert max(abs(B.exported_scores(export,x)-pf))<1e-10 and max(abs(B.exported_scores(export,xc)-pc))<1e-10;np.savez_compressed(O/'train_scores.npz',fit=pf,calibration=pc)
 oldrows=rowfile(B.O/'fit_rows.json.gz')+rowfile(B.O/'calibration_rows.json.gz');z=np.load(B.O/'train_scores.npz');old=scoremap(oldrows,np.concatenate([z['fit'],z['calibration']]));refs=B.refs('train',dates=d['calibration_dates']);tau,h=calibrate(refs,scoremap(cal,pc));oldt,oldh=calibrate(refs,old)
 save('calibration_history.json',h);save('old192_calibration_history.json',oldh);save('cutoff.json',dict(primary=tau,old192_calibrated=oldt,old_original=read(B.O/'cutoff.json')['primary_threshold'],fixed=.5,strict_greater=True,dev_scored=False));save('fit_receipt.json',dict(fits=1,rows=len(fit),continue_rows=10,fit_wall_s=elapsed,parameters=m.get_params()))
 names=['design.json','model.pkl','model_export.json','preprocessing.json','fit_weights.json','fit_receipt.json','train_scores.npz','calibration_history.json','old192_calibration_history.json','cutoff.json'];save('trained_freeze.json',{n:sha(O/n) for n in names});verify_trained();save('state.json',dict(stage='trained_before_dev'));archive('trained');print('GRID158_TRAINED',json.dumps(read(O/'cutoff.json')))
def publication():
 verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote;rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();assert hashlib.sha256(subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO)).hexdigest()==sha(O/'trained_freeze.json');save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote));print('GRID158_PREDEV_PUBLISHED')
def validate():
 verify_trained();assert read(O/'prevalidation_publication.json')['passed'];assert not (O/'validation_started.json').exists();save('validation_started.json',dict(fits=1));d=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json');va=B.eligible_data('validation');val=[r for r in va if r['meta']['visible_count']==192];x=B.transform(val,pre['features'],pre['medians'])
 with (O/'model.pkl').open('rb') as f:m=pickle.load(f)
 pv=m.predict_proba(x)[:,1];assert max(abs(B.exported_scores(read(O/'model_export.json'),x)-pv))<1e-10;np.savez_compressed(O/'validation_scores.npz',scores=pv);gzsave('validation_rows.json.gz',val)
 fit=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');z=np.load(O/'train_scores.npz');sm=scoremap(fit+cal+val,np.concatenate([z['fit'],z['calibration'],pv]));oldrows=rowfile(B.O/'fit_rows.json.gz')+rowfile(B.O/'calibration_rows.json.gz')+va;z=np.load(B.O/'train_scores.npz');old=scoremap(oldrows,np.concatenate([z['fit'],z['calibration'],np.load(B.O/'validation_scores.npz')['scores']]));replays=[];summaries=[];checks=collections.Counter()
 for part,refs in [('fit',B.refs('train',dates=d['fit_dates'])),('calibration',B.refs('train',dates=d['calibration_dates'])),('validation',B.refs('validation'))]:
  for method in ['FRAME','CAP192','CAP256','RF140_GLOBAL','RF140_192_ORIGINAL_CUTOFF','RF140_192_CALIBRATED','RF158_192_FIXED_0.5','RF158_192_CALIBRATED']:
   rr=[]
   for ref in refs:
    if method in ['FRAME','CAP192','CAP256']:a=B.replay(ref,method)
    elif method=='RF140_GLOBAL':a=B.replay(ref,'MODEL',cut['old_original'],old)
    else:
     control=method.startswith('RF140');threshold=cut['old_original'] if method.endswith('ORIGINAL_CUTOFF') else (cut['old192_calibrated'] if control else (cut['fixed'] if method.endswith('0.5') else cut['primary']));a=shadow(ref,threshold,old if control else sm);checks['shadow_policy_calls']+=1
    rr.append(dict(a,method=method,evaluation_partition=part))
   replays+=rr;summaries.append(dict(method=method,partition=part,**B.summarize(rr),joint_failures=sum(x['bad_learned_stop'] for x in rr)))
 vr=[r for r in summaries if r['partition']=='validation'];main=next(r for r in vr if r['method']=='RF158_192_CALIBRATED');cap=next(r for r in vr if r['method']=='CAP256');better=main['date_macro_query_reduction']>cap['date_macro_query_reduction'] and all(a['queries']<=b['queries'] for a,b in zip(main['by_date'],cap['by_date']));passed=main['action_differences']==main['joint_failures']==0 and better
 for rows in [fit,cal,val]:
  for row in rows:
   c=B.call_at(row['meta']['source_call']) if 'source_call' in row['meta'] else B.call_at(next(r['source'] for r in B.refs() if r['call_id']==row['meta']['call_id']));assert B.P.input_at(c,192)['features']==row['features'];checks['public_features']+=1
 verify_trained();save('replays.json',replays);save('summary.json',summaries);save('all_new_failures.json',[r for r in replays if r['method'].startswith('RF158') and r['bad_learned_stop']]);save('decision.json',dict(primary='RF158_192_CALIBRATED',passes_original_development_screen=passed,adopted=False,ready_to_scale=False,checkpoint_scan_forbidden=True));save('review.json',dict(passed=True,checks=dict(checks),fits=1,policy_replays=len(replays),new_physics=0,new_data=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(decision=read(O/'decision.json'),validation=[{k:r[k] for k in ['method','action_differences','queries','date_macro_query_reduction','joint_failures']} for r in vr]),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
