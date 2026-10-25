"""One inherited regularized logistic baseline for the unchanged global compression task."""
from pathlib import Path
import collections,datetime,hashlib,json,pickle,shutil,subprocess,sys,time,warnings
import numpy as np
sys.path.insert(0,'D:/Codex/GRL研究/work/grid144_class_mass')
import support as S
B=S.B;R=B.R;W=Path(__file__).parent;O=R/'outputs/grid150_linear_control';REPO=B.REPO
A=REPO/'research_archive/power-grid-2026/protocols/GRID150_linear_control';PREV=R/'outputs/grid146_single_entry'
read=B.read;sha=B.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
 S.verify_trained();S.diagnostic_api().verify()
 for n in ['source_hashes.json','code_freeze.json']:
  for p,h in read(O/n).items():assert sha(R/p)==h,p
def verify_trained():
 verify()
 for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def sigmoid(z):
 z=np.asarray(z,float);out=np.empty_like(z);mask=z>=0;out[mask]=1/(1+np.exp(-z[mask]));e=np.exp(z[~mask]);out[~mask]=e/(1+e);return out
def manual_scores(model,x):return sigmoid(np.asarray(x)@np.asarray(model['coef'])+model['intercept'])
def smap(rows,p):return {(r['meta']['call_id'],r['meta']['visible_count']):float(v) for r,v in zip(rows,p)}
def ranks(rows,p,tau):return S.diagnostic_api().rank_metrics(rows,np.asarray(p),tau)
def calibrate(refs,sm):
 tau=.5;history=[];initial=None
 for _ in range(200):
  rr=[B.replay(r,'MODEL',tau,sm) for r in refs]
  if initial is None:initial=rr
  bad=[r for r in rr if r['bad_learned_stop']]
  history.append(dict(threshold=tau,summary=B.summarize(rr),bad_stops=[dict(call_id=r['call_id'],score=r['stopped_score'],selected=r['selected'],original=r['original_selected']) for r in bad]))
  if not bad:return tau,history,initial,rr
  nxt=max(r['stopped_score'] for r in bad);assert nxt>tau;tau=nxt
 raise AssertionError('Preserve failed calibration;no retry')
def note(stage,text):
 entry=f'\n\n## 2026-10-25 — GRID150 {stage}\n\n{text}\n'
 for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
  if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
  else:old+=entry
  p.write_text(old,encoding='utf8',newline='\n')
 (REPO/'docs/NEXT_STEPS.md').write_text(f'# Next steps — GRID150 {stage}\n\n1. {text}\n2. 原18项/标签/386权重/12-4-4日期/搜索校准保持；不扩物理12项、网络或数据，不改目标救结果。\n3. 权重阈值评分前发布，保留全部旧对照和固定0.5读数，最终8封存；一次拟合不补训。\n',encoding='utf8',newline='\n')

def prepare():
 assert not O.exists();O.mkdir();S.verify_trained()
 assert read(R/'outputs/grid149_input_inventory/native_command_audit.json')['passed']
 d=read(B.O/'design.json');params=read(R/'outputs/grid101_gain_learning/design.json')['logistic']
 assert params==dict(C=1.,solver='lbfgs',max_iter=2000,random_state=20261022,class_weight=None)
 design=dict(id='GRID150_MATCHED_GLOBAL_LOGISTIC',utc=stamp(),authorization='User requests autonomous justified progress with progress and pre-fit notice.',
  question='Does inherited L2 regularized linear classifier generalize the same18-summary global compression target better than the existing forests?',
  features=d['features'],model_parameters=params,penalty='Installed1.7.2 defaultL2,intercept enabled,tol1e-4;record full estimator params at fit.',
  fit_dates=d['fit_dates'],calibration_dates=d['calibration_dates'],validation_dates=d['validation_dates'],
  raw_rows='Exact original386fit/198cal/142dev,all18values+18missing unchanged;old models never refit.',
  preprocessing='Copy original fit-only medians;StandardScaler mean/variance from386fit transformed rows only,unweighted;sample weights affect classifier loss only.',
  weights='Exact RF140 hierarchical date/call/prefix386 weights,no class balancing.',
  target=d['task'],search='Original multi-exit128/64 global checkpoints and unchanged conditional completion. Original force stops/no-incumbent behavior unchanged.',
  calibration='Same empirical zero-joint-failure from0.5,strict> and raise to highest wrong learned-stop score,replay full searches;not formal future guarantee.',
  primary='LR150_CALIBRATED',secondary='LR150_FIXED_0.5 retained',
  contrasts='Classifier family plus required fit-only standardization;not same-parameter-count or isolated architecture ablation. Logistic on old local target is not this control.',
  cap_s=120,fit_limit=1,validation_order='Freeze/push classifier,scaler,training scores and both readings/cutoff before new dev scoring.',
  screen='All62dev calls preserve original action,no non-saving learned exits;date macro query reduction>CAP256,each date queries<=CAP256 and at least1date strictly fewer. No adoption/scale/physics automatic.',
  prohibited=['second fit','C/solver/classweight/seed/threshold/feature grid','validation-selected cutoff','new label/quality margin','new state/scenario','physics/test8','network/RL/paper/scale'])
 save('design.json',design)
 for n in ['fit_rows.json.gz','calibration_rows.json.gz','fit_sample_weights.json','preprocessing.json']:shutil.copy2(B.O/n,O/('original_'+n))
 fit=S.rowfile(B.O/'fit_rows.json.gz');cal=S.rowfile(B.O/'calibration_rows.json.gz');assert S.public_checks(fit+cal)==584
 save('preflight.json',dict(passed=True,public_suffix_checks=584,features=36,new_fits=0,validation_scored=False,old_model_refits=0))
 paths=[O/'design.json',R/'outputs/grid101_gain_learning/design.json',R/'outputs/grid149_input_inventory/GRID149_report.md',R/'outputs/grid149_input_inventory/native_command_audit.json']
 paths+=[B.O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','fit_sample_weights.json','preprocessing.json']]
 paths+=[PREV/n for n in ['all_search_replays.json','summary.json','cutoff.json','final_verification.json']]
 paths+=[R/'work/grid140_global_compression/common.py',R/'work/grid144_class_mass/support.py']
 paths+=[O/('original_'+n) for n in ['fit_rows.json.gz','calibration_rows.json.gz','fit_sample_weights.json','preprocessing.json']]
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
 verify();save('state.json',dict(stage='frozen_before_one_fit'));archive('frozen')
 note('frozen','GRID149确认当前全局动作保持缺同输入低容量对照。冻结一次原18项逻辑回归：继承C1/lbfgs/2000/种子，无类别平衡，原386权重/12-4-4日期/标签/搜索校准保持；拟合行标准化。584后缀预检通过，尚未训练，训练前通知。')
 print('GRID150_FROZEN')

def fit():
 verify();assert not (O/'fit_started.json').exists();d=read(O/'design.json')
 rows=S.rowfile(O/'original_fit_rows.json.gz');cal=S.rowfile(O/'original_calibration_rows.json.gz');pre=read(O/'original_preprocessing.json')
 x=B.transform(rows,pre['features'],pre['medians']);xc=B.transform(cal,pre['features'],pre['medians']);y=np.asarray([r['target']['stop_label'] for r in rows],int)
 weights=np.asarray([r['weight'] for r in read(O/'original_fit_sample_weights.json')])
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.exceptions import ConvergenceWarning
 import sklearn
 scaler=StandardScaler().fit(x);xs=scaler.transform(x);cs=scaler.transform(xc)
 save('scaler.json',dict(mean=scaler.mean_.tolist(),variance=scaler.var_.tolist(),scale=scaler.scale_.tolist(),fit_rows=386,unweighted_fit=True))
 model=LogisticRegression(**d['model_parameters']);save('fit_started.json',dict(utc=stamp(),fits_permitted=1));t=time.perf_counter()
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');model.fit(xs,y,sample_weight=weights)
 elapsed=time.perf_counter()-t
 with (O/'model.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
 export=dict(coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),classes=model.classes_.tolist(),full_parameters=model.get_params())
 save('model_export.json',export)
 converged=not any(isinstance(w.message,ConvergenceWarning) for w in ww) and int(model.n_iter_[0])<d['model_parameters']['max_iter']
 save('fit_receipt.json',dict(new_fits=1,old_fits=0,rows=386,dimensions=36,elapsed_s=elapsed,iterations=int(model.n_iter_[0]),converged=converged,warnings=[str(w.message) for w in ww],sklearn=sklearn.__version__,validation_scored=False))
 assert elapsed<=d['cap_s'] and converged,'Preserve fitted artifact and stop;no retry'
 pf=model.predict_proba(xs)[:,1];pc=model.predict_proba(cs)[:,1]
 assert max(abs(manual_scores(export,xs)-pf))<1e-10 and max(abs(manual_scores(export,cs)-pc))<1e-10
 np.savez_compressed(O/'train_scores.npz',fit=pf,calibration=pc)
 tau,h,initial,final=calibrate(B.refs('train',set(d['calibration_dates'])),smap(cal,pc))
 save('calibration_history.json',h);save('calibration_fixed_calls.json',initial);save('calibration_final_calls.json',final)
 save('cutoff.json',dict(primary_threshold=tau,fixed_threshold=.5,strict_greater=True,empirical_only=True,validation_scored=False))
 save('fit_calibration_metrics.json',[dict(partition=p,reading=name,**ranks(rr,score,cut)) for p,rr,score in [('fit',rows,pf),('calibration',cal,pc)] for name,cut in [('fixed',.5),('calibrated',tau)]])
 names=['design.json','scaler.json','model.pkl','model_export.json','fit_receipt.json','train_scores.npz','calibration_history.json','calibration_fixed_calls.json','calibration_final_calls.json','cutoff.json','fit_calibration_metrics.json']
 save('trained_freeze.json',{n:sha(O/n) for n in names});verify_trained();save('state.json',dict(stage='trained_before_validation'));archive('trained')
 note('trained',f"一次逻辑回归拟合完成，{int(model.n_iter_[0])}迭代收敛，{elapsed:.4f}秒；经验阈值{tau:.17g}。模型/标准化/分数/阈值封存，开发验证尚未评分；不重训。")
 print(json.dumps(dict(fit=read(O/'fit_receipt.json'),cutoff=tau,calibration=h[-1]['summary']),ensure_ascii=False))

def publication():
 verify_trained();local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
 rel=(A/'trained/trained_freeze.json').relative_to(REPO).as_posix();data=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=REPO);assert hashlib.sha256(data).hexdigest()==sha(O/'trained_freeze.json')
 save('prevalidation_publication.json',dict(passed=True,local_sha=local,remote_sha=remote,trained_freeze_sha256=sha(O/'trained_freeze.json')));print('GRID150_PREVALIDATION_PUBLISHED',local)

def validate():
 verify_trained();pub=read(O/'prevalidation_publication.json');assert pub['passed'] and pub['trained_freeze_sha256']==sha(O/'trained_freeze.json');assert not (O/'validation_started.json').exists()
 save('validation_started.json',dict(utc=stamp(),published_sha=pub['local_sha'],fits_so_far=1))
 d=read(O/'design.json');pre=read(O/'original_preprocessing.json');sc=read(O/'scaler.json');cut=read(O/'cutoff.json')['primary_threshold'];val=B.eligible_data('validation');assert len(val)==142 and S.public_checks(val)==142
 x=B.transform(val,pre['features'],pre['medians']);x=(x-np.asarray(sc['mean']))/np.asarray(sc['scale'])
 with (O/'model.pkl').open('rb') as f:model=pickle.load(f)
 pv=model.predict_proba(x)[:,1];assert max(abs(pv-manual_scores(read(O/'model_export.json'),x)))<1e-10
 np.savez_compressed(O/'validation_scores.npz',scores=pv)
 save('validation_metrics.json',[dict(reading=name,**ranks(val,pv,t)) for name,t in [('fixed',.5),('calibrated',cut)]])
 fit=S.rowfile(O/'original_fit_rows.json.gz');cal=S.rowfile(O/'original_calibration_rows.json.gz');z=np.load(O/'train_scores.npz');sm=smap(fit+cal+val,np.concatenate([z['fit'],z['calibration'],pv]))
 prior=read(PREV/'all_search_replays.json');old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in prior};methods=list(dict.fromkeys(r['method'] for r in prior));assert len(methods)==15
 records=[];sums=[]
 for part,refs in [('fit',B.refs('train',set(d['fit_dates']))),('calibration',B.refs('train',set(d['calibration_dates']))),('validation',B.refs('validation'))]:
  for name in methods+['LR150_FIXED_0.5','LR150_CALIBRATED']:
   rr=[]
   for ref in refs:
    r=B.replay(ref,'MODEL',.5 if name.endswith('0.5') else cut,sm) if name.startswith('LR150') else dict(old[part,name,ref['call_id']])
    rr.append(dict(r,evaluation_partition=part,method=name))
   records+=rr;sums.append(dict(method=name,evaluation_partition=part,**B.summarize(rr),joint_failures=sum(r['bad_learned_stop'] for r in rr)))
 save('all_search_replays.json',records);save('summary.json',sums)
 metric=lambda name:next(s for s in sums if s['evaluation_partition']=='validation' and s['method']==name)
 a=metric('LR150_CALIBRATED');b=metric('CAP256');bd={r['date']:r for r in b['by_date']};dd=[dict(date=r['date'],action_differences=r['action_differences'],queries=r['queries'],cap256_queries=bd[r['date']]['queries']) for r in a['by_date']]
 passed=a['action_differences']==0 and a['joint_failures']==0 and a['date_macro_query_reduction']>b['date_macro_query_reduction'] and all(r['queries']<=r['cap256_queries'] for r in dd) and any(r['queries']<r['cap256_queries'] for r in dd)
 save('decision.json',dict(primary='LR150_CALIBRATED',passes_development_screen=passed,ready_for_frame_coverage_proposal=passed,ready_to_scale=False,adopted=False,
  action_differences=a['action_differences'],joint_failures=a['joint_failures'],query_reduction_macro=a['date_macro_query_reduction'],cap256_macro=b['date_macro_query_reduction'],old_rf140_macro=metric('RF140_CALIBRATED')['date_macro_query_reduction'],per_date=dd,
  new_fits=1,new_physical=0,final8_absent=True,scope='Already exposed FULL-teacher states,not independent test/actualFRAME/whole-week quality or measured time.'))
 verify_trained();save('state.json',dict(stage='validated_pending_audit'));print(json.dumps(read(O/'decision.json'),ensure_ascii=False))

def audit():
 verify_trained();assert read(O/'state.json')['stage']=='validated_pending_audit'
 d=read(O/'design.json');pre=read(O/'original_preprocessing.json');sc=read(O/'scaler.json');export=read(O/'model_export.json');cut=read(O/'cutoff.json')['primary_threshold']
 fit=S.rowfile(O/'original_fit_rows.json.gz');cal=S.rowfile(O/'original_calibration_rows.json.gz');val=B.eligible_data('validation');rows=fit+cal+val
 x=B.transform(rows,pre['features'],pre['medians']);assert max(abs(x[:386].mean(axis=0)-sc['mean']))<1e-10 and max(abs(x[:386].var(axis=0)-sc['variance']))<1e-10
 xs=(x-np.asarray(sc['mean']))/np.asarray(sc['scale']);z=np.load(O/'train_scores.npz');pv=np.load(O/'validation_scores.npz')['scores'];pp=np.concatenate([z['fit'],z['calibration'],pv]);manual=manual_scores(export,xs)
 assert max(abs(manual-pp))<1e-10;assert S.public_checks(rows)==726
 weights=np.asarray([r['weight'] for r in read(O/'original_fit_sample_weights.json')]);y=np.asarray([r['target']['stop_label'] for r in fit],float)
 gradient=(xs[:386].T@(weights*(z['fit']-y))+np.asarray(export['coef'])/d['model_parameters']['C'])/weights.sum();intercept_grad=float(np.sum(weights*(z['fit']-y))/weights.sum())
 assert max(abs(gradient).max(),abs(intercept_grad))<1.1e-4
 counts=collections.Counter(r['meta']['call_id'] for r in fit);dc=collections.Counter(r['meta']['date'] for r in fit)
 calls_per_date={date:len({r['meta']['call_id'] for r in fit if r['meta']['date']==date}) for date in d['fit_dates']}
 expected=np.asarray([1/(12*calls_per_date[r['meta']['date']]*counts[r['meta']['call_id']]) for r in fit]);expected*=386/expected.sum();assert np.max(abs(expected-weights))<1e-12
 sm=smap(rows,pp);tau,history,initial,final=calibrate(B.refs('train',set(d['calibration_dates'])),sm);assert tau==cut and history==read(O/'calibration_history.json')
 previous={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(PREV/'all_search_replays.json')};records=read(O/'all_search_replays.json');assert len(records)==305*17
 refs={r['call_id']:r for r in B.refs()};checks=collections.Counter();errors=[]
 for r in records:
  ref=refs[r['call_id']];c=B.call_at(ref['source']);n=r['global_queries'];first=B.P.best(c['outcomes'][:128]);best=B.P.best(c['outcomes'][:n]);extra=[]
  if best is not None and n<len(c['order']) and (first is None or first['rho']>=.9 or B.P.bid(best)!=B.P.bid(first)):
   station=B.subids()[best['pool_id']];ids={int(v) for k,v in c['aliases'].items() if B.subids()[int(k)]==station};extra=[v for v in c['outcomes'][n:] if v['pool_id'] in ids]
  assert B.P.bid(B.P.best(c['outcomes'][:n]+extra))==r['selected'] and n+len(extra)==r['queries'];checks['complete_searches']+=1
  if r['method'].startswith('LR150'):
   threshold=.5 if r['method'].endswith('0.5') else cut
   for q in r['model_decisions']:assert q['score']==sm[r['call_id'],q['visible_count']] and q['stop']==(q['score']>threshold);checks['logistic_decisions']+=1
   if r['bad_learned_stop']:errors.append(r)
  else:
   old=previous[r['evaluation_partition'],r['method'],r['call_id']];assert all(r[k]==v for k,v in old.items());checks['old_reference_rows_unchanged']+=1
 save('all_new_model_failures.json',errors)
 save('review.json',dict(passed=True,new_fits=1,old_fits=0,original_rows_labels_features_weights_same=True,fit_only_scaler=True,public_suffix_checks=726,
  manual_sigmoid_checks=726,max_score_error=float(max(abs(manual-pp))),fit_optimality_gradient_inf=float(max(abs(gradient).max(),abs(intercept_grad))),
  original_calibration_reexecuted=True,frozen_model_cutoff_published_before_validation=True,checks=dict(checks),no_new_physical=True,final8_absent=True))
 save('state.json',dict(stage='audited'));verify_trained();print(json.dumps(read(O/'review.json'),ensure_ascii=False))

if __name__=='__main__':{'prepare':prepare,'fit':fit,'publication':publication,'validate':validate,'audit':audit}[sys.argv[1]]()
