"""Frozen scores on existing FRAME-state prefixes. No refit or physical loop."""
from pathlib import Path
import collections,copy,hashlib,importlib.util,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid155_frozen_path_replay';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID155_frozen_path_replay'
spec=importlib.util.spec_from_file_location('g155_reference',R/'work/grid140_global_compression/common.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
spec=importlib.util.spec_from_file_location('g155_identification',R/'work/grid154_observed_label_coverage/run.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
MODELS={'RF140':'grid140_global_compression','LR150':'grid150_linear_control','MEAN151':'grid151_fixed_score_mean'}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
 L.verify()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def prepare():
 assert not O.exists();O.mkdir(parents=True);L.verify();assert read(L.O/'review.json')['passed']
 cutoffs={m:read(R/'outputs'/d/'cutoff.json')['primary_threshold'] for m,d in MODELS.items()}
 save('design.json',dict(id='GRID155_FROZEN_SAVED_FRAME_REPLAY',population='All30 existing original FRAME saved calls, all78 eligible checkpoints, four exposed development dates. No new physics or fitting.',
  policies=['FRAME','CAP256','RF140','LR150','MEAN151'],cutoffs=cutoffs,strict_greater=True,
  decision='Native/exported old scores only from original18 prefix features plus frozen preprocessing. First eligible prefix over old cutoff stops; otherwise original search unchanged. Original same-station completion, unknown feedback not fetched.',
  scoring='Extract inputs without labels. Freeze whole plan/cutoffs before scoring. Scores compared with native frozen models and previously scored exact teacher contexts. Hidden suffix poisoned.',
  outcomes='Known exact action agreement, differences, unresolved cases, non-saving exits separately. Count all30. Query count can be known even if selection unknown; do not silently fallback UNKNOWN to original and claim success.',
  screen='All30 known action agreement;zero unknown;zero non-saving early exits;date macro reduction greater than CAP256 and each date queries no greater. Development screen only, never independent confirmation.',
  limits='Saved state comparison; after a changed action later saved states no longer represent that policy trajectory. No fullweek quality/cost/control time inference. No final8, fit, cutoff change, collection, paper or scale.'))
 paths=[L.O/n for n in ['observed_targets.json','call_summary.json','source_hashes.json','review.json','complete.json']]
 for folder in ['grid140_global_compression','grid150_linear_control']:
  f=R/'outputs'/folder;paths +=[f/n for n in ['model.pkl','model_export.json','cutoff.json','validation_scores.npz']]
 paths +=[R/'outputs/grid140_global_compression/preprocessing.json',R/'outputs/grid150_linear_control/scaler.json',R/'outputs/grid150_linear_control/original_preprocessing.json',R/'outputs/grid151_fixed_score_mean/cutoff.json',O/'design.json']
 save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID155_FROZEN')
def analyse():
 verify();assert not (O/'review.json').exists();checks=collections.Counter();d=read(O/'design.json');targets=read(L.O/'observed_targets.json');inputs=[]
 for row in targets:
  c=read(R/row['source']);n=row['visible_count'];f=B.P.input_at(c,n);poison=copy.deepcopy(c)
  for x in poison['outcomes'][n:]:x.update(reward=-12345.,rho=9876.,strict_admissible=False,done=True)
  assert B.P.input_at(poison,n)==f;checks['suffix_poison']+=1
  inputs.append(dict(call_id=row['call_id'],date=row['date'],scenario=row['scenario'],visible_count=n,features=f['features']))
 save('public_inputs.json',inputs);pre=read(R/'outputs/grid140_global_compression/preprocessing.json');lrpre=read(R/'outputs/grid150_linear_control/original_preprocessing.json');assert pre==lrpre
 x=B.transform(inputs,pre['features'],pre['medians']);rf=B.exported_scores(read(R/'outputs/grid140_global_compression/model_export.json'),x)
 sc=read(R/'outputs/grid150_linear_control/scaler.json');xs=(x-np.asarray(sc['mean']))/np.asarray(sc['scale']);lm=read(R/'outputs/grid150_linear_control/model_export.json');z=xs@np.asarray(lm['coef'])+lm['intercept'];lr=1/(1+np.exp(-z));mean=(rf+lr)/2
 with (R/'outputs/grid140_global_compression/model.pkl').open('rb') as f:native_rf=pickle.load(f)
 with (R/'outputs/grid150_linear_control/model.pkl').open('rb') as f:native_lr=pickle.load(f)
 assert np.max(np.abs(native_rf.predict_proba(x)[:,1]-rf))<1e-10;assert np.max(np.abs(native_lr.predict_proba(xs)[:,1]-lr))<1e-10;checks['native_score_rows']=len(inputs)*2
 val=B.eligible_data('validation');val_lookup={(r['target']['call_id'],r['meta']['visible_count']):i for i,r in enumerate(val)}
 oldrf=np.load(R/'outputs/grid140_global_compression/validation_scores.npz')['scores'];oldlr=np.load(R/'outputs/grid150_linear_control/validation_scores.npz')['scores'];cov={r['source']:r for r in L.cases()}
 for i,row in enumerate(targets):
  for tid in cov[row['source']]['teacher_exact_hits']:
   k=val_lookup[(tid,row['visible_count'])];assert val[k]['features']==inputs[i]['features'];assert abs(oldrf[k]-rf[i])<1e-10 and abs(oldlr[k]-lr[i])<1e-10;checks['old_exact_context_scores']+=1
 smap={(r['call_id'],r['visible_count']):dict(RF140=float(rf[i]),LR150=float(lr[i]),MEAN151=float(mean[i])) for i,r in enumerate(inputs)}
 save('scores.json',[dict(call_id=r['call_id'],visible_count=r['visible_count'],**smap[(r['call_id'],r['visible_count'])]) for r in inputs]);replays=[]
 # Labels are accessed only after each policy's first-hit decision is fixed.
 for csum in read(L.O/'call_summary.json'):
  cr=[r for r in targets if r['call_id']==csum['call_id']];assert [r['visible_count'] for r in cr]==sorted(r['visible_count'] for r in cr)
  for method in ['FRAME','CAP256','RF140','LR150','MEAN151']:
   stop=None;visited=[]
   for r in cr:
    n=r['visible_count']
    if method=='FRAME':break
    score=smap[(r['call_id'],n)].get(method)
    yes=n>=256 if method=='CAP256' else score>d['cutoffs'][method];visited.append(dict(visible_count=n,score=score,stop=yes))
    if yes:stop=r;break
   if stop is None:q=csum['original_queries'];match=True;unknown=False;nonsaving=False;n=None;reason='ORIGINAL_SEARCH'
   else:
    q=stop['total_queries_if_stopped'];unknown=bool(stop['unobserved_completion_ids']);nonsaving=q>=csum['original_queries'];n=stop['visible_count'];reason=stop['reason']
    # NON_SAVING target did not need action calculation, but action agreement still must be audited if feedback exists.
    c=read(R/stop['source']);truth={r['pool_id']:r for r in c['outcomes']}
    if unknown:match=None
    else:match=B.P.bid(B.P.best([truth[i] for i in c['order'][:n]+stop['required_completion_ids']]))==c['chosen_pool_id']
   replays.append(dict(method=method,call_id=csum['call_id'],date=csum['date'],scenario=csum['scenario'],original_queries=csum['original_queries'],queries=q,stop_at=n,action_match=match,unknown=unknown,non_saving_early_exit=nonsaving,reason=reason,visited=visited))
 sums=[]
 for method in ['FRAME','CAP256','RF140','LR150','MEAN151']:
  rr=[r for r in replays if r['method']==method];ds=[]
  for date in sorted({r['date'] for r in rr}):
   aa=[r for r in rr if r['date']==date];qs=sum(r['queries'] for r in aa);oq=sum(r['original_queries'] for r in aa);ds.append(dict(date=date,queries=qs,original_queries=oq,reduction=1-qs/oq,matches=sum(r['action_match'] is True for r in aa),differences=sum(r['action_match'] is False for r in aa),unknown=sum(r['unknown'] for r in aa)))
  sums.append(dict(method=method,calls=len(rr),action_matches=sum(r['action_match'] is True for r in rr),action_differences=sum(r['action_match'] is False for r in rr),unknown=sum(r['unknown'] for r in rr),stops=sum(r['stop_at'] is not None for r in rr),non_saving_early_exits=sum(r['non_saving_early_exit'] for r in rr),queries=sum(r['queries'] for r in rr),date_macro_reduction=sum(r['reduction'] for r in ds)/4,by_date=ds))
 cap=next(x for x in sums if x['method']=='CAP256');dec=[]
 for x in sums[2:]:
  better=x['date_macro_reduction']>cap['date_macro_reduction'] and all(a['queries']<=b['queries'] for a,b in zip(x['by_date'],cap['by_date']))
  passed=x['action_matches']==30 and x['unknown']==0 and x['non_saving_early_exits']==0 and better;dec.append(dict(method=x['method'],passes_saved_state_development_screen=passed,adopted=False,ready_to_scale=False))
 verify();save('replays.json',replays);save('summary.json',sums);save('decision.json',dict(models=dec,new_fits=0,adopted=False,ready_to_scale=False));save('review.json',dict(passed=True,checks=dict(checks),new_fits=0,new_physics=0,new_cohort=0,score_rows=len(inputs),policy_call_replays=len(replays),final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(sums,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
