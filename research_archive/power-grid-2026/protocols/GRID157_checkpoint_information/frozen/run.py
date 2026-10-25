"""Read-only information available at checkpoints; no new learned stopping rule."""
from pathlib import Path
import collections,hashlib,importlib.util,json,shutil,sys
import numpy as np
spec=importlib.util.spec_from_file_location('g157_reference',Path('D:/Codex/GRL研究/work/grid156_date_holdout/run.py'));D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
B=D.B;R=D.R;W=Path(__file__).parent;O=R/'outputs/grid157_checkpoint_information';REPO=D.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID157_checkpoint_information'
read=D.read;sha=D.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def verify():
 D.verify()
 for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
 for p,h in read(O/'code_freeze.json').items():assert sha(R/p)==h,p
def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def prepare():
 assert not O.exists();O.mkdir(parents=True);D.verify();assert read(D.O/'review.json')['passed']
 save('design.json',dict(id='GRID157_CHECKPOINT_INFORMATION',population='Original12 fit dates only,386 prefixes and161 original search calls. No further cal/dev data accessed.',
  questions=['Which original18 features are constant or deterministic functions at128?','How does existing date-heldout score ranking differ at128,192,later? These labels/populations differ;no causal effect claim.','Keeping unchanged action-equality-and-query-saving target, how much saved-state hindsight opportunity remains if earliest eligible exit is192?'],
  groups=['128','192','>=256'],no_fits=True,no_new_scores=True,
  limits='Descriptive diagnosis using already computed heldout scores;not new model/threshold/exit-rule selection. Hindsight opportunity cannot be used online. Waiting changes evidence,labels and population;does not prove one extra batch causes predictive improvement.',
  prohibited=['new fitting','new predictions','new labels','threshold selection','new physics','new data','final8','paper','scale']))
 paths=[D.O/n for n in ['heldout_scores.npz','review.json','design.json','complete.json','replays.json']];paths +=[R/'outputs/grid150_linear_control/original_fit_rows.json.gz',B.O/'preprocessing.json',B.SOURCE/'call_ledger.json',O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('state.json',dict(stage='frozen'));archive('frozen');print('GRID157_FROZEN')
def analyse():
 verify();assert not (O/'review.json').exists();rows=D.rowfile(R/'outputs/grid150_linear_control/original_fit_rows.json.gz');p=np.load(D.O/'heldout_scores.npz')['scores'];features=read(B.O/'preprocessing.json')['features'];groups={'128':lambda n:n==128,'192':lambda n:n==192,'>=256':lambda n:n>=256};profiles=[];metrics=[];checks=collections.Counter()
 for name,pred in groups.items():
  ix=[i for i,r in enumerate(rows) if pred(r['meta']['visible_count'])];rr=[rows[i] for i in ix];profile={}
  for k in features:
   values=[r['features'][k] for r in rr];unique={json.dumps(v) for v in values};profile[k]=dict(unique_values=len(unique),missing=sum(v is None for v in values),constant=len(unique)==1,constant_value=values[0] if len(unique)==1 else None)
  profiles.append(dict(group=name,rows=len(rr),features=profile));metrics.append(dict(group=name,**D.metrics(rr,p[ix])))
 basis=['candidate_count','incumbent_reward','incumbent_rho','rho_limit','observed_admissible_fraction','best_two_reward_gap'];basischecks=[]
 for row in rows:
  f=row['features']
  if row['meta']['visible_count']!=128:continue
  g={k:f[k] for k in basis};m=f['candidate_count'];g.update(visible_count=128,queried_fraction=128/m,remaining_candidates=m-128,incumbent_available=True,prefix_available=True,prefix_safe=f['incumbent_rho']<.9,incumbent_changed_since_prefix=False,last_batch_reward_gain=0.,last_batch_repaired=False,improving_batch_fraction=0.,stale_batches=0,reward_gain_since_prefix=0.)
  assert g==f;basischecks.append(dict(sample_id=row['meta']['sample_id'],reconstructed=True));checks['six_field_exact_reconstruction']+=1
 save('entry128_basis_checks.json',basischecks);save('feature_profiles.json',profiles);save('depth_ranking.json',metrics)
 refs=B.refs(dates=read(D.O/'design.json')['dates']);records=[]
 for ref in refs:
  rr=[r for r in rows if r['meta']['call_id']==ref['call_id'] and r['target']['stop_label']];orig=ref['global_n']+ref['local_queries']
  for method,allowed in [('ALL_ELIGIBLE',rr),('EARLIEST192',[r for r in rr if r['meta']['visible_count']>=192])]:
   best=min(allowed,key=lambda r:(r['target']['stop_total_queries_known_now'],r['meta']['visible_count']),default=None);q=best['target']['stop_total_queries_known_now'] if best else orig;assert q<=orig
   records.append(dict(method=method,call_id=ref['call_id'],date=ref['date'],scenario=ref['scenario'],queries=q,original_queries=orig,action_match=True,stop_kind='HINDSIGHT_ONLY',model_decision_count=0,selected_checkpoint=best['meta']['visible_count'] if best else None))
 summaries={method:B.summarize([r for r in records if r['method']==method]) for method in ['ALL_ELIGIBLE','EARLIEST192']};replays=read(D.O/'replays.json');exits=collections.Counter(r['global_queries'] for r in replays if r['method']=='RF156_DATE_HELDOUT' and r['stop_kind']=='learned_early_stop')
 save('opportunity_calls.json',records);save('opportunity_summary.json',summaries);save('existing_exit_depths.json',dict(exits));verify();save('review.json',dict(passed=True,checks=dict(checks),rows=386,calls=161,new_fits=0,new_scores=0,new_labels=0,new_physics=0,final8_absent=True));save('state.json',dict(stage='analysed'));print(json.dumps(dict(metrics=metrics,constants128=[k for k,v in profiles[0]['features'].items() if v['constant']],opportunity={k:v['date_macro_query_reduction'] for k,v in summaries.items()}),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
