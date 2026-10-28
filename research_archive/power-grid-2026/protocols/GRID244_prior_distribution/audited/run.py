"""Read-only paired prior/source-support diagnosis; no new model or policy."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,collections
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid244_prior_distribution';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID244_prior_distribution'
P=R/'outputs/grid242_parent_prior_inputs';I=R/'outputs/grid236_shared_prior_inputs';C=R/'outputs/grid243_parent_coherent';H=R/'outputs/grid214_inner_head_generation';PLAN=R/'outputs/grid241_parent_prior_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p
 for root,name in [(P,'generated_freeze.json'),(I,'generated_freeze.json'),(C,'trained_freeze.json'),(C,'scored_freeze.json')]:
  for p,h in read(root/name).items():assert sha(root/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(C/'complete.json')['passed'];O.mkdir()
 save('design.json',dict(id='GRID244',scope='Read-only same-ID comparison of published parent12 and inner8 priors, all own12 rows, fixed fit8/cal4, originalSTOP/CONT, NETpositive/nonpositive. No model/scalar fit, no new headforward or new policy score.',
 comparisons='All rows before either b>0 selection; report counts and per-pool/seed/role/domain/label margin distributions, p/mu ratios and sign changes. Include parent-held originalSTOP source domain separately, not pooled with auxiliaryCONT. Labels are existing exposed development outcomes.',
 misses='22 published243 profitable misses, own-pool selected85 empirical positive constraints, all55 profitable208CONT and old239shared7; descriptive range checks, no new extremal alpha/threshold/strategy.',
 interpretation='Parent12 training includes own12 labels; inner8 excludes fixedcal4; both fit8. Sample count, head training and scaler differ. Paired contrasts describe support/ancestry, not a causal proof or whole-pipeline independent calibration. ParentNET/STOP stillouter12. Correlated prefixes/seeds/fold repetitions not independent.',
 forbidden='No fitted threshold/alpha, exception, seed selection, thirdpolicy, new data/physical/final8/paper/FRAME/dependencies. Retain230main,243seed1tradeoff,allcounterexamples. Original protocols unchanged.'))
 paths=[W/'run.py',O/'design.json',PLAN/'qualified_pools.json',H/'bundles.json',C/'gate_predictions.json',C/'profitable_misses.json',C/'effect_decomposition.json',C/'complete.json',C/'fit_receipts.json',R/'outputs/grid113_relational_learning/train_metadata.json']
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool']
  paths += [P/'public_full'/f'{pool}.npz',I/'public_full'/f'{pool}.npz',P/'calibration_inputs'/f'{pool}.npz',P/'fit_inputs'/f'{pool}.npz',P/'outer_parent_public'/f'{pool}.npz',C/'models'/f'{pool}.npz',C/'outer_predictions'/f'{pool}.npz']
 paths+=list((H/'labelled_samples').glob('*.npz'))
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID244_QUALIFIED')
def describe(x):
 x=np.asarray(x,dtype=float)
 if not len(x):return dict(n=0)
 return dict(n=len(x),mean=float(x.mean()),min=float(x.min()),q10=float(np.quantile(x,.1)),median=float(np.median(x)),q90=float(np.quantile(x,.9)),max=float(x.max()))
def group_summary(rows):
 b12=np.array([r['b12'] for r in rows]);b8=np.array([r['b8'] for r in rows]);d=b12-b8
 return dict(n=len(rows),b12=describe(b12),b8=describe(b8),delta=describe(d),parent_higher=int((d>0).sum()),parent_continue=int((b12>0).sum()),inner_continue=int((b8>0).sum()),parent_only_continue=int(((b12>0)&(b8<=0)).sum()),inner_only_continue=int(((b8>0)&(b12<=0)).sum()),event_p_delta=describe([r['p12']-r['p8'] for r in rows]),gross_logratio_delta=describe([r['loggross12']-r['loggross8'] for r in rows]))
def run():
 verify();assert not (O/'analysis_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'qualified/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('analysis_started.json',dict(source_published=head,new_model_fits=0,new_scalar_fits=0,new_forecasts=0,new_policy_scores=0));meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');bundles=read(H/'bundles.json');rows=[];poolsets={};target_access=[]
 for p in read(PLAN/'qualified_pools.json'):
  pool=p['pool'];f,s=[int(v[1:]) for v in pool.split('_')];t={}
  for r in bundles:
   if r['outer_fold']!=f or r['seed']!=s:continue
   z=np.load(H/'labelled_samples'/f"{r['bundle']}.npz");target_access.append(r['bundle'])
   for idx,n,st in zip(z['meta_ids'],z['actual_NET'],z['original_STOP_mask']):assert int(idx) not in t;t[int(idx)]=(float(n),bool(st))
  u=np.load(P/'public_full'/f'{pool}.npz');v=np.load(I/'public_full'/f'{pool}.npz')
  for name in ['ids','fit_role','cal_role','original_STOP','charge']:assert np.array_equal(u[name],v[name]),(pool,name)
  assert np.array_equal(u['X'][:,:41],v['X'][:,:41]);assert len(t)==len(u['ids']);pr=[]
  for j,i in enumerate(u['ids']):
   i=int(i);net,stop=t[i];assert stop==bool(u['original_STOP'][j]);m=meta[i];assert m['date'] not in p['outer_dates'];role='FIT8' if u['fit_role'][j] else 'CAL4';assert m['date'] in p['fit_dates' if role=='FIT8' else 'cal4_dates'];b12,b8=float(u['b'][j]),float(v['b'][j]);assert np.isfinite([b12,b8]).all();g12=float(u['event_p'][j]*u['mu'][j]);g8=float(v['event_p'][j]*v['mu4'][j]);assert g12>0 and g8>0
   r=dict(pool=pool,seed=s,id=i,sample_id=m['sample_id'],position=m['position'],date=m['date'],call=m['physical_state_group'],role=role,domain='STOP' if stop else 'CONT',positive=net>0,NET=net,b12=b12,b8=b8,p12=float(u['event_p'][j]),p8=float(v['event_p'][j]),loggross12=float(np.log(g12/u['charge'][j])),loggross8=float(np.log(g8/v['charge'][j])))
   rows.append(r);pr.append(r)
  poolsets[pool]=pr
 assert len(rows)==22653 and len(set(target_access))==36
 grouped=[]
 for role in ['FIT8','CAL4']:
  for domain in ['STOP','CONT']:
   for positive in [False,True]:
    subset=[r for r in rows if r['role']==role and r['domain']==domain and r['positive']==positive];grouped.append(dict(role=role,domain=domain,positive=positive,**group_summary(subset)))
 pools=[]
 for pool,rr in poolsets.items():
  for role in ['FIT8','CAL4']:
   for domain in ['STOP','CONT']:
    for positive in [False,True]:
     z=[r for r in rr if r['role']==role and r['domain']==domain and r['positive']==positive]
     pools.append(dict(pool=pool,role=role,domain=domain,positive=positive,**group_summary(z)))
 save('paired_rows.json',rows);save('group_summary.json',grouped);save('pool_summary.json',pools)
 # Range checks reproduce existing protected scores; no new policy is defined.
 gp=read(C/'gate_predictions.json');prof=[r for r in gp if r['base_continue'] and r['actual_next_NET']>0];assert len(prof)==55;outer=[];protection=[];checked=0
 for p in read(PLAN/'qualified_pools.json'):
  pool=p['pool'];f,s=[int(v[1:]) for v in pool.split('_')];c=np.load(P/'calibration_inputs'/f'{pool}.npz');model=np.load(C/'models'/f'{pool}.npz');theta=model['theta'];alpha=float(model['alpha'][0]);pos=c['y']>0;bpos=c['b'][pos];end=c['b']+np.sum(c['X']*theta[:-1],axis=1)+theta[-1];score=c['b']+alpha*(end-c['b']);assert np.all(score[pos]>=0);checked+=int(pos.sum());hp=np.load(P/'outer_parent_public'/f'{pool}.npz');sp=np.load(C/'outer_predictions'/f'{pool}.npz');lookup={int(i):j for j,i in enumerate(hp['ids'])};source={r['id']:r for r in poolsets[pool]}
  for i,b,e,sc,fit in zip(c['ids'][pos],bpos,end[pos],score[pos],c['fit_role'][pos]):
   protection.append(dict(pool=pool,id=int(i),sample_id=meta[int(i)]['sample_id'],call=meta[int(i)]['physical_state_group'],role='FIT8' if fit else 'CAL4',b12=float(b),endpoint=float(e),protected_score=float(sc),alpha=alpha))
  for r in prof:
   if r['fold']!=f or r['seed']!=s:continue
   ids=[i for i,j in lookup.items() if meta[i]['sample_id']==r['sample_id'] and meta[i]['position']==r['position']];assert len(ids)==1;i=ids[0];j=lookup[i];b=float(hp['b'][j]);e=b+float(np.sum(hp['X'][j]*theta[:-1]))+float(theta[-1]);published=float(sp['logit'][j]);assert abs(published-(b+alpha*(e-b)))<1e-12;assert bool(sp['continue_'][j])==r['continue_'];z=dict(r,pool=pool,id=i,b12=b,endpoint=e,score=published,alpha=alpha,protected_positive_count=len(bpos),below_protected_b_range=bool(len(bpos) and b<min(bpos)),above_protected_b_range=bool(len(bpos) and b>max(bpos)),protected_b=describe(bpos),protected_endpoint=describe(end[pos]),empirical_not_population_guarantee=True)
   outer.append(z)
 assert checked==85 and len(outer)==55 and sum(not r['continue_'] for r in outer)==22
 # Same originalSTOP source domain only; auxiliary states excluded in this contrast.
 support=[]
 for seed in range(3):
  for label in [False,True]:
   for role in ['FIT8','CAL4']:
    z=[r['b12'] for r in rows if r['seed']==seed and r['role']==role and r['domain']=='STOP' and r['positive']==label];support.append(dict(seed=seed,label=label,role=role,b12=describe(z)))
   z=[r for r in gp if r['seed']==seed and (r['actual_next_NET']>0)==label];margins=[]
   for pool in [p['pool'] for p in read(PLAN/'qualified_pools.json') if p['pool'].endswith(f'_s{seed}')]:
    h=np.load(P/'outer_parent_public'/f'{pool}.npz');margins.extend(float(b) for i,b in zip(h['ids'],h['b']) if (next(r for r in z if r['sample_id']==meta[int(i)]['sample_id'] and r['position']==meta[int(i)]['position']) if any(r['sample_id']==meta[int(i)]['sample_id'] and r['position']==meta[int(i)]['position'] for r in z) else None))
   support.append(dict(seed=seed,label=label,role='OUTER4_OLD_STOP',b12=describe(margins)))
 save('positive_support.json',outer);save('protected_positive_rows.json',protection);save('parent_support_domains.json',support);save('label_access.json',dict(existing_development_target_files=target_access,final8_read=False))
 misses=[r for r in outer if not r['continue_']];summary=dict(paired_source_rows=len(rows),pools=12,originalSTOP_selected_protected_positive85=checked,outer_profitable55=len(outer),outer_missed22=len(misses),misses_below_protected_b=sum(r['below_protected_b_range'] for r in misses),misses_above_protected_b=sum(r['above_protected_b_range'] for r in misses),misses_inside_protected_b=sum(not(r['below_protected_b_range'] or r['above_protected_b_range']) for r in misses),all7_old239_misses_still_missed=all(not r['new243_continue'] for r in read(C/'effect_decomposition.json')['old239_shared_miss_status']),newfits=0,newforecasts=0,newpolicyscores=0)
 save('summary.json',summary);verify();save('review.json',dict(passed=True,all22653sameids_roles_STOP_raw41_charge_checked=True,all36oldtargets_joined_once_perpool=True,all85savedpositiveprotections_reproduced=True,all55profitable_existing_predictions_checked=True,all22misses_retained=True,input_modelhashes_unchanged=True,new_fits=0,new_forecasts=0,new_policy_scores=0));save('complete.json',dict(passed=True,summary=summary));archive('analysed');print('GRID244_ANALYSED',json.dumps(dict(summary=summary,groups=grouped,support=support),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
