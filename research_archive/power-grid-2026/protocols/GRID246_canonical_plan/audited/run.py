"""Train-only representation/objective qualification; zero optimizer/held access."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid246_canonical_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID246_canonical_plan';Q=R/'outputs/grid242_parent_prior_inputs';P=R/'outputs/grid243_parent_coherent';PLAN=R/'outputs/grid241_parent_prior_plan';META=R/'outputs/grid113_relational_learning/train_metadata.json'
DROP=[14,18,19];KEEP=[j for j in range(42) if j not in DROP]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in read(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 if phase=='qualified':
  for folder in ['fit_inputs','calibration_inputs']:shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(R/'outputs/grid245_direction_support/complete.json')['passed'];O.mkdir()
 save('design.json',dict(id='GRID246',scope='Qualificationonly: all41featuredefinitions and OWNfit8 linearalgebra/risk/offsetaudit. No optimizerfit, new model/headforward, held score/target access or new policy.',
 semantic_pairs=[dict(drop=14,keep=15,relation='incumbent_reward=2-incumbent_rho up to float32 choice/ties'),dict(drop=18,keep=20,relation='admissible_reward_mean=2-admissible_rho_mean up to float32'),dict(drop=19,keep=21,relation='admissible_reward_std=admissible_rho_std up to float32')],
 selection='Three pairs fixed by publicMaxRhoReward and featuredefinitions, retainphysicalrho columns by convention. No correlation/heldoutcome/feature-ranking selection; all other38public+b columns retained unchanged. Notdropother empiricalconstant or dependentcolumns.',
 next_unique_candidate='GRID247 canonical38public+originalb12 =>39inputs40parameters. Fit8rows/weights/labels and union12STOP85positiveprotection IDENTICAL243. Same219meanNLL+unitL2all40 and221 maximalpositive step; nativeparententryunchanged. Exactly11linear+11scalar/600sonce,1KEEP. Primarymatched243;230historicalreference. No offsettransform,clip,OODgate,newfee orthreshold.',
 regularization_change='On exactly opposite/equal standardizedduplicatepairs, oldminimum L2 for combinedv is v^2/4; canonical singlecolumn penalty v^2/2. Explicit doubledfunctiondirectionpenalty,notonlycompression/notstrictsamecapacity andnot guaranteedimprovement. Floatingnearrelations not exacteveryfutureinput.',
 constraints='No newouter4 scores/readoutcomes, nofinal8/physics/newdata/paper/FRAME/install. 247source firstpublished,allweights/stepsbeforeheldscores,allpredictionsbeforelabeljoin. Allseeds/adverse kept;nofeaturesubsets/regularizationsearch.'))
 paths=[W/'run.py',O/'design.json',R/'work/grid101_gain_learning/features.py',R/'work/grid102_search_rl/observations.py',R/'work/grid02/ljn_nn/modules/rewards.py',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',R/'outputs/grid245_direction_support/feature_names.json',PLAN/'qualified_pools.json',META]
 for r in read(PLAN/'qualified_pools.json'):paths += [Q/'fit_inputs'/f"{r['pool']}.npz",Q/'calibration_inputs'/f"{r['pool']}.npz",R/r['residual_public41_scaler'],P/'models'/f"{r['pool']}.npz"]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID246_SOURCE_FROZEN')
def stats(x):
 return dict(min=float(np.min(x)),max=float(np.max(x)),mean=float(np.mean(x)))
def qualify():
 verify();assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('qualification_started.json',dict(source_published=head,optimizercalls=0,heldread=False));meta=read(META);names=read(R/'outputs/grid245_direction_support/feature_names.json');assert [names[k] for k in DROP]==['incumbent_reward','admissible_reward_mean','admissible_reward_std'];(O/'fit_inputs').mkdir();(O/'calibration_inputs').mkdir();records=[];checks=[];nfit=0;source_access=[]
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool'];z=np.load(Q/'fit_inputs'/f'{pool}.npz');c=np.load(Q/'calibration_inputs'/f'{pool}.npz');s=np.load(R/r['residual_public41_scaler']);model=np.load(P/'models'/f'{pool}.npz');x,y,w,b=z['X'],z['y'],z['weight'],z['b'];assert {meta[int(i)]['date'] for i in z['ids']}<=set(r['fit_dates']) and not ({meta[int(i)]['date'] for i in z['ids']}&set(r['outer_dates']));fallback=bool(z['fallback_KEEP'][0]);assert fallback==(pool=='f0_s1');nfit+=not fallback
  raw=x[:,:41]*s['scale41']+s['mean41'];relations=[float(abs(raw[:,14]+raw[:,15]-2).max()),float(abs(raw[:,18]+raw[:,20]-2).max()),float(abs(raw[:,19]-raw[:,21]).max())];assert max(relations)<2e-6
  aug=np.c_[x,np.ones(len(x))];sv=np.linalg.svd(aug,compute_uv=False);machine_tol=max(aug.shape)*np.finfo(float).eps*sv[0];rank=int((sv>machine_tol).sum());newx=x[:,KEEP];newaug=np.c_[newx,np.ones(len(newx))];newsv=np.linalg.svd(newaug,compute_uv=False);newrank=int((newsv>max(newaug.shape)*np.finfo(float).eps*newsv[0]).sum());prob=np.exp(-np.logaddexp(0,-b));nll=np.logaddexp(0,b)-y*b;grad=aug.T@(w*(prob-y))/w.sum() if w.sum()>0 else np.zeros(43);domains=[]
  for stop in [True,False]:
   mask=z['domain_STOP']==stop;ww=w[mask];yy=y[mask];domains.append(dict(domain='STOP' if stop else 'CONT',rows=int(mask.sum()),positive=int(yy.sum()),normalized_risk_mass=float(ww.sum()/w.sum()) if w.sum() else 0,positive_risk_share=float(ww[yy>0].sum()/ww.sum()) if ww.sum() else 0,parent_margin=stats(b[mask]),prior_sigmoid=stats(prob[mask]),parent_weightedNLL=float(ww@nll[mask]/ww.sum()) if ww.sum() else None))
  assert (b>0).all() and (b<1).all();g_over_c=(1+b)/(1-b);assert np.allclose(np.tanh(np.log(g_over_c)/2),b,atol=1e-14,rtol=1e-14)
  for folder,source in [('fit_inputs',z),('calibration_inputs',c)]:
   data={k:source[k].copy() for k in source.files};data['X']=source['X'][:,KEEP];assert data['X'].shape[1]==39;np.savez_compressed(O/folder/f'{pool}.npz',**data)
   for key in source.files:
    if key!='X':assert np.array_equal(data[key],source[key])
  # Check objective differentiability and fixed penalty bookkeeping on synthetic theta, not optimizer fit.
  theta=np.linspace(-.02,.02,40);off=b;logit=newaug@theta+off;p=np.exp(-np.logaddexp(0,-logit));err=w*(p-y)/w.sum() if w.sum() else np.zeros(len(y));g=newaug.T@err+theta
  def obj(t):
   v=newaug@t+off;return float(w@(np.logaddexp(0,v)-y*v)/w.sum()+.5*(t@t)) if w.sum() else float(.5*(t@t))
  fd=0.
  if not fallback:
   for k in range(40):
    eps=np.zeros(40);eps[k]=1e-6;fd=max(fd,abs((obj(theta+eps)-obj(theta-eps))/2e-6-g[k]))
   assert fd<1e-6
  records.append(dict(pool=pool,rows=len(x),features_old=42,parameters_old=43,features_new=39,parameters_new=40,new_linear_fit_planned=not fallback,new_scalar_fit_planned=not fallback,predeclaredKEEP=fallback,ownfit_dates=sorted({meta[int(i)]['date'] for i in z['ids']}),semantic_pair_max_errors=relations,standardized_pair_max_errors=[float(abs(x[:,14]+x[:,15]).max()),float(abs(x[:,18]+x[:,20]).max()),float(abs(x[:,19]-x[:,21]).max())],old_aug_machine_rank=rank,new_aug_machine_rank=newrank,old_singular_values=sv.tolist(),new_singular_values=newsv.tolist(),public41_scale=stats(s['scale41']),domains=domains,parent_margin=stats(b),prior_sigmoid=stats(prob),gross_charge_ratio=stats(g_over_c),zero_correction_gradient_inf=float(abs(grad).max()),zero_correction_bias_gradient=float(grad[-1]),fit_qualified_gradient_fd_error=fd,protected_positive=int((c['y']>0).sum())))
  source_access.append(dict(pool=pool,fit_fields=list(z.files),cal_fields=list(c.files),held_features_read=False,held_outcomes_read=False));checks.append(dict(pool=pool,all_labels_weights_ids_eligibility_unchanged=True,all85union_constraints_unchanged=True))
 assert nfit==11 and sum(r['protected_positive'] for r in records)==85 and sum(r['rows'] for r in records)==2851
 # Algebraic duplicate penalty identity is deterministic, not tuned to data.
 vv=np.linspace(-3,3,31);old=.5*((vv/2)**2+(-vv/2)**2);new=.5*vv**2;assert np.array_equal(2*old,new)
 save('training_audit.json',records);save('qualification.json',dict(passed=True,optimizer_fits=0,held_scores=0,held_outcome_access=0,source_access=source_access,checks=checks,new_linear_fits_planned=11,new_scalar_fits_planned=11,total_future_fits=22,total_future_cap_s=600,constantKEEP=1,all40finite_difference_checks=True,all_labels_weights_roles_parentprior_unionconstraints_unchanged=True,selected_indices=KEEP,feature_names=[names[j] for j in KEEP]+['intercept'],no_offsettransform=True,unitL2_representation_change_explicit=True));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_inputs').iterdir()),*list((O/'calibration_inputs').iterdir()),O/'qualification.json',O/'training_audit.json']});verify();save('complete.json',dict(passed=True,qualification_only=True,actual_fits=0,new_policy_scores=0,future_fits=22));archive('qualified');print('GRID246_QUALIFIED',json.dumps(dict(pools=12,active=11,meanSTOPpositiveRisk=float(np.mean([r['domains'][0]['positive_risk_share'] for r in records if not r['predeclaredKEEP']])),max_pair_error=max(max(r['semantic_pair_max_errors']) for r in records),ranks=[dict(pool=r['pool'],old=r['old_aug_machine_rank'],new=r['new_aug_machine_rank']) for r in records]),ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
