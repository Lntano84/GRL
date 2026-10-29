"""GRID264: only qualification of frozen261 guarded downward score bias; no true fit."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil
import numpy as np
sys.dont_write_bytecode=True
from runtime264 import base_score,guarded_shift,apply_shift
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid264_guarded_bias_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID264_guarded_bias_plan';Q=R/'outputs/grid260_bounded_basis_plan';N=R/'outputs/grid261_bounded_basis';OLD=R/'outputs/grid257_reference_labels';ROLE=R/'outputs/grid241_parent_prior_plan/qualified_pools.json';META=R/'outputs/grid113_relational_learning/train_metadata.json';PLAN=R/'outputs/grid263_training_residual/NEXT_PROTOCOL_PLAN.json'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a])
def verify():
 for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p

def archive(phase):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 if phase=='qualified':shutil.copytree(O/'qualified_inputs',d/'qualified_inputs')
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')

def prepare():
 assert not O.exists() and rd(PLAN)['id']=='GRID264';head=git('rev-parse','HEAD').decode().strip();assert not git('status','--porcelain=v1').strip() and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for folder,ar in [(Q/'calibration_inputs',REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan/qualified/calibration_inputs'),(N/'models',REPO/'research_archive/power-grid-2026/protocols/GRID261_bounded_basis/trained/models')]:
  assert len(list(folder.glob('*.npz')))==12
  for f in folder.glob('*.npz'):assert git('show',head+':'+(ar/f.name).relative_to(REPO).as_posix())==f.read_bytes()
 O.mkdir();save('design.json',dict(id='GRID264',scope='QualificationONLY0fits/truedelta/minimizer/shiftedrealpolicy/newouterdata',unique='Freeze261compositebase score. Future265 one maximaldownwardbiasperactivepool underoriginalown12positiveconstraints,KEEPdelta0. Explicit stoppingthreshold/empiricalobjectivechange,not unchangedNLL.',currentchecks='Original260vs257cal fields unchanged exceptfixedtanh38;role fit8/cal4withinown12 andoutsideouter4;261theta/alpha unchanged;all500oldpositive finite/nonnegative. Onlybooleanandhash ofoldpositive scores,no min/quantile/actualdelta.',synthetic='Only synthetic arrays call guarded_shift:ties/zero/oneULP/smallnumbers/KEEP/emptyguard/invalidnegativeguard;monotone noextraCONT andnonnegativefixedquerycost nonincrease.',future='26511scalar+1KEEP,0theta/NN,one600s. Original261publicscores may be byte-reused then subtract delta AFTERall12scalars publication;ALL12predictionspublishedbeforeoldresultjoin. No thresholdgrid/negative labels or outeroutcomes fordelta.',count='2640,branch62. Future265distinct+11=>73;old6continuations8285updates/604.703sactual/678.156cap retained.',limits='Same500own12empiricalpositives(active464) nonindependentcalibration,parentown12/internalcal4ancestry. No outergeneralization guarantee,allseeds/adverse retained;no final8/physical/grid/paper/FRAME/install.'))
 paths=[*W.glob('*.py'),O/'design.json',PLAN,ROLE,META,Q/'qualified_freeze.json',*sorted((Q/'calibration_inputs').glob('*.npz')),*sorted((OLD/'calibration_inputs').glob('*.npz')),*sorted((N/'models').glob('*.npz'))]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('input_publication.json',dict(passed=True,git_sha=head,all12cal_andmodels_specificblobs_verified=True));archive('source');print('GRID264_SOURCE_FROZEN')

def qualify():
 verify();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head and git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes()
 assert not (O/'qualification_started.json').exists();save('qualification_started.json',dict(source_published=head,actual_fits0=True));records=[];meta=rd(META);roles={p['pool']:p for p in rd(ROLE)};(O/'qualified_inputs').mkdir();positive_total=active_total=0
 for file in sorted((Q/'calibration_inputs').glob('*.npz')):
  z=np.load(file);old=np.load(OLD/'calibration_inputs'/file.name);m=np.load(N/'models'/file.name);p=file.stem;role=roles[p]
  assert set(z.files)==set(old.files)
  for k in z.files:
   expected=old[k].copy()
   if k=='X':expected[:,:38]=np.tanh(expected[:,:38])
   assert np.array_equal(z[k],expected),k
  assert np.isfinite(z['X']).all() and np.array_equal(z['X'][:,38],z['b']) and (z['X'][:,39]==1).all() and (z['b']>0).all()
  fitdates=set(role['fit_dates']);caldates=set(role['cal4_dates']);outer=set(role['outer_dates']);assert len(fitdates)==8 and len(caldates)==4 and not fitdates&caldates and not (fitdates|caldates)&outer
  for mid,isfit in zip(z['ids'],z['fit_role']):assert meta[int(mid)]['date'] in (fitdates if isfit else caldates)
  oldscore=base_score(z['X'],z['b'],m['theta'],float(m['alpha'][0]));pos=z['y']>0;assert np.isfinite(oldscore).all() and np.all(oldscore[pos]>=0)
  rowwise=np.array([base_score(z['X'][i:i+1],z['b'][i:i+1],m['theta'],float(m['alpha'][0]))[0] for i in np.flatnonzero(pos)],dtype=np.float64);assert np.array_equal(oldscore[pos],rowwise)
  keep=bool(m['fallback_KEEP'][0]);assert keep==(p=='f0_s1');positive_total+=int(pos.sum());active_total+=int(pos.sum())*int(not keep)
  records.append(dict(pool=p,rows=len(pos),positives=int(pos.sum()),fit8_positives=int((pos&z['fit_role']).sum()),cal4_positives=int((pos&~z['fit_role']).sum()),KEEP=keep,old_positive_scores_finite_nonnegative=True,oldscore_vector_hash=hashlib.sha256(oldscore[pos].tobytes()).hexdigest(),same_batch_and_row_expression=True,own12dates_sorted=sorted(fitdates|caldates),outer4excluded=True,real_delta_not_computed=True))
  shutil.copy2(file,O/'qualified_inputs'/f'{p}_cal.npz');shutil.copy2(N/'models'/file.name,O/'qualified_inputs'/f'{p}_model.npz')
 assert positive_total==500 and active_total==464 and len(records)==12
 tests=[]
 for name,scores,pos,cost in [('ties',[.125,.125,.5,-.2],[1,1,1,0],[1,2,64,3]),('zero',[0.,.2,-.1],[1,1,0],[1,5,2]),('tiny',[1e-300,2e-300,-1e-300],[1,1,0],[4,2,1]),('oneULP',[1.,float(np.nextafter(1.,np.inf)),2.,.9],[1,1,1,0],[2,3,4,5])]:
  arr=np.array(scores,dtype=np.float64);mask=np.array(pos,dtype=bool);q=np.array(cost);d,keep=guarded_shift(arr,mask);new,cont=apply_shift(arr,np.ones(len(arr),dtype=bool),d);assert not keep and np.all(new[mask]>=0) and not np.any(cont&~(arr>=0)) and q[cont].sum()<=q[arr>=0].sum()
  assert np.any((arr-float(np.nextafter(d,np.inf)))[mask]<0);tests.append(dict(name=name,passed=True,synthetic_delta=d,maximal_nextfloat_rejected=True))
 a=np.array([.2,-.1],dtype=np.float64);empty=np.zeros(2,dtype=bool);assert guarded_shift(a,empty)==(0.,True) and guarded_shift(a,np.array([True,False]),KEEP=True)==(0.,True)
 try:guarded_shift(np.array([-.1],dtype=np.float64),np.array([True]))
 except AssertionError:tests.append(dict(name='negativeprotectedprecondition',passed=True))
 else:raise AssertionError('Invalid protected score silently accepted')
 save('synthetic_checks.json',dict(passed=True,tests=tests,empty_andKEEP_zero=True,no_real_delta_fit=True));save('qualification.json',dict(passed=True,pools=records,total_positive500=positive_total,active_positive464=active_total,actual_fits=0,actual_scalar_minimum_calls=0,new_outer_inputs_scores_or_outcomes=0,NN_headforward=0,old_positive_score_checks_only=True,same_original_guard_set=True,not_independent_calibration=True));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'qualified_inputs').glob('*.npz')),O/'qualification.json',O/'synthetic_checks.json',W/'runtime264.py'] if p.is_relative_to(O)})
 # Runtime source is separately frozen; manifest keys stay output-relative.
 save('runtime_freeze.json',{'work/grid264_guarded_bias_plan/runtime264.py':sha(W/'runtime264.py')});save('complete.json',dict(passed=True,qualificationonly=True,actualfits0=True,protected500=True,active464=True,qualified_inputs24=True))
 archive('qualified');print('GRID264_QUALIFIED_NO_FITS',json.dumps(dict(positive500=positive_total,active464=active_total,actualfits0=True)))
if __name__=='__main__':globals()[sys.argv[1]]()
