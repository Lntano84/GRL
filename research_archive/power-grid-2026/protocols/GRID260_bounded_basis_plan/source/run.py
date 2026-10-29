"""Preregistered whole-public-basis qualification only. No optimizer or policy score."""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,importlib.util,traceback,collections
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid260_bounded_basis_plan';Q=R/'outputs/grid257_reference_labels';PLAN=R/'outputs/grid259_reference_diagnosis';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID260_bounded_basis_plan'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(Q/'qualified_freeze.json').items():assert sha(Q/rel)==h,rel

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def archive(phase,folders=[]):
 d=A/phase;d.mkdir(parents=True,exist_ok=False)
 for p in [*W.glob('*.py'),*O.iterdir()]:
  if p.is_file():shutil.copy2(p,d/p.name)
 for name in folders:shutil.copytree(O/name,d/name)
 save(d/'archive_manifest.json',{p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()})

def prepare():
 assert not O.exists() and read(PLAN/'complete.json')['passed'];O.mkdir();plan=read(PLAN/'NEXT_PROTOCOL_PLAN.json');assert plan['id']=='GRID260_NEXT_QUALIFICATION';head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for rel,h in read(Q/'qualified_freeze.json').items():assert sha(Q/rel)==h and hashlib.sha256(git('show',head+':'+(REPO/'research_archive/power-grid-2026/protocols/GRID257_reference_labels/qualified'/rel).relative_to(REPO).as_posix())).hexdigest()==h
 save(O/'pre_input_publication.json',dict(passed=True,all257qualified_inputs_published=head,plan259_published=True))
 save(O/'design.json',dict(id='GRID260',scope='Whole38publiccoordinate bounded-basis qualification ONLY; zero optimizer/scalar/NNforestforward/newouterpolicy/outcomes.',fixed_transform='copy original257X; X[:,:38]=np.tanh(X[:,:38]); preserve parentb column38/nativeSTOP39/offsetb and every other file field. Unitexistingnormalizedinputscale, no newscaler/temperature/cutoff or coordinate selection.',model='Sameparametercount40inputs41params,changedfunctionclassandunitL2functionalprior explicit,not strictsamecapacityclaim.',proof='Allaugmented41coordinatesabs<=1 -> abs(theta dot augmentedZ)<=norm1(theta)<=sqrt41 norm2(theta), including unchangedboundedb and0/1STOP. Bounddependslearnedtheta,notfixedoutput±2cap.',qualification='Onlyownfit8/cal4:allfinite,allunchangedfields bit-for-bit,exactnumericalcollisions/saturation logged withoutchanging rule,original21941gradient finite-difference on fixed theta,0optimizer. No newouterpublicinput/label or policy score.',KEEP='Samef0_s1KEEP;allother257eligibility retained,500positives(active464),2725activefitrows.',future='Ifqualified GRID261 one11linear+11original221positivealpha/one600s;original219meanweightedNLL+full41unitL2inclintercept/zero41/L-BFGS-Bmaxiter2000 gtol1e-6 ftol1e-12 maxls50,allsuccess rawgradinf<.001 before scoring. Primary258,250/GATE/230reuse;no alternativetransform/scale/seedsearch.',limits='Inputboundedness mayloseprecision/information anddoesnotguaranteequality. LegacyGRID48outputhardcapnot evidence for thisinputmap. Own12parent/internalcal4ancestryandoldexposeddata unchanged;localproxy notwholeweek/latency. No final8/physical/grid/paper/FRAME/install.'))
 paths=[W/'run.py',O/'design.json',O/'pre_input_publication.json',PLAN/'NEXT_PROTOCOL_PLAN.json',PLAN/'complete.json',PLAN/'own_fit_public_geometry.json',Q/'qualified_freeze.json',Q/'qualification.json',Q/'complete.json',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',R/'outputs/grid113_relational_learning/train_metadata.json']
 for folder in ['fit_inputs','calibration_inputs']:paths+=list((Q/folder).glob('*.npz'))
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID260_SOURCE_FROZEN')

def collisions(X,y,ids):
 # np.unique compares numeric coordinates,including +0/-0; no near-equality cutoff.
 _,inverse,count=np.unique(X,axis=0,return_inverse=True,return_counts=True);groups=[]
 for k in np.flatnonzero(count>1):
  ii=np.flatnonzero(inverse==k);groups.append(dict(rows=ii.tolist(),metadata_ids=ids[ii].astype(int).tolist(),count=len(ii),mixed_labels=len(set(y[ii].tolist()))>1,positive_rows=int(y[ii].sum())))
 return groups

def qualify():
 verify();assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save(O/'qualification_started.json',dict(source_published=head,optimizer_calls=0))
 spec=importlib.util.spec_from_file_location('g260_original219',R/'work/grid219_prior_objective/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);records=[];collision_records=[];saturation_records=[];preserved=[];totalfit=0;totalcal=0;positivecal=0;activefit=0;activepos=0
 for folder in ['fit_inputs','calibration_inputs']:(O/folder).mkdir()
 for file in sorted((Q/'fit_inputs').glob('*.npz')):
  pool=file.stem;maxfd=0.;metrics={}
  for folder in ['fit_inputs','calibration_inputs']:
   old=np.load(Q/folder/file.name);data={k:old[k].copy() for k in old.files};X=old['X'].copy();assert X.shape[1]==40 and np.isfinite(X).all();Z=X.copy();Z[:,:38]=np.tanh(X[:,:38]);assert np.isfinite(Z).all() and (abs(Z[:,:38])<=1).all() and (abs(Z[:,38])<=1).all() and np.isin(Z[:,39],[0.,1.]).all();assert np.array_equal(Z[:,38:40],X[:,38:40]) and np.array_equal(old['b'],Z[:,38]);data['X']=Z
   for key in old.files:
    if key!='X':assert data[key].dtype==old[key].dtype and data[key].shape==old[key].shape and data[key].tobytes()==old[key].tobytes()
   oldgroups=collisions(X,old['y'],old['ids']);newgroups=collisions(Z,old['y'],old['ids']);newlymixed=[]
   for group in newgroups:
    ii=group['rows']
    if group['mixed_labels'] and len(np.unique(X[ii],axis=0))>1:newlymixed.append(group)
   collision_records.append(dict(pool=pool,role=folder,original_duplicate_groups=oldgroups,transformed_duplicate_groups=newgroups,newly_mixed_due_to_mapping=newlymixed))
   exact=(abs(Z[:,:38])==1);saturation_records.append(dict(pool=pool,role=folder,total_public_coordinates=int(exact.size),exact_unit_coordinates=int(exact.sum()),rows_with_exact_unit=int(exact.any(1).sum()),positive_rows_with_exact_unit=int((exact.any(1)&(old['y']>0)).sum()),by_column=exact.sum(0).astype(int).tolist(),minimum_coordinate_derivative=float((1-Z[:,:38]**2).min()),original_public_absmax=float(abs(X[:,:38]).max()),transformed_public_absmax=float(abs(Z[:,:38]).max())))
   augmented=np.c_[Z,np.ones(len(Z))];assert np.linalg.norm(augmented,axis=1).max()<=np.sqrt(41)+1e-12;theta=np.linspace(-.02,.02,41);observed=float(abs(augmented@theta).max());l1=float(abs(theta).sum());l2=float(np.sqrt(41)*np.linalg.norm(theta));assert observed<=l1+1e-12 and l1<=l2+1e-12
   np.savez_compressed(O/folder/file.name,**data);preserved.append(dict(pool=pool,role=folder,all_nonX_fields_identical=True,b_and_nativeSTOP_identical=True,newX_sha256=hashlib.sha256(Z.tobytes()).hexdigest(),sourceX_sha256=hashlib.sha256(X.tobytes()).hexdigest()));metrics[folder]=dict(rows=len(Z),original_augmented_norm_max=float(np.linalg.norm(np.c_[X,np.ones(len(X))],axis=1).max()),bounded_augmented_norm_max=float(np.linalg.norm(augmented,axis=1).max()),exact_unit_coordinates=int(exact.sum()),old_mixed_collision_groups=sum(g['mixed_labels'] for g in oldgroups),new_mixed_collision_groups=sum(g['mixed_labels'] for g in newgroups),introduced_mixed_collision_groups=len(newlymixed))
   if folder=='fit_inputs':
    totalfit+=len(Z);fallback=bool(old['fallback_KEEP'][0]);assert np.array_equal(Z[:,39].astype(bool),old['domain_STOP'])
    if not fallback:
     activefit+=len(Z);activepos+=int(((old['y']>0)&old['domain_STOP']).sum());value,grad=m.objective(theta,Z,old['y'],old['weight'],old['b']);assert np.isfinite(value) and np.isfinite(grad).all()
     for k in range(41):
      d=np.zeros(41);d[k]=1e-6;fd=(m.objective(theta+d,Z,old['y'],old['weight'],old['b'])[0]-m.objective(theta-d,Z,old['y'],old['weight'],old['b'])[0])/2e-6;maxfd=max(maxfd,abs(fd-grad[k]))
     assert maxfd<1e-6
   else:totalcal+=len(Z);positivecal+=int(old['y'].sum());assert (Z[:,39]==1).all()
  records.append(dict(pool=pool,fit=metrics['fit_inputs'],calibration=metrics['calibration_inputs'],KEEP=fallback,future_linear_fit=not fallback,future_scalar_fit=not fallback,parameters=41,finite_difference_gradient_error=maxfd))
 assert totalfit==2859 and activefit==2725 and activepos==222 and totalcal==2054 and positivecal==500 and sum(r['future_linear_fit'] for r in records)==11;save(O/'field_preservation.json',preserved);save(O/'collision_checks.json',collision_records);save(O/'saturation_checks.json',saturation_records);save(O/'qualification.json',dict(passed=True,qualification_only=True,actual_fits=0,new_alpha_fits=0,new_NN_or_head_forward=0,new_outer_policy_scores=0,new_outer_outcomes_read=0,new_outer_inputs_exported=0,pools=records,fit8_candidate_rows=totalfit,activefit_rows=activefit,activefit_STOP_positive=activepos,own12_positive_constraints=positivecal,future_linear_fits=11,future_scalar_fits=11,future_total_fits=22,future_fit_cap_s=600,no_new_scaler_temperature_or_threshold=True,transformed_functionclass_and_L2_prior_changed=True,constant_output_hardcap=False))
 files=[p for folder in ['fit_inputs','calibration_inputs'] for p in (O/folder).iterdir()]+[O/'qualification.json',O/'field_preservation.json',O/'collision_checks.json',O/'saturation_checks.json'];save(O/'qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in files});verify();archive('qualified',['fit_inputs','calibration_inputs']);print('GRID260_QUALIFIED',activefit,positivecal,sum(r['future_linear_fit'] for r in records),sum(len(g['newly_mixed_due_to_mapping']) for g in collision_records))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if O.exists():save(O/(sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc(),actual_fits=0))
  raise
