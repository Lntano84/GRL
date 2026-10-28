"""Qualify fixed native q-gap extension; no optimizer/held outcomes."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,importlib.util
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid251_qgap_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID251_qgap_plan';Q=R/'outputs/grid249_context_bit_plan';FULL=R/'outputs/grid192_gate_support';HELD=R/'outputs/grid190_extension_gate';PLAN=R/'outputs/grid241_parent_prior_plan';BASE=R/'outputs/grid208_public_amount';S=R/'outputs/grid250_context_residual';META=R/'outputs/grid113_relational_learning/train_metadata.json'
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
  for folder in ['fit_inputs','calibration_inputs','outer_parent_public']:shutil.copytree(O/folder,d/folder)
 (d/'archive_manifest.json').write_text(json.dumps({p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8')
def prepare():
 assert not O.exists() and read(S/'complete.json')['passed'];O.mkdir();save('design.json',dict(id='GRID251',scope='Qualificationonly;ownfit8 publicnativeqgap/risk support,0optimizer/newmodel/NNforward/heldscore/outcomes. Fixed250NEXT_PROTOCOL_PLAN.',
 feature='ONLYappend1continuousqgap to25040inputs=>41inputs42params. gap=((nativeq1-q0)-(parentmean81-mean80))/sqrt(parentscale80^2+parentscale81^2). Samepublished208parent83scalerallroles. No newnormalizer/temperature/scalesearch;combinationreferencescale not fittedgapstd.',
 public='192raw83and190heldcolumns80/81 areoriginalNETstop/continuevaluesalreadycomputedbeforegate;verifySTOPbitandshared25040inputs. No valuecalibration/probabilityclaim. BaseNET/parentouter12,internalcal4notindependent.',
 unchanged='All250rows/labels/weights/riskmixture/b12/scalers/nativeSTOPbit/85positiveconstraints/parentCONT. No thresholds/fees/labeloralpha changes,Jan19/Jan5exceptions or featureselection.',
 future='GRID252singlematched250candidate:11new42paramlinear+11original221positive scalar,1KEEP,600sonce. Original219meanNLL+unitL2all42,zero42,L-BFGS-Bmaxiter2000/gtol1e-6/ftol1e-12/maxls50. Allsuccess/rawgradinf<.001 beforescore,no retry.',
 barriers='Code/inputpublishbeforequalification. Qualifiedinputs/modelcodebeforetraining;allmodels/stepspublishbeforeheldscores;allpredictionspublishbeforeoutcomejoin. Alreadyexposeddevelopment,allseeds/adverse retained.',
 forbidden='No newNN/physical/final8/data/grid/paper/FRAME/dependencyinstall; no metric/mixture/thresholdorregularizationsearch. Oneadditionaldimension/parameterexplicit,notstrictsamecapacity/guaranteedgain.'))
 paths=[W/'run.py',O/'design.json',S/'NEXT_PROTOCOL_PLAN.md',S/'complete.json',Q/'qualified_freeze.json',PLAN/'qualified_pools.json',META,R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',BASE/'trained_freeze.json',BASE/'preheld_publication.json']
 for r in read(PLAN/'qualified_pools.json'):
  n=f"{r['pool']}.npz";paths += [Q/'fit_inputs'/n,Q/'calibration_inputs'/n,Q/'outer_parent_public'/n,FULL/'fit_features'/n,HELD/f"held_{r['pool']}.npz",R/r['parent_scaler']]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID251_SOURCE_FROZEN')
def stats(x):
 return dict(n=len(x),min=float(x.min()),max=float(x.max()),mean=float(x.mean()),median=float(np.median(x))) if len(x) else dict(n=0)
def qualify():
 verify();assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('qualification_started.json',dict(sourcepublished=head,newfits=0,heldtargetsread=False));spec=importlib.util.spec_from_file_location('grid251loss',R/'work/grid219_prior_objective/run.py');loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss);recs=[];risks=[];meta=read(META);allfit=0;allpos=0;allheld=0;pub=read(BASE/'preheld_publication.json');frozen=read(BASE/'trained_freeze.json')
 for folder in ['fit_inputs','calibration_inputs','outer_parent_public']:(O/folder).mkdir()
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool'];src=np.load(FULL/'fit_features'/f'{pool}.npz');ids=src['ids'].astype(int);rawgap=src['x'][:,81]-src['x'][:,80];native=rawgap<=0;assert np.array_equal(native,np.arange(len(ids))<int(src['original_STOP_count']));pos={int(i):j for j,i in enumerate(ids)};scale_file=R/r['parent_scaler'];assert sha(scale_file)==frozen['models/'+scale_file.name];blob=git('show',pub['git_sha']+':research_archive/power-grid-2026/protocols/GRID208_public_amount/trained/models/'+scale_file.name);assert hashlib.sha256(blob).hexdigest()==sha(scale_file);sc=np.load(scale_file);center=float(sc['mean'][81]-sc['mean'][80]);denom=float(np.hypot(sc['scale'][80],sc['scale'][81]));assert denom>0 and np.isfinite([center,denom]).all();gap=(rawgap-center)/denom;assert np.isfinite(gap).all()
  for folder in ['fit_inputs','calibration_inputs']:
   z=np.load(Q/folder/f'{pool}.npz');ix=np.array([pos[int(i)] for i in z['ids']]);assert np.array_equal(native[ix].astype(float),z['X'][:,39]);data={k:z[k].copy() for k in z.files};data['X']=np.c_[z['X'],gap[ix]];assert data['X'].shape==(len(ix),41) and np.isfinite(data['X']).all();np.savez_compressed(O/folder/f'{pool}.npz',**data)
   for k in z.files:
    if k!='X':assert np.array_equal(data[k],z[k])
   if folder=='fit_inputs':
    assert {meta[int(i)]['date'] for i in z['ids']}<=set(r['fit_dates']);allfit+=len(ix)
   else:allpos+=int(z['y'].sum())
  h=np.load(HELD/f'held_{pool}.npz');p=np.load(Q/'outer_parent_public'/f'{pool}.npz');assert np.array_equal(h['ids'],p['ids']);heldgap=(h['features'][:,81]-h['features'][:,80]-center)/denom;assert (h['features'][:,81]<=h['features'][:,80]).all() and (p['X'][:,39]==1).all() and np.isfinite(heldgap).all();np.savez_compressed(O/'outer_parent_public'/f'{pool}.npz',ids=p['ids'],X=np.c_[p['X'],heldgap],b=p['b'],base_CONT=p['base_CONT']);allheld+=len(heldgap)
  z=np.load(O/'fit_inputs'/f'{pool}.npz');x,y,w,b=z['X'],z['y'],z['weight'],z['b'];fallback=bool(z['fallback_KEEP'][0]);theta=np.linspace(-.02,.02,42);v,g=loss.objective(theta,x,y,w,b);fd=0
  if not fallback:
   for k in range(42):
    e=np.zeros(42);e[k]=1e-6;fd=max(fd,abs((loss.objective(theta+e,x,y,w,b)[0]-loss.objective(theta-e,x,y,w,b)[0])/2e-6-g[k]))
   assert fd<1e-6
  for d in [True,False]:
   for label in ['ALL','POSITIVE','NONPOSITIVE']:
    mask=z['domain_STOP']==d
    if label!='ALL':mask&=(y>0)==(label=='POSITIVE')
    ww=w[mask];xx=x[mask,40];risks.append(dict(pool=pool,domain='STOP' if d else 'CONT',label=label,gap=stats(xx),risk_mass=float(ww.sum()),riskweighted_gap_mean=float(ww@xx/ww.sum()) if ww.sum() else None))
  zero=np.exp(-np.logaddexp(0,-b));newcoefgrad=float(x[:,40]@(w*(zero-y))/w.sum());recs.append(dict(pool=pool,parameters=42,fit_rows=len(y),protected_positive=int(np.load(O/'calibration_inputs'/f'{pool}.npz')['y'].sum()),public_held_rows=len(heldgap),published_parent_scaler_hash=sha(scale_file),gap_center=center,reference_scale=denom,ownfit_gap=stats(x[:,40]),zero_newcoef_gradient=newcoefgrad,gradient_fd_error=fd,new_linear_fit=not fallback,new_scalar_fit=not fallback,KEEP=fallback,no_newNETforward=True))
 assert allfit==2851 and allpos==85 and allheld==4681 and sum(r['new_linear_fit'] for r in recs)==11;save('training_gap_support.json',risks);save('qualification.json',dict(passed=True,pools=recs,nativeq_public_beforegate_verified=True,published_parent_scalers_verified=True,all40oldinputs_andlabels_weights_priors_constraints_unchanged=True,all42gradientchecks=True,newfits=0,newNETforward=0,held_scores=0,held_outcomes_read=0,new_linear_fits_planned=11,new_scalar_fits_planned=11,total_future_fits=22,future_cap_s=600));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_inputs').iterdir()),*list((O/'calibration_inputs').iterdir()),*list((O/'outer_parent_public').iterdir()),O/'qualification.json',O/'training_gap_support.json']});verify();save('complete.json',dict(passed=True,qualification_only=True,actualfits=0,heldscores=0,heldoutcomes=0,futurefits=22));archive('qualified');print('GRID251_QUALIFIED',allfit,allpos,allheld)
if __name__=='__main__':globals()[sys.argv[1]]()
