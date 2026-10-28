"""Qualify native pre-decision context bit, unchanged labels/risk/protection."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,importlib.util
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid249_context_bit_plan';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID249_context_bit_plan';Q=R/'outputs/grid246_canonical_plan';PUBLIC=R/'outputs/grid242_parent_prior_inputs';FULL=R/'outputs/grid192_gate_support';HELD=R/'outputs/grid190_extension_gate';KEEP=[j for j in range(42) if j not in [14,18,19]];PLAN=R/'outputs/grid241_parent_prior_plan'
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
 assert not O.exists() and read(R/'outputs/grid248_auxiliary_support/complete.json')['passed'];O.mkdir();save('design.json',dict(id='GRID249',qualification_only=True,question='Onecontextconditionedresidual,notauxiliarydiscard. Canonical39public inputs plus nativeNETSTOPbit,40inputs41params,sharedslopesplusregularizeddomainbias.',
 bit='native_STOP=(savedraw83[:,81]<=savedraw83[:,80]);originalfrozenNETq0/q1 alreadycomputed before gate. Verifyagainstoriginal_STOP_count/domain_STOP forOWNfit8,union12constraints andall4681oldheldSTOPstates. No NNforward/private future target toderivebit. Heldalways1becausedeploymententrySTOP.',
 unchanged='All247rows/labels/weights/canonical39features/scalers/b12/85STOPconstraints andparentCONTentry. Domainrisk0.5/0.5 unchanged. No feature/riskratio/threshold/alpha search.',
 future='Exactly11new41paramlinear+11original221scalar,1KEEP,600stotalonce. 219meanNLL+unitL2all41includingcontextcoefandintercept. SameL-BFGS-Bmaxiter2000/gtol1e-6/ftol1e-12/maxls50,zero41init,allsolver success rawgradinf<.001 before scoring;no retries. Primarymatched247/commonGATE,230historical.',
 capacity='Oneadditionalbinarypubliccontextandoneparameter,functionclass/regularizationpriorchangesexplicit. Indicatornotstandaloneunknownfuturelabel;not calibratedSTOPprobability or guaranteedimprovement.',
 barriers='Codeinputsfirstpublish. Qualifiedinputs before future250training. Allfutureweights/stepspublishbeforeheldscores,allpredictionsbeforeoutcomejoin. No heldoutcomes here.',
 boundaries='0optimizer/newmodel/NNforward/heldscores/physical/final8/paper/newFRAME/install;alloldexposeddatesseeds retained,outer12upstream notinnerindependent.'))
 paths=[W/'run.py',O/'design.json',Q/'qualified_freeze.json',R/'outputs/grid248_auxiliary_support/decision.json',PLAN/'qualified_pools.json',R/'work/grid192_gate_support/run.py',R/'work/grid190_extension_gate/run.py',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py']
 for r in read(PLAN/'qualified_pools.json'):
  n=f"{r['pool']}.npz";paths += [Q/'fit_inputs'/n,Q/'calibration_inputs'/n,FULL/'fit_features'/n,HELD/f"held_{r['pool']}.npz",PUBLIC/'outer_parent_public'/n]
 save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('source');print('GRID249_SOURCE_FROZEN')
def qualify():
 verify();assert not (O/'qualification_started.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(REPO).as_posix())==(O/'source_freeze.json').read_bytes();save('qualification_started.json',dict(sourcepublished=head,optimizercalls=0));spec=importlib.util.spec_from_file_location('grid249_fixedloss',R/'work/grid219_prior_objective/run.py');loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss);records=[];allheld=0;allpos=0;alltrain=0
 for d in ['fit_inputs','calibration_inputs','outer_parent_public']:(O/d).mkdir()
 for r in read(PLAN/'qualified_pools.json'):
  pool=r['pool'];f=np.load(FULL/'fit_features'/f'{pool}.npz');ids=f['ids'].astype(int);q=f['x'][:,80:82];native=q[:,1]<=q[:,0];original=np.arange(len(ids))<int(f['original_STOP_count']);assert np.array_equal(native,original);lookup={int(i):j for j,i in enumerate(ids)};trainbits=None
  for folder in ['fit_inputs','calibration_inputs']:
   z=np.load(Q/folder/f'{pool}.npz');bit=np.array([native[lookup[int(i)]] for i in z['ids']],float);data={k:z[k].copy() for k in z.files};data['X']=np.c_[z['X'],bit];assert data['X'].shape==(len(bit),40)
   if folder=='fit_inputs':assert np.array_equal(bit,z['domain_STOP']);trainbits=bit;alltrain+=len(bit)
   else:assert (bit==1).all();allpos+=int(z['y'].sum())
   np.savez_compressed(O/folder/f'{pool}.npz',**data)
   for k in z.files:
    if k!='X':assert np.array_equal(data[k],z[k])
  h=np.load(HELD/f'held_{pool}.npz');p=np.load(PUBLIC/'outer_parent_public'/f'{pool}.npz');assert np.array_equal(h['ids'],p['ids']);bit=(h['features'][:,81]<=h['features'][:,80]).astype(float);assert (bit==1).all();np.savez_compressed(O/'outer_parent_public'/f'{pool}.npz',ids=p['ids'],X=np.c_[p['X'][:,KEEP],bit],b=p['b'],base_CONT=p['base_CONT']);allheld+=len(bit)
  z=np.load(O/'fit_inputs'/f'{pool}.npz');x,y,w,b=z['X'],z['y'],z['weight'],z['b'];theta=np.linspace(-.02,.02,41);v,g=loss.objective(theta,x,y,w,b);fallback=bool(z['fallback_KEEP'][0]);fd=0.
  if not fallback:
   for k in range(41):
    eps=np.zeros(41);eps[k]=1e-6;fd=max(fd,abs((loss.objective(theta+eps,x,y,w,b)[0]-loss.objective(theta-eps,x,y,w,b)[0])/2e-6-g[k]))
   assert fd<1e-6
  # Synthetic parameter use checks the appended bit is a public domain bias, not a label.
  probe=np.zeros(41);probe[39]=.25;assert np.array_equal(x@probe[:-1]+probe[-1],.25*trainbits)
  records.append(dict(pool=pool,source_native_rows=len(ids),source_STOP_decisions_exact=True,fit_rows=len(trainbits),fit_STOP=int(trainbits.sum()),fit_CONT=int((trainbits==0).sum()),outer_rows=len(bit),outer_native_STOP_all=True,parameters=41,finite_difference_error=fd,future_linear_fit=not fallback,future_scalar_fit=not fallback,fallbackKEEP=fallback))
 assert alltrain==2851 and allpos==85 and allheld==4681 and sum(r['future_linear_fit'] for r in records)==11;save('qualification.json',dict(passed=True,pools=records,allnative_contextbits_verified=True,no_newNETforward=True,all_labels_weights_canonical39_parentoffset_eligibility_constraints_unchanged=True,ownfit_rows=alltrain,protectedpositive=allpos,public_held_rows=allheld,new_linear_fits_planned=11,new_scalar_fits_planned=11,total_future_fits=22,future_cap_s=600,actualfits=0,held_outcomes_read=0,contextbit_unstandardized=True));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_inputs').iterdir()),*list((O/'calibration_inputs').iterdir()),*list((O/'outer_parent_public').iterdir()),O/'qualification.json']});verify();save('complete.json',dict(passed=True,qualification_only=True,actual_fits=0,held_scores=0,held_outcomes=0,newNETforward=0,future_fits=22));archive('qualified');print('GRID249_QUALIFIED',alltrain,allpos,allheld)
if __name__=='__main__':globals()[sys.argv[1]]()
