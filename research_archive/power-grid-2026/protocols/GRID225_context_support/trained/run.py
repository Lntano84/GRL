"""Fixed mixed-context residual-support probe, consuming only224revision02."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import datetime,hashlib,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid225_context_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID225_context_support';Q=R/'outputs/grid224_context_support_plan/revision02';PARENT=R/'outputs/grid220_full_prior';CONTROL=R/'outputs/grid222_protected_step';BASE=R/'outputs/grid208_public_amount';META=R/'outputs/grid113_relational_learning/train_metadata.json'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
LOSS=module('grid225_fixed219',R/'work/grid219_prior_objective/run.py');STEP=module('grid225_fixed221',R/'work/grid221_step_plan/run.py');objective=LOSS.objective
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    for p,h in read(Q/'qualified_freeze.json').items():assert sha(Q/p)==h,p
    for p,h in read(Q.parent/'source_freeze.json').items():assert sha(R/p)==h,p
    for root,names in [(PARENT,['trained_freeze.json','scored_freeze.json']),(CONTROL,['calibrated_freeze.json','scored_freeze.json'])]:
        for name in names:
            for p,h in read(root/name).items():assert sha(root/p)==h,p

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    for folder in {'trained':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,dest/folder)
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')

def prepare():
    assert not O.exists() and read(Q.parent/'independent_audit.json')['passed'] and read(Q/'qualification.json')['passed'];O.mkdir();save('design.json',dict(id='GRID225_CONTEXT_SUPPORT',authorization='User continuous autonomous research/training untilstopped;GRLheartbeat executes frozen224revision02.',protocol=read(Q/'design.json'),new_arm='MIXED_CONTEXT ANCHORED only. Existing222ANCHORED labelledSTOP_ONLY and208GATE controls reused bytefrozen,notrefitted.',fits=dict(new_linear=11,new_scalar_steps=11,constant_KEEP=1,total_cap_s=600,solver='L-BFGS-B,maxiter2000,gtol1e-6,ftol1e-12,maxls50,zero126initialization;allsolver_successandgradinf<1e-3 beforeheld'),calibration='Onlycalibration_X/b/y/ids originalSTOP constraints,NOTauxiliarylabels. Original221choose_alpha;ULPdown only.',input='Only224revision02 qualified inputs. Frozen220public_held_inputs used after allmodels published. No newNNforward orfeature normalization.',boundaries='Allownoutertrain dates,all3seeds/4folds. Fixedfees/controllers/128/64/stagnation2/.9/order/aliases/station. Domainrisk0.5STOP/0.5CONT isexplicitchangedauxobjective;notrawpooledutility/calibratedSTOPprobability/equalexposure.',barriers='Source/inputsfirstpublished;all11models+11steps+KEEPpublished beforeheld scores;all12predfilespublished beforeoutcomejoin. Exposedhistoricalteacherpathdevelopmentonly. No newphysics/final8/paper/newFRAME,nostepsweep/refit/extension.'))
    records=[]
    for r in read(Q/'qualification.json')['records']:
        pool=r['pool'];z=np.load(Q/'qualified_inputs'/f'{pool}.npz');old=np.load(PARENT/'fit_inputs'/f'{pool}.npz');assert z['X'].shape[1]==125 and z['calibration_X'].shape==old['X'].shape;assert np.array_equal(z['calibration_X'],old['X']) and np.array_equal(z['calibration_y'],old['y']) and np.array_equal(z['calibration_ids'],old['ids']);assert np.all(z['b']>0) and np.all(z['calibration_b']>0);assert np.isfinite(z['X'].astype(np.float32)).all();assert abs(z['weight'][z['domain_STOP']].sum()/z['weight'].sum()-.5)<1e-12;assert not set(r['allowed_outer_fit_dates'])&set(r['held_dates']);active=bool(z['active_fit'][0]);assert active==(pool!='f0_s1');theta=np.linspace(-.01,.01,126);value,grad=objective(theta,z['X'],z['y'],z['weight'],z['b']);err=0.
        for k in [0,41,82,83,123,124,125]:
            e=np.zeros(126);e[k]=1e-6;estimate=(objective(theta+e,z['X'],z['y'],z['weight'],z['b'])[0]-objective(theta-e,z['X'],z['y'],z['weight'],z['b'])[0])/2e-6;err=max(err,abs(estimate-grad[k]));assert abs(estimate-grad[k])<1e-6
        records.append(dict(pool=pool,active=active,mixed_rows=len(z['y']),primary_STOP_rows=int(z['primary_rows'][0]),primary_positive=int(z['calibration_y'].sum()),gradient_error=err,allowed_fit_dates=r['allowed_outer_fit_dates'],held_dates=r['held_dates']))
    assert sum(r['active'] for r in records)==11;save('qualification.json',dict(passed=True,records=records,actual_fits=0,new_linear_fits=11,new_scalar_fits=11));files=[W/'run.py',O/'design.json',O/'qualification.json',Q/'design.json',Q/'qualification.json',Q/'qualified_freeze.json',Q.parent/'source_freeze.json',Q.parent/'complete.json',R/'work/grid219_prior_objective/run.py',R/'work/grid221_step_plan/run.py',CONTROL/'complete.json',CONTROL/'independent_audit.json',META,BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',CONTROL/'oof_trajectories.json',CONTROL/'gate_predictions.json',CONTROL/'search_metrics.json',CONTROL/'family_summary.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']+list((Q/'qualified_inputs').glob('*.npz'))+list((PARENT/'public_held_inputs').glob('*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('qualified');print('GRID225_QUALIFIED_11MODELS_11SCALARS')

def train():
    verify();assert not (O/'training_started.json').exists();from scipy.optimize import minimize
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),linear_fits=11,scalar_fits=11,cap_s=600));(O/'models').mkdir();start=time.perf_counter();records=[];access=[]
    for r in read(O/'qualification.json')['records']:
        assert time.perf_counter()-start<600,'Original600s cap:retain partials,no retry/extension.';pool=r['pool'];path=Q/'qualified_inputs'/f'{pool}.npz';z=np.load(path);access.append(dict(path=path.relative_to(R).as_posix(),type='own_fit_inputs_only'));theta=np.zeros(126);active=bool(z['active_fit'][0]);t=time.perf_counter();initial=objective(theta,z['X'],z['y'],z['weight'],z['b'])[0]
        if active:
            result=minimize(objective,theta,args=(z['X'],z['y'],z['weight'],z['b']),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,gtol=1e-6,ftol=1e-12,maxls=50));theta=result.x.copy();loss,grad=objective(theta,z['X'],z['y'],z['weight'],z['b']);success=bool(result.success);nit=int(result.nit);message=str(result.message);assert loss<=initial+1e-10 and abs(loss-result.fun)<1e-10
        else:loss=initial;grad=np.zeros(126);success=True;nit=0;message='Originalf0_s1KEEP unchanged;no primarySTOPpositive.'
        end=z['calibration_X']@theta[:-1]+theta[-1]+z['calibration_b'];a,bound=STEP.choose_alpha(z['calibration_b'],end,z['calibration_y']>0,fallback=not active);score=z['calibration_b']+a*(end-z['calibration_b']);assert np.all(score[z['calibration_y']>0]>=0);np.savez_compressed(O/'models'/f'{pool}.npz',theta=theta,alpha=np.array([a]),fallback_KEEP=np.array([not active]));records.append(dict(pool=pool,new_linear_fit=active,new_scalar_fit=active,rows=len(z['y']),primary_positive=int(z['calibration_y'].sum()),fit_positive_kept=int((score[z['calibration_y']>0]>=0).sum()),alpha=a,analytical_bound=bound,initial_loss=initial,final_loss=loss,gradient_inf=float(abs(grad).max()),solver_success=success,iterations=nit,message=message,fit_and_checks_wall_s=time.perf_counter()-t));save('fit_receipts.json',records)
    assert sum(r['new_linear_fit'] for r in records)==11 and sum(r['new_scalar_fit'] for r in records)==11;save('fits.json',dict(passed=True,new_linear_fits=11,new_scalar_fits=11,constant_KEEP=1,total_fit_and_checks_wall_s=time.perf_counter()-start,all_solvers_success=all(r['solver_success'] and r['gradient_inf']<1e-3 for r in records),new_NN_fits=0));save('fit_access.json',access);save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json',O/'fit_access.json']});verify();archive('trained');print('GRID225_ALL22_FITS_DONE',read(O/'fits.json')['all_solvers_success'])

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    trained();assert read(O/'fits.json')['all_solvers_success'];head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all_new_models_and_steps_published=True));print('GRID225_MODELS_PUBLISHED')

def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all_models_steps_published=True));(O/'held_predictions').mkdir();records=[]
    for r in read(O/'qualification.json')['records']:
        pool=r['pool'];z=np.load(PARENT/'public_held_inputs'/f'{pool}.npz');cp=np.load(O/'models'/f'{pool}.npz');theta=cp['theta'];alpha=float(cp['alpha'][0]);endpoint=z['X']@theta[:-1]+theta[-1]+z['b'];logit=z['b']+alpha*(endpoint-z['b']);keep=logit>=0;assert np.isfinite(logit).all();np.savez_compressed(O/'held_predictions'/f'{pool}.npz',ids=z['ids'],logit=logit,keep=keep,base_CONT=z['base_CONT'],endpoint=endpoint);records.append(dict(pool=pool,rows=len(logit),new_continue=int((keep&z['base_CONT']).sum()),outcomes_parsed=False))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID225_ALL12_SCORES_SEALED')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12_predictions_published=True));print('GRID225_PREDICTIONS_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
