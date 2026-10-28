"""Separately frozen one-step numeric repair; no restart of qualified endpoints."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid226_context_numeric';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID226_context_numeric'
sp=importlib.util.spec_from_file_location('frozen225_parent',R/'work/grid225_context_support/run.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
Q=P.Q;INPUT_PUBLIC=P.PARENT/'public_held_inputs';CONTROL=P.CONTROL;BASE=P.BASE;META=P.META;objective=P.objective;STEP=P.STEP

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args])
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
    P.trained()
def hessian(theta,x,y,w,b):
    aug=np.c_[x,np.ones(len(x))];z=x@theta[:-1]+theta[-1]+b;p=np.exp(-np.logaddexp(0,-z));return np.eye(len(theta))+aug.T@((w*p*(1-p)/w.sum())[:,None]*aug)

def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    for folder in {'trained':['models'],'scored':['held_predictions']}.get(phase,[]):shutil.copytree(O/folder,dest/folder)
    (dest/'archive_manifest.json').write_text(json.dumps({p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')

def prepare():
    assert not O.exists();P.trained();assert read(P.O/'complete.json')['status']=='numerical_qualification_failed_before_held_scoring' and not (P.O/'score_started.json').exists();O.mkdir();bad=[r for r in read(P.O/'fit_receipts.json') if not r['solver_success'] or r['gradient_inf']>=1e-3];assert [r['pool'] for r in bad]==['f2_s0'];save('design.json',dict(id='GRID226_CONTEXT_NUMERIC',source_protocol=read(P.O/'design.json'),reason=read(P.O/'qualification_stop.json'),explicit_amendment='GRID225staysnumericallyunqualifiedandunscored. New protocol,one fixed exactNewtonpolish only on failed f2_s0 checkpoint,then sameSTOP-positive scalar recalibration. Other10passedmodels+scalars andKEEP copiedbyteexactly. No L-BFGSrestart/retry/extraiterationsonpasseditems;nooriginal225windowextension.',unchanged='Same224rev02inputs/scalers/labels/weights/meanobjective/fullunitL2/125features/126params/fees/constraint/deploymask. Numericalprocedure amended ononepool beforeany mixed-context heldscore. Do not claim original225solverprotocolpassed.',repair='theta_new=theta_old-solve(exact126Hessian,originalgradient). Exactly one fullNewtonstep, no linesearch/grid or secondattempt. Require originalobjective notincrease >1e-10,originalrawgradinf<1e-3,and originalprimarypositive constraints. If fails stopunscored,do notrelaxcriteria.',budget='New bounded repair stage:one modelupdate+one scalarfit<=22,600s singlewindow. Original225endedat22fits and0.1869s,unchanged. Preserveallsourceendpoints and publishedstop.',barriers='New source firstpublished. All12effective model+stepfilespublishedbefore anyheldscores;all12predictionspublishedbeforeoutcomejoin. Existing222/208 controls reused,allseeds/date/adverse retained,developmentonly. No newphysics/final8/paper/NN/FRAME.'))
    z=np.load(Q/'qualified_inputs/f2_s0.npz');theta=np.load(P.O/'models/f2_s0.npz')['theta'];loss,g=objective(theta,z['X'],z['y'],z['weight'],z['b']);h=hessian(theta,z['X'],z['y'],z['weight'],z['b']);eigen=np.linalg.eigvalsh(h);assert eigen.min()>0 and np.allclose(h,h.T,atol=1e-8,rtol=1e-10);v=np.linspace(-1,1,126);v/=np.linalg.norm(v);aug=np.c_[z['X'],np.ones(len(z['X']))];eps=1e-4/max(1,float(abs(aug@v).max()));difference=(objective(theta+eps*v,z['X'],z['y'],z['weight'],z['b'])[1]-objective(theta-eps*v,z['X'],z['y'],z['weight'],z['b'])[1])/(2*eps);err=float(abs(difference-h@v).max()/max(1,float(abs(h@v).max())));assert err<1e-5;save('qualification.json',dict(passed=True,repair_pool='f2_s0',raw_gradient_before=float(abs(g).max()),minimum_Hessian_eigenvalue=float(eigen.min()),maximum_Hessian_eigenvalue=float(eigen.max()),Hessian_vector_difference_error=err,actual_new_updates=0,held_scores=0))
    paths=[W/'run.py',O/'design.json',O/'qualification.json',R/'work/grid225_context_support/run.py',P.O/'complete.json',P.O/'qualification_stop.json',P.O/'fits.json',P.O/'fit_receipts.json',P.O/'trained_freeze.json',META,CONTROL/'oof_trajectories.json',CONTROL/'gate_predictions.json',CONTROL/'search_metrics.json',CONTROL/'family_summary.json',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']+list((P.O/'models').glob('*.npz'))+list(INPUT_PUBLIC.glob('*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID226_ONE_STEP_QUALIFIED',err)

def polish():
    verify();assert not (O/'repair_started.json').exists();save('repair_started.json',dict(one_newton_step=1,one_scalar=1,no_retries=True,cap_s=600));start=time.perf_counter();(O/'models').mkdir();records=[]
    for source in sorted((P.O/'models').glob('*.npz')):
        pool=source.stem;z=np.load(Q/'qualified_inputs'/source.name);cp=np.load(source);theta=cp['theta'];alpha=float(cp['alpha'][0]);fallback=bool(cp['fallback_KEEP'][0]);before,g=objective(theta,z['X'],z['y'],z['weight'],z['b'])
        if pool=='f2_s0':
            h=hessian(theta,z['X'],z['y'],z['weight'],z['b']);step=np.linalg.solve(h,g);newtheta=theta-step;after,newg=objective(newtheta,z['X'],z['y'],z['weight'],z['b']);assert after<=before+1e-10 and abs(newg).max()<1e-3;end=z['calibration_X']@newtheta[:-1]+newtheta[-1]+z['calibration_b'];alpha,bound=STEP.choose_alpha(z['calibration_b'],end,z['calibration_y']>0);assert np.all((z['calibration_b']+alpha*(end-z['calibration_b']))[z['calibration_y']>0]>=0);np.savez_compressed(O/'models'/source.name,theta=newtheta,alpha=np.array([alpha]),fallback_KEEP=np.array([fallback]));records.append(dict(pool=pool,reused=False,newton_updates=1,scalar_fits=1,loss_before=before,loss_after=after,gradient_before=float(abs(g).max()),gradient_after=float(abs(newg).max()),max_theta_change=float(abs(step).max()),alpha=alpha,analytical_bound=bound))
        else:
            shutil.copy2(source,O/'models'/source.name);assert sha(source)==sha(O/'models'/source.name);assert fallback or abs(g).max()<1e-3;records.append(dict(pool=pool,reused=True,newton_updates=0,scalar_fits=0,gradient_after=float(abs(g).max()) if not fallback else 0.,alpha=alpha,model_bytes_unchanged=True))
    assert time.perf_counter()-start<600 and len(records)==12 and sum(r['newton_updates'] for r in records)==1;save('repair_receipts.json',records);save('fits.json',dict(passed=True,new_model_updates=1,new_scalar_fits=1,reused_passed_models=10,constant_KEEP=1,repair_and_checks_wall_s=time.perf_counter()-start,all_effective_pools_qualified=True,no_source225_edit=True));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'repair_receipts.json']});verify();archive('trained');print('GRID226_POLISHED_ONCE',json.dumps(next(r for r in records if not r['reused'])))

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    trained();assert read(O/'fits.json')['all_effective_pools_qualified'];head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12effective_models_published=True));print('GRID226_MODELS_PUBLISHED')

def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all_models_published=True));(O/'held_predictions').mkdir();records=[]
    for source in sorted((O/'models').glob('*.npz')):
        pool=source.stem;z=np.load(INPUT_PUBLIC/source.name);cp=np.load(source);theta=cp['theta'];alpha=float(cp['alpha'][0]);endpoint=z['X']@theta[:-1]+theta[-1]+z['b'];logit=z['b']+alpha*(endpoint-z['b']);np.savez_compressed(O/'held_predictions'/source.name,ids=z['ids'],logit=logit,keep=logit>=0,base_CONT=z['base_CONT'],endpoint=endpoint);records.append(dict(pool=pool,rows=len(logit),new_continue=int(((logit>=0)&z['base_CONT']).sum()),outcomes_parsed=False))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID226_ALL12_SCORES_SEALED')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID226_PREDICTIONS_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
