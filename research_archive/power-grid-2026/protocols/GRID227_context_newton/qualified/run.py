"""Separately frozen safeguarded numerical solve; qualified endpoints stay frozen."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import hashlib,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid227_context_newton';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID227_context_newton'
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


def stable_math(theta,x,y,w,b):
    aug=np.c_[x,np.ones(len(x))];z=aug@theta+b;plus=np.exp(-np.logaddexp(0.,-z));minus=np.exp(-np.logaddexp(0.,z));rho=w/w.sum();loss=float(rho@np.where(y>0,np.logaddexp(0.,-z),np.logaddexp(0.,z))+.5*(theta@theta));residual=np.where(y>0,-minus,plus);gradient=aug.T@(rho*residual)+theta;h=np.eye(len(theta))+aug.T@((rho*plus*minus)[:,None]*aug);return loss,gradient,h

def safeguarded(theta,x,y,w,b,cap_s=600):
    start=time.perf_counter();theta=theta.copy();trace=[]
    for iteration in range(51):
        assert time.perf_counter()-start<cap_s,'Fixed numerical-stage window;do not extend.';value,g,h=stable_math(theta,x,y,w,b);norm=float(abs(g).max());trace.append(dict(iteration=iteration,loss=value,gradient_inf=norm))
        if norm<1e-6:return theta,trace
        assert iteration<50,'Fixed50 safeguarded Newton iterations exhausted;stop unscored.'
        d=np.sqrt(np.diag(h));step=np.linalg.solve(h/d[:,None]/d[None,:],g/d)/d;descent=float(g@step);assert descent>0 and np.isfinite(step).all();accepted=False;roundoff=64*np.finfo(float).eps*max(1.,abs(value))
        for backtrack in range(61):
            rate=2.**(-backtrack);candidate=theta-rate*step;nv,ng,_=stable_math(candidate,x,y,w,b)
            if nv<=value-1e-4*rate*descent or (abs(nv-value)<=roundoff and float(abs(ng).max())<norm):
                trace[-1].update(rate=rate,backtracks=backtrack);theta=candidate;accepted=True;break
        assert accepted,'Fixed line search failed;do not retry with another procedure.'
    raise AssertionError('Unreachable')

def prepare():
    assert not O.exists();P.trained();assert read(R/'outputs/grid226_context_numeric/complete.json')['passed'] is False and not (P.O/'score_started.json').exists();O.mkdir();save('design.json',dict(id='GRID227_CONTEXT_NEWTON',scientific_source_protocol=read(P.O/'design.json'),history='225strictgradientfailure f2s0,226singleNewtonfailure preserved,neither evaluated. Thisstage replaces only failed optimization solve,notmethod/data/objective or threshold.',numerical_protocol='One safeguardedNewton solve from fixed225unqualifiedcheckpoint,atmost50Newtoniterations and61fixedhalvingbacktracks each,Armijo1e-4,stablegradientinf<1e-6 internally AND original219rawgradientinf<1e-3. Exactsame meanrisk/fullunitL2 via stable BCE;diagonalpreconditioning only for linear equation,notfeaturetransform orpenaltychange. No alternate solver/retry/parametersearch.',reuse='10qualified225models+scalars and1KEEP copiedbyteexactly;onefailed modelpolish+oneSTOPscalar recalibration. No restartcompleteditems,original225/226stoppedprotocols unchanged.',numeric_tests='Before real modelupdate,2fixed synthetic test optimizations withlarge columns,analyticgradient/Hessian equivalence checked. Synthetic tests notheld modelselection.',budget='One boundednumericstage<=600s,newrealmodelsolve1+scalar1;50fixedNewtoniterations,noautomaticextension. Failed225originalwindow andendpoint receipts remainunchanged.',barriers='Source/numericqualificationfirstpublished;all12effective models publishedbeforeheldscore;all12scorespublishedbeforeoutcomejoin. Existing222 and208controls reused. All3seeds16exposed dates. No newphysics/NN/final8/paper/FRAME;developmentonly.'))
    tests=[]
    problems=[(np.array([[-2.,-2e6],[-1.,-1e6],[1.,1e6],[2.,2e6]]),np.array([0,0,1,1]),np.ones(4),np.zeros(4)),(np.array([[-2.,-1e6],[-1.,2e6],[0.,-.5e6],[1.,.1e6],[2.,1e6],[3.,-2e6]]),np.array([0,1,0,1,1,0]),np.arange(1.,7.),np.linspace(-.2,.3,6))]
    for i,(x,y,w,b) in enumerate(problems):
        theta,trace=safeguarded(np.zeros(x.shape[1]+1),x,y,w,b,60);value,g,h=stable_math(theta,x,y,w,b);ov,og=objective(theta,x,y,w,b);assert abs(value-ov)<1e-10 and abs(og).max()<1e-3;tests.append(dict(test=i,iterations=len(trace)-1,gradient_inf=float(abs(g).max()),original_gradient_inf=float(abs(og).max()),loss=value,theta=theta.tolist()))
    z=np.load(Q/'qualified_inputs/f2_s0.npz');t=np.load(P.O/'models/f2_s0.npz')['theta'];v,g,h=stable_math(t,z['X'],z['y'],z['weight'],z['b']);ov,og=objective(t,z['X'],z['y'],z['weight'],z['b']);assert abs(v-ov)<1e-10 and np.allclose(g,og,atol=1e-7,rtol=1e-7);save('qualification.json',dict(passed=True,synthetic_tests=tests,stable_original_objective_gradient_equivalence=True,actual_model_updates=0,held_scores=0))
    paths=[W/'run.py',O/'design.json',O/'qualification.json',R/'work/grid225_context_support/run.py',R/'outputs/grid226_context_numeric/complete.json',R/'outputs/grid226_context_numeric/repair_failure.json',P.O/'complete.json',P.O/'qualification_stop.json',P.O/'fit_receipts.json',P.O/'trained_freeze.json',META,CONTROL/'oof_trajectories.json',CONTROL/'gate_predictions.json',CONTROL/'search_metrics.json',CONTROL/'family_summary.json',BASE/'gate_predictions.json',BASE/'oof_trajectories.json',BASE/'search_metrics.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']+list((P.O/'models').glob('*.npz'))+list(INPUT_PUBLIC.glob('*.npz'));save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in paths});verify();archive('qualified');print('GRID227_NUMERIC_QUALIFIED',json.dumps(tests))

def polish():
    verify();assert not (O/'repair_started.json').exists();save('repair_started.json',dict(new_real_model_solve=1,new_scalar=1,max_Newton_iterations=50,max_backtracks=60,cap_s=600,no_retries=True));start=time.perf_counter();(O/'models').mkdir();records=[]
    for source in sorted((P.O/'models').glob('*.npz')):
        pool=source.stem;z=np.load(Q/'qualified_inputs'/source.name);cp=np.load(source);theta=cp['theta'];alpha=float(cp['alpha'][0]);fallback=bool(cp['fallback_KEEP'][0]);before,g=objective(theta,z['X'],z['y'],z['weight'],z['b'])
        if pool=='f2_s0':
            newtheta,trace=safeguarded(theta,z['X'],z['y'],z['weight'],z['b'],max(.001,600-(time.perf_counter()-start)));after,newg=objective(newtheta,z['X'],z['y'],z['weight'],z['b']);assert after<=before+1e-10 and abs(newg).max()<1e-3;end=z['calibration_X']@newtheta[:-1]+newtheta[-1]+z['calibration_b'];alpha,bound=STEP.choose_alpha(z['calibration_b'],end,z['calibration_y']>0);assert np.all((z['calibration_b']+alpha*(end-z['calibration_b']))[z['calibration_y']>0]>=0);np.savez_compressed(O/'models'/source.name,theta=newtheta,alpha=np.array([alpha]),fallback_KEEP=np.array([fallback]));save('Newton_trace.json',trace);records.append(dict(pool=pool,reused=False,new_model_solve=1,scalar_fits=1,Newton_iterations=len(trace)-1,loss_before=before,loss_after=after,gradient_before=float(abs(g).max()),original_gradient_after=float(abs(newg).max()),max_theta_change=float(abs(newtheta-theta).max()),alpha=alpha,analytical_bound=bound))
        else:
            shutil.copy2(source,O/'models'/source.name);assert sha(source)==sha(O/'models'/source.name);assert fallback or abs(g).max()<1e-3;records.append(dict(pool=pool,reused=True,new_model_solve=0,scalar_fits=0,original_gradient_after=float(abs(g).max()) if not fallback else 0.,alpha=alpha,model_bytes_unchanged=True))
    assert time.perf_counter()-start<600 and len(records)==12 and sum(r['new_model_solve'] for r in records)==1;save('repair_receipts.json',records);save('fits.json',dict(passed=True,new_model_solves=1,new_scalar_fits=1,reused_passed_models=10,constant_KEEP=1,repair_and_checks_wall_s=time.perf_counter()-start,all_effective_pools_qualified=True,no_source225_or226_edits=True));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'repair_receipts.json',O/'Newton_trace.json']});verify();archive('trained');print('GRID227_NUMERIC_REPAIRED',json.dumps(next(r for r in records if not r['reused'])))

def trained():
    verify()
    for p,h in read(O/'trained_freeze.json').items():assert sha(O/p)==h,p

def publication():
    trained();assert read(O/'fits.json')['all_effective_pools_qualified'];head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'trained/trained_freeze.json').relative_to(REPO).as_posix())==(O/'trained_freeze.json').read_bytes();save('pre_score_publication.json',dict(passed=True,git_sha=head,all12effective_models_published=True));print('GRID227_MODELS_PUBLISHED')

def score():
    trained();assert read(O/'pre_score_publication.json')['passed'] and not (O/'score_started.json').exists();save('score_started.json',dict(all_models_published=True));(O/'held_predictions').mkdir();records=[]
    for source in sorted((O/'models').glob('*.npz')):
        pool=source.stem;z=np.load(INPUT_PUBLIC/source.name);cp=np.load(source);theta=cp['theta'];alpha=float(cp['alpha'][0]);endpoint=z['X']@theta[:-1]+theta[-1]+z['b'];logit=z['b']+alpha*(endpoint-z['b']);np.savez_compressed(O/'held_predictions'/source.name,ids=z['ids'],logit=logit,keep=logit>=0,base_CONT=z['base_CONT'],endpoint=endpoint);records.append(dict(pool=pool,rows=len(logit),new_continue=int(((logit>=0)&z['base_CONT']).sum()),outcomes_parsed=False))
    assert sum(r['rows'] for r in records)==4681;save('score_receipts.json',records);save('scored_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'held_predictions').iterdir()),O/'score_receipts.json']});archive('scored');print('GRID227_ALL12_SCORES_SEALED')

def scored():
    trained()
    for p,h in read(O/'scored_freeze.json').items():assert sha(O/p)==h,p

def score_publication():
    scored();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head;assert git('show','HEAD:'+(A/'scored/scored_freeze.json').relative_to(REPO).as_posix())==(O/'scored_freeze.json').read_bytes();save('pre_join_publication.json',dict(passed=True,git_sha=head,all12predictions_published=True));print('GRID227_PREDICTIONS_PUBLISHED')
if __name__=='__main__':globals()[sys.argv[1]]()
