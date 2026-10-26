"""New from-zero numerical control with exactly the GRID192 logistic objective."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import contextlib,datetime,importlib.util,io,json,pickle,shutil,sys,time,traceback
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid193_gate_precondition';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID193_gate_precondition';SRC=R/'outputs/grid192_gate_support'
sp=importlib.util.spec_from_file_location('grid193_previous_support',R/'work/grid192_gate_support/run.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);B=M.B;read=M.read;sha=M.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item())+'\n',encoding='utf8',newline='\n')
def bind():B.O=O;B.W=W;B.A=A
def verify():
    for n in ['source_freeze.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    B.H.sealed()
def archive(p):bind();B.archive(p)
def objective(z,u,d):
    x=z['standard']/d;w=z['sample_weight'];y=z['y'];S=w.sum();logit=x@u[:-1]+u[-1];beta=u[:-1]/d;p=np.exp(-np.logaddexp(0.,-logit));e=w*(p-y)/S
    value=float(w@(np.logaddexp(0.,logit)-y*logit)/S+beta@beta/(2*S));grad=np.r_[x.T@e+u[:-1]/(S*d*d),e.sum()]
    return value,grad
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed();assert read(SRC/'complete.json')['passed']
    save('design.json',dict(id='GRID193_EXACT_OBJECTIVE_PRECONDITION',authorization='Sustained learning exploration authorized;pretrainingnotice separate12fromzero numericalcontrol atsame2000iterationcap. PreserveGRID192 all12originalendpoints.',
        cause='GRID192two maxiteration fits have originalgradientinfinity .115/.828 and CONTfeatures reach~2.09million originalSTOP-standardunits. Do not label support hypothesis failure before numerical control.',
        same='SameGRID192data,y,weights,totalweight,STOPnormalizer,C1,L2,83features,84params,.5readout,all12foldseed pairs. No oldendpointwarmstart or resumedfit.',
        transform='Fit-only d=max(1,sqrt(weightedmean(Z^2))). Let v=d*beta,Ztilde=Z/d. Same logit=Zbeta+b=Ztildev+b. Penalty sum((v/d)^2)/(2C*S),NOT ordinaryisotropicL2onv.',
        optimizer='ScipyL-BFGS-B,zero84initialparameters,maxiter2000,maxfun15000,maxls50,ftol0,gtol1e-4/max(d). Callbackacceptsonly originalbeta-gradient infinity<=1e-4 (same originalLRtol);certificationreportedseparatelyfromscipystatus. Fixed600sec total,noautomaticextensions/retries.',
        qualification='Samefullfitloss andgradientchain atfixedrandomprobe;centralfinite differences eightcoordinates. No optimization or heldgate scoring beforequalification/freeze.',
        evaluation='All12finalweights published beforeheld scoring,reusefixedGRID190heldNETfeatures. Sameprespecifiedexpected-query comparison asGRID192 vsoriginalGRID190;allmodels/seeds/dates retained. No optimizerparameter search.',
        limitations='Numericalcontrol is a new fixedoptimizationprotocol,not originalsameoptimizationpath;if objective certified,compare same statistical loss. ExposedteacherOOF,not independenttest/physicalquality.'))
    files=[O/'design.json',SRC/'complete.json',SRC/'optimization_and_scale.json',SRC/'fits.json']+list((SRC/'fit_features').iterdir())+[R/p for p in read(SRC/'source_freeze.json')];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});code=[W/'run.py',R/'work/grid192_gate_support/run.py']+[R/p for p in read(SRC/'code_freeze.json')];save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});verify();archive('planned');print('GRID193_PLANNED')
def qualify():
    verify();assert not (O/'preflight.json').exists();(O/'fit_features').mkdir();checks=[];rng=np.random.default_rng(193)
    for f in range(4):
        for seed in range(3):
            src=SRC/'fit_features'/f'f{f}_s{seed}.npz';shutil.copy2(src,O/'fit_features'/src.name);z=np.load(src);mass=z['sample_weight']/z['sample_weight'].sum();d=np.maximum(1.,np.sqrt(mass@(z['standard']**2)));u=rng.normal(0,.01,84);val,gu=objective(z,u,d);beta=u[:-1]/d;logit=z['standard']@beta+u[-1];S=z['sample_weight'].sum();v=float(z['sample_weight']@(np.logaddexp(0.,logit)-z['y']*logit)/S+beta@beta/(2*S));p=np.exp(-np.logaddexp(0.,-logit));e=z['sample_weight']*(p-z['y'])/S;gb=np.r_[z['standard'].T@e+beta/S,e.sum()];assert abs(val-v)<1e-11 and np.allclose(gu*np.r_[d,1.],gb,atol=1e-9,rtol=1e-9)
            fd=[]
            for j in [0,16,48,64,80,81,82,83]:
                eps=1e-5;up=u.copy();um=u.copy();up[j]+=eps;um[j]-=eps;der=(objective(z,up,d)[0]-objective(z,um,d)[0])/(2*eps);assert np.isclose(der,gu[j],atol=1e-7,rtol=1e-5);fd.append(abs(der-gu[j]))
            np.savez_compressed(O/f'precondition_f{f}_s{seed}.npz',d=d);checks.append(dict(fold=f,seed=seed,original_data_sha256=sha(src),max_scale=float(d.max()),loss_difference=abs(val-v),max_finite_difference_error=max(fd)));print('QUALIFIED_COORDINATES',f,seed,flush=True)
    save('preflight.json',dict(passed=True,pools=checks,all12loss_and_gradient_identity=True,finite_difference_coordinates=96,new_fits=0,new_NET_forwards=0));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),*O.glob('precondition_*.npz'),O/'preflight.json']});verify();archive('qualified');print('GRID193_QUALIFIED')
class CertifiedStop(Exception):
    def __init__(self,u,n):self.u=u.copy();self.n=n
def train():
    verify();assert read(O/'preflight.json')['passed'];assert not (O/'training_started.json').exists()
    for p,h in read(O/'qualified_freeze.json').items():assert sha(O/p)==h
    from scipy.optimize import minimize
    from sklearn.linear_model import LogisticRegression
    save('training_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fromzero_fits=12,wall_cap_s=600));(O/'models').mkdir();receipts=[];start=time.perf_counter()
    for f in range(4):
        for seed in range(3):
            if time.perf_counter()-start>600:raise TimeoutError('Fixed600s newcontrol;noautomaticextension')
            z=np.load(O/'fit_features'/f'f{f}_s{seed}.npz');d=np.load(O/f'precondition_f{f}_s{seed}.npz')['d'];iteration=[0];t=time.perf_counter()
            def fun(u):return objective(z,u,d)
            def callback(u):
                iteration[0]+=1;_,gu=fun(u)
                if float(abs(gu*np.r_[d,1.]).max())<=1e-4:raise CertifiedStop(u,iteration[0])
            try:
                res=minimize(fun,np.zeros(84),method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=2000,maxfun=15000,maxls=50,ftol=0.,gtol=1e-4/float(d.max())));u=res.x;status=str(res.message);nit=int(res.nit)
            except CertifiedStop as stop:u=stop.u;status='Original-gradient criterion certified by callback';nit=stop.n
            value,gu=fun(u);original_grad=float(abs(gu*np.r_[d,1.]).max());beta=u[:-1]/d;b=float(u[-1]);model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=2000);model.coef_=beta.reshape(1,-1);model.intercept_=np.array([b]);model.classes_=np.array([0,1]);model.n_features_in_=83;model.n_iter_=np.array([nit],dtype=np.int32);logit=z['standard']@beta+b
            assert np.allclose(logit,(z['standard']/d)@u[:-1]+u[-1],atol=1e-8,rtol=1e-10);assert np.allclose(logit,model.decision_function(z['standard']),atol=1e-10,rtol=1e-10);name=f'f{f}_s{seed}';np.savez_compressed(O/'models'/f'{name}.npz',coef=model.coef_,intercept=model.intercept_,mean=z['mean'],scale=z['scale'],classes=model.classes_,n_iter=model.n_iter_);(O/'models'/f'{name}.pkl').write_bytes(pickle.dumps(model,protocol=4));receipts.append(dict(fold=f,seed=seed,n_iter=nit,converged=original_grad<=1e-4,original_gradient_infinity=original_grad,objective=value,status=status,fit_wall_s=time.perf_counter()-t,weight_sha256=sha(O/'models'/f'{name}.npz')));save('fit_receipts.json',receipts);print('FIT_NUMERICAL_CONTROL',f,seed,nit,original_grad,flush=True)
    save('fits.json',dict(passed=True,new_fits=12,from_zero=True,records=receipts,all_converged=all(v['converged'] for v in receipts),total_wall_s=time.perf_counter()-start));save('trained_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'models').iterdir()),O/'fits.json',O/'fit_receipts.json']});verify();archive('trained');print('GRID193_ALL12_TRAINED')
def delegate(phase):
    bind();s=io.StringIO()
    try:
        with contextlib.redirect_stdout(s):getattr(B,phase)()
    except Exception:
        (O/f'{phase}_console.log').write_text(s.getvalue(),encoding='utf8');(O/f'{phase}_error.log').write_text(traceback.format_exc(),encoding='utf8');raise
    (O/f'{phase}_console.log').write_text(s.getvalue(),encoding='utf8');print(s.getvalue().replace('GRID190_','GRID193_'))
def publication():delegate('publication')
def evaluate():B.features=M.cached_features;delegate('evaluate')
if __name__=='__main__':globals()[sys.argv[1]]()
