"""Same positive gain link,matched mean-targeted losses; saved data only."""
from pathlib import Path
import sys,datetime,collections,shutil
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid126_known_cost_gain'))
import common126 as base
src=base.src;np=base.np;torch=base.torch
W=Path(__file__).parent;O=R/'outputs/grid127_mean_gain_objective';S=base.S;OLD=base.OLD;CV=base.CV;BASE=base.O
REPO=base.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID127_mean_gain_objective'
read=base.read;save=base.save;sha=base.sha;build=base.build;digest=base.digest
ARMS=['POS_MSE','POS_KL'];ALL=['NET_ZERO','POS_HUBER']+ARMS

def verify():
    for fn in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/fn).items():assert sha(R/n)==h,n
    base.verify();src.sealed()

def getdata(split,part='train'):
    return base.getdata(split,part)

def torch_gain(r,c,arm):return base.torch_gain(r,c,'COST_POSITIVE')
def numpy_gain(r,c,arm):return base.numpy_gain(r,c,'NET_ZERO' if arm=='NET_ZERO' else 'COST_POSITIVE')
def actions(q,c,arm):return q[:,1]>0

def losses(q,g,c,arm):
    pred=torch_gain(q[:,1],c,arm)
    if arm=='POS_MSE':main=.5*(pred-g).square().mean()
    else:
        assert arm=='POS_KL';logpred=torch.log(c)+torch.asinh(q[:,1]/c);logtarget=torch.log(g.clamp_min(torch.finfo(g.dtype).tiny));main=.5*(pred-g+g*(logtarget-logpred)).mean()
    stop=.5*torch.nn.functional.smooth_l1_loss(q[:,0],torch.zeros_like(g),beta=1.)
    return main,stop

def profile(net,x,y,g,c,ids,p,arm):
    q=src.forward(net,x[ids]);r=q[:,1].astype(float);cost=c[ids];pg=numpy_gain(r,cost,arm);err=pg-g[ids];h=np.hypot(r,cost);jac=1+r/h;pos=y[ids]>0;high=y[ids]>1;hub=np.where(abs(err)<1,.5*err**2,abs(err)-.5);grad=err*jac if arm=='POS_MSE' else .5*err/h
    objective=.5*err**2 if arm=='POS_MSE' else .5*(pg-g[ids]+g[ids]*(np.log(np.maximum(g[ids],np.finfo(float).tiny))-np.log(cost)-np.arcsinh(r/cost)))
    go=r>0
    return dict(main_huber=float(p@hub),main_objective=float(p@objective),continue_mass=float(p@go),sign_regret=float(p@(np.maximum(y[ids],0)-go*y[ids])),gain_negative_mass=float(p@(pg<0)),jacobian_mean=float(p@jac),positive_target_jacobian_mean=float(p[pos]@jac[pos]/p[pos].sum()),minimum_jacobian=float(jac.min()),positive_main_gradient_abs=float(p[pos]@abs(grad[pos])/p[pos].sum()),large_positive_main_gradient_abs=float(p[high]@abs(grad[high])/p[high].sum()),all_main_gradient_abs=float(p@abs(grad)))

def archive(phase):
    a=A/phase;a.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,a/p.name)
        elif p.name=='models':shutil.copytree(p,a/p.name)
    save(a/'archive_manifest.json',{p.relative_to(a).as_posix():sha(p) for p in a.rglob('*') if p.is_file()})

def docs(message,steps):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(steps,encoding='utf8')
