"""Fixed-size gain/known-cost learning on saved teacher feedback only."""
from pathlib import Path
import sys,datetime,collections,shutil
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid123_event_auxiliary'))
import common123 as src
import numpy as np
import torch
W=Path(__file__).parent;O=R/'outputs/grid126_known_cost_gain';S=src.O;OLD=src.OLD;CV=src.CV
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID126_known_cost_gain'
read=src.read;save=src.save;sha=src.sha;build=src.build;digest=src.digest
ARMS=['GAIN_DIRECT','COST_POSITIVE'];ALL=['NET_ZERO']+ARMS

def verify():
    for fn in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/fn).items():assert sha(R/n)==h,n
    src.verify();src.sealed()

def getdata(split,part='train'):
    x,y,event,ids,p=src.getdata(split,part);z=np.load(O/f'{part}_targets.npz')
    return x,y,z['gain'],z['charge'],ids,p

def torch_gain(r,c,arm):
    if arm=='GAIN_DIRECT':return r
    assert arm=='COST_POSITIVE';h=torch.hypot(r,c)
    # Equivalent to r+sqrt(r^2+c^2),with stable negative branch and no zero denominator.
    return torch.where(r>=0,r+h,c.square()/(h-torch.minimum(r,torch.zeros_like(r))))

def numpy_gain(r,c,arm):
    r=r.astype(float);c=c.astype(float)
    if arm=='NET_ZERO':return r+c
    if arm=='GAIN_DIRECT':return r
    assert arm=='COST_POSITIVE';h=np.hypot(r,c)
    return np.where(r>=0,r+h,c*c/(h-np.minimum(r,0.)))

def actions(q,c,arm):
    # Analytically COST_POSITIVE net>0 iff latent r>0,avoids subtractive cancellation.
    return q[:,1]>c if arm=='GAIN_DIRECT' else q[:,1]>0

def losses(q,g,c,arm):
    main=.5*torch.nn.functional.smooth_l1_loss(torch_gain(q[:,1],c,arm),g,beta=1.)
    stop=.5*torch.nn.functional.smooth_l1_loss(q[:,0],torch.zeros_like(g),beta=1.)
    return main,stop

def profile(net,x,y,g,c,ids,p,arm):
    q=src.forward(net,x[ids]);raw=q[:,1].astype(float);cost=c[ids];gg=numpy_gain(raw,cost,arm);err=gg-g[ids];hub=np.where(abs(err)<1,.5*err**2,abs(err)-.5);go=actions(q,cost,arm);jac=np.ones(len(ids)) if arm=='GAIN_DIRECT' else 1+raw/np.hypot(raw,cost)
    pos=y[ids]>0
    return dict(main_huber=float(p@hub),continue_mass=float(p@go),sign_regret=float(p@(np.maximum(y[ids],0)-go*y[ids])),gain_negative_mass=float(p@(gg<0)),jacobian_mean=float(p@jac),positive_target_jacobian_mean=float(p[pos]@jac[pos]/p[pos].sum()),minimum_jacobian=float(jac.min()))

def archive(phase):
    a=A/phase;a.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,a/p.name)
        elif p.name=='models':shutil.copytree(p,a/p.name)
    save(a/'archive_manifest.json',{p.relative_to(a).as_posix():sha(p) for p in a.rglob('*') if p.is_file()})

def docs(message,next_steps):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_steps,encoding='utf8')
