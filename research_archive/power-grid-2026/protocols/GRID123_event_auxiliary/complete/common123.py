"""Same-capacity auxiliary supervision; saved feedback only."""
from pathlib import Path
import sys,datetime,collections,shutil
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid120_decision_objective'))
from common120 import *
from common120 import verify as verify120
sys.path.insert(0,str(R/'work/grid121_date_transfer'))
import common121 as folds
W=Path(__file__).parent;O=R/'outputs/grid123_event_auxiliary';OLD=R/'outputs/grid107_direct_value';CV=R/'outputs/grid121_date_transfer'
ARMS=['AUX_NETSIGN','AUX_GAIN']

def verify():
    for filename in ['code_freeze.json','data_freeze.json']:
        for name,h in read(O/filename).items():assert sha(R/name)==h,name
    verify120();folds.verify();sealed()

def getdata(split,part='train'):
    with np.load(O/f'{part}_data.npz') as z:x=z['X'];y=z['net'];event=z['gain_event'];prob=z['probability']
    if split=='full':return x,y,event,np.arange(len(y)),prob
    assert part=='train'
    fold=int(split[1:]);xx,yy,ids,held,p,_=folds.fold_data(fold);assert np.array_equal(y,yy)
    return xx,y,event,ids,p

def components(q,y,event,arm):
    main=.5*torch.nn.functional.smooth_l1_loss(q[:,1],y,beta=1.)
    label=event if arm=='AUX_GAIN' else (y>0).float()
    aux=.5*torch.nn.functional.binary_cross_entropy_with_logits(q[:,0],label)
    return main,aux

def diagnostic_gradients(net,main,aux):
    shared=[p for name,p in net.named_parameters() if not name.startswith('q_net.4.')]
    a=torch.autograd.grad(main,shared,retain_graph=True);b=torch.autograd.grad(aux,shared,retain_graph=True)
    a=torch.cat([v.reshape(-1).detach().double() for v in a]);b=torch.cat([v.reshape(-1).detach().double() for v in b]);an=float(a.norm());bn=float(b.norm())
    return dict(main_shared_norm=an,aux_shared_norm=bn,aux_over_main=bn/an if an else None,cosine=float(a@b)/(an*bn) if an*bn else None,
        main_half_loss=float(main.detach()),aux_half_loss=float(aux.detach()))
