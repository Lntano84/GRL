"""Fixed decision-objective screen on existing data; no environment."""
from pathlib import Path
import sys
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from common import *
from common import verify as verify113
sys.path.insert(0,str(R/'work/grid115_matched_module_plan'))
from matched_model import OriginalBackbone
import torch
W=Path(__file__).parent;O=R/'outputs/grid120_decision_objective';S=R/'outputs/grid113_relational_learning'
ARMS=['SIGN_BCE','UTILITY_BCE']

def build(seed):
    torch.manual_seed(seed)
    return OriginalBackbone()

def loss_per_state(q,target,arm,normalizer):
    margin=q[:,1]-q[:,0]
    loss=torch.nn.functional.binary_cross_entropy_with_logits(margin,(target>0).float(),reduction='none')
    if arm=='UTILITY_BCE':loss=loss*target.abs()/normalizer
    return loss

def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for n,h in read(O/name).items():assert sha(R/n)==h,n
    verify113();sealed()

def data(part):
    meta=read(S/f'{part}_metadata.json');raw=pickle.loads((S/f'{part}_public.pkl').read_bytes())
    return np.stack([r['global41'] for r in raw]),np.load(S/f'{part}_targets.npz')['target'],meta

def forward(net,x):
    net.eval()
    with torch.no_grad():return torch.cat([net(torch.from_numpy(x[i:i+64])) for i in range(0,len(x),64)]).numpy()

def trajectory(c,name,actions,fee,scale):
    p=0;seq=[]
    while p<len(c['states']):
        go=int(actions[p]);seq.append(go)
        if not go:break
        p+=1
    gain=float(c['gains'][:p].sum());queries=int(c['queries'][:p].sum())
    return dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],scenario=c['scenario'],method=name,horizon=len(c['states']),actions=seq,batches=p,gain=gain,queries=queries,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),net_utility=(gain-fee*queries)/scale,chosen_pool_id=c['chosen'][p])

def summaries(trajectories):
    ans={}
    for name in sorted({r['method'] for r in trajectories}):
        rows=[r for r in trajectories if r['method']==name];ans[name]={}
        for label,ss in [('all',rows),('single',[r for r in rows if r['horizon']==1]),('multi',[r for r in rows if r['horizon']>1])]:
            if ss:ans[name][label]=summarize(ss)
    return ans
