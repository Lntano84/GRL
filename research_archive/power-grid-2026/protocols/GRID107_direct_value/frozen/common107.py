"""Same-network fixed-budget supervised value diagnosis; no physical environment."""
from pathlib import Path
import hashlib,json,os,pickle
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid107_direct_value'
REPO=ROOT/'work/GRL-results-integration';ARCHIVE=REPO/'research_archive/power-grid-2026/protocols/GRID107_direct_value'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def scalar(v):
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=scalar);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def sealed():
    d=read(ROOT/'outputs/grid100_data_foundation/design.json')
    for r in d['scenario_plan']:
        if r['partition']=='test':
            assert not (ROOT/d['environment']/'chronics'/r['scenario']).exists()
            assert not (ROOT/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists()
def verify():
    for n in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
    sealed()
def loadcases():
    with (ROOT/'outputs/grid102_search_rl/cases.pkl').open('rb') as f:return pickle.load(f)
def normalization():
    with np.load(ROOT/'outputs/grid102_search_rl/normalization.npz') as z:return z['mean'],z['scale']
def normalized(x,mean,std):return np.clip((x-mean)/std,-20,20).astype(np.float32)
def weight_digest(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()
def carrier(seed,params):
    import gymnasium as gym
    from stable_baselines3 import DQN
    class NeverStepped(gym.Env):
        def __init__(self):
            self.action_space=gym.spaces.Discrete(2)
            self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(41,),dtype=np.float32)
        def reset(self,**kw):raise RuntimeError('Carrier spaces only; no environment transition is permitted')
        def step(self,a):raise RuntimeError('No environment transitions in supervised fitting')
    return DQN('MlpPolicy',NeverStepped(),seed=seed,**params)
def predict(net,X):
    import torch
    with torch.no_grad():return net(torch.as_tensor(X,dtype=torch.float32)).cpu().numpy()
def state_metrics(target,q,metadata):
    action=q[:,1]>q[:,0];positive=target>0;regret=np.maximum(target,0)-action*target
    dates=sorted({r['date'] for r in metadata});groups=sorted({r['physical_state_group'] for r in metadata})
    date_mask={d:np.array([r['date']==d for r in metadata]) for d in dates}
    group_mask={g:np.array([r['physical_state_group']==g for r in metadata]) for g in groups}
    stats=dict(states=len(target),positive=int(positive.sum()),false_stop=int(np.sum(positive&~action)),
        unprofitable_continue=int(np.sum(~positive&action)),positive_recall=float(action[positive].mean()) if positive.any() else None,
        missed_positive_net=float(target[positive&~action].sum()),negative_query_net=float(-target[~positive&action].sum()),
        mean_realized_sign_regret=float(regret.mean()),
        date_macro_realized_sign_regret=float(np.mean([regret[m].mean() for m in date_mask.values()])),
        physical_group_macro_realized_sign_regret=float(np.mean([regret[m].mean() for m in group_mask.values()])),
        per_date={d:dict(states=int(m.sum()),positive=int(positive[m].sum()),false_stop=int(np.sum(positive[m]&~action[m])),
            unprofitable_continue=int(np.sum(~positive[m]&action[m])),mean_realized_sign_regret=float(regret[m].mean())) for d,m in date_mask.items()})
    return stats
def summarize(rows):
    result={}
    for date in sorted({r['date'] for r in rows}):
        rs=[r for r in rows if r['date']==date];gain=sum(r['full_gain'] for r in rs);queries=sum(r['full_queries'] for r in rs)
        gs=sorted({r['physical_state_group'] for r in rs})
        result[date]=dict(cases=len(rs),mean_net_utility=float(np.mean([r['net_utility'] for r in rs])),
            mean_gain=float(np.mean([r['gain'] for r in rs])),mean_queries=float(np.mean([r['queries'] for r in rs])),
            query_fraction=sum(r['queries'] for r in rs)/queries,gain_capture=sum(r['gain'] for r in rs)/gain if gain else None,
            physical_call_balanced_utility=float(np.mean([np.mean([r['net_utility'] for r in rs if r['physical_state_group']==g]) for g in gs])))
    return dict(per_date=result,macro={k:float(np.mean([v[k] for v in result.values() if v[k] is not None])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_call_balanced_utility']})
