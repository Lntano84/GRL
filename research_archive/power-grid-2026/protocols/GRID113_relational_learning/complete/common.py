"""GRID113 fixed paired learning; saved public inputs only, no environment."""
from pathlib import Path
import os,sys,json,hashlib,pickle
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid113_relational_learning';REPO=R/'work/GRL-results-integration'
sys.path.insert(0,str(R/'work/grid112_relational_plan'))
from schema import pack,canonical
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import state_metrics,summarize,sealed
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item());f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def digest(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def configure():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def verify():
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(O/name).items():assert sha(R/p)==h,p
    sealed()
def collate(rows):
    import torch
    n=max(len(r['context_action']) for r in rows);m=max(len(r['next_action']) for r in rows);b=len(rows)
    d={k:np.zeros((b,n,w),np.float32) for k,w in [('context_action',68),('context_feedback',8)]}
    d['next_action']=np.zeros((b,m,68),np.float32);d['context_mask']=np.zeros((b,n),bool);d['next_mask']=np.zeros((b,m),bool)
    for i,r in enumerate(rows):
        ni=len(r['context_action']);mi=len(r['next_action'])
        for k in ['context_action','context_feedback']:d[k][i,:ni]=r[k]
        d['next_action'][i,:mi]=r['next_action'];d['context_mask'][i,:ni]=True;d['next_mask'][i,:mi]=True
    for k in ['anchor_action','global41','known_query_cost']:d[k]=np.stack([r[k] for r in rows])
    return {k:torch.from_numpy(v) for k,v in d.items()}
def tensor_cache(rows):
    # Retain all rows; slice each minibatch to its own maximum, no topK/truncation.
    d=collate(rows);nc=d['context_mask'].sum(1);nq=d['next_mask'].sum(1)
    def take(ids):
        ids=np.asarray(ids);n=int(nc[ids].max());m=int(nq[ids].max());out={}
        for k,v in d.items():
            a=v[ids]
            if k in ['context_action','context_feedback','context_mask']:a=a[:,:n]
            if k in ['next_action','next_mask']:a=a[:,:m]
            out[k]=a
        return out
    return take
def predict(net,rows,batch=32):
    import torch
    net.eval();ans=[]
    with torch.no_grad():
        for j in range(0,len(rows),batch):ans.append(net(**collate(rows[j:j+batch])).numpy())
    return np.concatenate(ans)
