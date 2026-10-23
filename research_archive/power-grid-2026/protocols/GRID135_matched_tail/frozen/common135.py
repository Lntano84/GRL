"""Fixed four-arm full-completion classification, public inputs only."""
from pathlib import Path
import os,sys,json,hashlib,pickle,math
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid135_matched_tail';W=Path(__file__).parent
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID135_matched_tail'
sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from model import RelationalValue
from common import collate,tensor_cache
sys.path.insert(0,str(R/'work/grid112_relational_plan'))
from schema import pack,FIELDS
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed
ARMS=['FIRST_JOINT','FIRST_FACTOR','FULL_JOINT','FULL_FACTOR']
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):
    p=O/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');q.replace(p)
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def digest(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    sealed()
def net_for(arm,seed):
    torch.manual_seed(seed);return RelationalValue(arm.split('_')[1])
def encoded(c,x,norm,extent,fee,scale):
    assert set(c)==FIELDS and len(c['next_ids'])>64
    g=np.zeros(41,np.float32);g[:37]=np.clip((x[:37]-norm['mean'])/norm['scale'],-20,20)
    wanted=len(c['next_ids']) if extent=='FULL' else 64;chunks=[]
    for j in range(0,wanted,64):
        part=dict(c,next_ids=c['next_ids'][j:min(j+64,wanted)],next_plans=c['next_plans'][j:min(j+64,wanted)])
        chunks.append(pack(part,g,fee,scale))
    out=dict(chunks[0]);out['next_action']=np.concatenate([v['next_action'] for v in chunks])
    # Known full count/cost are identical across FIRST/FULL. No future outcome.
    out['known_query_cost']=np.array([fee*len(c['next_ids'])/scale],np.float32)
    return out
def predict(net,rows):
    ans=[];net.eval()
    with torch.no_grad():
        for j in range(0,len(rows),32):ans.append(net(**collate(rows[j:j+32])).numpy())
    return np.concatenate(ans)
def no_gain(logits):
    a=np.asarray(logits,dtype=np.float64);z=a-a.max(1,keepdims=True);p=np.exp(z);return p[:,0]/p.sum(1)
def threshold(scores,bad):return float(scores[bad].max()) if bad.any() else 1.
def numpy_forward(w,row,mode):
    def lin(x,k):return x@w[k+'.weight'].T+(w[k+'.bias'] if k+'.bias' in w else 0.)
    def enc(x,k):return np.maximum(lin(np.maximum(lin(x,k+'.0'),0),k+'.2'),0)
    a=enc(row['context_action'],'A');f=enc(row['context_feedback'],'F');q=enc(row['next_action'],'A');anchor=enc(row['anchor_action'],'A')
    if mode=='FACTOR':f=np.broadcast_to(f.mean(0),a.shape)
    v=enc(np.concatenate([a,f],-1),'V');att=lin(q,'Q')@lin(a,'K').T/math.sqrt(8);att=np.exp(att-att.max(1,keepdims=True));att/=att.sum(1,keepdims=True)
    g=enc(np.concatenate([row['global41'],anchor,row['known_query_cost']]),'G')
    h=enc(np.concatenate([q,att@v,np.broadcast_to(g,(len(q),16))],-1),'H')
    return lin(np.maximum(lin(np.concatenate([h.mean(0),h.max(0),g]),'head.0'),0),'head.2')
