"""Shared action access and causal offline/online replay before fitting."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid77';sys.path.insert(0,str(Path(__file__).parent));import metacontrol as M
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
class Constant(nn.Module):
    def __init__(self,action):super().__init__();self.action=action
    def forward(self,x):
        y=torch.zeros((*x.shape[:-1],3));y[...,self.action]=10.;return y
def main():
    records=read(OUT/'records.json');subs=read(OUT/'data_receipts.json')['subids'];checks=0;suffix=0
    for rec in records:
        rec['subids']=subs;rec['aliases']={int(k):int(v) for k,v in rec['aliases'].items()}
        rec['context']={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in rec['context'].items()}
        rec['prefixes']=M.prefixes(rec)
        for action in [0,1,2]:
            model=Constant(action);seen=[];truth={r['pool_id']:r for r in rec['rows']}
            def observe(i,p):
                assert i not in seen;seen.append(i);return {**truth[i],'phase':p}
            got=M.run(rec['order'],observe,0,subs,rec['aliases'],rec['rho_limit'],rec['context'],model)
            ref=M.replay(model,rec)
            assert got['public_queries']==ref['end'] and got['chosen_pool_id']==ref['chosen']
            assert seen==[r['pool_id'] for r in got['outcomes']]
            if action==2 and len(rec['rows'])>128:assert got['chosen_pool_id']==rec['guard_choice'] and got['public_queries']==rec['guard_queries']
            checks+=1
        first=rec['prefixes'][0]
        altered=[dict(r,reward=-999.,rho=999.,strict_admissible=False) for r in rec['rows'][first['end']:]]
        x=M.state((rec['rows'][:first['end']]+altered)[:first['end']],rec['order'],subs,rec['aliases'],rec['rho_limit'],rec['context'],initial=True)
        assert np.array_equal(x,first['state']);suffix+=1
    q=M.mask_q(torch.zeros(3),torch.tensor([0.,0.,0.]+[0.]*22));assert q[0]<-1e8 and q[2]<-1e8 and q[1]==0
    result=dict(passed=True,shared_mode_checks=checks,hidden_suffix_checks=suffix,new_queries=0,new_fits=0,
        role='Source-style DQN has exact guarded mode available and same public operating context. No evaluation data.')
    (OUT/'learning_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
