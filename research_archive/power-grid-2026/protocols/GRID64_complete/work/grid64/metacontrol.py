"""Bhatia et al. (ICAPS 2022) stopping-only grid adaptation, not full A*/RRT* reproduction."""
import math,struct
import numpy as np
import torch
from torch import nn

DIM=12
NAMES=['query_fraction','pool_size_909','has_admissible','best_reward_half','best_rho_half',
       'last_quality_gain','stale_fraction','admissible_fraction','mean_quality','std_quality',
       'top_two_quality_gap','public_rho_limit_half']
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def quality(row):
    return 0. if row is None else float(np.clip(float(row['reward'])/2.,0.,1.))
def features(rows,n,limit,last_gain=0.,stale=0):
    valid=[r for r in rows if r['strict_admissible']]
    ranked=sorted(valid,key=key,reverse=True);best=ranked[0] if ranked else None
    qs=[quality(r) for r in valid]
    return np.asarray([len(rows)/max(1,n),n/909.,float(best is not None),quality(best),
        min(2.,best['rho']/2.) if best else 2.,last_gain,stale/16.,len(valid)/max(1,len(rows)),
        float(np.mean(qs)) if qs else 0.,float(np.std(qs)) if qs else 0.,
        quality(best)-quality(ranked[1]) if len(ranked)>1 else 0.,min(2.,limit/2.)],dtype=np.float32)
def utility(q,fraction,beta):return q-math.expm1(beta*fraction)
class QModel(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):return self.net(x)
def best_row(rows):return max((r for r in rows if r['strict_admissible']),key=key,default=None)
def prefixes(rows,limit):
    n=len(rows);ends=list(range(min(128,n),n,64))+[n]
    ends=sorted(set(ends));out=[];previous=None;stale=0
    for end in ends:
        subset=rows[:end];best=best_row(subset)
        improved=best is not None and (previous is None or key(best)>key(previous))
        gain=quality(best)-quality(previous) if previous is not None else 0.
        stale=0 if previous is None or improved else stale+1
        out.append({'end':end,'state':features(subset,n,limit,gain,stale),'quality':quality(best),
                    'chosen':None if best is None else best['pool_id']})
        previous=best
    return out
def mask_q(q,states):
    q=q.clone();q[...,0]=q[...,0].masked_fill(states[...,2]<.5,-1e9);return q
def greedy(model,state):
    with torch.inference_mode():q=model(torch.as_tensor(state));masked=mask_q(q,torch.as_tensor(state))
    return int(masked.argmax()),[float(v) for v in q]
def replay(model,rows,limit):
    ps=prefixes(rows,limit)
    for j,p in enumerate(ps):
        if j==len(ps)-1:return p
        if greedy(model,p['state'])[0]==0:return p
def run(order,observe,model,limit):
    rows=[];events=[];initial=min(128,len(order));best=None;previous=None;stale=0;forward=0
    prefix=None;reason='meta_pool_exhausted'
    for offset in range(0,len(order),64):
        # First 128 observations are one indivisible initial monitoring interval.
        if offset==64:continue
        ids=order[:initial] if offset==0 else order[offset:offset+64]
        for i in ids:
            row=observe(i,'meta_initial' if offset==0 else 'meta_continue');assert row['pool_id']==i
            rows.append(row)
        best=best_row(rows)
        if offset==0:prefix=None if best is None else best['pool_id']
        improved=best is not None and (previous is None or key(best)>key(previous))
        gain=quality(best)-quality(previous) if previous is not None else 0.
        stale=0 if previous is None or improved else stale+1
        state=features(rows,len(order),limit,gain,stale)
        if len(rows)==len(order):break
        action,q=greedy(model,state);forward+=1
        events.append({'queries':len(rows),'state':state.tolist(),'q_values':q,'action':action,
                       'forced_continue':best is None,'chosen':None if best is None else best['pool_id']})
        previous=best
        if action==0:reason='meta_learned_stop';break
    return {'outcomes':rows,'chosen_pool_id':None if best is None else best['pool_id'],
            'shortlist_choice':prefix,'fallback':len(rows)>initial,'stop_reason':reason,'events':events,
            'public_queries':len(rows),'local_queries':0,'local_substation':None,'meta_forward_count':forward}
