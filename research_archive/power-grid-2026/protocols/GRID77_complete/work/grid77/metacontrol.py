"""ICAPS2022-style DQN with common context and an explicit verification mode.

RETURN / GLOBAL64 / DELEGATE_GUARD(initial only) are grid-mode adaptations,
not the original weighted-A*/RRT* task. Same public context as cheap gate.
"""
from pathlib import Path
import functools,importlib.util,json,math
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
G=module('grid77_context_features','work/grid75/feature_gate.py')
P=module('grid77_guard_reference','work/grid70/policies.py')
quality=G.M.quality;key=G.M.key;utility=G.M.utility
DIM=25
class QModel(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):return self.net(x)
def state(rows,order,subs,aliases,limit,context,gain=0.,stale=0,initial=False):
    x=G.features(rows,order,subs,aliases,limit,context)[:24].copy();x[5]=gain;x[6]=stale/16.
    return np.concatenate((x,np.asarray([initial],dtype=np.float32)))
def mask_q(q,s):
    q=q.clone();q[...,0]=q[...,0].masked_fill(s[...,2]<.5,-1e9);q[...,2]=q[...,2].masked_fill(s[...,24]<.5,-1e9);return q
def greedy(model,s):
    t=torch.as_tensor(s)
    with torch.inference_mode():q=mask_q(model(t),t)
    return int(q.argmax()),q.tolist()
def prefixes(rec):
    rows=rec['rows'];order=rec['order'];ends=sorted(set([*range(min(128,len(rows)),len(rows),64),len(rows)]));result=[];prev=None;stale=0
    for j,end in enumerate(ends):
        best=G.M.best_row(rows[:end]);improved=best is not None and (prev is None or key(best)>key(prev))
        gain=quality(best)-quality(prev) if prev is not None else 0.;stale=0 if prev is None or improved else stale+1
        result.append(dict(end=end,quality=quality(best),chosen=None if best is None else best['pool_id'],
            state=state(rows[:end],order,rec['subids'],rec['aliases'],rec['rho_limit'],rec['context'],gain,stale,j==0)))
        prev=best
    return result
def replay(model,rec):
    ps=rec['prefixes']
    for j,p in enumerate(ps):
        if j==len(ps)-1:return dict(p,mode='exhausted')
        action,_=greedy(model,p['state'])
        if action==0:return dict(p,mode='return')
        if action==2:
            assert j==0
            return dict(end=rec['guard_queries'],quality=rec['guard_quality'],chosen=rec['guard_choice'],mode='guard')
@functools.lru_cache(maxsize=2)
def load(seed):
    selected=json.loads((ROOT/'outputs/grid77/selected.json').read_text(encoding='utf-8'))
    model=QModel();model.load_state_dict(torch.load(ROOT/f"outputs/grid77/models/cost{selected['costindex']}_seed{seed}/final.pt",map_location='cpu',weights_only=True));model.eval();return model
def run(order,observe,seed,subs,aliases,limit,context,model=None):
    model=load(seed) if model is None else model
    rows=[];events=[];best=None;prev=None;stale=0;n=min(128,len(order));prefix=None
    def buy(ids,phase):
        for i in ids:
            r=observe(i,phase);assert r['pool_id']==i;rows.append(r)
    buy(order[:n],'meta_initial');prefix=G.M.best_row(rows);reason='meta_context_exhausted'
    while True:
        best=G.M.best_row(rows)
        if len(rows)==len(order):break
        improved=best is not None and (prev is None or key(best)>key(prev));gain=quality(best)-quality(prev) if prev is not None else 0.
        stale=0 if prev is None or improved else stale+1
        initial=len(rows)==n
        x=state(rows,order,subs,aliases,limit,context,gain,stale,initial);action,q=greedy(model,x)
        events.append(dict(queries=len(rows),state=x.tolist(),q_values=q,action=action,chosen=None if best is None else best['pool_id']))
        if action==0:reason='meta_context_return';break
        if action==2:
            assert initial;cached={r['pool_id']:r for r in rows}
            def reveal(i,phase):
                if i in cached:return cached.pop(i)
                return observe(i,'meta_guard_'+phase)
            result=P.run(order,reveal,'FRAME_GUARD',subs,aliases,limit=limit);assert not cached
            result.update(stop_reason='meta_delegate_'+result['stop_reason'],meta_events=events,meta_forward_count=len(events))
            return result
        assert action==1;prev=best;end=len(rows);buy(order[end:end+64],'meta_global')
    return dict(outcomes=rows,chosen_pool_id=None if best is None else best['pool_id'],shortlist_choice=None if prefix is None else prefix['pool_id'],
        fallback=len(rows)>n,stop_reason=reason,events=[],public_queries=len(rows),local_substation=None,local_queries=0,
        completion_budget=0,global_extra_queries=0,pre_extra_choice=None if best is None else best['pool_id'],completion_candidate_ids=[],extra_selected_ids=[],
        meta_events=events,meta_forward_count=len(events),verification_guard=False,prefix_safe_at_entry=prefix is not None and prefix['rho']<.9)
