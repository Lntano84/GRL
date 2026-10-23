"""Draft input contract only: NumPy, no model, optimizer, target or simulator."""
import numpy as np
MAX_DEVICES=17
FEEDBACK=['reward_gap_asinh','reward_finite','legal_feedback','strict_admissible','done','illegal','ambiguous','exception_present']
FIELDS={'seen_ids','seen_plans','seen_feedback','next_ids','next_plans','incumbent_id','incumbent_reward','incumbent_plan'}

def canonical(plan):
    p=np.asarray(plan,dtype=int);assert 0<len(p)<=MAX_DEVICES and set(p.tolist())<={-1,1,2}
    active=p>0;bits=np.zeros(MAX_DEVICES,dtype=np.float32);live=np.zeros(MAX_DEVICES,dtype=np.float32);present=np.zeros(MAX_DEVICES,dtype=np.float32)
    if active.any():bits[:len(p)]=((p!=p[np.flatnonzero(active)[0]])&active).astype(float)
    live[:len(p)]=active;present[:len(p)]=1
    return bits,live,present

def action(plan,incumbent_plan):
    b,a,m=canonical(plan);ib,_,_=canonical(incumbent_plan)
    return np.concatenate([b,b-ib,a,m]).astype(np.float32)

def feedback(row,incumbent_reward,gain_scale):
    finite=bool(np.isfinite(row['reward']));rho_ok=bool(np.isfinite(row['rho']) and row['rho']>0)
    legal=not(row['done'] or row['illegal'] or row['ambiguous'] or row['exceptions']) and finite and rho_ok
    gap=np.arcsinh((float(np.float32(row['reward']))-incumbent_reward)/gain_scale) if finite else 0.
    return np.array([gap,finite,legal,row['strict_admissible'],row['done'],row['illegal'],row['ambiguous'],bool(row['exceptions'])],dtype=np.float32)

def pack(c,normalized_global41,cost_per_query,gain_scale):
    assert set(c)==FIELDS and len(c['seen_ids'])>0 and 0<len(c['next_ids'])<=64
    g=np.asarray(normalized_global41,dtype=np.float32);assert g.shape==(41,)
    a=np.stack([action(p,c['incumbent_plan']) for p in c['seen_plans']]);f=np.stack([feedback(r,c['incumbent_reward'],gain_scale) for r in c['seen_feedback']])
    q=np.stack([action(p,c['incumbent_plan']) for p in c['next_plans']]);anchor=action(c['incumbent_plan'],c['incumbent_plan'])
    result=dict(context_action=a,context_feedback=f,next_action=q,anchor_action=anchor,global41=g,
        known_query_cost=np.array([cost_per_query*len(q)/gain_scale],dtype=np.float32))
    assert all(np.isfinite(v).all() for v in result.values())
    return result
