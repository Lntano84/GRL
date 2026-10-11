"""Public-only replacements for ranking, leaving masks/dispatch/stop/caches intact."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/grid35b'))
import numpy as np
from physics_order import order_from_public_edges
def rank(self,obs,base,eligible,reps,aliases):
    times={'feature_s':0.,'prior_s':0.,'forward_rank_s':0.}
    start=time.perf_counter()
    if self.rule=='STATIC_ID':order=sorted(reps)
    elif self.rule=='LOCAL_STRESS':
        pressure=np.zeros(self.env.n_sub);live=np.flatnonzero(obs.line_status)
        for ends in [self.env.line_or_to_subid,self.env.line_ex_to_subid]:np.maximum.at(pressure,ends[live],obs.rho[live])
        order=sorted(reps,key=lambda i:(-float(pressure[self.subids[i]]),i))
    elif self.rule=='PHYSICS_TAIL':
        live=np.flatnonzero(obs.line_status);u=self.env.line_or_to_subid[live];v=self.env.line_ex_to_subid[live]
        edges=np.stack([np.concatenate([u,v]),np.concatenate([v,u])])
        order=order_from_public_edges(edges,obs.rho[live],self.env.n_sub,eligible,self.subids,aliases)
    else:raise ValueError(self.rule)
    assert len(order)==len(reps) and set(order)==set(reps)
    times['forward_rank_s']=time.perf_counter()-start
    return order,None,None,times
