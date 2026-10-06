"""Deterministic legal observations, without state identifiers or action outcomes."""
import re
import numpy as np
from core import TAU, system, vector_of

VTYPES=['X','Y','Z','I','L']
RTYPES=['inv_balance','lost_ub','capacity','compat','carry_implies_setup','no_two_carry','prod_activation','minlot','minlot_carry','one_carry','stability']

def finite(x): return np.where(np.isfinite(x),x,0.)

def make_features(bm,anchor,nominal,rlx,fr,masks,slots,stab,original,with_edges=True):
    a,lo,up=system(bm,stab); n=bm.idx.n; nr=a.shape[0]
    # Public per-row scaling, independent of train/val/test and labels.
    scale=np.maximum(1.,np.asarray(abs(a).max(axis=1).toarray()).ravel())
    available=bool(rlx['optimal'] and rlx['primal_valid'] and rlx['dual_valid'])
    x=rlx['x'] if available else np.zeros(n)
    cd=rlx['col_dual'] if available else np.zeros(n)
    rd=rlx['row_dual'] if available else np.zeros(nr)
    ref=vector_of(anchor,bm)
    # Type(5), bounds/missing, objective, ref, LP/RC/missing, FR/structural,
    # time/short-Y, fault/capacity, observable incremental loss heuristic.
    v=np.zeros((n,23),np.float32)
    incremental=np.maximum(0.,anchor.L-nominal.L)*np.asarray(bm.inst.l)[:,None]
    future=np.flip(np.cumsum(np.flip(incremental,axis=1),axis=1),axis=1)
    for c,(kind,i,j,t) in slots.items():
        v[c,VTYPES.index(kind)]=1.
        cap=bm.inst.c[j][max(0,t-1)] if j>=0 else sum(row[max(0,t-1)] for row in bm.inst.c)
        orig=original.c[j][max(0,t-1)] if j>=0 else sum(row[max(0,t-1)] for row in original.c)
        u=t+1 if kind=='Z' else t
        v[c,5:]=[finite(bm.var_lb[c]),finite(bm.var_ub[c]),not np.isfinite(bm.var_ub[c]),
                   bm.c[c],ref[c],x[c],cd[c],not available,c in fr,
                   bm.var_lb[c]==bm.var_ub[c],t/bm.inst.T,kind=='Y' and 1<=t<=TAU,
                   cap/max(1.,orig),cap<orig, (orig-cap)/max(1.,orig),
                   float(future[i,u-1]) if 1<=u<=bm.inst.T else 0.,
                   bm.inst.l[i],float(anchor.L[i,max(0,t-1)])]
    # Row type(11), bounds/missing, direction, activity, lower/upper slack,
    # dual, missing LP, time and known fault capacity change.
    r=np.zeros((nr,24),np.float32); activity=a@x
    names=bm.row_names+['stability']
    for row,name in enumerate(names):
        typ=name.split('[')[0]; r[row,RTYPES.index(typ)]=1.
        tm=re.search(r't(\d+)',name); t=int(tm.group(1)) if tm else 0
        jm=re.search(r'j(\d+)',name); j=int(jm.group(1)) if jm else -1
        fault=0.
        if typ=='capacity' and j>=0 and t>=1:
            fault=(original.c[j][t-1]-bm.inst.c[j][t-1])/max(1.,original.c[j][t-1])
        s=scale[row]
        r[row,11:]=[finite(lo[row])/s,finite(up[row])/s,not np.isfinite(lo[row]),not np.isfinite(up[row]),
                     np.isfinite(lo[row]) and lo[row]==up[row],activity[row]/s,
                     (activity[row]-lo[row])/s if np.isfinite(lo[row]) and available else 0.,
                     (up[row]-activity[row])/s if np.isfinite(up[row]) and available else 0.,
                     rd[row]*s,not available,t/bm.inst.T,0<t<=TAU,fault]
    out=dict(v=v,r=r,masks=masks.astype(np.float32))
    if with_edges:
        co=a.tocoo(); edge=(co.data/scale[co.row]).astype(np.float32)
        out.update(ei=np.stack([co.row,co.col]).astype(np.int64),
                   ev=np.stack([edge,np.sign(edge)],axis=1))
    assert np.isfinite(v).all() and np.isfinite(r).all()
    return out

def pooled(x,mask=None):
    x=x if mask is None else x[mask>0]
    if not len(x): return np.zeros(x.shape[1]*2,np.float32)
    return np.r_[x.mean(axis=0),x.max(axis=0)].astype(np.float32)

def table(g):
    glob=np.r_[pooled(g['v']),pooled(g['r'])]
    return np.stack([np.r_[glob,pooled(g['v'],mask),mask.mean(),mask.sum()/max(1,len(mask))] for mask in g['masks']]).astype(np.float32)

class Normalize:
    def fit(self,graphs):
        self.params={}
        for k in ['v','r']:
            count=sum(len(g[k]) for g in graphs)
            mean=sum(g[k].astype(np.float64).sum(axis=0) for g in graphs)/count
            var=sum(((g[k].astype(np.float64)-mean)**2).sum(axis=0) for g in graphs)/count
            self.params[k]=(mean.astype(np.float32),np.maximum(np.sqrt(var),1e-6).astype(np.float32))
        return self
    def transform(self,g):
        return {**g,**{k:((g[k]-m)/s).astype(np.float32) for k,(m,s) in self.params.items()}}
