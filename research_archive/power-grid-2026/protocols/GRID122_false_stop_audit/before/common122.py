"""Read-only support/visibility audit. No model/optimizer/environment construction."""
from pathlib import Path
import sys,collections,datetime,shutil
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid121_date_transfer'))
from common121 import *
from common121 import verify as verify121
W=Path(__file__).parent;O=R/'outputs/grid122_false_stop_audit';SOURCE=R/'outputs/grid121_date_transfer'
FEATURES=read(R/'outputs/grid102_search_rl/design.json')['input']['features']
EXTRA=['known_same_station','known_strict','known_legal','legal_reward_gap_asinh_mean','legal_reward_gap_asinh_std','next_anchor_pair_distance_mean','next_seen_pair_distance_mean','next_exact_seen_fraction']

def verify():
    for filename in ['code_freeze.json','data_freeze.json']:
        for name,h in read(O/filename).items():assert sha(R/name)==h,name
    verify121();sealed()

def public_descriptors(row):
    """Only existing paid feedback and publicly known action plans."""
    feedback=row['context_feedback'];legal=feedback[:,2].astype(bool)
    active=row['anchor_action'][34:51].astype(bool)&row['anchor_action'][51:68].astype(bool)
    assert np.all(row['context_action'][:,34:68]==row['anchor_action'][34:68]) and np.all(row['next_action'][:,34:68]==row['anchor_action'][34:68])
    ij=np.asarray([(i,j) for i in range(17) for j in range(i+1,17) if active[i] and active[j]],int)
    assert len(ij)>0
    def signature(a):return (a[...,ij[:,0]]==a[...,ij[:,1]]).astype(np.int16)
    seen=signature(row['context_action']);pending=signature(row['next_action']);anchor=signature(row['anchor_action'])
    dist=np.not_equal(pending[:,None,:],seen[None,:,:]).mean(-1)
    vals=feedback[legal,0]
    return np.asarray([len(seen),feedback[:,3].sum(),legal.sum(),vals.mean() if len(vals) else 0.,vals.std() if len(vals) else 0.,
        np.not_equal(pending,anchor).mean(),dist.min(1).mean(),np.mean(dist.min(1)==0)],float)

def nearest_batch(query,reference,query_ids,reference_ids,groups=None,exclude_same_call=False):
    """Exact Euclidean candidates; refine near-minimums by direct subtraction."""
    result=[];rr=(reference*reference).sum(1)
    for start in range(0,len(query),96):
        block=query[start:start+96];d=(block*block).sum(1)[:,None]+rr[None,:]-2*block@reference.T
        for j,row in enumerate(block):
            qi=int(query_ids[start+j]);allowed=np.ones(len(reference),bool)
            if exclude_same_call:allowed=np.array([groups[qi]!=groups[int(i)] for i in reference_ids])
            assert allowed.any();dd=d[j].copy();dd[~allowed]=np.inf;approx=dd.min()
            close=np.flatnonzero(allowed&(dd<=approx+1e-8));exact=((reference[close]-row)**2).sum(1);minimum=float(exact.min());ties=close[exact==minimum]
            # The refinement tolerance only guards roundoff in candidate selection; final ties are exact.
            result.append(dict(index=qi,distance_squared=minimum,nearest_indices=[int(reference_ids[t]) for t in ties]))
    return result

def weighted_quantile(values,weights,q):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weights)[order];return float(v[np.searchsorted(np.cumsum(w),q*w.sum(),side='left')])
