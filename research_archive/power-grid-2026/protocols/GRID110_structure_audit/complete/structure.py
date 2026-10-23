"""Public command intentions and current-state pressure; never forecast outcomes."""
import numpy as np
NAMES=['next_best_partition_distance_min','next_best_partition_distance_mean','next_best_partition_distance_max',
       'next_seen_nearest_distance_mean','next_seen_nearest_distance_max','next_novel_partition_fraction',
       'next_current_partition_distance_mean','current_max_rho','current_overload_excess_mean',
       'station_incident_max_rho','station_incident_overload_share']

def signature(topology):
    a=np.asarray(topology);u,v=np.triu_indices(len(a),1)
    return ((a[u]>0)&(a[v]>0)&(a[u]==a[v])).astype(np.uint8)

def compute(c):
    assert set(c)=={'plans','incumbent_id','seen_station_ids','next_ids','current_station_topology','rho','line_status','incident_lines'}
    plans=c['plans']; incumbent=c['incumbent_id'];seen=c['seen_station_ids'];nxt=c['next_ids']
    assert incumbent in seen and not set(seen)&set(nxt) and nxt
    sig={i:signature(plans[i]) for i in set(seen)|set(nxt)}
    assert len(sig[incumbent])>0
    a=np.stack([sig[i] for i in nxt]);b=sig[incumbent]
    db=np.mean(a!=b,axis=1)
    ds=np.min(np.mean(a[:,None,:]!=np.stack([sig[i] for i in seen])[None,:,:],axis=2),axis=1)
    dc=np.mean(a!=signature(c['current_station_topology']),axis=1)
    rho=np.asarray(c['rho']);live=np.asarray(c['line_status'],bool);inc=np.asarray(c['incident_lines'],bool)&live;over=(rho>1)&live
    x=np.array([db.min(),db.mean(),db.max(),ds.mean(),ds.max(),np.mean(ds>0),dc.mean(),
        rho[live].max() if live.any() else 0,np.maximum(rho[live]-1,0).mean() if live.any() else 0,
        rho[inc].max() if inc.any() else 0,np.sum(over&inc)/over.sum() if over.any() else 0],dtype=float)
    assert x.shape==(11,) and np.isfinite(x).all()
    return x
