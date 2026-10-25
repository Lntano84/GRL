"""Array implementation of the same qualified physical descriptors."""
import numpy as np
FIELDS={'terminal_power','rho','line_status','incident_lines','anchor_plan','next_plans'}
def state4(c):
    rho=np.asarray(c['rho'],float);live=np.asarray(c['line_status'],bool);incident=np.asarray(c['incident_lines'],bool)&live;over=(rho>1)&live
    return np.array([rho[live].max() if live.any() else 0.,np.maximum(rho[live]-1,0).mean() if live.any() else 0.,rho[incident].max() if incident.any() else 0.,np.sum(over&incident)/np.sum(over) if over.any() else 0.],dtype=np.float32)
def power8(c):
    assert set(c)==FIELDS
    p=np.asarray(c['terminal_power'],float);plans=np.vstack([np.asarray(c['anchor_plan'],int),np.asarray(c['next_plans'],int)]);assert plans.shape[1]==len(p) and 2<=len(plans)<=65
    denominator=1+np.abs(p).sum();imbalance=(np.abs(((plans==1)*p).sum(1))+np.abs(((plans==2)*p).sum(1)))/denominator;anchor=imbalance[0];values=imbalance[1:]
    return np.r_[state4(c),np.log1p(np.abs(p).sum()),anchor,values.min()-anchor,values.mean()-anchor].astype('f')
