"""Current-state action descriptors, without forecast, feedback outcomes or labels."""
import numpy as np
NAMES=['current_live_max_rho','current_overload_excess_mean','station_incident_max_rho','station_incident_overload_share',
       'log_station_absolute_active_power','anchor_frozen_flow_imbalance','next_min_minus_anchor_imbalance','next_mean_minus_anchor_imbalance']
FIELDS={'terminal_power','rho','line_status','incident_lines','anchor_plan','next_plans'}

def imbalance(plan,power):
    plan=np.asarray(plan);power=np.asarray(power,float)
    assert plan.shape==power.shape and set(plan.tolist())<={-1,1,2}
    return sum(abs(float(power[plan==b].sum())) for b in [1,2])/(1+float(np.abs(power).sum()))

def compute(c):
    assert set(c)==FIELDS
    power=np.asarray(c['terminal_power'],float);rho=np.asarray(c['rho'],float);live=np.asarray(c['line_status'],bool);incident=np.asarray(c['incident_lines'],bool)&live
    assert 0<len(power)<=17 and rho.shape==live.shape==incident.shape and np.isfinite(power).all() and np.isfinite(rho).all()
    anchor=imbalance(c['anchor_plan'],power);values=np.array([imbalance(p,power) for p in c['next_plans']]);assert 0<len(values)<=64
    over=(rho>1)&live
    result=[float(rho[live].max()) if live.any() else 0.,float(np.maximum(rho[live]-1,0).mean()) if live.any() else 0.,
            float(rho[incident].max()) if incident.any() else 0.,float(np.sum(over&incident)/np.sum(over)) if over.any() else 0.,
            float(np.log1p(np.abs(power).sum())),anchor,float(values.min()-anchor),float(values.mean()-anchor)]
    out=np.asarray(result,np.float32);assert out.shape==(8,) and np.isfinite(out).all();return out

def independent(c):
    p=[float(v) for v in c['terminal_power']];r=[float(v) for v in c['rho']];live=[bool(v) for v in c['line_status']];incident=[bool(a and b) for a,b in zip(live,c['incident_lines'])]
    def imb(plan):return sum(abs(sum(v for v,b in zip(p,plan) if b==bus)) for bus in [1,2])/(1+sum(abs(v) for v in p))
    values=[imb(plan) for plan in c['next_plans']];a=imb(c['anchor_plan']);lr=[v for v,b in zip(r,live) if b];ir=[v for v,b in zip(r,incident) if b];over=sum(v>1 for v in lr)
    return np.asarray([max(lr,default=0.),sum(max(v-1,0) for v in lr)/len(lr) if lr else 0.,max(ir,default=0.),sum(v>1 for v in ir)/over if over else 0.,np.log1p(sum(abs(v) for v in p)),a,min(values)-a,sum(values)/len(values)-a],np.float32)
