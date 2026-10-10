"""Untrained shortlist controls; original controller/search/optimizer stay intact."""
import hashlib
import time
import numpy as np

SEED='GRID04-cheap-v1'

def random_ids(step,n=352,k=20):
    return sorted(range(n),key=lambda i:hashlib.sha256(f'{SEED}:{int(step)}:{i}'.encode()).digest())[:k]

def local_ids(rho,incident_lines,n=352,k=20):
    def key(i):
        values=np.asarray(rho,dtype=np.float64)[incident_lines[i]]
        maximum=float(values.max()) if len(values) else 0.0
        mass=float(np.square(np.maximum(values-0.9,0.0)).sum())
        return (-maximum,-mass,hashlib.sha256(f'{SEED}:{i}'.encode()).digest())
    return sorted(range(n),key=key)[:k]

def attach_shortlist(agent,policy,env,trace):
    """Override only candidate enumeration on the existing NN controller."""
    module=agent.topo_12_unsafe
    if policy in ['NN20','NN352']:
        original=module.get_top_k
        def traced(gym_obs,top_k):
            start=time.perf_counter()
            ids=original(gym_obs,top_k)
            trace.events.append({'kind':'shortlist','step':trace.step,'module':'topo_12_unsafe',
                'policy':policy,'ids':ids.tolist(),'wall_s':time.perf_counter()-start})
            return ids
        module.get_top_k=traced
        return
    n=module.gym_env.action_space.n
    assert n==352 and policy in ['LOCAL20','RANDOM20']
    # Pool entries are topology actions; affected buses define incident line sets.
    topo_to_sub=np.repeat(np.arange(env.n_sub),env.sub_info)
    incident=[]
    for i in range(n):
        action=module.gym_env.action_space.from_gym(i)
        affected=np.unique(topo_to_sub[np.asarray(action.set_bus)!=0])
        assert len(affected)>0
        incident.append(np.flatnonzero(np.isin(env.line_or_to_subid,affected)|np.isin(env.line_ex_to_subid,affected)))
    def candidates(obs):
        start=time.perf_counter()
        ids=random_ids(obs.current_step,n,20) if policy=='RANDOM20' else local_ids(obs.rho,incident,n,20)
        trace.events.append({'kind':'shortlist','step':trace.step,'module':'topo_12_unsafe',
            'policy':policy,'ids':ids,'wall_s':time.perf_counter()-start})
        return [module.gym_env.action_space.from_gym(i) for i in ids]
    module._get_tested_action=candidates
    module.grid04_incident=incident
