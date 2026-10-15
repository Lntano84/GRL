"""Prefix + actual public grid context; no candidate suffix information."""
from pathlib import Path
import functools,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid75_legal_prefix',ROOT/'work/grid72/feature_gate.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
M=old.M
GLOBAL=['root_rho_max','root_abs_dispatch','root_abs_target_dispatch','root_total_demand',
    'graph_average_degree','active_bus_fraction','radial_load_bus_fraction','mean_node_rho',
    'hot_bus_fraction','substation_cooldown_fraction','root_generation','root_curtailment']
LOCAL=['incumbent_generation','incumbent_demand','incumbent_rho','incumbent_flow','incumbent_degree','incumbent_cooldown']
NAMES=old.NAMES[:12]+GLOBAL+old.NAMES[12:]+LOCAL
SUPPORT=[1,3,8,9,10,12,15,25]
predict=old.predict;out_of_support=old.out_of_support;decide=old.decide
def features(rows,order,subids,aliases,limit,context):
    prefix=old.features(rows,order,subids,aliases,limit)
    x=np.asarray(context['x']);edges=np.asarray(context['edges']);glob=np.asarray(context['glob'])
    n=len(x);assert x.shape==(n,14) and n%2==0 and len(glob)==11
    active=(x[:,10]>0)|(np.abs(x[:,0])+np.abs(x[:,6])>0);loaded=x[:,6]>0
    global_values=[float(glob[2]),float(glob[3]),float(glob[4]),float(glob[5]),
        edges.shape[1]/max(1,n),float(active.mean()),float((x[loaded,10]<=.1+1e-6).mean()) if loaded.any() else 0.,
        float(x[active,9].mean()) if active.any() else 0.,float((x[:,9]>.9).mean()),
        float((x[:,12]>0).mean()),float(x[:,0].sum()),float(x[:,5].sum())]
    best=M.best_row(rows)
    if best is None:local=[0.]*6
    else:
        s=int(subids[best['pool_id']]);v=x[[s,s+n//2]]
        local=[float(v[:,0].sum()),float(v[:,6].sum()),float(v[:,9].max()),float(v[:,11].sum()),float(v[:,10].sum()),float(v[:,12].max())]
    result=np.asarray([*prefix[:12],*global_values,*prefix[12:],*local],dtype=np.float32)
    assert result.shape==(38,) and np.isfinite(result).all()
    return result
@functools.lru_cache(maxsize=4)
def load(name):
    assert name in ('plain','struct')
    return json.loads((ROOT/f'outputs/grid75/models/{name}.json').read_text(encoding='utf-8'))
