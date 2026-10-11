"""Shared parameterized safe-QP reuse, derived from LJN(MPL-2.0).

Preserve every original objective term, constraint, solver qualification and
post-solve state update. Scalar coefficients become numeric constants of one
cache entry, with their exact values included in its invalidation key.
"""
from collections import OrderedDict
import cvxpy as cp
import numpy as np
from cache_dc import key_for

CAPACITY=32
COEFFICIENTS=['_margin_th_limit','_alpha_por_error','_penalty_curtailment_safe',
    '_penalty_storage_safe','_penalty_redispatching_safe','_weight_redisp_target',
    '_weight_storage_target','_weight_curtail_target']


def coefficient_values(mod):
    values=[]
    for name in COEFFICIENTS:
        attr=getattr(mod,name);arr=np.asarray(attr.value if hasattr(attr,'value') else attr)
        if arr.shape!=() or not np.isfinite(arr).all():raise ValueError(('Invalid scalar coefficient',name))
        values.append(float(arr))
    return values


def compute_safe_cached(mod,obs,l_id=None):
    # Original method changes these indices for the proposed reconnection.
    if l_id is not None:
        mod.bus_ex.value[l_id]=obs.line_ex_to_subid[l_id]
        mod.bus_or.value[l_id]=obs.line_or_to_subid[l_id]
    values=coefficient_values(mod)
    key=key_for(mod)+(('safe_coefficients',tuple(values),float(obs.delta_time)),)
    cache=getattr(mod,'_grid30_safe_programs',None)
    if cache is None:
        cache=mod._grid30_safe_programs=OrderedDict()
        mod._grid30_safe_counts={'calls':0,'hit':0,'miss':0,'eviction':0}
    counts=mod._grid30_safe_counts;counts['calls']+=1
    if key in cache:
        counts['hit']+=1;entry=cache.pop(key);cache[key]=entry
    else:
        counts['miss']+=1
        margin,alpha,penalty_c,penalty_s,penalty_r,weight_r,weight_s,weight_c=values
        theta=cp.Variable(shape=mod.nb_max_bus)
        curtailment_mw=cp.Variable(shape=mod.nb_max_bus)
        storage=cp.Variable(shape=mod.nb_max_bus)
        redispatching=cp.Variable(shape=mod.nb_max_bus)
        origin,end=np.asarray(mod.bus_or.value),np.asarray(mod.bus_ex.value)
        for ids in (origin,end):
            if not np.isfinite(ids).all() or not np.array_equal(ids,np.rint(ids)):
                raise ValueError('Non-integral topology')
        reactance=mod._powerlines_x.value
        f_or=cp.multiply(1.0/reactance,theta[origin.astype(np.int64)]-theta[end.astype(np.int64)])
        f_or_corr=f_or-alpha*mod._prev_por_error
        inj_bus=(mod.load_per_bus+storage)-(mod.gen_per_bus+redispatching-curtailment_mw)
        energy_added=cp.sum(curtailment_mw)+cp.sum(storage)-cp.sum(redispatching)-mod._storage_power_obs
        equations=mod._aux_compute_kcl(inj_bus,f_or)
        dispatch_after=mod._past_dispatch+redispatching
        soc_after=mod._past_state_of_charge+storage/(60.0/obs.delta_time)
        constraints=([theta[mod._mask_theta_zero()]==0]+[el==0 for el in equations]+
            [f_or_corr<=margin*mod._th_lim_mw,f_or_corr>=-margin*mod._th_lim_mw]+
            [redispatching<=mod.redisp_up,redispatching>=-mod.redisp_down]+
            [curtailment_mw<=mod.curtail_up,curtailment_mw>=-mod.curtail_down]+
            [storage<=mod.storage_up,storage>=-mod.storage_down]+[energy_added==0])
        cost=(penalty_c*cp.sum_squares(curtailment_mw)+penalty_s*cp.sum_squares(storage)+
            penalty_r*cp.sum_squares(redispatching)+weight_r*cp.sum_squares(dispatch_after)+
            weight_s*cp.sum_squares(soc_after-mod._storage_target_bus)+
            weight_c*cp.sum_squares(curtailment_mw+mod.curtail_down))
        problem=cp.Problem(cp.Minimize(cost),constraints)
        if not problem.is_dcp(dpp=True):raise RuntimeError('Safe QP is not DPP compliant')
        entry=(problem,f_or,curtailment_mw,storage,redispatching);cache[key]=entry
        if len(cache)>CAPACITY:cache.popitem(last=False);counts['eviction']+=1
    problem,f_or,curtailment_mw,storage,redispatching=entry
    converged=mod._solve_problem(problem,solver_type=cp.OSQP)
    if converged:
        mod.flow_computed[:]=f_or.value
        result=(curtailment_mw.value,storage.value,redispatching.value)
        mod._storage_power_obs.value=0.
    else:
        mod.flow_computed[:]=np.nan
        result=tuple(np.zeros(shape=mod.nb_max_bus) for _ in range(3))
    return result
