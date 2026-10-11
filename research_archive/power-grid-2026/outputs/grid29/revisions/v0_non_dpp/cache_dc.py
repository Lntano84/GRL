"""Shared DC-program reuse, derived from LJN's run_dc (MPL-2.0).

No changed objective, constraint, solver, iteration cap, warm-start option,
continuous-control cadence or learned weight. Cache belongs to one optimizer.
Topology-dependent indices/masks and reactances are part of the cache key.
"""
from collections import OrderedDict
import cvxpy as cp
import numpy as np

CAPACITY = 32


def key_for(mod):
    values = [mod.bus_or.value, mod.bus_ex.value, mod.bus_load.value,
              mod.bus_gen.value, mod._mask_theta_zero(), mod._powerlines_x]
    if mod.bus_storage is not None:
        values.append(mod.bus_storage.value)
    return tuple((np.asarray(v).shape, np.asarray(v).dtype.str,
                  np.asarray(v).tobytes()) for v in values)


def run_dc_cached(mod, obs):
    # These original public-state updates must happen even on a cache hit.
    mod._update_topo_param(obs)
    mod._update_inj_param(obs)
    return solve_current(mod)


def solve_current(mod):
    key = key_for(mod)
    cache = getattr(mod, '_grid29_dc_programs', None)
    if cache is None:
        cache = mod._grid29_dc_programs = OrderedDict()
        mod._grid29_dc_counts = {'calls': 0, 'hit': 0, 'miss': 0, 'eviction': 0}
    counts = mod._grid29_dc_counts
    counts['calls'] += 1
    if key in cache:
        counts['hit'] += 1
        prob, f_or = cache.pop(key)
        cache[key] = (prob, f_or)
    else:
        counts['miss'] += 1
        theta = cp.Variable(shape=mod.nb_max_bus)
        origin = np.asarray(mod.bus_or.value)
        end = np.asarray(mod.bus_ex.value)
        for value in (origin, end):
            if not np.isfinite(value).all() or not np.array_equal(value, np.rint(value)):
                raise ValueError('Non-integral topology index')
        f_or = cp.multiply(1.0 / mod._powerlines_x,
                           theta[origin.astype(np.int64)] - theta[end.astype(np.int64)])
        inj_bus = mod.load_per_bus - mod.gen_per_bus
        equations = mod._aux_compute_kcl(inj_bus, f_or)
        constraints = [theta[mod._mask_theta_zero()] == 0] + [el == 0 for el in equations]
        prob = cp.Problem(cp.Minimize(1.0), constraints)
        if not prob.is_dcp(dpp=True):
            raise RuntimeError('DC program is not DPP compliant')
        cache[key] = (prob, f_or)
        if len(cache) > CAPACITY:
            cache.popitem(last=False)
            counts['eviction'] += 1
    # This calls the original optimizer's solver wrapper, including any existing
    # status qualification. warm_start=False remains inside that wrapper.
    converged = mod._solve_problem(prob, solver_type=cp.OSQP)
    if converged:
        mod.flow_computed[:] = f_or.value
    else:
        mod.flow_computed[:] = np.nan
    return converged
