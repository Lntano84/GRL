"""Bounded engineering diagnosis: reproduce prefix and persist actual/shadow mismatch.

Never continue after the first selected event, regardless of its numerical outcome.
"""
import run_probe as r
from common import *
import cvxpy as cp
OUT = ROOT / 'outputs/grid14/shadow_delivery_diagnosis'
OUT.mkdir(exist_ok=False)
started = time.perf_counter()
env = make_env(); env.seed(0); env.set_id(r.D['training_weeks'][0]); obs = env.reset()
ctl = ResidualControl(env, obs); reward = 0.; rows = []; physical = 0
original = ctl.base.topo_12_unsafe.get_act
saved = {}
def get_act(observation, base_action, reward, done=False, **kwargs):
    topology = original(observation, base_action, reward, done=done, **kwargs)
    if observation.current_step != 200: return topology
    combined = base_action.copy() + topology.copy()
    future, _, _, info = observation.simulate(combined, time_step=1)
    shadow = r.clone_optim(ctl.base.optim, env)
    statuses = []; solver = shadow._solve_problem
    def solve(prob, solver_type=None):
        answer = solver(prob, solver_type)
        statuses.append({'status': str(prob.status), 'answer': bool(answer), 'solver': solver_type})
        return answer
    shadow._solve_problem = solve
    delivered = shadow.get_act(observation, combined, reward)
    saved.update(shadow=delivered.copy(), topology=topology.copy(), statuses=statuses,
                 before=observation.to_vect().copy(), before_mem=r.memory(ctl.base.optim))
    return topology
ctl.base.topo_12_unsafe.get_act = get_act
statuses = []; solver = ctl.base.optim._solve_problem
def solve(prob, solver_type=None):
    answer = solver(prob, solver_type)
    statuses.append({'status': str(prob.status), 'answer': bool(answer), 'solver': solver_type})
    return answer
ctl.base.optim._solve_problem = solve
try:
    for step in range(1, 202):
        assert time.perf_counter()-started < 120
        base, restore, p = ctl.propose(obs, reward, cadence=6)
        act = restore if p['offered'] else base
        if step == 201:
            av = action_vector(act); sv = action_vector(saved['shadow'])
            differences = {}
            for name in type(act).attr_list_vect:
                a = np.asarray(act._get_array_from_attr_name(name), dtype=float)
                s = np.asarray(saved['shadow']._get_array_from_attr_name(name), dtype=float)
                unequal = ~(np.equal(a,s) | (np.isnan(a) & np.isnan(s)))
                if np.any(unequal):
                    differences[name] = {'indices': np.flatnonzero(unequal).tolist(),
                                         'actual': a[unequal].tolist(), 'shadow': s[unequal].tolist()}
            forecasts = {}
            fv = {}
            for key, action in [('actual', act), ('shadow', saved['shadow'])]:
                future, _, done, info = obs.simulate(action, time_step=1)
                forecasts[key] = {'rho': float(future.rho.max()), 'done': bool(done), **flags(info)}
                fv[key] = future.to_vect().copy()
            np.savez_compressed(OUT / 'vectors.npz', before=obs.to_vect(), actual_action=av,
                                shadow_action=sv, **fv)
            write_json(OUT / 'result.json', {'differences': differences,
                       'max_finite_abs_diff': float(np.nanmax(np.abs(av-sv))),
                       'actual_solver': statuses, 'shadow_solver': saved['statuses'],
                       'forecasts': forecasts, 'physical_steps': physical,
                       'public_forecasts': int(env.nb_highres_called),
                       'wall_s': time.perf_counter()-started,
                       'scope': 'Engineering numerical diagnosis only, no intervention physical step or training.'})
            print(json.dumps({'differences': differences, 'actual_solver': statuses,
                              'shadow_solver': saved['statuses'], 'forecasts': forecasts}, indent=2), flush=True)
            break
        obs, reward, done, info = env.step(act); physical += 1; assert not done
finally:
    env.close()
