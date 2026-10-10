"""Frozen public-forecast test of earlier existing emergency QP, no training."""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, np, json, hashlib, time, make_env, write_json, digest, flags, OptimModule
from trace_agent import action_vector

OUT = ROOT / 'outputs/grid12'
assert not OUT.exists(); OUT.mkdir()
source_item = json.loads((ROOT / 'outputs/grid11_failure_context/design.json').read_text())['source']
source = ROOT / source_item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f: rows = list(map(json.loads, f))
with np.load(source / 'vectors.npz') as f: vectors = f['action'].copy()
design = {'source': source_item, 'states': [1714, 1716], 'forecast_horizon': 6,
          'physical_cap': 1716, 'forecast_step_cap': 100, 'wall_cap_s': 180,
          'candidates': ['saved_base', 'isolated_default_unsafe_QP_on_saved_base'],
          'selection': 'Fixed earlier warning state and first overflow state from already-observed training failure; not natural occurrence or independent evaluation.',
          'scope': 'Public forecast environment only; no actual-future branch, model fit, parameter search or deployed policy.'}
write_json(OUT / 'design.json', design)
write_json(OUT / 'code_freeze.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'work/grid08/common.py', ROOT / 'work/grid02/ljn_nn/modules/convex_optim.py']})
steps = forecasts = solves = 0; env = None; start = time.perf_counter(); records = []; saved = {}
try:
    env = make_env(); env.seed(0); env.set_id(source_item['scenario']); obs = env.reset()
    for i in range(1717):
        assert time.perf_counter() - start < design['wall_cap_s']
        assert digest(obs.to_vect()) == rows[i]['before_hash']
        if i in design['states']:
            before = digest(obs.to_vect()); base = env.action_space(); base.from_vect(vectors[i])
            module = OptimModule(env, env.action_space); module.reset(obs)
            statuses = []; original = module._solve_problem
            def checked_solve(prob, solver_type=None):
                ok = original(prob, solver_type); statuses.append(str(prob.status))
                return bool(ok and prob.status in ('optimal', 'optimal_inaccurate'))
            module._solve_problem = checked_solve
            count_before = int(env.nb_highres_called); t = time.perf_counter()
            qp = module.get_act(obs, base.copy(), reward=0.)
            qp_wall = time.perf_counter() - t; solves += 1
            internal_forecasts = int(env.nb_highres_called) - count_before
            forecasts += internal_forecasts
            assert internal_forecasts == 1 and digest(obs.to_vect()) == before
            qualified = bool(statuses and statuses[-1] in ('optimal', 'optimal_inaccurate'))
            state = {'state_step': i, 'current_rho': float(obs.rho.max()), 'source_proposal': rows[i]['proposal'],
                     'qp_solve_status': statuses, 'qp_wall_s': qp_wall, 'qp_qualified': qualified,
                     'qp_differs': not np.array_equal(action_vector(qp), action_vector(base)), 'branches': {}}
            saved[f's{i}_public_observation'] = obs.to_vect().copy()
            for name, action in [('saved_base', base), ('default_unsafe_QP', qp)]:
                saved[f's{i}_{name}_action'] = action_vector(action).copy()
                forecast = obs.get_forecast_env(); branch = []
                try:
                    forecast.reset()
                    for h in range(design['forecast_horizon']):
                        assert forecasts < design['forecast_step_cap']
                        future, _, done, info = forecast.step(action if h == 0 else forecast.action_space()); forecasts += 1
                        fl = flags(info)
                        branch.append({'horizon_step': h + 1, 'done': bool(done), 'rho': float(future.rho.max()),
                                       'line_11_rho': float(future.rho[11]), 'line_11_overflow': int(future.timestep_overflow[11]), **fl})
                        saved[f's{i}_{name}_forecast{h+1}'] = future.to_vect().copy()
                        if done: break
                finally:
                    forecast.close()
                state['branches'][name] = branch
                assert digest(obs.to_vect()) == before and digest(env.get_obs().to_vect()) == before
            records.append(state); print(json.dumps(state), flush=True)
        if i == 1716: break
        action = env.action_space(); action.from_vect(vectors[i])
        obs, _, done, _ = env.step(action); steps += 1
        assert not done and digest(obs.to_vect()) == rows[i]['after_hash']
    assert len(records) == 2
    np.savez_compressed(OUT / 'public_vectors.npz', **saved)
    write_json(OUT / 'results.json', {'passed': True, 'physical_steps': steps, 'forecast_steps_including_QP': forecasts,
               'qp_solves': solves, 'wall_s': time.perf_counter() - start, 'records': records,
               'scope': design['scope'], 'warning': 'Only the first action differs; subsequent forecast actions are no-op. Not a closed-loop MPC or causal full-episode survival comparison.'})
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'forecasts': forecasts, 'wall_s': time.perf_counter() - start, 'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
