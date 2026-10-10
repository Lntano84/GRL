"""GRID00: bounded engineering smoke test, never a performance experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid00'

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

import_start = time.perf_counter()
import numpy as np
import grid2op
from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent
from lightsim2grid import LightSimBackend
import_s = time.perf_counter() - import_start
(OUT / 'import_ready.json').write_text(json.dumps({'import_s': import_s}), encoding='utf-8')
start = time.perf_counter()

def check_time():
    if time.perf_counter() - start > 180:
        raise TimeoutError('Frozen 180-second smoke cap reached')

def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def flags(info):
    return {k: bool(info.get(k, False)) for k in ['is_illegal', 'is_ambiguous']} | {
        'exceptions': [str(e) for e in info.get('exception', [])]}

def main():
    available = grid2op.list_available_test_env()
    save('test_environments.json', available)
    target = 'l2rpn_idf_2023'
    assert target in available, 'Bundled target absent: do not silently substitute'
    data = Path(grid2op.__file__).parent / 'data' / target
    files = [{'path': str(p.relative_to(data)), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(data.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    save('bundled_assets.json', {'root': str(data), 'files': files})
    check_time()
    make_start = time.perf_counter()
    # Pass the bundled absolute directory: no dataset discovery/download fallback.
    env = grid2op.make(str(data), test=True, backend=LightSimBackend())
    make_s = time.perf_counter() - make_start
    try:
        env.seed(0)
        env.set_id(0)
        obs = env.reset()
        meta = {
            'scope': 'official bundled test data; not research/heldout evaluation',
            'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {n: importlib.metadata.version(n) for n in ['grid2op','lightsim2grid','pandapower','numpy','scipy','pandas']},
            'target': target, 'backend': type(env.backend).__name__,
            'seed': 0, 'scenario_id': env.chronics_handler.get_id(),
            'bundled_chronics': sorted(p.name for p in (data/'chronics').iterdir() if p.is_dir()),
            'dimensions': {n:int(getattr(env,n)) for n in ['n_sub','n_line','n_gen','n_load','n_storage','dim_topo']},
            'action_class': type(env.action_space()).__name__,
            'authorized_action_keys': sorted(env.action_space().authorized_keys),
            'observation_class': type(obs).__name__, 'reward_class': type(env._reward_helper.template_reward).__name__,
            'rules_class': type(env._game_rules.legal_action).__name__,
            'parameters': env.parameters.to_dict(),
            'forecast_timestamps': [str(x[0]) for x in obs._forecasted_inj],
            'import_s': import_s, 'make_s': make_s,
        }
        save('environment.json', meta)
        before_obs = env.get_obs().to_vect().copy()
        before_backend = tuple(np.array(x,copy=True) for x in env.backend.lines_or_info())
        before_step = int(env.nb_time_step)
        simulations = []
        vectors = {'initial_observation': before_obs}
        sim1 = None
        for horizon in [1,12]:
            check_time()
            count_before = int(env.nb_highres_called)
            t = time.perf_counter()
            forecast, rew, done, info = obs.simulate(env.action_space(), time_step=horizon)
            simulation_s = time.perf_counter() - t
            if horizon == 1:
                sim1 = forecast.to_vect().copy()
            unchanged = np.array_equal(before_obs, env.get_obs().to_vect(), equal_nan=True)
            backend_unchanged = all(np.array_equal(a,b,equal_nan=True) for a,b in zip(before_backend,env.backend.lines_or_info()))
            assert unchanged and backend_unchanged and env.nb_time_step == before_step
            simulations.append({'horizon_steps':horizon, 'wall_s':simulation_s, 'done':bool(done),
                'rho_max':float(forecast.rho.max()), 'reward':float(rew), **flags(info),
                'physical_obs_unchanged':unchanged,'backend_origin_flows_unchanged':backend_unchanged,
                'step_unchanged':env.nb_time_step == before_step,
                'counter_before':count_before, 'counter_after':int(env.nb_highres_called)})
            vectors[f'forecast_h{horizon}'] = forecast.to_vect().copy()
        save('simulation_checks.json', simulations)
        all_runs = []
        # One bundled chronic, two interface controls, exact repetition: 4 x <=24 steps.
        for agent_type in [DoNothingAgent, RecoPowerlineAgent]:
            for repeat in range(2):
                check_time()
                env.seed(0)
                env.set_id(0)
                obs = env.reset()
                agent = agent_type(env.action_space)
                agent.seed(0)
                agent.reset(obs)
                records = []
                observations = [obs.to_vect().copy()]
                actions = []
                reward = 0.0
                done = False
                counter_start = int(env.nb_highres_called)
                t_run = time.perf_counter()
                for step in range(24):
                    check_time()
                    counter_before = int(env.nb_highres_called)
                    t_act = time.perf_counter()
                    act = agent.act(obs, reward, done)
                    act_s = time.perf_counter() - t_act
                    t_step = time.perf_counter()
                    obs, reward, done, info = env.step(act)
                    step_s = time.perf_counter() - t_step
                    records.append({'step':step+1,'act_s':act_s,'step_s':step_s,
                        'rho_max':float(obs.rho.max()),'reward':float(reward),'done':bool(done),
                        'highres_calls':int(env.nb_highres_called)-counter_before, **flags(info),
                        'observation_hash':digest(obs.to_vect()),'action_hash':digest(act.to_vect())})
                    observations.append(obs.to_vect().copy())
                    actions.append(act.to_vect().copy())
                    if done:
                        break
                key = f'{agent_type.__name__}_{repeat}'
                vectors[key+'_obs'] = np.array(observations)
                vectors[key+'_actions'] = np.array(actions)
                all_runs.append({'agent':agent_type.__name__,'repeat':repeat,'scenario_id':env.chronics_handler.get_id(),
                    'wall_s':time.perf_counter()-t_run,'steps':len(records),'terminal':done,
                    'highres_calls':int(env.nb_highres_called)-counter_start,'records':records})
                save('trajectories.json',all_runs)
                np.savez_compressed(OUT/'vectors.npz', **vectors)
        replay = []
        for cls in [DoNothingAgent, RecoPowerlineAgent]:
            prefix = cls.__name__
            obs_match = np.array_equal(vectors[prefix+'_0_obs'],vectors[prefix+'_1_obs'],equal_nan=True)
            act_match = np.array_equal(vectors[prefix+'_0_actions'],vectors[prefix+'_1_actions'],equal_nan=True)
            assert obs_match and act_match
            replay.append({'agent':prefix,'observations_equal':obs_match,'actions_equal':act_match})
        save('replay_checks.json', replay)
        # Descriptive only: matching forecasts are not required for qualification.
        actual1 = vectors['DoNothingAgent_0_obs'][1]
        forecast_difference = float(np.max(np.abs(sim1.astype(float)-actual1.astype(float))))
        save('result.json',{'status':'PASS_INTERFACE_SMOKE_ONLY','actual_steps':sum(r['steps'] for r in all_runs),
            'smoke_wall_s':time.perf_counter()-start, 'import_s':import_s,
            'illegal_count':sum(s['is_illegal'] for r in all_runs for s in r['records']),
            'ambiguous_count':sum(s['is_ambiguous'] for r in all_runs for s in r['records']),
            'exception_steps':sum(bool(s['exceptions']) for r in all_runs for s in r['records']),
            'forecast1_vs_actual1_max_observation_difference':forecast_difference,
            'warning':'No strong baseline executed; no learning, survival or comparative quality conclusion.'})
    finally:
        env.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', {'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                              'elapsed_s':time.perf_counter()-start})
        raise
