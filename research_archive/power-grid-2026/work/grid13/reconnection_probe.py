"""Failure-conditioned public reconnection/busbar diagnostic; no new policy."""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, np, json, hashlib, time, make_env, write_json, digest, flags
from trace_agent import action_vector

OUT = ROOT / 'outputs/grid13'
assert not OUT.exists(); OUT.mkdir()
source_item = json.loads((ROOT / 'outputs/grid11_failure_context/design.json').read_text())['source']
source = ROOT / source_item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f: rows = list(map(json.loads, f))
with np.load(source / 'vectors.npz') as f: vectors = f['action'].copy()
design = {'source': source_item, 'states': [1717, 1718], 'line': 81,
    'candidate_specs': ['saved_base', 'default_remembered_bus_reconnection', '1_1', '1_2', '2_1', '2_2'],
    'public_forecast_horizon': 6, 'physical_cap': 1718, 'forecast_cap': 100, 'wall_cap_s': 180,
    'motivation': 'GRID12 archived state line 81 cooldown falls from 3 to 1; check the subsequent two public states. Direct reconnection is outside the GRID11 bus-topology candidate pool.',
    'scope': 'One already-seen training failure; public one-action-then-noop forecasts only. Not a deployed rescue, closed-loop policy or independent evaluation.'}
write_json(OUT / 'design.json', design)
write_json(OUT / 'code_freeze.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'work/grid08/common.py', ROOT / 'work/grid02/ljn_nn/modules/topology_heuristic.py']})
steps = forecasts = 0; env = None; start = time.perf_counter(); records = []; saved = {}
try:
    env = make_env(); env.seed(0); env.set_id(source_item['scenario']); obs = env.reset()
    for i in range(1719):
        assert time.perf_counter() - start < design['wall_cap_s']
        assert digest(obs.to_vect()) == rows[i]['before_hash']
        if i in design['states']:
            before = digest(obs.to_vect()); base = env.action_space(); base.from_vect(vectors[i])
            specs = [('saved_base', base), ('default_remembered_bus', env.action_space({'set_line_status': [(81, 1)]}))]
            for bus_or, bus_ex in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                specs.append((f'{bus_or}_{bus_ex}', env.action_space.reconnect_powerline(line_id=81, bus_or=bus_or, bus_ex=bus_ex)))
            record = {'state': i, 'rho': float(obs.rho.max()), 'line_81_status': bool(obs.line_status[81]),
                'line_81_cooldown': int(obs.time_before_cooldown_line[81]),
                'line_81_maintenance': int(obs.time_next_maintenance[81]), 'branches': {}}
            saved[f's{i}_public_observation'] = obs.to_vect().copy()
            for name, action in specs:
                saved[f's{i}_{name}_action'] = action_vector(action).copy()
                forecast = obs.get_forecast_env(); branch = []
                try:
                    forecast.reset()
                    for h in range(design['public_forecast_horizon']):
                        assert forecasts < design['forecast_cap']
                        future, _, done, info = forecast.step(action if h == 0 else forecast.action_space()); forecasts += 1
                        row = {'horizon_step': h + 1, 'done': bool(done), 'rho': float(future.rho.max()),
                            'line_11_rho': float(future.rho[11]), 'line_81_status': bool(future.line_status[81]), **flags(info)}
                        branch.append(row); saved[f's{i}_{name}_forecast{h+1}'] = future.to_vect().copy()
                        if done: break
                finally:
                    forecast.close()
                record['branches'][name] = branch
                assert digest(obs.to_vect()) == before and digest(env.get_obs().to_vect()) == before
            records.append(record); print(json.dumps(record), flush=True)
        if i == 1718: break
        action = env.action_space(); action.from_vect(vectors[i])
        obs, _, done, _ = env.step(action); steps += 1
        assert not done and digest(obs.to_vect()) == rows[i]['after_hash']
    assert len(records) == 2
    np.savez_compressed(OUT / 'public_vectors.npz', **saved)
    write_json(OUT / 'results.json', {'passed': True, 'physical_steps': steps, 'forecast_steps': forecasts,
                                    'wall_s': time.perf_counter() - start, 'records': records, 'scope': design['scope']})
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'forecasts': forecasts, 'wall_s': time.perf_counter() - start, 'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
