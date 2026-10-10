"""Public pre-terminal topology and forecasts for the frozen January trace."""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, time, make_env, write_json, digest, flags

OUT = ROOT / 'outputs/grid11_failure_context/bridge_check'
assert not OUT.exists(); OUT.mkdir()
source_item = json.loads((OUT.parent / 'design.json').read_text())['source']
source = ROOT / source_item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f: rows = list(map(json.loads, f))
with np.load(source / 'vectors.npz') as f: vectors = f['action'].copy()
write_json(OUT / 'design.json', {'source': source_item, 'replay_steps': 1718,
    'physical_cap': 1718, 'wall_cap_s': 180, 'forecast_actions': ['saved_actual_next_action', 'noop'],
    'scope': 'Failure-conditioned training diagnostic. Only public current topology/forecasts, no actual-future branch or fitting.'})
write_json(OUT / 'code_freeze.json', {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
env = None; steps = 0; start = time.perf_counter()
try:
    env = make_env(); env.seed(0); env.set_id(source_item['scenario']); obs = env.reset()
    for i in range(1718):
        assert time.perf_counter() - start < 180
        assert digest(obs.to_vect()) == rows[i]['before_hash']
        action = env.action_space(); action.from_vect(vectors[i])
        obs, _, done, _ = env.step(action); steps += 1
        assert not done and digest(obs.to_vect()) == rows[i]['after_hash']
    def node(sub, bus): return np.asarray(sub) + env.n_sub * (np.asarray(bus) - 1)
    u = node(env.line_or_to_subid, obs.line_or_bus); v = node(env.line_ex_to_subid, obs.line_ex_bus)
    gn = node(env.gen_to_subid, obs.gen_bus); ln = node(env.load_to_subid, obs.load_bus)
    live = np.asarray(obs.line_status, dtype=bool)
    vertices = set(map(int, u[live])) | set(map(int, v[live]))
    vertices.update(map(int, gn[np.asarray(obs.gen_bus) > 0])); vertices.update(map(int, ln[np.asarray(obs.load_bus) > 0]))
    def components(exclude=None):
        neighbors = {x: [] for x in vertices}
        for i in np.flatnonzero(live):
            if i == exclude: continue
            neighbors[int(u[i])].append(int(v[i])); neighbors[int(v[i])].append(int(u[i]))
        seen = set(); groups = []
        for root in sorted(vertices):
            if root in seen: continue
            stack = [root]; seen.add(root); group = []
            while stack:
                x = stack.pop(); group.append(x)
                for y in neighbors[x]:
                    if y not in seen: seen.add(y); stack.append(y)
            group = sorted(group)
            generators = [int(i) for i, x in enumerate(gn) if int(x) in group and obs.gen_bus[i] > 0]
            loads = [int(i) for i, x in enumerate(ln) if int(x) in group and obs.load_bus[i] > 0]
            groups.append({'nodes': group, 'generators': generators, 'loads': loads,
                'generation_mw': float(obs.gen_p[generators].sum()), 'load_mw': float(obs.load_p[loads].sum())})
        return groups
    forecasts = []; before = digest(obs.to_vect()); highres_start = int(env.nb_highres_called)
    for name, vector in [('saved_actual_next_action', vectors[1718]), ('noop', None)]:
        action = env.action_space()
        if vector is not None: action.from_vect(vector)
        future, _, done, info = obs.simulate(action, time_step=1)
        forecasts.append({'action': name, 'done': bool(done), 'max_rho': float(future.rho.max()), **flags(info)})
        assert digest(obs.to_vect()) == before
    result = {'passed': True, 'physical_steps': steps, 'prefix_hashes_matched': steps,
        'wall_s': time.perf_counter() - start, 'current_step': int(obs.current_step),
        'line_11': {'from_node': int(u[11]), 'to_node': int(v[11]), 'rho': float(obs.rho[11]),
                    'overflow_steps': int(obs.timestep_overflow[11]), 'status': bool(live[11])},
        'public_line_endpoints': [{'line': int(i), 'u': int(u[i]), 'v': int(v[i])} for i in np.flatnonzero(live)],
        'components_before': components(), 'components_without_line_11': components(11),
        'forecasts': forecasts, 'native_highres_count': int(env.nb_highres_called) - highres_start,
        'note': 'Post-game-over all-zero/cleared observations must not be interpreted as 186 actual line trips. Static removal is a graph connectivity calculation, not an AC-feasibility certificate or a causal policy experiment.'}
    write_json(OUT / 'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'public_line_endpoints'}, indent=2))
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'wall_s': time.perf_counter() - start,
                                    'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
