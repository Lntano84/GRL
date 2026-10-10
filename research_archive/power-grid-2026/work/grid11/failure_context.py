"""Replay one frozen training failure; inspect public maintenance/topology context.

No counterfactual actual-future branch, fitting, or model change. A trace replay
can identify observed events, not prove why an earlier policy action failed.
"""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, time, make_env, write_json, digest, flags

OUT = ROOT / 'outputs/grid11_failure_context'
assert not OUT.exists(); OUT.mkdir()
manifest = json.loads((ROOT / 'outputs/grid08/v2/train/gnn/manifest.json').read_text())
item = [r for r in manifest[-4:] if r['scenario'] == '2035-01-29_4'][0]
source = ROOT / item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as handle:
    rows = list(map(json.loads, handle))
with np.load(source / 'vectors.npz') as data:
    actions = data['action'].copy()
assert len(rows) == len(actions) == 1719 and not item['complete']
design = {'source': item, 'environment_seed': 0, 'physical_cap': 2000, 'wall_cap_s': 300,
    'scope': 'One previously observed training failure; chronological public-context replay, no new policy or causal diagnosis.',
    'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [source / 'steps.jsonl.gz', source / 'vectors.npz']}}
write_json(OUT / 'design.json', design)
write_json(OUT / 'code_freeze.json', {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

def public_context(obs, env):
    def nodes(sub, bus):
        return np.asarray(sub) + env.n_sub * (np.maximum(np.asarray(bus), 1) - 1)
    u = nodes(env.line_or_to_subid, obs.line_or_bus); v = nodes(env.line_ex_to_subid, obs.line_ex_bus)
    live = np.asarray(obs.line_status, dtype=bool)
    gn = nodes(env.gen_to_subid, obs.gen_bus); ln = nodes(env.load_to_subid, obs.load_bus)
    active = set(map(int, u[live])) | set(map(int, v[live]))
    active.update(map(int, gn[np.asarray(obs.gen_bus) > 0])); active.update(map(int, ln[np.asarray(obs.load_bus) > 0]))
    parent = {x: x for x in active}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in zip(u[live], v[live]):
        parent[find(int(a))] = find(int(b))
    groups = {}
    for x in sorted(active): groups.setdefault(find(x), []).append(x)
    top = np.argsort(obs.rho)[-8:][::-1]
    soon = np.flatnonzero((obs.time_next_maintenance >= 0) & (obs.time_next_maintenance <= 12))
    return {'current_step': int(obs.current_step), 'max_rho': float(obs.rho.max()),
        'top_rho': [{'line': int(i), 'rho': float(obs.rho[i]), 'live': bool(live[i]),
                     'overflow_steps': int(obs.timestep_overflow[i])} for i in top],
        'disconnected_lines': np.flatnonzero(~live).tolist(),
        'maintenance_within_hour': [{'line': int(i), 'starts_in_steps': int(obs.time_next_maintenance[i]),
                                    'duration_steps': int(obs.duration_next_maintenance[i])} for i in soon],
        'active_bus_components': list(groups.values()),
        'generator_bus': gn.tolist(), 'load_bus': ln.tolist(),
        'actual_dispatch_l1': float(np.abs(obs.actual_dispatch).sum()),
        'target_dispatch_l1': float(np.abs(obs.target_dispatch).sum())}

env = None; steps = 0; start = time.perf_counter(); contexts = []
try:
    env = make_env(); env.seed(0); env.set_id(item['scenario']); obs = env.reset()
    for i, (row, vector) in enumerate(zip(rows, actions)):
        assert steps < design['physical_cap'] and time.perf_counter() - start < design['wall_cap_s']
        assert digest(obs.to_vect()) == row['before_hash'], ('before', i + 1)
        before = public_context(obs, env) if i >= len(rows) - 25 else None
        action = env.action_space(); action.from_vect(vector)
        obs, reward, done, info = env.step(action); steps += 1
        assert digest(obs.to_vect()) == row['after_hash'], ('after', i + 1)
        assert bool(done) == row['done']
        if before is not None:
            contexts.append({'step': i + 1, 'before': before, 'after': public_context(obs, env),
                'choice': row['choice'], 'proposal': row['proposal'], 'feedback': flags(info),
                'cascade_disconnections': np.asarray(info.get('disc_lines', [])).tolist()})
        if (i + 1) % 256 == 0 or done: print(json.dumps({'step': i + 1, 'matched': True}), flush=True)
        if done: break
    assert steps == 1719
    result = {'passed': True, 'physical_steps': steps, 'wall_s': time.perf_counter() - start,
        'prefix_hashes_matched': steps, 'contexts': contexts, 'complete': False,
        'scope': design['scope'], 'forecast_calls': 0}
    write_json(OUT / 'results.json', result)
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'wall_s': time.perf_counter() - start,
                                    'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
