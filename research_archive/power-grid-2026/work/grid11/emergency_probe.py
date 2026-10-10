"""Public-forecast search over author topology pools at two training-risk states.

Failure-conditioned diagnostic only. No controller is deployed, no actual-future
branch is used, and one-step viable actions do not certify episode survival.
"""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, time, make_env, write_json, digest, flags, public_cost_ledger
from trace_agent import action_vector

OUT = ROOT / 'outputs/grid11_failure_context/emergency_probe'
assert not OUT.exists(); OUT.mkdir()
source_item = json.loads((OUT.parent / 'design.json').read_text())['source']
source = ROOT / source_item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f: rows = list(map(json.loads, f))
with np.load(source / 'vectors.npz') as f: vectors = f['action'].copy()
pools = [ROOT / 'work/grid02/ljn_nn/assets' / name for name in ['action_12_unsafe.npz', 'action_N1_unsafe.npz']]
design = {'source': source_item, 'public_states': [1714, 1718], 'physical_cap': 1718,
    'forecast_cap': 6500, 'wall_cap_s': 300, 'combinations': ['topology_only', 'saved_base_plus_topology'],
    'pools': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pools},
    'scope': 'Two failure-conditioned training states; legal current observations and one-step public forecasts. No new policy or actual-future branch.'}
write_json(OUT / 'design.json', design)
write_json(OUT / 'code_freeze.json', {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
steps = forecasts = 0; env = None; start = time.perf_counter(); records = []; candidates = {}
try:
    env = make_env(); env.seed(0); env.set_id(source_item['scenario']); obs = env.reset()
    actions = {}; pool_counts = {}
    for path in pools:
        with np.load(path) as data: matrix = data['action_space']
        pool_counts[path.name] = len(matrix)
        for vector in matrix:
            a = env.action_space(); a.from_vect(vector)
            actions.setdefault(digest(action_vector(a)), a)
    for i in range(1719):
        assert time.perf_counter() - start < 300
        assert digest(obs.to_vect()) == rows[i]['before_hash']
        if i in design['public_states']:
            before = digest(obs.to_vect()); next_actual = env.action_space(); next_actual.from_vect(vectors[i])
            tested = {}; details = []; best = None; native_start = int(env.nb_highres_called)
            for name, a in [('saved_base', next_actual), ('noop', env.action_space())]:
                tested.setdefault(digest(action_vector(a)), (name, a))
            filtered_cooldown = 0
            for h, topo in actions.items():
                affected = np.flatnonzero(topo.get_topological_impact()[1])
                if any(obs.time_before_cooldown_sub[int(s)] > 0 for s in affected):
                    filtered_cooldown += 1; continue
                tested.setdefault(h, ('topology_only', topo))
                combined = next_actual + topo
                tested.setdefault(digest(action_vector(combined)), ('saved_base_plus_topology', combined))
            for j, (h, (name, a)) in enumerate(tested.items()):
                assert forecasts < design['forecast_cap'] and time.perf_counter() - start < 300
                future, _, done, info = obs.simulate(a, time_step=1); forecasts += 1
                fl = flags(info); rho = float(future.rho.max())
                valid = not done and not fl['illegal'] and not fl['ambiguous'] and not fl['exceptions'] and np.isfinite(rho) and rho > 0
                cost = public_cost_ledger(future, env, float(obs.curtailment_mw.sum(dtype=np.float64)))['recomputed_raw_cost'] if valid else None
                row = {'index': j, 'origin': name, 'action_hash': h, 'valid': bool(valid),
                       'done': bool(done), 'max_rho': rho, 'cost': cost, **fl}
                details.append(row)
                if valid and (best is None or (rho, cost) < (best['max_rho'], best['cost'])):
                    best = row; candidates[f's{i}_action'] = action_vector(a).copy()
                    candidates[f's{i}_forecast_observation'] = future.to_vect().copy()
                assert digest(obs.to_vect()) == before
            write_json(OUT / f'state_{i}_candidates.json', details)
            record = {'state_step': i, 'current_rho': float(obs.rho.max()), 'source_choice': rows[i]['choice'],
                'source_proposal': rows[i]['proposal'], 'unique_candidates': len(details), 'cooldown_filtered': filtered_cooldown,
                'valid': sum(r['valid'] for r in details), 'valid_rho_lt_1': sum(r['valid'] and r['max_rho'] < 1. for r in details),
                'best': best, 'base_and_noop': [r for r in details if r['origin'] in ['saved_base', 'noop']],
                'native_highres_count': int(env.nb_highres_called) - native_start}
            records.append(record); print(json.dumps(record), flush=True)
        if i == 1718:
            break  # Diagnose the terminal pre-state without advancing its actual future.
        action = env.action_space(); action.from_vect(vectors[i])
        obs, _, done, _ = env.step(action); steps += 1
        assert not done and digest(obs.to_vect()) == rows[i]['after_hash']
    assert len(records) == 2
    np.savez_compressed(OUT / 'best_public_candidates.npz', **candidates)
    write_json(OUT / 'results.json', {'passed': True, 'physical_steps': steps, 'forecasts': forecasts,
        'wall_s': time.perf_counter() - start, 'pool_sizes': pool_counts, 'unique_pool_size': len(actions),
        'records': records, 'scope': design['scope'],
        'warning': 'Viable one-step forecasts are not multi-step survival evidence or learned-policy results. Pooled author action sets are a stronger search diagnostic than the original agent branch, not a faithful replay of its selection.'})
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'forecasts': forecasts, 'wall_s': time.perf_counter() - start,
                                    'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
