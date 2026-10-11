"""After both fits, same exposed states and unchanged semantic stopping policy."""
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'work/grid40'), str(ROOT / 'work/grid20')]
from adapter import OUT as BASE_OUT, Predictor, raw_features, read
OUT = ROOT / 'outputs/grid40b'
from common import np, make_env, digest, write_json
from components import fingerprint
spec = importlib.util.spec_from_file_location('grid40_stopping', ROOT / 'work/grid28/policy.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
assert len(read(OUT / 'training_finished.json')['fits']) == 2
assert not (OUT / 'offline_review.json').exists()
for entry in read(OUT / 'model_manifest.json')['files']:
    assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
env = make_env(); env.seed(0); initial = env.reset()
library = __import__('common').ResidualControl(env, initial).base.topo_n1_unsafe.topo_act_list
subids = [int(a.as_dict()['set_bus_vect']['modif_subs_id'][0]) for a in library]
models = {seed: Predictor(OUT / 'models' / f'softgnn_seed{seed}' / 'final.pt') for seed in [0, 1]}
rows = []; inputs = {}; begun = time.perf_counter()
old = {(r['week'], r['call'], r['ranker']): r for r in read(ROOT / 'outputs/grid33/review.json')['rows']}
for week in read(ROOT / 'outputs/grid38/design.json')['development_weeks']:
    folder = ROOT / 'outputs/grid26/runs' / f'r0__{week}__FULL'
    for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
        full = read(path); truth = {r['pool_id']: r for r in full['outcomes']}
        with np.load(path.with_name(path.stem + '_state.npz')) as z:
            obs = initial.copy(); obs.from_vect(z['observation'])
            base = env.action_space(); base.from_vect(z['base_action'])
        assert digest(obs.to_vect()) == full['state_hash']
        features = raw_features(obs, base); h = fingerprint(features)
        reps = full['representatives']; aliases = {int(k): int(v) for k, v in full['aliases'].items()}
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for seed, predictor in models.items():
            started = time.perf_counter(); scores = predictor.scores(features)
            order = sorted(reps, key=lambda i: (-float(scores[i]), i)); inference_s = time.perf_counter() - started
            result = P.local_complete(order, lambda i, phase: {**truth[i], 'phase': phase}, 'PATIENCE64', subids, aliases, .9)
            winner = full['chosen_pool_id']; choice = result['chosen_pool_id']
            original = old[(week, path.name, f'GNN{seed}')]
            rows.append({'week': week, 'call': path.name, 'step': full['step'], 'ranker': f'SOFTGNN{seed}',
                'input_hash': h, 'queries': result['public_queries'], 'original_queries': original['queries'],
                'choice': choice, 'full_choice': winner, 'same_full_choice': choice == winner,
                'rho_regret': None if winner is None or choice is None else truth[choice]['rho'] - truth[winner]['rho'],
                'inference_s': inference_s, 'order': order, 'stop_reason': result['stop_reason']})
summary = []
for seed in [0, 1]:
    part = [r for r in rows if r['ranker'] == f'SOFTGNN{seed}']; assert len(part) == 26
    critical = next(r for r in part if r['week'] == '2035-03-05_0' and r['step'] == 1733)
    summary.append({'ranker': f'SOFTGNN{seed}', 'queries': sum(r['queries'] for r in part),
        'original_queries': sum(r['original_queries'] for r in part), 'same_full_choice': sum(r['same_full_choice'] for r in part),
        'max_rho_regret': max(r['rho_regret'] or 0. for r in part),
        'known_critical_choice_match': critical['same_full_choice'],
        'mean_forward_rank_s': sum(r['inference_s'] for r in part) / len(part)})
write_json(OUT / 'offline_review.json', {'passed_execution': True, 'rows': rows, 'summaries': summary, 'source_hashes': inputs,
    'new_teacher_queries': 0, 'reserved_evaluations': 0, 'wall_s': time.perf_counter() - begun,
    'warning': 'Existing26exposed states, not closed-loop quality/time or independent transfer. Official-core resource-matched adaptation; both seeds retained. Training convergence reviewed separately.'})
with (OUT / 'offline_per_state.csv').open('w', encoding='utf-8', newline='') as f:
    fields = [k for k in rows[0] if k != 'order']; w = csv.DictWriter(f, fields); w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
print(json.dumps(summary, indent=2)); env.close()
