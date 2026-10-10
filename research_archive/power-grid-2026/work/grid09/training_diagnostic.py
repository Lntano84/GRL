"""Read-only final-policy/critic checks on each arm's last training pass.

No simulator, fitting, or evaluation data. Realized returns are from changing
training policies, so critic errors do not establish final-policy calibration.
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, np, torch, json, time, tensor_features, write_json
from networks_v1 import ActorCritic

out = ROOT / 'outputs/grid09'
design = json.loads((out / 'design.json').read_text())
started = time.perf_counter()
results = {}
for version in ['physical', 'decision']:
    folder = out / version / 'train/gnn'
    assert (folder / 'finished.json').exists()
    saved = torch.load(folder / 'final.pt', weights_only=False)
    model = ActorCritic('gnn', saved['n_nodes'], max_edges=saved['max_edges'])
    model.load_state_dict(saved['state'])
    model.eval()
    episodes = []
    all_p, all_v, all_mc = [], [], []
    manifest = json.loads((folder / 'manifest.json').read_text())
    for item in manifest[-4:]:
        assert item['scenario'] in design['training_weeks']
        path = ROOT / item['path']
        with gzip.open(path / 'steps.jsonl.gz', 'rt', encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        returns = np.zeros(len(rows))
        future = 0.
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            reward = -row['raw_cost'] / design['ppo']['cost_scale']
            if row['done'] and not row['complete']:
                reward -= design['ppo']['blackout_penalty']
            future = reward + design['ppo']['gamma'] * (0. if row['done'] else future)
            returns[i] = future
        offered = [i for i, row in enumerate(rows) if row['proposal']['offered']]
        probabilities, values = [], []
        with np.load(path / 'teacher.npz') as features:
            assert len(features['actions']) == len(offered)
            for j in range(len(offered)):
                with torch.no_grad():
                    logits, value = model(*tensor_features((features['x'][j], features[f'e{j}'], features['g'][j])))
                probabilities.append(float(torch.softmax(logits, -1)[0, 1]))
                values.append(float(value.item()))
        p, v, truth = np.array(probabilities), np.array(values), returns[offered]
        error = v - truth
        variance = float(np.var(truth))
        episodes.append({
            'week': item['scenario'], 'complete': item['complete'], 'samples': len(offered),
            'restore_probability_min_median_max': [float(p.min()), float(np.median(p)), float(p.max())],
            'final_argmax_restores': int((p > .5).sum()),
            'mean_value': float(v.mean()), 'mean_realized_return': float(truth.mean()),
            'critic_rmse': float(np.sqrt(np.mean(error ** 2))),
            'critic_explained_variance': 1. - float(np.var(error)) / variance if variance > 1e-12 else None,
        })
        all_p.extend(p.tolist()); all_v.extend(v.tolist()); all_mc.extend(truth.tolist())
    v, truth = np.array(all_v), np.array(all_mc)
    results[version] = {'episodes': episodes, 'samples': len(all_p),
                        'pooled_critic_rmse': float(np.sqrt(np.mean((v - truth) ** 2)))}
result = {'results': results, 'wall_s': time.perf_counter() - started,
          'physical_steps': 0, 'fitting': False,
          'scope': 'Training-only descriptive diagnostic on changing-policy realized trajectories; no unbiased final-policy or causal claim.'}
write_json(out / 'training_diagnostic.json', result)
print(json.dumps(result, indent=2))
