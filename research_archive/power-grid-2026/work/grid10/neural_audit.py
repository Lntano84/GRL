"""Recompute final-checkpoint evaluation choices from saved legal features.

Uses the shared model forward implementation; does not claim an independent
implementation, retraining or electrical replay.
"""
import gzip
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, torch, tensor_features, write_json
from networks import ActorCritic

OUT = ROOT / 'outputs/grid10'
assert json.loads((OUT / 'finished.json').read_text())['passed']
records = []
for version in ['raw', 'normalized']:
    saved = torch.load(OUT / version / 'train/gnn/final.pt', weights_only=False)
    model = ActorCritic('gnn', saved['n_nodes'], max_edges=saved['max_edges'])
    model.load_state_dict(saved['state']); model.eval()
    for item in json.loads((OUT / version / 'evaluate/gnn/manifest.json').read_text()):
        folder = ROOT / item['path']
        with gzip.open(folder / 'steps.jsonl.gz', 'rt', encoding='utf-8') as handle:
            rows = list(map(json.loads, handle))
        offered = [r for r in rows if r['proposal']['offered']]
        assert all(r['choice'] == 0 for r in rows if not r['proposal']['offered'])
        probabilities = []; values = []
        with np.load(folder / 'teacher.npz') as data:
            assert len(data['actions']) == len(offered)
            for j, row in enumerate(offered):
                with torch.no_grad():
                    logits, value = model(*tensor_features((data['x'][j], data[f'e{j}'], data['g'][j])))
                assert int(logits.argmax(-1)) == row['choice'], (version, item['scenario'], row['step'])
                probabilities.append(float(torch.softmax(logits, -1)[0, 1])); values.append(float(value))
        p = np.array(probabilities)
        records.append({'version': version, 'week': item['scenario'], 'offered_recomputed': len(offered),
            'restore_probability_min_median_max': [float(p.min()), float(np.median(p)), float(p.max())],
            'native_value_min_max': [min(values), max(values)],
            'mu': float(model.value_mu), 'sigma': float(model.value_sigma)})
write_json(OUT / 'neural_audit.json', {'passed': True, 'records': records,
    'total_choices_recomputed': sum(r['offered_recomputed'] for r in records), 'physical_steps': 0,
    'scope': 'Re-prediction with the shared model implementation; no independent forward implementation, retraining or physics replay.'})
print(json.dumps({'passed': True, 'total_choices_recomputed': sum(r['offered_recomputed'] for r in records)}))
