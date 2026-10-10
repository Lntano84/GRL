"""Describe paired stochastic training trajectories; not a policy comparison."""
import gzip
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid10'

def load(version):
    folder = OUT / version / 'train/gnn'
    assert (folder / 'finished.json').exists()
    return json.loads((folder / 'manifest.json').read_text())

records = []
for a, b in zip(load('raw'), load('normalized')):
    assert a['scenario'] == b['scenario']
    with gzip.open(ROOT / a['path'] / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f:
        x = list(map(json.loads, f))
    with gzip.open(ROOT / b['path'] / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f:
        y = list(map(json.loads, f))
    pairs = list(zip(x, y))
    state_diffs = [i + 1 for i, (p, q) in enumerate(pairs) if p['after_hash'] != q['after_hash']]
    records.append({'raw_path': a['path'], 'normalized_path': b['path'],
        'week': a['scenario'], 'paired_prefix': len(pairs),
        'raw_steps': len(x), 'normalized_steps': len(y),
        'different_choices': sum(p['choice'] != q['choice'] for p, q in pairs),
        'different_action_vectors': sum(p['action_hash'] != q['action_hash'] for p, q in pairs),
        'different_observations': len(state_diffs), 'first_observation_difference': state_diffs[0] if state_diffs else None,
        'raw_complete': a['complete'], 'normalized_complete': b['complete'],
        'raw_cost': a['raw_cost'], 'normalized_cost': b['raw_cost']})
assert len(records) == 8
result = {'records': records, 'physical_steps': 0,
    'scope': 'Descriptive paired prefixes under changing stochastic training policies, not controlled causal effects of individual actions or final-policy evaluation.'}
(OUT / 'training_trajectory_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
