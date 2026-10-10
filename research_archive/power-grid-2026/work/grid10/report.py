"""Compare fixed final controllers on jointly completed development weeks."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid10'
D = json.loads((OUT / 'design.json').read_text())
records = []
for version in D['versions']:
    audit = json.loads((OUT / version / 'audit.json').read_text())
    assert audit['passed']
    records.extend({**r, 'version': version} for r in audit['records'] if '/evaluate/' in r['path'].replace('\\', '/'))
base = json.loads((ROOT / 'outputs/grid08/v0/evaluate/baselines/manifest.json').read_text())
v2 = json.loads((ROOT / 'outputs/grid08/v2/evaluate/gnn/manifest.json').read_text())
refs = {name: {r['scenario']: r for r in base if r['policy'] == name}
        for name in ['NN20', 'ALWAYS_RESTORE', 'IMMEDIATE_COST']}
refs['V2_GNN'] = {r['scenario']: r for r in v2}
refs['RAW_HUBER'] = {r['scenario']: r for r in records if r['version'] == 'raw'}

def trajectory(item):
    with gzip.open(ROOT / item['path'] / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f:
        return [(r['action_hash'], r['after_hash']) for r in map(json.loads, f)]

comparisons = []
for version in D['versions']:
    models = {r['scenario']: r for r in records if r['version'] == version}
    for name, reference in refs.items():
        if version == 'raw' and name == 'RAW_HUBER': continue
        pairs = []
        for week in D['evaluation_weeks']:
            a, b = models[week], reference[week]
            pairs.append({'week': week, 'model_steps': a['steps'], 'reference_steps': b['steps'],
                'model_complete': a['complete'], 'reference_complete': b['complete'],
                'gain': (b['raw_cost'] - a['raw_cost']) / abs(b['raw_cost']) if a['complete'] and b['complete'] else None,
                'identical_trajectory': trajectory(a) == trajectory(b)})
        gains = [p['gain'] for p in pairs if p['gain'] is not None]
        comparisons.append({'version': version, 'reference': name, 'pairs': pairs,
            'joint_complete': len(gains), 'mean_gain': float(np.mean(gains)) if gains else None,
            'positive': sum(g > 1e-9 for g in gains), 'negative': sum(g < -1e-9 for g in gains),
            'identical': sum(p['identical_trajectory'] for p in pairs)})
resources = {v: json.loads((OUT / v / 'audit.json').read_text())['physical_steps'] for v in D['versions']}
resources['total_physical'] = sum(resources.values())
resources['bytes'] = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
assert resources['total_physical'] <= 50000 and resources['bytes'] <= 1073741824
for path, expected in json.loads((OUT / 'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
result = {'comparisons': comparisons, 'records': records, 'resources': resources, 'final_test_used': False,
          'limits': ['One seed, four already-used development weeks; not independent confirmation.',
                     'Fixed normalization and MSE loss change together; no adaptive PopArt implementation.',
                     'Warm-fit targets are changing-policy training returns; not unbiased final-policy targets.',
                     'Saved-vector arithmetic audit is not independent physics replay or retraining.']}
(OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
lines = ['# GRID10: paired critic training package', '',
         '| Version | Reference | Jointly complete | Mean cost reduction | Positive / negative | Identical trajectories |',
         '|---|---|---:|---:|---:|---:|']
for c in comparisons:
    g = 'NA' if c['mean_gain'] is None else f"{100*c['mean_gain']:.4f}%"
    lines.append(f"| {c['version']} | {c['reference']} | {c['joint_complete']} | {g} | {c['positive']} / {c['negative']} | {c['identical']} |")
lines += ['', 'Same V2 actor and equal-budget critic warm-fits; native physical discount, reward and actor advantages retained. Fixed final checkpoints only.', '', json.dumps(resources)]
lines += ['- ' + x for x in result['limits']]
(OUT / 'GRID10_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'comparisons': comparisons, 'resources': resources}, indent=2))
