"""Training-only configuration choice; not downstream/test selection."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid40b'; OLD = ROOT / 'outputs/grid40'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
for p, h in read(OUT / 'code_freeze.json').items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
F = read(OUT / 'training_finished.json'); assert len(F['fits']) == 2
old = {r['seed']: r for r in read(OLD / 'training_audit.json')['convergence']}
rows = []
for fit in F['fits']:
    seed = fit['seed']; folder = OUT / 'models' / f'softgnn_seed{seed}'
    a = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
    b = torch.load(OLD / 'models' / f'softgnn_seed{seed}' / 'initial.pt', map_location='cpu', weights_only=True)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    curve = read(folder / 'curve.json'); assert len(curve) == 64
    assert all(math.isfinite(r[k]) for r in curve for k in ['training_kl', 'eval_training_kl', 'eval_training_top1'])
    initial = fit['first']['eval_training_kl']; final = fit['last']['eval_training_kl']; previous = old[seed]['last_kl']
    rows.append({'seed': seed, 'initial_weights_exact_original': True, 'old_final_training_kl': previous,
        'new_final_training_kl': final, 'kl_improvement': previous - final,
        'frequency_kl': old[seed]['frequency_kl'], 'below_frequency': final < old[seed]['frequency_kl'],
        'last_top1': fit['last']['eval_training_top1']})
qualified = all(r['kl_improvement'] >= .05 and r['below_frequency'] for r in rows)
R = {'passed_accounting': True, 'fits': rows, 'configuration_qualified': qualified,
     'selected_uniform_config': 'GRID40B' if qualified else None,
     'selection_uses_only_training_fit': True, 'development_predictions_read': 0, 'reserved_evaluations': 0,
     'warning': 'Improved training KL is neither downstream quality nor published-method superiority. One optimization adjustment only; original fits and both seeds preserved.'}
(OUT / 'audit.json').write_text(json.dumps(R, indent=2), encoding='utf-8')
print(json.dumps(R, indent=2))
