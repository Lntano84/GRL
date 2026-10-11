"""Independent target/scaling accounting and training convergence qualification."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid40'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
D = read(OUT / 'design.json'); finished = read(OUT / 'training_finished.json')
assert len(finished['fits']) == 2
for p, h in read(OUT / 'code_freeze.json').items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
all_nodes = []; targets = []; masks = []; positive = 0
for r in read(OUT / 'inputs_manifest.json')['records']:
    p = OUT / 'inputs' / f"{r['id']}.npz"; assert hashlib.sha256(p.read_bytes()).hexdigest() == r['npz_sha256']
    source = read(ROOT / r['source']); oldp = ROOT / 'outputs/grid21/inputs/train' / f"{r['id']}.npz"
    with np.load(oldp) as z: old_target, old_mask = z['target'].copy(), z['mask'].copy()
    with np.load(p) as z:
        target, mask = z['target'], z['mask']
        assert np.array_equal(target, old_target) and np.array_equal(mask, old_mask)
        assert z['x'].shape[1] == 27 and z['global_features'].shape == (1605,)
        candidate = [(i, s) for i, s in zip(source['pool_ids'], source['outcomes']) if mask[i] and s['strict_admissible']]
        expected = np.zeros(D['n_actions'])
        if candidate:
            best = min(s['rho'] for i, s in candidate)
            raw = {i: math.exp(-(s['rho'] - best) / .01) for i, s in candidate}; denominator = math.fsum(raw.values())
            for i, v in raw.items(): expected[i] = v / denominator
        assert np.max(np.abs(expected - target)) < 1e-7
        assert not np.any(target[~mask])
        if target.sum() > 0:
            all_nodes.append(z['x'].copy()); targets.append(target.copy()); masks.append(mask.copy()); positive += 1
assert positive == 148
x = np.concatenate(all_nodes).astype(np.float64)
with np.load(OUT / 'normalization.npz') as z:
    assert np.array_equal(x.mean(0).astype(np.float32), z['mean'])
    sd = x.std(0); assert np.array_equal(np.where(sd > 1e-8, sd, 1.).astype(np.float32), z['scale'])
q = np.stack(targets).astype(np.float64); m = np.stack(masks)
frequency = q.mean(0)
frequency = np.maximum(frequency, 1e-15)[None, :] * m
frequency /= frequency.sum(1, keepdims=True)
nonzero = q > 0
frequency_kl = float(np.sum(q[nonzero] * np.log(q[nonzero] / frequency[nonzero])) / positive)
convergence = []
for fit in finished['fits']:
    p = OUT / 'models' / f"softgnn_seed{fit['seed']}"; curve = read(p / 'curve.json')
    assert len(curve) == D['epochs'] and [r['epoch'] for r in curve] == list(range(1, D['epochs'] + 1))
    assert all(math.isfinite(r[k]) for r in curve for k in ['training_kl', 'eval_training_kl', 'eval_training_top1', 'wall_s'])
    last = curve[-1]['eval_training_kl']
    convergence.append({'seed': fit['seed'], 'first_kl': curve[0]['eval_training_kl'], 'last_kl': last,
        'min_kl': min(r['eval_training_kl'] for r in curve), 'frequency_kl': frequency_kl,
        'last_vs_frequency': frequency_kl - last,
        'early8_mean_kl': float(np.mean([r['eval_training_kl'] for r in curve[:8]])),
        'late8_mean_kl': float(np.mean([r['eval_training_kl'] for r in curve[-8:]])),
        'last_top1': curve[-1]['eval_training_top1']})
R = {'passed_accounting': True, 'targets': 167, 'positive': positive, 'training_only_scaling_verified': True,
     'frequency_kl': frequency_kl, 'convergence': convergence,
     'warning': 'Training fit only, no validation/dev/final score read. Frequency KL is a fit diagnostic,not downstream baseline. Plateau/poorfit requires a bounded optimization adaptation before published-method comparison,not a claim that SoftGNN is weak.'}
(OUT / 'training_audit.json').write_text(json.dumps(R, indent=2), encoding='utf-8')
print(json.dumps(R, indent=2))
