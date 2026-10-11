"""Both fixed seeds, same existing targets; no validation/final-test reading."""
import hashlib
import os
import sys
import time
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/grid40'))
from adapter import OUT, GAT, read, graph, model_config
from common import np, torch, write_json
from torch_geometric.data import Batch

D = read(OUT / 'design.json'); assert read(OUT / 'preflight.json')['passed']
assert not (OUT / 'training_finished.json').exists()
for p, h in read(OUT / 'code_freeze.json').items():
    assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
with np.load(OUT / 'normalization.npz') as z: mean, scale = z['mean'].copy(), z['scale'].copy()
records = []
for row in read(OUT / 'inputs_manifest.json')['records']:
    p = OUT / 'inputs' / f"{row['id']}.npz"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['npz_sha256']
    with np.load(p) as z:
        if z['target'].sum() <= 0: continue
        features = {k: z[k].copy() for k in ['x', 'edges', 'global_features']}
        records.append((graph(features, mean, scale), torch.as_tensor(z['target'].copy()), torch.as_tensor(z['mask'].copy())))
assert len(records) == 148
begun = time.perf_counter(); summaries = []
for seed in D['seeds']:
    folder = OUT / 'models' / f'softgnn_seed{seed}'; folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); model = GAT(model_config(D))
    torch.save(model.state_dict(), folder / 'initial.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=D['lr'], weight_decay=D['weight_decay'])
    random = np.random.default_rng(seed + 20261010); curve = []; start = time.perf_counter()
    for epoch in range(D['epochs']):
        model.train(); order = random.permutation(len(records)); total = 0.
        for offset in range(0, len(order), D['batch_size']):
            selected = [records[i] for i in order[offset:offset + D['batch_size']]]
            prediction = model(Batch.from_data_list([r[0] for r in selected]))
            mask = torch.stack([r[2] for r in selected]); target = torch.stack([r[1] for r in selected])
            logs = torch.nn.functional.log_softmax(prediction.masked_fill(~mask, -1e9), dim=1)
            loss = torch.nn.functional.kl_div(logs, target, reduction='batchmean')
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(selected)
            assert time.perf_counter() - begun < D['caps']['training_s']
        model.eval()
        with torch.no_grad():
            clean_loss = 0.; top1 = 0
            for offset in range(0, len(records), D['batch_size']):
                part = records[offset:offset + D['batch_size']]
                pred = model(Batch.from_data_list([r[0] for r in part]))
                masks = torch.stack([r[2] for r in part]); targets = torch.stack([r[1] for r in part])
                logs = torch.nn.functional.log_softmax(pred.masked_fill(~masks, -1e9), 1)
                clean_loss += float(torch.nn.functional.kl_div(logs, targets, reduction='sum'))
                top1 += int((logs.argmax(1) == targets.argmax(1)).sum())
        curve.append({'epoch': epoch + 1, 'training_kl': total / len(records), 'eval_training_kl': clean_loss / len(records),
                      'eval_training_top1': top1 / len(records), 'wall_s': time.perf_counter() - start})
        write_json(folder / 'curve.json', curve)
        if (epoch + 1) % 8 == 0: print(__import__('json').dumps({'seed': seed, **curve[-1]}), flush=True)
    torch.save(model.state_dict(), folder / 'final.pt')
    summary = {'seed': seed, 'epochs': D['epochs'], 'positive_training_states': len(records),
               'trainable_parameters': sum(p.numel() for p in model.parameters()), 'training_s': time.perf_counter() - start,
               'first': curve[0], 'last': curve[-1], 'checkpoint': (folder / 'final.pt').relative_to(ROOT).as_posix()}
    write_json(folder / 'summary.json', summary); summaries.append(summary)
manifest = []
for p in (OUT / 'models').rglob('*.pt'):
    manifest.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size})
write_json(OUT / 'model_manifest.json', {'files': manifest, 'warning': 'Initial/final models retained locally; resource-matched adaptation, not author pretrained weights.'})
write_json(OUT / 'training_finished.json', {'fits': summaries, 'wall_s': time.perf_counter() - begun, 'new_teacher_queries': 0, 'reserved_evaluations': 0})
(OUT / 'RUN_STATE.md').write_text('BOTH FITS COMPLETE:convergence and cached-state review pending; final dates sealed.\n', encoding='utf-8')
print(__import__('json').dumps(summaries, indent=2))
