"""Freeze official-core adaptation and build only existing training observations."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'work/grid40'), str(ROOT / 'work/grid20')]
from adapter import OUT, GAT, raw_features, graph, model_config, read
from common import np, torch, make_env, ResidualControl, digest, write_json
from components import fingerprint
from torch_geometric.data import Batch

assert not (OUT / 'design.json').exists()
OUT.mkdir(parents=True, exist_ok=True)
for row in read(ROOT / 'outputs/grid_softgnn_reference/manifest.json')['files']:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']
begun = time.perf_counter()
env = make_env(); env.seed(0); initial = env.reset(); ctl = ResidualControl(env, initial)
n_actions = len(ctl.base.topo_n1_unsafe.topo_act_list)
ids = read(ROOT / 'outputs/grid21/training_finished.json')['training_input_ids']
records = []; feature_hashes = []; usable = []
for identifier in ids:
    folder = ROOT / 'outputs/grid21/inputs/train'
    meta = read(folder / f'{identifier}.json'); source = ROOT / meta['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['source_sha256']
    with np.load(source.with_name(source.stem + '_state.npz')) as z:
        obs = initial.copy(); obs.from_vect(z['observation'])
        base = env.action_space(); base.from_vect(z['base_action'])
    assert digest(obs.to_vect()) == meta['state_hash']
    features = raw_features(obs, base)
    with np.load(folder / f'{identifier}.npz') as z:
        target, mask, representatives = z['target'].copy(), z['mask'].copy(), z['representatives'].copy()
        assert np.array_equal(features['global_features'], z['base'])
    assert mask.shape == target.shape == (n_actions,)
    assert np.array_equal(np.flatnonzero(mask), representatives)
    assert not np.any(target[~mask])
    if target.sum() > 0:
        assert abs(float(target.sum()) - 1.) < 1e-6
        usable.append(features['x'])
    else:
        assert target.sum() == 0
    dest = OUT / 'inputs' / f'{identifier}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **features, target=target, mask=mask)
    records.append({'id': identifier, 'week': meta['week'], 'step': meta['step'],
                    'source': meta['source'], 'source_sha256': meta['source_sha256'],
                    'state_hash': meta['state_hash'], 'feature_hash': fingerprint(features),
                    'positive_target': bool(target.sum() > 0),
                    'npz_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                    'nodes': len(features['x']), 'directed_edges': features['edges'].shape[1]})
assert len(records) == 167 and len(usable) == 148
node_matrix = np.concatenate(usable).astype(np.float64)
mean = node_matrix.mean(0).astype(np.float32)
std = node_matrix.std(0); scale = np.where(std > 1e-8, std, 1.).astype(np.float32)
np.savez_compressed(OUT / 'normalization.npz', mean=mean, scale=scale)
architecture = {
    'in_channels': 27,
    'gat_layers': [
        {'out_channels': 16, 'heads': 1, 'dropout': .17710654571580198, 'concat': True},
        {'out_channels': 32, 'heads': 4, 'dropout': .13918996625925875, 'concat': True},
        {'out_channels': 64, 'heads': 2, 'dropout': .1739807941185772, 'concat': True},
        {'out_channels': 128, 'heads': 4, 'dropout': .09648865126183648, 'concat': False}],
    'linear_layers': [{'out_features': 1024, 'dropout': .49583389349521434},
                      {'out_features': 1024, 'dropout': .4891896004401196}, {'out_features': n_actions}],
    'dropout': 0., 'pooling_type': 'max', 'gat_activation': 'elu', 'linear_activation': 'relu',
    'global_features_size': len(features['global_features'])}
D = {'stage': 'GRID40_RESOURCE_MATCHED_SOFTGNN_ADAPTATION',
     'source_commit': read(ROOT / 'outputs/grid_softgnn_reference/manifest.json')['commit'],
     'source_repository': 'https://github.com/ainetus/soft_label_gnn',
     'architecture': architecture, 'n_actions': n_actions, 'seeds': [0, 1], 'epochs': 64, 'batch_size': 8,
     'lr': .0041438789382345345, 'weight_decay': 8.47714945228751e-06,
     'training_weeks': read(ROOT / 'outputs/grid38/design.json')['training_weeks'],
     'training_ids': ids, 'normalization_ids': [r['id'] for r in records if r['positive_target']],
     'target': 'Existing canonical strictly-admissible exp(-(rho-best)/.01), identical GRID21 targets; all legal representatives in softmax; KLDivLoss batchmean.',
     'adaptation': 'Official GAT and27feature converter imported byte-for-byte. WCCI2022->IDF2023;2030->909classes; concatenate same1605public base-action vector through author-supported global feature input; training-only node-feature scaling. No PPO prior in this standalone imitation ranker. Shared controller/PPO/guard/search remains identical.',
     'training_scope': 'Resource-matched initial64epochs/batch8 equal original learner passes/updates, retain both seeds and final not best checkpoint. Author batch256/800max/validation scheduler and distributed tuning not rerun. No author checkpoint supplied. Inspect convergence before claiming a strong final paper baseline; weak pilot cannot refute published method.',
     'evaluation': 'After both fixed fits,26existing cached FULL states and known1733critical action; then common-cache full-controller matched live comparison. No final8date families.',
     'caps': {'training_s': 2400, 'new_teacher_queries': 0, 'reserved_evaluations': 0}}
write_json(OUT / 'design.json', D); write_json(OUT / 'inputs_manifest.json', {'records': records})
torch.manual_seed(937)
model = GAT(model_config(D)); model.eval()
test = graph(features, mean, scale)
with torch.no_grad(): y = model(Batch.from_data_list([test]))
assert y.shape == (1, n_actions) and torch.isfinite(y).all()
permutation = torch.randperm(test.num_nodes); inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(test.num_nodes)
permuted = test.clone(); permuted.x = test.x[permutation]; permuted.edge_index = inverse[test.edge_index]
with torch.no_grad(): yp = model(Batch.from_data_list([permuted]))
error = float((y - yp).abs().max()); assert error < 2e-5
loss = torch.nn.KLDivLoss(reduction='batchmean')(y, torch.softmax(y.detach(), 1)); assert abs(float(loss)) < 1e-5
assert raw_features(obs, base).keys() == {'x', 'edges', 'global_features'}
write_json(OUT / 'preflight.json', {'passed': True, 'records': len(records), 'positive': len(usable),
            'official_source_hashes_checked': True, 'public_base_matches_original': True,
            'node_permutation_max_logit_error': error, 'finite_forward': True,
            'parameters': sum(p.numel() for p in model.parameters()), 'wall_s': time.perf_counter() - begun})
files = list((ROOT / 'work/grid40').glob('*.py')) + [ROOT / r['path'] for r in read(ROOT / 'outputs/grid_softgnn_reference/manifest.json')['files']]
files += [OUT / 'design.json', OUT / 'normalization.npz', OUT / 'inputs_manifest.json']
write_json(OUT / 'code_freeze.json', {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
write_json(OUT / 'environment.json', {'torch': torch.__version__, 'pyg': __import__('torch_geometric').__version__, 'threads': torch.get_num_threads(), 'device': 'cpu'})
(OUT / 'RUN_STATE.md').write_text('FROZEN:official-core adapted baseline, both fixed fits not started; final dates sealed.\n', encoding='utf-8')
print(json.dumps(read(OUT / 'preflight.json'), indent=2))
env.close()
