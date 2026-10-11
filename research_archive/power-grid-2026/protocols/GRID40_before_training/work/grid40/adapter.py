"""Resource-matched SoftGNN adaptation; official GAT/converter are imported unchanged.

Original MPL-2.0 sources live in outputs/grid_softgnn_reference/official.
This glue adds public base-action context, training-only scaling, common legal
canonical masks, and the unchanged staged-search interface. It is not a faithful
WCCI2022 reproduction or the author's supplied pretrained model.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/grid08'))
from common import np, torch
from torch_geometric.data import Data, Batch

OUT = ROOT / 'outputs/grid40'
OFFICIAL = ROOT / 'outputs/grid_softgnn_reference/official/gnn'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GAT = import_file('grid40_official_GAT', OFFICIAL / 'gnn_models.py').GAT
converter = import_file('grid40_official_obs', OFFICIAL / 'obs_converter.py')


def raw_features(obs, base):
    x, edges = converter.collect_node_and_edge_features_separated(obs)
    parts = []
    for name in type(base).attr_list_vect:
        values = np.asarray(base._get_array_from_attr_name(name), dtype=np.float32)
        values = np.nan_to_num(values, nan=0., posinf=100., neginf=-100.)
        scale = 100. if name in ['_redispatch', '_storage_power'] else 2.
        parts.append(np.clip(values, -100., 100.) / scale)
    result = {'x': x.numpy().copy(), 'edges': edges.numpy().copy(),
              'global_features': np.concatenate(parts).astype(np.float32)}
    assert result['x'].shape == (obs.dim_topo, 27)
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def graph(features, mean, scale):
    normalized = (features['x'] - mean) / scale
    assert np.isfinite(normalized).all()
    return Data(x=torch.as_tensor(normalized, dtype=torch.float32),
                edge_index=torch.as_tensor(features['edges'], dtype=torch.long),
                global_features=torch.as_tensor(features['global_features'][None], dtype=torch.float32))


def model_config(design):
    # Only declarative plain config is transcribed; downloaded pickle is never executed.
    return {**design['architecture'], 'loss': torch.nn.KLDivLoss(reduction='batchmean')}


class Predictor:
    def __init__(self, checkpoint):
        self.design = read(OUT / 'design.json')
        with np.load(OUT / 'normalization.npz') as z:
            self.mean, self.scale = z['mean'].copy(), z['scale'].copy()
        self.model = GAT(model_config(self.design))
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.model.eval()

    def scores(self, features):
        with torch.no_grad():
            values = self.model(Batch.from_data_list([graph(features, self.mean, self.scale)]))[0].numpy().copy()
        assert len(values) == self.design['n_actions'] and np.isfinite(values).all()
        return values


def install_search_adapter(Search):
    original_init, original_rank = Search.__init__, Search.rank

    def init(self, env, ctl, rule, budget, model_path=None):
        if rule != 'SOFTGNN':
            return original_init(self, env, ctl, rule, budget, model_path)
        original_init(self, env, ctl, rule, budget, None)
        assert model_path is not None and len(self.library) == read(OUT / 'design.json')['n_actions']
        self.checkpoint_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        self.softgnn = Predictor(model_path)

    def rank(self, obs, base, eligible, reps, aliases):
        if self.rule != 'SOFTGNN':
            return original_rank(self, obs, base, eligible, reps, aliases)
        begun = time.perf_counter()
        features = raw_features(obs, base)
        feature_s = time.perf_counter() - begun
        begun = time.perf_counter()
        values = self.softgnn.scores(features)
        order = sorted(reps, key=lambda i: (-float(values[i]), i))
        assert set(order) == set(reps) and len(order) == len(reps)
        return order, np.zeros(len(self.library), dtype=np.float32), features, {
            'feature_s': feature_s, 'prior_s': 0., 'forward_rank_s': time.perf_counter() - begun}

    Search.__init__, Search.rank = init, rank
    return original_init, original_rank
