"""Native-value units, loss derivatives and unchanged actor checks; no physics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, torch, np, json, hashlib, tensor_features, write_json
from networks_v1 import ActorCritic as BaseActorCritic
from networks import ActorCritic, value_loss
from trace_math import preflight as trace_preflight

OUT = ROOT / 'outputs/grid10'
D = json.loads((OUT / 'design.json').read_text())
for path, expected in json.loads((OUT / 'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
source = torch.load(ROOT / D['source_checkpoint']['path'], weights_only=False)
base = BaseActorCritic('gnn', source['n_nodes'], max_edges=source['max_edges'])
base.load_state_dict(source['state']); base.eval()
models = {}
warm_metrics = {}
with np.load(ROOT / D['critic_data']['path']) as data:
    X = torch.from_numpy(data['x']); targets = torch.from_numpy(data['returns'])
    assert len(X) == D['critic_data']['samples'] and set(data['weeks']).issubset(D['training_weeks'])
    for version in D['versions']:
        model = ActorCritic('gnn', source['n_nodes'], max_edges=source['max_edges'])
        missing = model.load_state_dict(source['state'], strict=False)
        assert missing.missing_keys == ['value_mu', 'value_sigma'] and not missing.unexpected_keys
        warm = torch.load(ROOT / D['critic_sources'][version]['path'], weights_only=False)
        model.value.load_state_dict(warm['state'])
        model.value_mu.fill_(warm['mu']); model.value_sigma.fill_(warm['sigma']); model.eval()
        for name, tensor in base.policy.state_dict().items():
            assert torch.equal(tensor, model.policy.state_dict()[name]), name
        with torch.no_grad():
            pred = model.value(X).flatten() * model.value_sigma + model.value_mu
        warm_metrics[version] = {'mu': float(model.value_mu), 'sigma': float(model.value_sigma),
                                 'training_rmse': float(((pred - targets) ** 2).mean().sqrt())}
        models[version] = model

states = 0; maximum_logit_error = 0.; maximum_value_error = 0.
manifest = json.loads((ROOT / 'outputs/grid08/v2/train/gnn/manifest.json').read_text())
for item in manifest[-4:]:
    assert item['scenario'] in D['training_weeks']
    with np.load(ROOT / item['path'] / 'teacher.npz') as data:
        for j in range(len(data['actions'])):
            feature = (data['x'][j], data[f'e{j}'], data['g'][j])
            x, _, g = tensor_features(feature)
            pooled = torch.cat([x.mean(0), x.max(0).values, g[:6]])
            with torch.no_grad():
                reference, _ = base(*tensor_features(feature))
                for model in models.values():
                    logits, actual = model(*tensor_features(feature))
                    expected = model.value(pooled).flatten() * model.value_sigma + model.value_mu
                    maximum_logit_error = max(maximum_logit_error, float((logits - reference).abs().max()))
                    maximum_value_error = max(maximum_value_error, float((actual - expected).abs().max()))
                    assert torch.equal(logits, reference)
                    assert torch.allclose(actual, expected, atol=3e-5, rtol=1e-5)
            states += 1
assert states == 532

# Independently hand-derived mean-square value and gradient in native units.
pred = torch.tensor([3., 8.], dtype=torch.float64, requires_grad=True)
truth = torch.tensor([1., 2.], dtype=torch.float64)
loss = value_loss(pred, truth, torch.tensor(-7.), torch.tensor(4.), 'normalized')
assert abs(float(loss.detach()) - 1.25) < 1e-12
loss.backward()
assert torch.allclose(pred.grad, torch.tensor([.125, .375], dtype=torch.float64), atol=1e-12, rtol=0)
# Translation of target units preserves loss; native unscaled targets/bootstraps remain intact.
assert torch.allclose(loss.detach(), value_loss(pred.detach(), truth, torch.tensor(19.), torch.tensor(4.), 'normalized'))
raw = value_loss(pred.detach(), truth, torch.tensor(0.), torch.tensor(1.), 'raw')
assert float(raw) == 3.5
report = {'passed': True, 'physical_steps': 0, 'actor_equivalent_training_states': states,
          'max_logit_error': maximum_logit_error, 'max_native_value_error': maximum_value_error,
          'normalized_loss_and_gradient': True, 'raw_huber_hand_check': True,
          'warm_critic_training_metrics': warm_metrics, 'physical_gae': trace_preflight(),
          'scope': 'Math, architecture and training-only warm-start qualification. No policy-performance evidence.'}
write_json(OUT / 'preflight.json', report)
print(json.dumps(report, indent=2))
