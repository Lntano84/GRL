"""Fit archived training returns only; diagnose capacity/optimization, not policy gains."""
import copy,gzip
from common import *
from networks_v1 import ActorCritic
FOLDER=OUT/'critic_capacity_probe';assert not FOLDER.exists();FOLDER.mkdir()
config={'checkpoint':'outputs/grid08/v2/train/gnn/final.pt','episodes':'Last completed training pass only',
        'objectives':['raw_huber','static_normalized_mse'],'epochs':64,'minibatch':64,'learning_rate':.0003,
        'seed':20261009,'wall_cap_s':120,
        'scope':'Training-set capacity check. No held-out policy evaluation, no simulator, no actor updates. Not adaptive PopArt implementation or a new RL version.'}
write_json(FOLDER/'design.json',config)
saved=torch.load(ROOT/config['checkpoint'],weights_only=False)
model=ActorCritic('gnn',saved['n_nodes'],max_edges=saved['max_edges']);model.load_state_dict(saved['state']);model.eval()
D=json.loads((OUT/'v2/design.json').read_text());CFG=D['ppo']
manifest=json.loads((OUT/'v2/train/gnn/manifest.json').read_text());features=[];targets=[];weeks=[]
for item in manifest[-4:]:
    path=ROOT/item['path'];rows=[json.loads(x) for x in gzip.open(path/'steps.jsonl.gz','rt',encoding='utf-8')]
    returns=np.zeros(len(rows));future=0.
    for i in reversed(range(len(rows))):
        row=rows[i];reward=-row['raw_cost']/CFG['cost_scale']
        if row['done'] and not row['complete']:reward-=CFG['blackout_penalty']
        future=reward+CFG['gamma']*(0. if row['done'] else future);returns[i]=future
    indices=[i for i,r in enumerate(rows) if r['proposal']['offered']]
    if not indices:continue
    with np.load(path/'teacher.npz') as f:
        assert len(indices)==len(f['actions'])
        for j,i in enumerate(indices):
            x=torch.from_numpy(f['x'][j]);g=torch.from_numpy(f['g'][j])
            pooled=torch.cat([x.mean(0),x.max(0).values,g[:6]])
            with torch.no_grad():_,v=model(*tensor_features((f['x'][j],f[f'e{j}'],f['g'][j])))
            assert torch.allclose(model.value(pooled).flatten(),v.flatten(),atol=1e-5,rtol=1e-5)
            features.append(pooled);targets.append(returns[i]);weeks.append(item['scenario'])
X=torch.stack(features);Y=torch.tensor(targets,dtype=torch.float32);mu=Y.mean();sigma=Y.std(unbiased=False).clamp_min(1e-6)
np.savez_compressed(FOLDER/'training_data.npz',x=X.numpy(),returns=Y.numpy(),weeks=np.array(weeks))
def metrics(pred):
    error=pred-Y
    return {'rmse':float(torch.mean(error**2).sqrt()),'mae':float(error.abs().mean()),
            'mean_bias':float(error.mean()),'explained_variance':1.-float(error.var(unbiased=False)/Y.var(unbiased=False))}
with torch.no_grad():initial=model.value(X).flatten();initial_metrics=metrics(initial)
start=time.perf_counter();results=[]
for mode in config['objectives']:
    torch.manual_seed(config['seed']);value=copy.deepcopy(model.value)
    if mode=='static_normalized_mse':
        with torch.no_grad():value[-1].weight.div_(sigma);value[-1].bias.sub_(mu).div_(sigma)
        with torch.no_grad():assert torch.allclose(value(X).flatten()*sigma+mu,initial,atol=3e-5,rtol=1e-5)
    optimizer=torch.optim.Adam(value.parameters(),lr=config['learning_rate']);begin=time.perf_counter()
    for epoch in range(config['epochs']):
        for ids in torch.randperm(len(X)).split(config['minibatch']):
            assert time.perf_counter()-start<config['wall_cap_s']
            pred=value(X[ids]).flatten()
            loss=torch.nn.functional.smooth_l1_loss(pred,Y[ids]) if mode=='raw_huber' else torch.nn.functional.mse_loss(pred,(Y[ids]-mu)/sigma)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),.5);optimizer.step()
    with torch.no_grad():
        pred=value(X).flatten()
        if mode=='static_normalized_mse':pred=pred*sigma+mu
        measured=metrics(pred)
    torch.save({'state':value.state_dict(),'mu':float(mu) if mode=='static_normalized_mse' else 0.,
                'sigma':float(sigma) if mode=='static_normalized_mse' else 1.,'scope':config['scope']},FOLDER/f'{mode}_critic_only.pt')
    results.append({'objective':mode,'wall_s':time.perf_counter()-begin,**measured})
report={'config':config,'samples':len(X),'initial_metrics':initial_metrics,'results':results,
    'target_mean':float(mu),'target_std':float(sigma),
    'warning':'All errors are on the fitted training data. Realized returns arise from a changing stochastic policy and are not unbiased final-policy value labels. Better fit does not prove better actions, safe control, or a failure cause. No trained controller is replaced.'}
write_json(FOLDER/'results.json',report);print(json.dumps(report,indent=2))
