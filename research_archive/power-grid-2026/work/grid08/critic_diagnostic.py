"""Training-only final-critic diagnostic on last-pass rollout Monte Carlo returns."""
import argparse,gzip
from common import *
from networks_v1 import ActorCritic
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['grid08','grid09'],default='grid08');ap.add_argument('--version',default='v2');ap.add_argument('--kind',default='gnn');a=ap.parse_args()
folder=ROOT/'outputs'/a.stage/a.version/'train'/a.kind
assert (folder/'finished.json').exists()
design=json.loads((ROOT/'outputs'/a.stage/('v2/design.json' if a.stage=='grid08' else 'design.json')).read_text())
cfg=design['ppo'];saved=torch.load(folder/'final.pt',weights_only=False)
model=ActorCritic(a.kind,saved['n_nodes'],max_edges=saved['max_edges']);model.load_state_dict(saved['state']);model.eval()
manifest=json.loads((folder/'manifest.json').read_text());records=[];all_values=[];all_returns=[]
for item in manifest[-4:]:
    path=ROOT/item['path'];rows=[json.loads(x) for x in gzip.open(path/'steps.jsonl.gz','rt',encoding='utf-8')]
    returns=np.zeros(len(rows));future=0.
    for i in reversed(range(len(rows))):
        row=rows[i];r=-row['raw_cost']/cfg['cost_scale']
        if row['done'] and not row['complete']:r-=cfg['blackout_penalty']
        future=r+cfg['gamma']*(0. if row['done'] else future);returns[i]=future
    indices=[i for i,r in enumerate(rows) if r['proposal']['offered']]
    if not indices:continue
    vals=[]
    with np.load(path/'teacher.npz') as f:
        assert len(f['actions'])==len(indices)
        for j in range(len(indices)):
            with torch.no_grad():_,v=model(*tensor_features((f['x'][j],f[f'e{j}'],f['g'][j])))
            vals.append(float(v.item()))
    truth=returns[indices];values=np.array(vals);error=values-truth
    variance=float(np.var(truth));ev=1.-float(np.var(error))/variance if variance>1e-12 else None
    records.append({'week':item['scenario'],'samples':len(indices),'complete':item['complete'],
        'mean_value':float(values.mean()),'mean_mc_return':float(truth.mean()),
        'rmse':float(np.sqrt(np.mean(error**2))),'mean_bias':float(error.mean()),'explained_variance':ev})
    all_values.extend(vals);all_returns.extend(truth.tolist())
values=np.array(all_values);truth=np.array(all_returns);error=values-truth
result={'stage':a.stage,'version':a.version,'kind':a.kind,'records':records,
    'pooled_rmse':float(np.sqrt(np.mean(error**2))),
    'pooled_mean_bias':float(error.mean()),
    'scope':'Training-only diagnostic of final critic versus realized discounted returns from the last training pass. No fitting or simulator calls. Rollouts come from a changing stochastic policy, so errors are not an unbiased final-policy value-function test or proof of a failure cause.'}
write_json(ROOT/'outputs'/a.stage/a.version/f'{a.kind}_critic_diagnostic.json',result)
print(json.dumps(result,indent=2))
