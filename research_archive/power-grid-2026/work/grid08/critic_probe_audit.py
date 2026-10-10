"""Reconstruct MC targets by discounted sums and independently rescore saved fits."""
import gzip,copy
from common import *
from networks_v1 import ActorCritic
FOLDER=OUT/'critic_capacity_probe';report=json.loads((FOLDER/'results.json').read_text())
with np.load(FOLDER/'training_data.npz') as f:X=np.array(f['x']);Y=np.array(f['returns']);weeks=np.array(f['weeks'])
design=json.loads((OUT/'v2/design.json').read_text());cfg=design['ppo']
target=[];week_labels=[]
for item in json.loads((OUT/'v2/train/gnn/manifest.json').read_text())[-4:]:
    rows=[json.loads(x) for x in gzip.open(ROOT/item['path']/'steps.jsonl.gz','rt',encoding='utf-8')]
    rewards=np.array([-r['raw_cost']/cfg['cost_scale'] for r in rows],dtype=float)
    if rows[-1]['done'] and not rows[-1]['complete']:rewards[-1]-=cfg['blackout_penalty']
    for i,row in enumerate(rows):
        if row['proposal']['offered']:
            target.append(float(np.dot(rewards[i:],cfg['gamma']**np.arange(len(rows)-i))));week_labels.append(item['scenario'])
assert np.array_equal(weeks,np.array(week_labels)) and len(target)==len(Y)==report['samples']
error=float(np.max(np.abs(Y-np.array(target))));assert np.allclose(Y,target,rtol=2e-6,atol=2e-5)
source=torch.load(ROOT/report['config']['checkpoint'],weights_only=False)
model=ActorCritic('gnn',source['n_nodes'],max_edges=source['max_edges']);model.load_state_dict(source['state'])
checks=[]
for item in report['results']:
    saved=torch.load(FOLDER/f"{item['objective']}_critic_only.pt",weights_only=False)
    value=copy.deepcopy(model.value);value.load_state_dict(saved['state'])
    with torch.no_grad():prediction=(value(torch.from_numpy(X)).flatten()*saved['sigma']+saved['mu']).numpy().astype(float)
    delta=prediction-Y.astype(float)
    measures={'rmse':float(np.sqrt(np.mean(delta**2))),'mae':float(np.mean(np.abs(delta))),
              'mean_bias':float(np.mean(delta)),'explained_variance':1.-float(np.var(delta))/float(np.var(Y.astype(float)))}
    for key,number in measures.items():assert abs(number-item[key])<=2e-5
    checks.append({'objective':item['objective'],'metrics':measures})
audit={'passed':True,'closed_discounted_sum_targets':len(Y),'max_abs_float32_target_rounding':error,
       'rescored_critic_fits':checks,'scope':'Independent target arithmetic and checkpoint rescoring; no independent refits or policy-performance evidence.'}
write_json(FOLDER/'audit.json',audit)
write_json(FOLDER/'artifact_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in FOLDER.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
print(json.dumps(audit,indent=2))
