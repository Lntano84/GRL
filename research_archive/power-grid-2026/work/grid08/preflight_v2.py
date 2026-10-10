"""Verify disclosed prior shift preserves critic, mask and checkpoint sources."""
from common import *
from networks_v1 import ActorCritic
D=json.loads((OUT/'v2/design.json').read_text());checks={}
for kind in ['gnn','mlp']:
    source=OUT/'v1/train'/kind/'final.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==D['warm_start']['input_hashes'][kind]
    saved=torch.load(source,weights_only=False)
    model=ActorCritic(kind,saved['n_nodes'],max_edges=saved['max_edges']);model.load_state_dict(saved['state'])
    file=next((OUT/'v1/train'/kind).glob('e*_/teacher.npz'),None)
    files=sorted((OUT/'v1/train'/kind).glob('e*_*/teacher.npz'));assert files
    with np.load(files[0]) as z:
        features=[(z['x'][i].copy(),z[f'e{i}'].copy(),z['g'][i].copy()) for i in range(min(20,len(z['g'])))]
    with torch.no_grad():old=[model(*tensor_features(f)) for f in features]
    shift=D['warm_start']['bias_shifts'][kind]
    with torch.no_grad():model.policy.actor.bias[1]+=shift
    probabilities=[]
    with torch.no_grad():
        for f,(previous_logits,previous_value) in zip(features,old):
            logits,value=model(*tensor_features(f));assert torch.equal(value,previous_value)
            assert abs(float(logits[0,1]-previous_logits[0,1])-shift)<1e-6
            probabilities.append(float(logits.softmax(-1)[0,1]))
            inactive=(f[0],f[1],f[2].copy());inactive[2][6]=0.
            masked,_=model(*tensor_features(inactive));assert masked.argmax(-1).item()==0
    checks[kind]={'critic_unchanged':True,'input_dependent_logit_differences_preserved':True,
        'mask_passed':True,'sample_restore_probabilities':probabilities}
write_json(OUT/'v2/preflight.json',{'passed':True,'checks':checks,'physical_steps':0,'fits':0})
print(json.dumps(checks,default=json_default))
