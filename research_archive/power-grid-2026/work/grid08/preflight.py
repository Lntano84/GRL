import json,time,traceback
from common import *
from torch.distributions import Categorical
start=time.perf_counter(); env=None
try:
    d=json.loads((OUT/'design.json').read_text()); env=make_env()
    scenario='2035-01-08_5'; env.seed(0);env.set_id(scenario);obs=env.reset()
    ctl=ResidualControl(env,obs)
    model=ActorCritic('gnn',2*env.n_sub)
    original=json.loads((OUT/'original_assets.json').read_text())
    reference=[json.loads(x) for x in (ROOT/'outputs/grid06/runs'/f'{scenario}__NN20'/'steps.jsonl').read_text().splitlines()]
    offered=0;eligible=0; checks=0;statuses={}; reward=0.
    with (OUT/'preflight_proposals.jsonl').open('x',encoding='utf-8') as f:
        for k in range(1,385):
            a,r,p=ctl.propose(obs,reward,d['cadence'])
            feat=graph_features(obs,env,p); logits,value=model(*tensor_features(feat))
            loss=(value.square().mean()+Categorical(logits=logits).entropy().mean())
            loss.backward();model.zero_grad()
            if not p['offered']: assert torch.softmax(logits,dim=-1)[0,1].item()==0
            f.write(json.dumps({'step':k,**p})+'\n')
            eligible+=p['eligible'];offered+=p['offered']
            for s in p['solve_status']:statuses[s]=statuses.get(s,0)+1
            # Qualification deliberately executes unmodified baseline, not proposed restoration.
            obs,reward,done,info=env.step(a)
            assert digest(action_vector(a))==reference[k-1]['action_hash']
            assert digest(obs.to_vect())==reference[k-1]['observation_hash']
            checks+=1
            if done:break
    # Independent permutation test for graph encoder only (no MLP comparison claim).
    feat=graph_features(obs,env,p);x,e,g=tensor_features(feat)
    perm=torch.randperm(len(x));inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
    with torch.no_grad():
        z1,v1=model(x,e,g);z2,v2=model(x[perm],inverse[e],g)
    err=max((z1-z2).abs().max().item(),(v1-v2).abs().max().item())
    assert err<1e-5
    for path,h in original.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    write_json(OUT/'preflight.json',{'passed':True,'baseline_prefix_exact_steps':checks,
        'eligible':eligible,'offered':offered,'safe_solve_statuses':statuses,
        'gnn_forward_backward':True,'mask_checked':True,'permutation_max_error':err,
        'original_files_unchanged':True,'wall_s':time.perf_counter()-start,
        'scope':'No restoration executed; forecast legality screened; not a training performance result.'})
    print(json.dumps(json.loads((OUT/'preflight.json').read_text()),indent=2))
except Exception:
    write_json(OUT/'preflight_failure.json',{'traceback':traceback.format_exc(),'wall_s':time.perf_counter()-start})
    raise
finally:
    if env:env.close()
