"""Train only one target variant from immutable original initial checkpoints."""
import hashlib,json,math,os,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid38'
sys.path[:0]=[str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]
import numpy as np
import torch
from prior import ResidualScorer
from components import tensors,batch,fingerprint
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
D=read(OUT/'design.json');assert (ROOT/'outputs/grid37/finished.json').exists(),'Do not fit during fair timing workers.'
assert not (OUT/'training_finished.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
started=time.perf_counter();records=[];target_records=[];names=['x','edges','edge_attr','glob','base','plans']
input_ids=read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']
assert len(input_ids)==167 and len(set(input_ids))==167
for identifier in input_ids:
    path=ROOT/'outputs/grid21/inputs/train'/f'{identifier}.json'
    meta=read(path);assert meta['week'] in D['training_weeks'];c=read(ROOT/meta['source']);scores=dict(zip(c['pool_ids'],c['outcomes']))
    with np.load(path.with_suffix('.npz')) as z:
        features={k:z[k].copy() for k in names};original=z['target'].copy();mask=z['mask'].copy();prior=z['prior'].copy();reps=z['representatives'].tolist()
    assert fingerprint(features)==meta['input_hash'];valid=[i for i in reps if scores[i]['strict_admissible']]
    optimal=[];target=np.zeros_like(original)
    if valid:
        maximum=max(float(np.float32(scores[i]['reward'])) for i in valid)
        optimal=[i for i in valid if float(np.float32(scores[i]['reward']))==maximum]
        target=(1-D['mixture'])*original;target[optimal]+=D['mixture']/len(optimal)
        assert abs(float(target.sum(dtype=np.float64))-1)<1e-6 and np.all(target[~mask]==0)
        records.append({'id':path.stem,'features':features,'tensor':tensors(features),'mask':torch.as_tensor(mask),'target':torch.as_tensor(target),'prior':torch.as_tensor(prior)})
    else:assert not original.any()
    target_records.append({'id':path.stem,'source':meta['source'],'original_input_sha256':hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),'valid_count':len(valid),'reward_optimal':optimal,'target':target.tolist()})
save(OUT/'targets.json',target_records);assert len(target_records)==167 and len(records)==148
fits=[]
for kind in D['kinds']:
    for seed in D['seeds']:
        initial_path=ROOT/f'outputs/grid21/models/{kind}_seed{seed}/initial.pt';state=torch.load(initial_path,map_location='cpu',weights_only=True)
        n_nodes=len(records[0]['features']['x']);max_edges=(state['inner.core.flat.0.weight'].shape[1]-n_nodes*14)//6 if kind=='mlp' else 372
        torch.manual_seed(seed)
        model=ResidualScorer(kind,n_nodes,max_edges,len(records[0]['features']['base']),records[0]['features']['plans'].shape[1],state['inner.subids'].tolist())
        model.load_state_dict(state);folder=OUT/'models'/f'{kind}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
        torch.save(model.state_dict(),folder/'initial.pt')
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=D['lr'],weight_decay=D['weight_decay'])
        rng=np.random.default_rng(seed+20261010);curve=[];begun=time.perf_counter()
        for epoch in range(D['epochs']):
            order=rng.permutation(len(records));total=0.;seen=0
            for offset in range(0,len(order),D['batch_size']):
                rows=[records[i] for i in order[offset:offset+D['batch_size']]];residual=model(**batch(rows))
                prior=torch.stack([r['prior'] for r in rows]);mask=torch.stack([r['mask'] for r in rows]);target=torch.stack([r['target'] for r in rows])
                logs=torch.nn.functional.log_softmax((residual+prior).masked_fill(~mask,-1e9),dim=1)
                plogs=torch.nn.functional.log_softmax(prior.masked_fill(~mask,-1e9),dim=1)
                kl=(logs.exp()*(logs-plogs)).sum(1).mean()
                loss=-(target*logs).sum(1).mean()+D['prior_KL_weight']*kl;assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(rows);seen+=len(rows)
                assert time.perf_counter()-started<D['caps']['training_s']
            curve.append({'epoch':epoch+1,'training_objective':total/seen,'wall_s':time.perf_counter()-begun});save(folder/'curve.json',curve)
            if (epoch+1)%16==0:print(json.dumps({'kind':kind,'seed':seed,**curve[-1]}),flush=True)
        # The unused original GRL actor/critic/trunk remain frozen and archived.
        frozen=[k for k in state if any(part in k for part in ['core.actor.','core.critic.','core.trunk.'])]
        assert all(torch.equal(model.state_dict()[k],state[k]) for k in frozen)
        torch.save(model.state_dict(),folder/'final.pt');model.eval()
        stat={'kind':kind,'seed':seed,'preference_rows':len(records),'epochs':D['epochs'],'training_s':time.perf_counter()-begun,'final_training_objective':curve[-1]['training_objective'],'same_original_initial_state':True,'frozen_unused_tensors_checked':len(frozen)}
        save(folder/'summary.json',stat);fits.append(stat)
save(OUT/'training_finished.json',{'fits':fits,'wall_s':time.perf_counter()-started,'target_rows':167,'valid_rows':148,'new_forecasts':0,'development_labels_opened':0,'reserved_evaluations':0})
(OUT/'RUN_STATE.md').write_text('ALL4 FITS SAVED; development ranking/closed-loop not yet evaluated.\n',encoding='utf-8')
print(json.dumps({'fits':4,'training_wall_s':time.perf_counter()-started}))
