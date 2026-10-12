"""Two fixed64epoch fits; preserve source initial/final checkpoints and all seeds."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid50';HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'work/grid21'),str(ROOT/'work/grid20'),str(HERE)]
import numpy as np
import torch
from compressed_residual import CompressedResidualScorer
from components import tensors,batch,fingerprint
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
D=read(OUT/'design.json');assert not (OUT/'training_finished.json').exists()
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
begun=time.perf_counter();records=[];names=['x','edges','edge_attr','glob','base','plans']
reserved={r['date'] for r in read(ROOT/'outputs/grid_final_holdout/reserved_date_families.json')['date_families']}
for identifier in read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']:
    p=ROOT/'outputs/grid21/inputs/train'/f'{identifier}.npz';meta=read(p.with_suffix('.json'))
    assert meta['week'] in D['training_weeks'] and meta['week'][:10] not in reserved
    with np.load(p) as z:features={k:z[k].copy() for k in names};target=z['target'].copy();mask=z['mask'].copy();prior=z['prior'].copy()
    assert fingerprint(features)==meta['input_hash']
    if target.any():records.append({'id':identifier,'features':features,'tensor':tensors(features),'target':torch.as_tensor(target),'mask':torch.as_tensor(mask),'prior':torch.as_tensor(prior)})
assert len(records)==148
fits=[]
for seed in D['seeds']:
    source=ROOT/f'outputs/grid21/models/gnn_seed{seed}/initial.pt';state=torch.load(source,map_location='cpu',weights_only=True)
    torch.manual_seed(seed);f=records[0]['features'];model=CompressedResidualScorer('gnn',len(f['x']),372,len(f['base']),f['plans'].shape[1],state['inner.subids'].tolist())
    model.load_state_dict(state);assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    with torch.no_grad():assert torch.count_nonzero(model(**records[0]['tensor']))==0
    folder=OUT/'models'/f'gnn_seed{seed}';folder.mkdir(parents=True,exist_ok=False);shutil.copy2(source,folder/'initial.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256((folder/'initial.pt').read_bytes()).hexdigest()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=D['lr'],weight_decay=D['weight_decay'])
    rng=np.random.default_rng(seed+20261010);curve=[];start=time.perf_counter()
    for epoch in range(D['epochs']):
        order=rng.permutation(len(records));total=0.;seen=0
        for offset in range(0,len(order),D['batch_size']):
            rs=[records[i] for i in order[offset:offset+D['batch_size']]];residual=model(**batch(rs))
            prior=torch.stack([r['prior'] for r in rs]);mask=torch.stack([r['mask'] for r in rs]);target=torch.stack([r['target'] for r in rs])
            logs=torch.nn.functional.log_softmax((residual+prior).masked_fill(~mask,-1e9),dim=1)
            plogs=torch.nn.functional.log_softmax(prior.masked_fill(~mask,-1e9),dim=1)
            kl=(logs.exp()*(logs-plogs)).sum(1).mean();loss=-(target*logs).sum(1).mean()+D['prior_KL_weight']*kl
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();total+=float(loss.detach())*len(rs);seen+=len(rs);assert time.perf_counter()-begun<D['caps']['training_s']
        curve.append({'epoch':epoch+1,'training_objective':total/seen,'wall_s':time.perf_counter()-start});save(folder/'curve.json',curve)
        if (epoch+1)%16==0:print(json.dumps({'seed':seed,**curve[-1]}),flush=True)
    frozen=[k for k in state if any(t in k for t in ['core.actor.','core.critic.','core.trunk.'])]
    assert all(torch.equal(model.state_dict()[k],state[k]) for k in frozen)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.dtype.is_floating_point)
    torch.save(model.state_dict(),folder/'final.pt')
    stats={'kind':'gnn','seed':seed,'epochs':64,'preference_rows':148,'training_s':time.perf_counter()-start,
       'final_training_objective':curve[-1]['training_objective'],'same_original_initial_bytes':True,
       'frozen_unused_tensors_checked':len(frozen),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save(folder/'summary.json',stats);fits.append(stats)
save(OUT/'training_finished.json',{'fits':fits,'wall_s':time.perf_counter()-begun,'new_forecasts':0,'development_labels_opened':0,'GRID44_evaluations':0})
(OUT/'RUN_STATE.md').write_text('BOTH FIXED64EPOCH FITS SAVED;olddevelopment cached comparison pending.\n',encoding='utf-8')
print(json.dumps({'fits':2,'wall_s':time.perf_counter()-begun}))
