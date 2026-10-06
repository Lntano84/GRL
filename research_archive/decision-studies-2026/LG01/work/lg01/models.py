"""Two rounds of bipartite messages; joint graph/neighborhood set scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import pickle, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from features import Normalize, table
from core import OUT, save, read, sha, ACTIONS
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def tensors(g): return {k:torch.from_numpy(v) for k,v in g.items()}

def pool(x):
    if not len(x): return torch.zeros(x.shape[1]*2,dtype=x.dtype)
    return torch.cat([x.mean(0),x.max(0).values])

class SetValue(nn.Module):
    def __init__(self,vd=23,rd=24,message=True):
        super().__init__(); self.message=message
        self.vin=nn.Sequential(nn.Linear(vd,32),nn.ReLU())
        self.rin=nn.Sequential(nn.Linear(rd,32),nn.ReLU())
        if message:
            self.vmsg=nn.ModuleList([nn.Linear(34,32) for _ in range(2)])
            self.rmsg=nn.ModuleList([nn.Linear(34,32) for _ in range(2)])
            self.vup=nn.ModuleList([nn.Linear(64,32) for _ in range(2)])
            self.rup=nn.ModuleList([nn.Linear(64,32) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(193,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,g):
        v,r=self.vin(g['v']),self.rin(g['r'])
        if self.message:
            ri,vi=g['ei']; edge=g['ev']
            vd=torch.bincount(vi,minlength=len(v)).clamp_min(1)[:,None]
            rd=torch.bincount(ri,minlength=len(r)).clamp_min(1)[:,None]
            for k in range(2):
                msg=torch.relu(self.vmsg[k](torch.cat([v[vi],edge],1)))
                agg=torch.zeros_like(r).index_add(0,ri,msg)/rd
                r=torch.relu(self.rup[k](torch.cat([r,agg],1)))
                msg=torch.relu(self.rmsg[k](torch.cat([r[ri],edge],1)))
                agg=torch.zeros_like(v).index_add(0,vi,msg)/vd
                v=torch.relu(self.vup[k](torch.cat([v,agg],1)))
        glob=torch.cat([pool(v),pool(r)])
        # A0 is identically zero; A1..A4 use the SAME set head.
        outputs=[torch.zeros((),dtype=v.dtype)]
        for mask in g['masks'][1:]:
            outputs.append(self.head(torch.cat([glob,pool(v[mask>0]),mask.mean().reshape(1)]))[0])
        return torch.stack(outputs)

def choice(p):
    p=np.asarray(p,float)
    return int(np.argmax(p)) if np.isfinite(p).all() and np.max(p)>0 else 0

class NeuralScorer:
    def __init__(self,path):
        data=torch.load(path,map_location='cpu',weights_only=False)
        self.model=SetValue(message=data['message']); self.model.load_state_dict(data['state']); self.model.eval()
        self.norm=data['norm']; self.requires_edges=data['message']
    def choose(self,g):
        with torch.inference_mode(): p=self.model(tensors(self.norm.transform(g))).numpy()
        return choice(p),p

class TreeScorer:
    requires_edges=False
    def __init__(self,path):
        with open(path,'rb') as f: self.model=pickle.load(f)
    def choose(self,g):
        p=np.r_[0.,self.model.predict(table(g)[1:])]; return choice(p),p

def atomic_torch(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def train_neural(name,graphs,costs,denoms,families,train_ids,val_ids):
    dest=OUT/'models'; dest.mkdir(exist_ok=True,parents=True)
    path=dest/f'{name}.pt'; progress=dest/f'{name}.progress.pt'
    if (dest/f'{name}.training.json').exists():
        return read(dest/f'{name}.training.json')
    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    norm=Normalize().fit([graphs[i] for i in train_ids]); gs=[tensors(norm.transform(g)) for g in graphs]
    message=name=='gnn'; model=SetValue(message=message)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    target=(costs[:,[0]]-costs)/denoms[:,None]; targets=torch.from_numpy(target.astype(np.float32))
    criterion=nn.HuberLoss(delta=.02)
    start_wall=time.time(); epoch=0; batch=0; history=[]; best=float('inf'); bad=0; losses=[]; order=[]
    if progress.exists():
        ck=torch.load(progress,weights_only=False,map_location='cpu')
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer'])
        norm=ck['norm'];start_wall=ck['start_wall'];epoch=ck['epoch'];batch=ck['batch'];history=ck['history']
        best,bad,losses,order=ck['best'],ck['bad'],ck['losses'],ck['order']
        random.setstate(ck['python_rng']);torch.set_rng_state(ck['torch_rng'])
    def evaluate():
        model.eval(); pred=[]
        with torch.inference_mode():
            for i in val_ids: pred.append(model(gs[i]).numpy())
        picks=[choice(p) for p in pred]
        regret=np.asarray([(costs[i,a]-min(costs[i]))/denoms[i] for i,a in zip(val_ids,picks)])
        valfamilies=np.asarray(families)[val_ids]
        macro=float(np.mean([np.mean(regret[valfamilies==f]) for f in sorted(set(valfamilies))]))
        return macro,picks,pred
    if not path.exists():
        # Initial model is saved only for diagnostics; validation selection concerns trained checkpoints.
        initial=evaluate()
        atomic_torch(dest/f'{name}.initial.pt',dict(state=model.state_dict(),norm=norm,message=message))
        save(dest/f'{name}.initial.json',dict(regret=initial[0],actions=initial[1]))
    while epoch<50 and bad<5 and time.time()-start_wall<1800:
        if not order:
            order=list(train_ids);random.shuffle(order);batch=0;losses=[]
        model.train()
        while batch<len(order) and time.time()-start_wall<1800:
            i=order[batch];opt.zero_grad(set_to_none=True)
            output=model(gs[i]);loss=criterion(output[1:],targets[i,1:]);loss.backward();opt.step()
            losses.append(float(loss.detach()));batch+=1
            atomic_torch(progress,dict(state=model.state_dict(),optimizer=opt.state_dict(),norm=norm,
                 start_wall=start_wall,epoch=epoch,batch=batch,history=history,best=best,bad=bad,
                 losses=losses,order=order,python_rng=random.getstate(),torch_rng=torch.get_rng_state()))
        if batch<len(order): break
        regret,picks,pred=evaluate();epoch+=1
        improved=regret<best-1e-12
        if improved:
            best=regret;bad=0
            atomic_torch(path,dict(state=model.state_dict(),norm=norm,message=message,epoch=epoch,regret=regret))
        else: bad+=1
        atomic_torch(dest/f'{name}.epoch{epoch:02}.pt',dict(state=model.state_dict(),norm=norm,message=message,epoch=epoch,regret=regret))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_regret=regret,
                            selected=improved,actions=picks,scores=[p.tolist() for p in pred],elapsed_s=time.time()-start_wall))
        save(dest/f'{name}.epochs.json',history)
        order=[];batch=0;losses=[]
        atomic_torch(progress,dict(state=model.state_dict(),optimizer=opt.state_dict(),norm=norm,
             start_wall=start_wall,epoch=epoch,batch=batch,history=history,best=best,bad=bad,
             losses=losses,order=order,python_rng=random.getstate(),torch_rng=torch.get_rng_state()))
        print(f'TRAIN {name} epoch={epoch} regret={regret:.6f} patience={bad} elapsed={time.time()-start_wall:.1f}',flush=True)
    if not path.exists(): raise RuntimeError(f'{name}: training cap without a complete validated epoch')
    result=dict(name=name,seed=7,epochs=epoch,wall_s=time.time()-start_wall,best_regret=best,
                selected_epoch=torch.load(path,weights_only=False)['epoch'],
                stopped='patience' if bad>=5 else 'max_epochs' if epoch>=50 else 'time_cap',
                validation_training_cap_ok=time.time()-start_wall<=1805,partial_batch=batch,
                best_hash=sha(path),message=message)
    save(dest/f'{name}.training.json',result); return result

def fit_all(graphs,costs,denoms,families,train_ids,val_ids):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    dest=OUT/'models';dest.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in ['gnn','mlp']: results.append(train_neural(name,graphs,costs,denoms,families,train_ids,val_ids))
    path=dest/'tree.pkl'
    if not path.exists():
        t0=time.perf_counter()
        x=np.concatenate([table(graphs[i])[1:] for i in train_ids])
        y=np.concatenate([((costs[i,0]-costs[i,1:])/denoms[i]) for i in train_ids])
        model=HistGradientBoostingRegressor(max_iter=100,max_depth=3,learning_rate=.05,l2_regularization=1,random_state=7,early_stopping=False)
        with threadpool_limits(limits=1): model.fit(x,y)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f: pickle.dump(model,f,protocol=5);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        save(dest/'tree.training.json',dict(wall_s=time.perf_counter()-t0,params=model.get_params(),hash=sha(path)))
    result={r['name']:r for r in results};result['tree']=read(dest/'tree.training.json')
    save(dest/'training.json',result)
    return result
