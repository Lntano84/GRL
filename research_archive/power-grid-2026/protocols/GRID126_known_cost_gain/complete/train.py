"""30 bounded paired fits; original sampling streams, no held-date inference."""
import os,time,hashlib,traceback
from common126 import *

def main():
    assert not (O/'training_started.json').exists();verify();src.configure();assert read(O/'frozen_push.json')['passed'];D=read(O/'design.json');P=read(O/'preflight.json');(O/'models').mkdir(exist_ok=False);start=time.perf_counter();records=[];net=None;opt=None;losslog=[]
    save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800))
    try:
        for split,seed,arm in D['training']['fit_order']:
            x,y,g,c,ids,p=getdata(split);X=torch.from_numpy(x);G=torch.from_numpy(g.astype('f'));C=torch.from_numpy(c.astype('f'));net=build(seed);initial=digest(net);assert initial==P['original_initial_hashes'][str(seed)];init_profile=profile(net,x,y,g,c,ids,p,arm);opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(len(y),np.int64);losslog=[];norms=[];name=f'{split}__{arm}_seed{seed}';fitstart=time.perf_counter()
            for update in range(1,7251):
                if time.perf_counter()-start>=1800:raise TimeoutError('Frozen30fit total1800s cap')
                ix=ids[rng.choice(len(ids),32,replace=True,p=p)];np.add.at(counts,ix,1);h.update(ix.astype('<i8').tobytes());net.train();q=net(X[ix]);main,stop=losses(q,G[ix],C[ix],arm);loss=main+stop;assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters());norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losslog.append([float(main.detach()),float(stop.detach())]);norms.append(float(norm))
                if update%2000==0 or update==7250:
                    progress=dict(stage='training',name=name,update=update,completed_fits=len(records),wall_s=time.perf_counter()-start);save(O/'state.json',progress);print(__import__('json').dumps(progress),flush=True)
            key=str(seed) if split=='full' else f'fold{split[1:]}_seed{seed}';expected=read(src.S/'preflight.json')['sampling_streams'][key]['index_sha256'] if split=='full' else read(CV/'preflight.json')['sampling_streams'][key];oldcounts=np.load(OLD/f'training_exposure_seed{seed}.npz')['state_draws'] if split=='full' else np.load(CV/f'{key}_expected_exposure.npz')['state_draws'];assert h.hexdigest()==expected and np.array_equal(counts,oldcounts)
            final_profile=profile(net,x,y,g,c,ids,p,arm);path=O/'models'/f'{name}.pt';torch.save(dict(split=split,seed=seed,arm=arm,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=7250,initial_hash=initial,final_hash=digest(net)),path);np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,half_main_stop_losses=np.asarray(losslog),gradient_norm_before_clip=np.asarray(norms))
            records.append(dict(name=name,split=split,seed=seed,arm=arm,updates=7250,parameters=6978,initial_hash=initial,final_hash=digest(net),checkpoint_sha256=sha(path),index_sha256=h.hexdigest(),initial_fit_profile=init_profile,final_fit_profile=final_profile,clipped_updates=int(np.sum(np.asarray(norms)>10)),wall_s=time.perf_counter()-fitstart));save(O/'fit_progress.json',dict(completed=records,wall_s=time.perf_counter()-start))
        verify();save(O/'fits.json',dict(passed=True,new_fits=30,records=records,wall_s=time.perf_counter()-start,new_model_validation_or_holdout_forwards_during_training=0,new_physical_calls=0));save(O/'state.json',dict(stage='all30_endpoints_complete_waiting_evaluation',fits=30))
    except BaseException as e:
        if net is not None:torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,updates_completed=len(losslog),name=name),O/'partial_checkpoint.pt')
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=records,wall_s=time.perf_counter()-start));raise

if __name__=='__main__':main()
