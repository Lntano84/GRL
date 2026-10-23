"""Exactly six final-only fits; train-data reads only."""
import time,traceback
from common120 import *

def main():
    assert not (O/'training_started.json').exists(),'Never repeat complete/partial fits silently';verify();configure();assert read(O/'frozen_push.json')['passed']
    D=read(O/'design.json');P=read(O/'preflight.json');x,y,_=data('train');prob=np.load(S/'train_targets.npz')['sampling_probability'];X=torch.from_numpy(x);Y=torch.from_numpy(y);Z=D['objectives']['normalizer']
    (O/'models').mkdir(exist_ok=False);started=time.perf_counter();save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800));records=[];net=None;opt=None;losses=[]
    try:
        for arm,seed in D['train']['fit_order']:
            assert time.perf_counter()-started<1800,'Total fit cap';name=f'{arm}_seed{seed}';net=build(seed);assert digest(net)==P['original_initial_hashes'][str(seed)]
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            rng=np.random.default_rng(20261022+seed);counts=np.zeros(len(y),np.int64);h=hashlib.sha256();losses=[];norms=[];fs=time.perf_counter()
            for update in range(1,7251):
                if time.perf_counter()-started>=1800:raise TimeoutError('Original total1800s window reached')
                ids=rng.choice(len(y),32,replace=True,p=prob);np.add.at(counts,ids,1);h.update(ids.astype('<i8').tobytes())
                net.train();q=net(X[ids]);loss=loss_per_state(q,Y[ids],arm,Z).mean();assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                if update%1000==0 or update==7250:
                    p=dict(stage='training',arm=arm,seed=seed,updates=update,completed_fits=len(records),fit_wall_s=time.perf_counter()-fs,total_wall_s=time.perf_counter()-started,mean_last_loss=float(np.mean(losses[-500:])));save(O/'state.json',p);print(json.dumps(p),flush=True)
            assert h.hexdigest()==P['sampling_streams'][str(seed)]['index_sha256']
            assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
            path=O/'models'/f'{name}.pt';torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=7250,initial_hash=P['original_initial_hashes'][str(seed)],final_hash=digest(net),rng=rng.bit_generator.state),path)
            np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,minibatch_losses=np.asarray(losses),gradient_norm_before_clip=np.asarray(norms))
            records.append(dict(arm=arm,seed=seed,updates=7250,parameters=6978,initial_hash=P['original_initial_hashes'][str(seed)],final_hash=digest(net),checkpoint_sha256=sha(path),index_sha256=h.hexdigest(),state_draws=232000,classification_targets=232000,fit_wall_s=time.perf_counter()-fs,clipped_updates=int(np.sum(np.asarray(norms)>10)),validation_used_for_fit=False))
            save(O/'fit_progress.json',dict(completed=records,total_wall_s=time.perf_counter()-started))
        save(O/'fits.json',dict(passed=True,new_fits=6,records=records,wall_s=time.perf_counter()-started,validation_used_for_fit=False,new_physical_calls=0));verify();save(O/'state.json',dict(stage='six_fits_complete_awaiting_evaluation',fits=6))
    except BaseException as e:
        if net is not None:torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,arm=arm,seed=seed,updates_completed=len(losses),rng=rng.bit_generator.state),O/'partial_checkpoint.pt')
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=records,wall_s=time.perf_counter()-started));raise
if __name__=='__main__':main()
