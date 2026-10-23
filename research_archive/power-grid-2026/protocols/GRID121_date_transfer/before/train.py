"""24 serial fits on12dates each, no held-date predictions in training."""
import time,traceback
from common121 import *

def main():
    assert not (O/'training_started.json').exists();verify();configure();assert read(O/'frozen_push.json')['passed'];D=read(O/'design.json');P=read(O/'preflight.json')
    (O/'models').mkdir(exist_ok=False);start=time.perf_counter();save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800));records=[];net=None;opt=None;losses=[]
    try:
        for fold,seed,arm in D['fit_order']:
            if time.perf_counter()-start>=1800:raise TimeoutError('Original24fit total1800s window reached')
            x,y,ids,held,prob,Z=fold_data(fold);X=torch.from_numpy(x);Y=torch.from_numpy(y);net=build(seed);assert digest(net)==P['original_initial_hashes'][str(seed)]
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.);rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(len(y),np.int64);losses=[];norms=[];name=f'fold{fold}__{arm}_seed{seed}';fs=time.perf_counter()
            for update in range(1,7251):
                if time.perf_counter()-start>=1800:raise TimeoutError('Original24fit total1800s window reached')
                ix=ids[rng.choice(len(ids),32,replace=True,p=prob)];np.add.at(counts,ix,1);h.update(ix.astype('<i8').tobytes());net.train();q=net(X[ix])
                loss=torch.nn.functional.smooth_l1_loss(q,torch.stack([torch.zeros_like(Y[ix]),Y[ix]],1),beta=1.) if arm=='HUBER' else loss_per_state(q,Y[ix],arm,Z).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
                if update%2000==0 or update==7250:
                    p=dict(stage='training',fold=fold,arm=arm,seed=seed,updates=update,completed_fits=len(records),wall_s=time.perf_counter()-start);save(O/'state.json',p);print(json.dumps(p),flush=True)
            assert counts[held].sum()==0 and h.hexdigest()==P['sampling_streams'][f'fold{fold}_seed{seed}']
            assert np.array_equal(counts,np.load(O/f'fold{fold}_seed{seed}_expected_exposure.npz')['state_draws'])
            path=O/'models'/f'{name}.pt';torch.save(dict(fold=fold,arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=7250,initial_hash=P['original_initial_hashes'][str(seed)],final_hash=digest(net),rng=rng.bit_generator.state),path)
            np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,minibatch_losses=np.asarray(losses),gradient_norm_before_clip=np.asarray(norms))
            records.append(dict(fold=fold,seed=seed,arm=arm,updates=7250,parameters=6978,initial_hash=P['original_initial_hashes'][str(seed)],final_hash=digest(net),checkpoint_sha256=sha(path),index_sha256=h.hexdigest(),fit_wall_s=time.perf_counter()-fs,held_states_drawn=0,clipped_updates=int(np.sum(np.array(norms)>10))))
            save(O/'fit_progress.json',dict(completed=records,wall_s=time.perf_counter()-start))
        save(O/'fits.json',dict(passed=True,new_fits=24,records=records,wall_s=time.perf_counter()-start,held_date_predictions_during_fit=0,new_physical_calls=0));verify();save(O/'state.json',dict(stage='all24_fits_complete_awaiting_evaluation',fits=24))
    except BaseException as e:
        if net is not None:torch.save(dict(state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,fold=fold,seed=seed,arm=arm,updates_completed=len(losses)),O/'partial_checkpoint.pt')
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=records,wall_s=time.perf_counter()-start));raise
if __name__=='__main__':main()
