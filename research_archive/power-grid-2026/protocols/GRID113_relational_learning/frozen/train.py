"""Exactly six final-only serial fits, global30min wall cap; train-only IO."""
import time,traceback
import numpy as np,torch
from common import *
from model import RelationalValue
def main():
    assert not (O/'training_started.json').exists(),'Never silently repeat fits'
    verify();configure();D=read(O/'design.json');P=read(O/'preflight.json');assert P['passed']
    assert (O/'frozen_push.json').exists()
    rows=pickle.loads((O/'train_public.pkl').read_bytes());meta=read(O/'train_metadata.json');assert all(r['partition']=='train' for r in meta)
    z=np.load(O/'train_targets.npz');target=z['target'];prob=z['sampling_probability'];yt=torch.from_numpy(np.column_stack([np.zeros_like(target),target]));take=tensor_cache(rows)
    (O/'models').mkdir(exist_ok=False);start=time.perf_counter();fits=[];save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800))
    net=None;opt=None;update=0
    try:
        for arm,seed in D['fit_order']:
            if time.perf_counter()-start>=1800:raise TimeoutError('Total six-fit wall cap reached before next fit')
            torch.manual_seed(seed);net=RelationalValue(arm);assert digest(net)==P['initial_hashes'][str(seed)]
            opt=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);losses=[];h=hashlib.sha256();fs=time.perf_counter();update=0
            for update in range(1,7251):
                if time.perf_counter()-start>=1800:raise TimeoutError('Total six-fit wall cap1800s reached; partial retained')
                ids=rng.choice(3582,size=32,replace=True,p=prob);np.add.at(counts,ids,1);h.update(ids.astype('<i8').tobytes())
                net.train();q=net(**take(ids));loss=torch.nn.functional.smooth_l1_loss(q,yt[ids],beta=1.)
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
                assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
                if update%500==0 or update==7250:
                    progress=dict(arm=arm,seed=seed,updates=update,mean_last_loss=float(np.mean(losses[-500:])),fit_wall_s=time.perf_counter()-fs,total_wall_s=time.perf_counter()-start,completed_fits=len(fits))
                    save(O/'state.json',dict(stage='training',**progress));print(json.dumps(progress),flush=True)
            assert h.hexdigest()==P['sampling_streams'][str(seed)]['index_sha256']
            assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
            name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt'
            torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=opt.state_dict(),updates=7250,
                initial_hash=P['initial_hashes'][str(seed)],final_hash=digest(net),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),path)
            np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,minibatch_losses=np.asarray(losses))
            fits.append(dict(arm=arm,seed=seed,updates=7250,parameters=5938,draws=int(counts.sum()),scalar_targets=int(counts.sum()*2),
                index_sha256=h.hexdigest(),initial_hash=P['initial_hashes'][str(seed)],final_hash=digest(net),checkpoint_sha256=sha(path),
                fit_wall_s=time.perf_counter()-fs,validation_read_for_fit=False,new_physical_steps=0))
            save(O/'fit_progress.json',dict(completed=fits,total_wall_s=time.perf_counter()-start));print('FIT_DONE '+name,flush=True)
        save(O/'fits.json',dict(passed=True,records=fits,new_fits=6,wall_s=time.perf_counter()-start,validation_used_for_fit=False,new_physical_steps=0))
        save(O/'state.json',dict(stage='six_fits_complete_awaiting_evaluation',fits=6));verify()
    except BaseException as e:
        if net is not None:torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=opt.state_dict() if opt else None,updates_completed=len(losses),rng=rng.bit_generator.state),O/'partial_checkpoint.pt')
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=fits,total_wall_s=time.perf_counter()-start,partial_preserved=True));raise
if __name__=='__main__':main()
