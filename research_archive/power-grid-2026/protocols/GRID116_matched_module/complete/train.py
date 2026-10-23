"""Exactly six fresh fits; first-step gradient qualification counted inside7250."""
import time,traceback
import torch,numpy as np
from common116 import *
def gradient_flow(net):
    ans={}
    for name,p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        key='base' if name.startswith('base.') else 'delta.'+name.split('.')[1]
        ans[key]=ans.get(key,0.)+float(p.grad.detach().double().square().sum())
    return {k:v**.5 for k,v in ans.items()}
def main():
    assert not (O/'training_started.json').exists(),'No silent repeated fits';verify();configure();assert read(O/'frozen_push.json')['passed']
    D=read(O/'design.json');P=read(O/'preflight.json');rows=pickle.loads((S/'train_public.pkl').read_bytes());meta=read(S/'train_metadata.json');assert all(r['partition']=='train' for r in meta)
    z=np.load(S/'train_targets.npz');y=z['target'];prob=z['sampling_probability'];yt=torch.from_numpy(np.column_stack([np.zeros_like(y),y]));take=tensor_cache(rows)
    (O/'models').mkdir(exist_ok=False);started=time.perf_counter();save(O/'training_started.json',dict(pid=os.getpid(),unix=time.time(),wall_cap_s=1800));records=[];net=None;optimizer=None
    try:
        for arm,seed in D['fit_order']:
            if time.perf_counter()-started>=1800:raise TimeoutError('Original total1800s cap reached')
            name=f'{arm}_seed{seed}';net=build(seed,arm);assert digest(net)==P['initial_hashes'][name]
            optimizer=torch.optim.Adam(net.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);losses=[];h=hashlib.sha256();first_steps=[];fit_start=time.perf_counter()
            for update in range(1,7251):
                if time.perf_counter()-started>=1800:raise TimeoutError('Original total1800s cap reached; partial preserved')
                ids=rng.choice(3582,32,replace=True,p=prob);np.add.at(counts,ids,1);h.update(ids.astype('<i8').tobytes())
                net.train();q=net(**take(ids));loss=torch.nn.functional.smooth_l1_loss(q,yt[ids],beta=1.);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if update<=2:
                    flow=gradient_flow(net);upstream={k:v for k,v in flow.items() if k not in ['base','delta.head']}
                    if update==1:assert all(v==0 for v in upstream.values())
                    else:assert all(v>0 for v in upstream.values()),('Upstream residual gradient absent after first scheduled update',flow)
                    first_steps.append(dict(update=update,gradient_norms_before_clip=flow));save(O/f'first_steps_{name}.json',first_steps)
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(norm);optimizer.step();losses.append(float(loss.detach()))
                if update%500==0 or update==7250:
                    progress=dict(arm=arm,seed=seed,updates=update,mean_last_loss=float(np.mean(losses[-500:])),fit_wall_s=time.perf_counter()-fit_start,total_wall_s=time.perf_counter()-started,completed_fits=len(records))
                    save(O/'state.json',dict(stage='training',**progress));print(json.dumps(progress),flush=True)
            assert h.hexdigest()==P['sampling_streams'][str(seed)]['index_sha256'];assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
            path=O/'models'/f'{name}.pt';torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=optimizer.state_dict(),updates=7250,
                initial_hash=P['initial_hashes'][name],final_hash=digest(net),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),path)
            np.savez_compressed(O/f'exposure_{name}.npz',state_draws=counts,minibatch_losses=np.asarray(losses))
            records.append(dict(arm=arm,seed=seed,updates=7250,parameters=12916,initial_hash=P['initial_hashes'][name],final_hash=digest(net),checkpoint_sha256=sha(path),
                draws=int(counts.sum()),scalar_targets=int(2*counts.sum()),index_sha256=h.hexdigest(),fit_wall_s=time.perf_counter()-fit_start,
                first_two_steps_in_budget=True,upstream_gradient_qualified_after_first_step=True,validation_used_for_fit=False,new_physical_steps=0))
            save(O/'fit_progress.json',dict(completed=records,total_wall_s=time.perf_counter()-started));print('FIT_DONE '+name,flush=True)
        save(O/'fits.json',dict(passed=True,new_fits=6,records=records,wall_s=time.perf_counter()-started,validation_used_for_fit=False,new_physical_steps=0));verify()
        save(O/'state.json',dict(stage='six_fits_complete_awaiting_evaluation',fits=6))
    except BaseException as e:
        if net is not None:
            torch.save(dict(arm=arm,seed=seed,state_dict=net.state_dict(),optimizer=optimizer.state_dict(),updates_completed=len(losses),rng=rng.bit_generator.state),O/'partial_checkpoint.pt')
            np.savez_compressed(O/'partial_exposure.npz',state_draws=counts,minibatch_losses=np.asarray(losses))
        save(O/'training_error.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=records,total_wall_s=time.perf_counter()-started,partial_preserved=True));raise
if __name__=='__main__':main()
