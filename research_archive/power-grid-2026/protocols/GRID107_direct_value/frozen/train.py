"""Three final-only direct full-information value regressions, no env stepping."""
import json,time
import numpy as np,torch
from common107 import ROOT,OUT,read,save,verify,carrier,weight_digest,predict
def main():
    assert not (OUT/'models').exists(),'Never silently repeat a fit'
    verify();D=read(OUT/'design.json');meta=read(OUT/'training_rows.json');assert all(r['partition']=='train' for r in meta)
    with np.load(OUT/'train.npz') as z:X=z['X'];target=z['target'];prob=z['sampling_probability']
    assert X.shape==(3582,41);targets=np.column_stack([np.zeros_like(target),target])
    xt=torch.as_tensor(X);yt=torch.as_tensor(targets)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (OUT/'models').mkdir();fits=[]
    for seed in D['seeds']:
        model=carrier(seed,D['dqn_initialization']);start=time.perf_counter()
        assert weight_digest(model.q_net)==read(OUT/'preflight.json')['initial_q_hashes'][str(seed)]
        rng=np.random.default_rng(D['training']['sampler_seed_offset']+seed);counts=np.zeros(len(X),dtype=np.int64);trace=[];losses=[]
        def training_snapshot(update):
            q=predict(model.q_net,X);regret=np.maximum(target,0)-(q[:,1]>q[:,0])*target
            trace.append(dict(update=update,train_huber=float(torch.nn.functional.smooth_l1_loss(torch.as_tensor(q),yt)),
                train_weighted_huber=float(np.sum(prob*np.mean(np.where(np.abs(q-targets)<1,.5*(q-targets)**2,np.abs(q-targets)-.5),axis=1))),
                positive_misses=int(np.sum((target>0)&(q[:,1]<=q[:,0]))),negative_continues=int(np.sum((target<=0)&(q[:,1]>q[:,0]))),
                mean_training_realized_sign_regret=float(np.mean(regret)),wall_s=time.perf_counter()-start))
        training_snapshot(0)
        for update in range(1,D['training']['updates']+1):
            ids=rng.choice(len(X),size=D['training']['states_per_batch'],replace=True,p=prob)
            np.add.at(counts,ids,1);model.policy.set_training_mode(True)
            q=model.q_net(xt[ids]);loss=torch.nn.functional.smooth_l1_loss(q,yt[ids])
            model.policy.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),D['dqn_initialization']['max_grad_norm'])
            model.policy.optimizer.step();losses.append(float(loss.detach()))
            if update%500==0 or update==7250:
                training_snapshot(update);save(OUT/f'trace_seed{seed}.json',trace)
        model._n_updates=D['training']['updates'];assert model.num_timesteps==0
        model.save(OUT/'models'/f'direct_seed{seed}_final')
        np.savez_compressed(OUT/f'training_exposure_seed{seed}.npz',state_draws=counts,minibatch_losses=np.asarray(losses))
        fits.append(dict(seed=seed,updates=7250,paired_state_draws=int(counts.sum()),scalar_action_targets=int(counts.sum()*2),
            states_never_drawn=int(np.sum(counts==0)),positive_states_never_drawn=int(np.sum((target>0)&(counts==0))),
            fit_s=time.perf_counter()-start,parameters=sum(p.numel() for p in model.q_net.parameters()),
            environment_steps=0,optimizer=type(model.policy.optimizer).__name__,final_q_hash=weight_digest(model.q_net)))
        save(OUT/'fit_progress.json',dict(completed_seeds=[r['seed'] for r in fits],new_fits=len(fits)))
        print(json.dumps(fits[-1]),flush=True);model.env.close()
    save(OUT/'fits.json',dict(passed=True,new_fits=3,records=fits,validation_used_for_fit=False,
        critical_probe_inputs_used_for_fit=False,new_physical_steps=0,final_test_materialized=False))
if __name__=='__main__':main()
