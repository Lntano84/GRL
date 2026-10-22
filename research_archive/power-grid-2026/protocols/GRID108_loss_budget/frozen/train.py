"""Three exact Huber continuations and three MSE paths, fixed two endpoints only."""
import hashlib,json,time
import numpy as np,torch
from stable_baselines3 import DQN
from common108 import ROOT,OUT,read,save,verify,carrier,predict,weight_digest
def main():
    assert not (OUT/'models').exists();verify();D=read(OUT/'design.json')
    with np.load(ROOT/D['source_train']) as z:X=z['X'];target=z['target'];prob=z['sampling_probability']
    assert X.shape==(3582,41);Y=np.column_stack([np.zeros_like(target),target]);xt=torch.as_tensor(X);yt=torch.as_tensor(Y)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (OUT/'models').mkdir();allstart=time.perf_counter();records=[];checkpoints=[];draw_hashes={}
    for loss_name in D['losses']:
        for seed in D['seeds']:
            start=time.perf_counter();rng=np.random.default_rng(D['training']['sampler_seed_offset']+seed)
            counts=np.zeros(len(X),dtype=np.int64);stream=hashlib.sha256();trace=[];miniloss=[]
            start_step=7250 if loss_name=='HUBER' else 0
            if start_step:
                model=DQN.load(ROOT/f'outputs/grid107_direct_value/models/direct_seed{seed}_final.zip',device='cpu')
                assert all(float(v['step'])==7250 for v in model.policy.optimizer.state.values())
                for _ in range(start_step):
                    ids=rng.choice(len(X),size=32,replace=True,p=prob);np.add.at(counts,ids,1);stream.update(ids.tobytes())
                draw_hashes[seed,7250]=stream.hexdigest()
            else:model=carrier(seed,D['dqn_initialization'])
            def snapshot(step):
                q=predict(model.q_net,X);e=q-Y;ae=np.abs(e)
                trace.append(dict(updates=step,train_weighted_mse=float(prob@np.mean(e**2,axis=1)),
                    train_weighted_huber=float(prob@np.mean(np.where(ae<1,.5*e**2,ae-.5),axis=1)),
                    positive_misses=int(np.sum((target>0)&(q[:,1]<=q[:,0]))),
                    negative_continues=int(np.sum((target<=0)&(q[:,1]>q[:,0]))),wall_s=time.perf_counter()-start))
            def checkpoint(step):
                model._n_updates=step;assert model.num_timesteps==0
                assert all(float(v['step'])==step for v in model.policy.optimizer.state.values())
                method=f'{loss_name}_{step}_{seed}';model.save(OUT/'models'/method)
                if loss_name=='HUBER':draw_hashes[seed,step]=stream.hexdigest()
                else:assert draw_hashes[seed,step]==stream.hexdigest(),'Minibatch sequences not matched'
                np.savez_compressed(OUT/f'draws_{method}.npz',state_draws=counts)
                checkpoints.append(dict(method=method,updates=step,loss=loss_name,seed=seed,q_hash=weight_digest(model.q_net),
                    stream_sha256=stream.hexdigest(),paired_state_draws=int(counts.sum()),scalar_targets=int(counts.sum()*2)))
            snapshot(start_step)
            for step in range(start_step+1,29001):
                assert time.perf_counter()-allstart<D['caps']['fitting_wall_s'] and time.perf_counter()-start<D['caps']['one_path_s']
                ids=rng.choice(len(X),size=32,replace=True,p=prob);np.add.at(counts,ids,1);stream.update(ids.tobytes())
                model.policy.set_training_mode(True);q=model.q_net(xt[ids])
                loss=torch.nn.functional.smooth_l1_loss(q,yt[ids]) if loss_name=='HUBER' else torch.nn.functional.mse_loss(q,yt[ids])
                model.policy.optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(),10.);model.policy.optimizer.step();miniloss.append(float(loss.detach()))
                if step%1000==0 or step in D['budgets']:
                    snapshot(step);save(OUT/f'trace_{loss_name}_{seed}.json',trace)
                if step in D['budgets']:checkpoint(step)
            np.savez_compressed(OUT/f'minibatch_losses_{loss_name}_{seed}.npz',losses=np.array(miniloss))
            records.append(dict(loss=loss_name,seed=seed,start_updates=start_step,end_updates=29000,gradient_updates_executed=29000-start_step,
                wall_s=time.perf_counter()-start,final_q_hash=weight_digest(model.q_net),environment_steps=0))
            save(OUT/'fit_progress.json',dict(completed_paths=records,new_checkpoints=len(checkpoints)))
            print(json.dumps(records[-1]),flush=True)
    save(OUT/'fits.json',dict(passed=True,new_training_paths=6,new_checkpoints=len(checkpoints),records=records,checkpoints=checkpoints,
        old_short_huber_reused=3,matched_draw_sequences=True,validation_used_for_training=False,new_physical_steps=0,final_test_materialized=False))
if __name__=='__main__':main()
