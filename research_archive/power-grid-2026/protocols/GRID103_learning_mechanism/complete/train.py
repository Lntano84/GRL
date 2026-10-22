"""One next-batch regressor and three matched gamma0 DQN ablations."""
import json,time
import joblib,numpy as np,torch
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from common import OUT,read,save,load,verify,LocalSearchEnv

class Trace(BaseCallback):
    def __init__(self,seed):super().__init__();self.seed=seed;self.rows=[];self.started=time.perf_counter()
    def _on_step(self):
        if self.num_timesteps%1000==0:
            eps=list(self.model.ep_info_buffer)
            self.rows.append(dict(timesteps=self.num_timesteps,wall_s=time.perf_counter()-self.started,
                recent_episode_reward=float(np.mean([r['r'] for r in eps])) if eps else None,
                recent_episode_length=float(np.mean([r['l'] for r in eps])) if eps else None,
                exploration_rate=float(self.model.exploration_rate)))
            save(OUT/f'gamma0_trace_seed{self.seed}.json',self.rows)
            save(OUT/'run_state.json',dict(stage='gamma0_training',seed=self.seed,steps=self.num_timesteps))
            if self.num_timesteps%10000==0:print(json.dumps(dict(seed=self.seed,steps=self.num_timesteps)),flush=True)
        return True

def main():
    assert not (OUT/'models').exists(),'Never silently repeat fits'
    verify();D=read(OUT/'design.json');cases,mean,scale,old=load()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (OUT/'models').mkdir();receipts=[]
    with np.load(OUT/'regression_train.npz') as z:X=z['X'];y=z['y'];weight=z['weight']
    params={k:D['regression'][k] for k in ['n_estimators','max_depth','min_samples_leaf','max_features','random_state','n_jobs']}
    start=time.perf_counter();model=RandomForestRegressor(**params).fit(X,y,sample_weight=weight)
    joblib.dump(model,OUT/'models/next_batch_rf.joblib')
    receipts.append(dict(model='MYOPIC_RF64',fit_s=time.perf_counter()-start,train_states=len(y)))
    for seed in D['temporal_ablation']['seeds']:
        env=LocalSearchEnv([c for c in cases if c['partition']=='train'],mean,scale,D['cost_per_query'],D['gain_scale'])
        model=DQN('MlpPolicy',env,seed=seed,**D['temporal_ablation']['dqn'])
        start=time.perf_counter();model.learn(total_timesteps=D['temporal_ablation']['timesteps'],callback=Trace(seed),progress_bar=False)
        model.save(OUT/'models'/f'dqn_gamma0_seed{seed}_final')
        assert model.gamma==0 and model.num_timesteps==30000
        receipts.append(dict(model=f'DQN_G0_{seed}',fit_s=time.perf_counter()-start,seed=seed,
            timesteps=int(model.num_timesteps),parameters=sum(p.numel() for p in model.q_net.parameters()),updates=int(model._n_updates)))
        env.close()
    save(OUT/'fits.json',dict(model_fits=4,records=receipts,validation_used_for_fit=False,new_physical_calls=0,final_test_materialized=False))
    save(OUT/'run_state.json',dict(stage='training_complete',model_fits=4))
if __name__=='__main__':main()
