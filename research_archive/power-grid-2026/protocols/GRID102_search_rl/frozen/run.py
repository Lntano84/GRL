"""Cost-aware supervised baseline and three fixed small DQN training runs."""
import collections,json,pickle,time
import joblib,numpy as np,torch
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from common import ROOT,OUT,read,save,verify,test_sealed
from environment import LocalSearchEnv


class Trace(BaseCallback):
    def __init__(self,seed):super().__init__();self.seed=seed;self.records=[];self.started=time.perf_counter()
    def _on_step(self):
        if self.num_timesteps%1000==0:
            episodes=list(self.model.ep_info_buffer)
            self.records.append(dict(timesteps=self.num_timesteps,wall_s=time.perf_counter()-self.started,
                recent_episode_reward=float(np.mean([r['r'] for r in episodes])) if episodes else None,
                recent_episode_length=float(np.mean([r['l'] for r in episodes])) if episodes else None,
                exploration_rate=float(self.model.exploration_rate)))
            save(OUT/f'train_trace_seed{self.seed}.json',self.records)
            save(OUT/'run_state.json',dict(stage='training_dqn',seed=self.seed,timesteps=self.num_timesteps,pid=__import__('os').getpid()))
            print(json.dumps(dict(seed=self.seed,timesteps=self.num_timesteps,wall_s=self.records[-1]['wall_s'])),flush=True)
        if self.num_timesteps%10000==0:self.model.save(OUT/'models'/f'dqn_seed{self.seed}_step{self.num_timesteps}')
        return True


def main():
    assert not (OUT/'fits.json').exists(),'Do not repeat fits'
    verify();test_sealed();assert read(OUT/'preflight.json')['passed']
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    D=read(OUT/'design.json')
    with (OUT/'cases.pkl').open('rb') as f:cases=pickle.load(f)
    train=[c for c in cases if c['partition']=='train']
    with np.load(OUT/'normalization.npz') as z:mean=z['mean'];scale=z['scale']
    (OUT/'models').mkdir(exist_ok=False);receipts=[]
    with np.load(OUT/'supervised_train.npz') as z:X=z['X'];y=z['y'];weights=z['weight']
    started=time.perf_counter();lr=LogisticRegression(**D['logistic']).fit(X,y,sample_weight=weights)
    assert lr.n_iter_[0]<D['logistic']['max_iter'],'Keep unconverged fit, do not silently refit'
    joblib.dump(lr,OUT/'models/cost_logistic.joblib')
    receipts.append(dict(model='COST_LOGISTIC',fit_s=time.perf_counter()-started,train_states=len(y),iterations=int(lr.n_iter_[0])))
    for seed in D['seeds']:
        env=LocalSearchEnv(train,mean,scale,D['cost_per_query'],D['gain_scale'])
        model=DQN('MlpPolicy',env,seed=seed,**D['dqn'])
        started=time.perf_counter();trace=Trace(seed)
        model.learn(total_timesteps=D['timesteps'],callback=trace,progress_bar=False)
        model.save(OUT/'models'/f'dqn_seed{seed}_final')
        assert model.num_timesteps==D['timesteps']
        receipts.append(dict(model=f'DQN{seed}',seed=seed,fit_s=time.perf_counter()-started,
            timesteps=int(model.num_timesteps),parameter_count=sum(p.numel() for p in model.q_net.parameters()),
            gradient_updates=int(model._n_updates)))
        env.close()
    save(OUT/'fits.json',dict(model_fits=4,records=receipts,train_dates=sorted({c['date'] for c in train}),
         validation_used_for_fit=False,checkpoint_selection='Final30000 only; no validation selection',final_test_materialized=False))
    save(OUT/'run_state.json',dict(stage='training_complete',model_fits=4,new_physical_calls=0))


if __name__=='__main__':main()
