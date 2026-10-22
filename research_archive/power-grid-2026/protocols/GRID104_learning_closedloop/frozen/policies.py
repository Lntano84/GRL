"""Unchanged FRAME global search; learned control of completion batches only."""
from pathlib import Path
import importlib.util,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
G=load('grid104_global_core','work/grid91_mechanism/policies.py')
F=load('grid104_visible_projection','work/grid102_search_rl/observations.py')
key=G.key

class CompletionPolicy:
    def __init__(self,method,seed=None):
        import json,joblib
        self.method=method;self.model=None
        d=json.loads((ROOT/'outputs/grid102_search_rl/design.json').read_text(encoding='utf8'))
        self.cost=d['cost_per_query'];self.gain_scale=d['gain_scale']
        with np.load(ROOT/'outputs/grid102_search_rl/normalization.npz') as z:self.mean=z['mean'];self.scale=z['scale']
        if method=='LOCAL_RF':self.model=joblib.load(ROOT/'outputs/grid103_learning_mechanism/models/next_batch_rf.joblib')
        elif method.startswith('LOCAL_DQN0_'):
            from stable_baselines3 import DQN
            assert seed in [0,1,2] and method==f'LOCAL_DQN0_{seed}'
            self.model=DQN.load(ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{seed}_final.zip',device='cpu')
            assert self.model.gamma==0
        else:assert method in ['FRAME_GUARD','LOCAL_BATCH1']
        if self.model is not None:self.predict(np.zeros(41,dtype=np.float32),1) # model-only warm-up outside control
    def predict(self,obs,next_queries):
        if self.method=='LOCAL_RF':
            value=float(self.model.predict(obs[None,:])[0]);threshold=self.cost*next_queries/self.gain_scale
            return int(value>threshold),dict(predicted_normalized_gain=value,normalized_query_cost=threshold)
        action=int(self.model.predict(obs,deterministic=True)[0])
        return action,{}

    def run(self,order,reveal,subids,aliases,limit=None,context=None):
        if self.method=='FRAME_GUARD':
            result=G.run(order,reveal,'FRAME_GUARD',subids,aliases,limit=limit,context=context)
            result.update(learned_decisions=[],local_feature_s=0.,local_inference_s=0.,learning_method=self.method)
            return result
        result=G.run(order,reveal,'NO_LOCAL',subids,aliases,limit=limit,context=context)
        rows=result['outcomes'];best=F.best(rows);initial=None if best is None else F.reward(best)
        pending=result['completion_candidate_ids'].copy();budget=len(pending);purchased=[];decisions=[]
        feature_s=inference_s=0.;batch=0
        if budget:
            targets={int(rep) for alias,rep in aliases.items() if int(subids[int(alias)])==result['local_substation']}
        while pending:
            next_ids=pending[:64];details={};raw=None;obs=None;clipped=0
            if self.method=='LOCAL_BATCH1':action=int(batch==0)
            else:
                t=time.perf_counter()
                raw=F.observe(rows,order,aliases,subids,limit,targets,pending,budget,batch)
                standardized=(raw-self.mean)/self.scale;clipped=int(np.sum(np.abs(standardized)>20))
                obs=np.clip(standardized,-20,20).astype(np.float32);feature_s+=time.perf_counter()-t
                t=time.perf_counter();action,details=self.predict(obs,len(next_ids));inference_s+=time.perf_counter()-t
            decisions.append(dict(batch=batch,paid_queries=len(rows),next_ids=next_ids,remaining=len(pending),action=action,
                raw_features=None if raw is None else raw.tolist(),normalized_features=None if obs is None else obs.tolist(),
                clipped_features=clipped,**details))
            if action==0:break
            before=key(best)
            for i in next_ids:
                r=reveal(i,'local_completion');assert r['pool_id']==i
                rows.append(r);purchased.append(i)
                if r['strict_admissible'] and key(r)>key(best):best=r
            result['events'].append(dict(queries=len(rows),phase='local_completion',chosen=best['pool_id'],improved=key(best)>before))
            pending=pending[len(next_ids):];batch+=1
        result.update(chosen_pool_id=None if best is None else best['pool_id'],public_queries=len(rows),local_queries=len(purchased),
            extra_selected_ids=purchased,stop_reason=result['stop_reason']+('_learned_stop' if pending else '_local_exhausted') if budget else result['stop_reason'],
            learned_decisions=decisions,local_feature_s=feature_s,local_inference_s=inference_s,learning_method=self.method,
            local_reward_gain=0. if initial is None else F.reward(best)-initial)
        return result
