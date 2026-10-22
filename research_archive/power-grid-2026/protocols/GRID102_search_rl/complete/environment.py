"""Finite saved-feedback local search. Actions0=stop,1=purchase next64."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LocalSearchEnv(gym.Env):
    metadata={}
    def __init__(self,cases,mean,scale,cost,gain_scale):
        super().__init__();self.cases=cases;self.mean=mean;self.scale=scale
        self.cost=cost;self.gain_scale=gain_scale
        self.action_space=spaces.Discrete(2)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(41,),dtype=np.float32)
        self.groups={}
        for i,c in enumerate(cases):self.groups.setdefault(c['date'],[]).append(i)
        self.dates=sorted(self.groups);self.done=True
    def observation(self):
        if self.done:return np.zeros(41,dtype=np.float32)
        return np.clip((self.case['states'][self.position]-self.mean)/self.scale,-20,20).astype(np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options is not None and 'index' in options:index=options['index']
        else:
            date=self.dates[int(self.np_random.integers(len(self.dates)))];group=self.groups[date]
            index=group[int(self.np_random.integers(len(group)))]
        self.case=self.cases[index];self.position=0;self.done=False
        return self.observation(),{}
    def step(self,action):
        assert not self.done and self.action_space.contains(action)
        value=0.
        if int(action)==0:self.done=True
        else:
            j=self.position
            value=(self.case['gains'][j]-self.cost*self.case['queries'][j])/self.gain_scale
            self.position+=1;self.done=self.position==len(self.case['queries'])
        info=dict(batches=self.position,paid_queries=int(self.case['queries'][:self.position].sum()),
                  observed_gain=float(self.case['gains'][:self.position].sum()))
        return self.observation(),float(value),bool(self.done),False,info
