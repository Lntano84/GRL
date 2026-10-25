"""Checkpoint-compatible deployment definitions without training-module imports."""
from pathlib import Path
import hashlib,importlib.util,json,types
import numpy as np
import torch
from torch import nn
R=Path('D:/Codex/GRL研究')
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
# Every dependency here uses a unique module name, never `common` or `model`.
V=load('grid184_relational_definition','work/grid113_relational_learning/model.py')
S=load('grid184_schema','work/grid112_relational_plan/schema.py')
class Q(nn.Module):
    def __init__(self,n):super().__init__();self.q_net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):return self.q_net(x)
class Factor(nn.Module):
    def __init__(self):super().__init__();self.base=Q(42);self.delta=V.RelationalValue('FACTOR')
    def forward(self,**public):return self.base(torch.cat([public['global41'],torch.zeros_like(public['known_query_cost'])],-1))+self.delta(**public)
def pack_one(row):
    out={k:torch.from_numpy(np.asarray(v)[None]) for k,v in row.items()}
    out['context_mask']=torch.ones((1,len(row['context_action'])),dtype=torch.bool);out['next_mask']=torch.ones((1,len(row['next_action'])),dtype=torch.bool)
    return out
class FrozenModels:
    def __init__(self,seed):
        assert seed in [0,1,2];self.seed=seed;self.nets={};folder=R/'outputs/grid182_deployment_bundle'
        manifest=json.loads((folder/'deployment_manifest.json').read_text(encoding='utf8'))
        def sha(p):
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
        # Preserve caller RNG state; all learned parameters are then overwritten.
        with torch.random.fork_rng(devices=[]):
            for arm in ['STATE4','POWER8','FACTOR']:
                p=folder/'models'/f'{arm}_seed{seed}.pt';assert sha(p)==manifest['models'][p.name]
                net=Factor() if arm=='FACTOR' else Q(50);net.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict']);net.eval();self.nets[arm]=net
        assert sha(folder/'normalization.npz')==manifest['normalization_sha256']
        with np.load(folder/'normalization.npz') as z:self.norm={k:z[k].copy() for k in z.files}
        task=json.loads((R/'outputs/grid102_search_rl/design.json').read_text(encoding='utf8'));self.fee=task['cost_per_query'];self.scale=task['gain_scale']
        # Pure model/tensor warm-up, outside the controller timer.
        synthetic=dict(context_action=np.zeros((1,68),'f'),context_feedback=np.zeros((1,8),'f'),next_action=np.zeros((1,68),'f'),anchor_action=np.zeros(68,'f'),global41=np.zeros(41,'f'),known_query_cost=np.ones(1,'f'))
        for _ in range(3):
            self.small('STATE4',np.zeros(50,'f'));self.small('POWER8',np.zeros(50,'f'));self.factor(synthetic)
    def inputs(self,raw,power,n):
        g=np.clip((raw-self.norm['mean41'])/self.norm['scale41'],-20,20).astype('f');p=np.clip((power-self.norm['mean8'])/self.norm['scale8'],-20,20).astype('f')
        x8=np.r_[g,np.float32(self.fee*n/self.scale),p].astype('f');x4=x8.copy();x4[46:]=0;return g,x4,x8
    def small(self,arm,x):
        with torch.inference_mode():return self.nets[arm](torch.from_numpy(x[None]))[0].numpy().copy()
    def factor(self,pub):
        with torch.inference_mode():return self.nets['FACTOR'](**pack_one(pub))[0].numpy().copy()
    def decide(self,mode,raw,c,power):
        assert mode in ['FACTOR','CONSENSUS'];scores={};delegated=mode=='FACTOR';g=np.clip((raw-self.norm['mean41'])/self.norm['scale41'],-20,20).astype('f')
        if mode=='CONSENSUS':
            g,x4,x8=self.inputs(raw,power,len(c['next_ids']));scores['STATE4']=self.small('STATE4',x4);scores['POWER8']=self.small('POWER8',x8)
            a,b=[bool(scores[k][1]>scores[k][0]) for k in ['STATE4','POWER8']];delegated=a!=b
            if not delegated:return int(a),dict(delegated=False,scores={k:v.tolist() for k,v in scores.items()})
        scores['FACTOR']=self.factor(S.pack(c,g,self.fee,self.scale));return int(scores['FACTOR'][1]>scores['FACTOR'][0]),dict(delegated=delegated,scores={k:v.tolist() for k,v in scores.items()})
