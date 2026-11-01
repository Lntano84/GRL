"""Frozen owner-f model payload. No warm-up, training or environment creation."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
R=Path('D:/Codex/GRL研究')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class FrozenModels:
 def __init__(self,pool):
  import torch
  from guard import O,read,sha
  self.pool=pool;ref=next(r for r in read(O/'model_references.json') if r['pool']==pool)
  for p,h in ref['hashes'].items():assert sha(R/p)==h,p
  self.defs=load('g291_checkpoint_definitions','work/grid184_search_adapter/models.py');self.S=self.defs.S
  with torch.random.fork_rng(devices=[]):
   self.net=self.defs.Factor();self.net.load_state_dict(torch.load(R/ref['checkpoint'],map_location='cpu',weights_only=False)['state_dict']);self.net.eval()
  state=hashlib.sha256()
  for k,v in self.net.state_dict().items():state.update(k.encode());state.update(v.detach().cpu().numpy().tobytes())
  assert state.hexdigest()==ref['state_hash']
  with np.load(R/ref['normalization']) as z:self.mean=z['mean'].copy();self.scale=z['scale'].copy()
  self.parent=load('g291_parent','work/grid288_parent208_interface/runtime.py');self.residual=load('g291_residual','work/grid287_frozen261_interface/runtime.py');combo=load('g291_combo','work/grid289_combined_single_state/runtime.py');self.gate=combo.Frozen208261(R,ref,self.parent,self.residual)
  task=read(R/'outputs/grid102_search_rl/design.json');self.fee=task['cost_per_query'];self.gain_scale=task['gain_scale']
 def native(self,raw,context):
  import torch
  g=np.clip((raw-self.mean)/self.scale,-20,20).astype('f');public=self.S.pack(context,g,self.fee,self.gain_scale);got={}
  hooks=[self.net.base.q_net[3].register_forward_hook(lambda m,i,o:got.__setitem__('main',o.detach().numpy().copy())),self.net.delta.head[1].register_forward_hook(lambda m,i,o:got.__setitem__('delta',o.detach().numpy().copy()))]
  try:
   with torch.inference_mode():q=self.net(**self.defs.pack_one(public))[0].numpy().copy()
  finally:
   for h in hooks:h.remove()
  h=np.concatenate([got['main'],got['delta'],q[None],public['known_query_cost'][None]],axis=1).astype(float)
  return dict(continue_=bool(q[1]>q[0]),raw83=h,log=dict(owner_pool=self.pool,native_q=q.tolist(),NN_input={k:v.tolist() for k,v in public.items()},raw83=h[0].tolist()))
 def extension(self,raw,raw83,arm):
  x=self.parent.pack_parent(raw83,raw[None],self.gate.parent_scale);p=self.parent.predict_parent(x,raw83[:,82],self.gate.event,self.gate.amount);cont=p['base_CONT'];log=dict(parent_input124=x[0].tolist(),parent_p=float(p['event_p'][0]),parent_mu=float(p['mu'][0]),known_charge=float(p['charge'][0]),parent_b=float(p['b'][0]),parent_CONT=bool(cont[0]),residual_KEEP=self.gate.KEEP)
  if arm=='BOUNDED261':
   X=self.residual.pack_public(raw[None],self.gate.mean41,self.gate.scale41,p['b']);score,cont=self.residual.score_public40(X,p['b'],cont,self.gate.theta,self.gate.alpha);log.update(residual_input40=X[0].tolist(),residual_logit=float(score[0]),alpha=self.gate.alpha)
  return dict(continue_=bool(cont[0]),log=log)
