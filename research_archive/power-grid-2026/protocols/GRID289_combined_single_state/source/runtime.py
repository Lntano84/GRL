"""Frozen208+261 composite; explicit owner identity; no teacher or environment."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
class Frozen208261:
 def __init__(self,root,reference,parent_api,residual_api):
  self.owner=reference["pool"];self.parent_api=parent_api;self.residual_api=residual_api;root=Path(root)
  for rel,h in reference["hashes"].items():
   if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError("model/reference hash")
  self.event,self.amount=[pickle.loads((root/p).read_bytes()) for p in reference["parent_heads"]]
  with np.load(root/reference["parent_scaler"]) as z:self.parent_scale={k:z[k].copy() for k in ["mean","scale","raw_mean","raw_scale"]}
  with np.load(root/reference["residual_public41_scaler"]) as z:self.mean41=z["mean41"].copy();self.scale41=z["scale41"].copy()
  with np.load(root/reference["residual_model"]) as z:self.theta=z["theta"].copy();self.alpha=float(z["alpha"][0]);self.KEEP=bool(z["fallback_KEEP"][0])
  if self.theta.shape!=(41,) or self.event.n_features_in_!=124 or self.amount.n_features_in_!=124:raise ValueError("model width")
  if self.KEEP and (self.alpha!=0 or np.any(self.theta!=0)):raise ValueError("KEEP")
  for z in [*self.parent_scale.values(),self.mean41,self.scale41,self.theta]:z.flags.writeable=False
 def decide(self,raw83,raw41,*,owner_pool):
  if owner_pool!=self.owner:raise ValueError("owner pool differs; no cross-owner hidden substitution")
  raw83=np.asarray(raw83);raw41=np.asarray(raw41)
  if raw83.shape!=(1,83) or raw41.shape!=(1,41):raise ValueError("single state only")
  if not (raw83[:,81]<=raw83[:,80]).all():raise ValueError("must enter at original native NETSTOP")
  t=time.perf_counter();x=self.parent_api.pack_parent(raw83,raw41,self.parent_scale);pack_s=time.perf_counter()-t
  t=time.perf_counter();parent=self.parent_api.predict_parent(x,raw83[:,82],self.event,self.amount);head_s=time.perf_counter()-t
  t=time.perf_counter();X=self.residual_api.pack_public(raw41,self.mean41,self.scale41,parent["b"]);res_pack_s=time.perf_counter()-t
  t=time.perf_counter();logit,cont=self.residual_api.score_public40(X,parent["b"],parent["base_CONT"],self.theta,self.alpha);linear_s=time.perf_counter()-t
  return dict(**parent,logit=logit,continue_=cont,timing=dict(parent_pack_s=pack_s,parent_heads_s=head_s,residual_pack_s=res_pack_s,residual_linear_s=linear_s))
