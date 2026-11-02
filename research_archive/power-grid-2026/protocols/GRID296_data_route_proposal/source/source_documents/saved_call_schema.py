"""Same physical controller and feedback; explicit order/parameter interventions."""
from pathlib import Path
import hashlib,importlib.util,time
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('review_policy',ROOT/'work/grid89_review/policies.py');P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
def expected_order(item,reps,subids=None,obs=None,env=None):
 mode=item.get('order_mode','STATIC_ID')
 if mode=='STATIC_ID':return sorted(reps)
 if mode.startswith('RANDOM_'):
  # Fixed global priority, independent of call count, state outcomes and week.
  seed=int(mode.split('_')[1]);return sorted(reps,key=lambda i:(hashlib.sha256(f'{seed}:{i}'.encode()).digest(),i))
 if mode=='LOCAL_STRESS':
  import numpy as np
  pressure=np.zeros(env.n_sub);live=np.flatnonzero(obs.line_status)
  for ends in [env.line_or_to_subid,env.line_ex_to_subid]:np.maximum.at(pressure,ends[live],obs.rho[live])
  return sorted(reps,key=lambda i:(-float(pressure[subids[i]]),i))
 raise ValueError(mode)
def install(Search,item):
 from common import flags,graph_features
 def rank(self,obs,base,eligible,reps,aliases):
  t=time.perf_counter();order=expected_order(item,reps,self.subids,obs,self.env)
  return order,None,None,dict(feature_s=0.,prior_s=0.,forward_rank_s=time.perf_counter()-t)
 def choose(self,obs,base,reward,done=False,**kwargs):
  start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter()
  order,prior,features,times=rank(self,obs,base,eligible,reps,aliases);limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())))
  cache={};forecast_s=0.
  def reveal(i,phase):
   nonlocal forecast_s
   assert i not in cache;t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base);forecast_s+=time.perf_counter()-t
   f=flags(info);rho=float(future.rho.max());valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous'])
   r=dict(pool_id=i,rho=rho,reward=float(rr),done=bool(ended),**f,strict_admissible=valid,forecast_s=time.perf_counter()-t,phase=phase);cache[i]=r;return r
  context=None
  if item['policy']!='FULL':
   xx,ee,gg=graph_features(obs,self.env,dict(offered=False,predictions=[None,None]));context=dict(x=xx,edges=ee,glob=gg)
  result=P.run(order,reveal,item['policy'],self.subids,aliases,item.get('parameters'),context=context,limit=limit,model_path=item.get('meta_model'))
  if context is not None:features=context
  prefix=result['shortlist_choice'];safe=(item.get('parameters') or {}).get('safe',.9)
  reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=safe else 'shortlist_below_author_safe'
  log=dict(rule=self.rule,policy=self.policy,budget=(item.get('parameters') or {}).get('p',128),eligible=eligible,representatives=reps,aliases=aliases,order=order,order_mode=item.get('order_mode','STATIC_ID'),shortlist=order[:(item.get('parameters') or {}).get('p',128)],masked_ids=changed,rho_limit=limit,guard_rho=safe,guard_reason=reason,**result,source_s=time.perf_counter()-start,prepare_s=prepared-start,forecast_s=forecast_s,**times)
  chosen=result['chosen_pool_id'];return None if chosen is None else actions[chosen],log,features,prior
 Search.rank=rank;Search.choose=choose
