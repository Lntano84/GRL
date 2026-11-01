"""Original Search integration; install only after physical authorization guard."""
from pathlib import Path
import importlib.util,time
R=Path('D:/Codex/GRL研究')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def install(Search,item,models):
 N=load('g291_public_runtime','work/grid183_online_contract/runtime.py');P=load('g291_completion','work/grid291_control_preparation/policy.py');policy=P.CompletionPolicy(item['arm'],models,N)
 from common import flags,graph_features
 def choose(self,obs,base,reward,done=False,**kwargs):
  start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter();order=sorted(reps);limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())));cache={};forecast_s=0.
  def reveal(i,phase):
   nonlocal forecast_s
   assert i not in cache;t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base);elapsed=time.perf_counter()-t;forecast_s+=elapsed;f=flags(info);rho=float(future.rho.max());valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous']);row=dict(pool_id=i,rho=rho,reward=float(rr),done=bool(ended),**f,strict_admissible=valid,forecast_s=elapsed,phase=phase);cache[i]=row;return row
  t=time.perf_counter();xx,ee,gg=graph_features(obs,self.env,dict(offered=False,predictions=[None,None]));graph_s=time.perf_counter()-t;context=dict(x=xx,edges=ee,glob=gg)
  t=time.perf_counter();provider=N.PublicState(obs,base,actions,self.env);provider_s=time.perf_counter()-t
  result=policy.run(order,reveal,self.subids,aliases,limit=limit,provider=provider,context=context);result['local_feature_s']+=provider_s
  prefix=result['shortlist_choice'];reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=.9 else 'shortlist_below_author_safe'
  log=dict(rule=self.rule,policy=self.policy,budget=128,eligible=eligible,representatives=reps,aliases=aliases,order=order,order_mode='STATIC_ID',shortlist=order[:128],masked_ids=changed,rho_limit=limit,guard_rho=.9,guard_reason=reason,**result,source_s=time.perf_counter()-start,prepare_s=prepared-start,forecast_s=forecast_s,feature_s=graph_s+result['local_feature_s'],prior_s=0.,forward_rank_s=0.,common_graph_s=graph_s,completion_mode=item['arm'],model_seed=item['model_seed'],model_pool=item['pool'])
  selected=result['chosen_pool_id'];return None if selected is None else actions[selected],log,context,None
 Search.choose=choose
 return policy
