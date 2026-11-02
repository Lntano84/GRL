"""Reuse original261 policy/numerics; override one public nativeSTOP per episode."""
from pathlib import Path
import importlib.util,time,types
from intervention import EpisodeIntervention,EpisodeModels,digest
R=Path('D:/Codex/GRL研究')
def load(n,p):
 s=importlib.util.spec_from_file_location(n,R/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def install(Search,item,models,expected=None):
 N=load('g297_public_runtime','work/grid183_online_contract/runtime.py'); P=load('g297_original_completion','work/grid291_control_preparation/policy.py')
 state=EpisodeIntervention(item['arm'],expected);payload=EpisodeModels(models,state)
 def observe(*args):state.observe(*args);return N.observe(*args)
 api=types.SimpleNamespace(**{k:getattr(N,k) for k in ['G','best','reward','InputUnavailable']},observe=observe)
 policy=P.CompletionPolicy('BOUNDED261',payload,api)
 from common import flags,graph_features,action_vector,digest as vector_digest
 def choose(self,obs,base,reward,done=False,**kwargs):
  start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter();order=sorted(reps);limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())));cache={};forecast_s=0.
  state.begin(dict(step=int(obs.current_step)+1,state_hash=vector_digest(obs.to_vect()),base_action_hash=vector_digest(action_vector(base)),order_hash=digest(order),aliases_hash=digest(aliases),rho_limit=limit))
  def reveal(i,phase):
   nonlocal forecast_s
   assert i not in cache;t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base);elapsed=time.perf_counter()-t;forecast_s+=elapsed;f=flags(info);rho=float(future.rho.max());valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous']);row=dict(pool_id=i,rho=rho,reward=float(rr),done=bool(ended),**f,strict_admissible=valid,forecast_s=elapsed,phase=phase);cache[i]=row;return row
  t=time.perf_counter();xx,ee,gg=graph_features(obs,self.env,dict(offered=False,predictions=[None,None]));graph_s=time.perf_counter()-t;context=dict(x=xx,edges=ee,glob=gg)
  t=time.perf_counter();provider=N.PublicState(obs,base,actions,self.env);provider_s=time.perf_counter()-t
  result=policy.run(order,reveal,self.subids,aliases,limit=limit,provider=provider,context=context);result['local_feature_s']+=provider_s
  prefix=result['shortlist_choice'];reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=.9 else 'shortlist_below_author_safe'
  log=dict(rule=self.rule,policy=self.policy,budget=128,eligible=eligible,representatives=reps,aliases=aliases,order=order,order_mode='STATIC_ID',shortlist=order[:128],masked_ids=changed,rho_limit=limit,guard_rho=.9,guard_reason=reason,**result,source_s=time.perf_counter()-start,prepare_s=prepared-start,forecast_s=forecast_s,feature_s=graph_s+result['local_feature_s'],prior_s=0.,forward_rank_s=0.,common_graph_s=graph_s,completion_mode='BOUNDED261_SINGLE_INTERVENTION',model_seed=item['model_seed'],model_pool=item['pool'])
  selected=result['chosen_pool_id'];return None if selected is None else actions[selected],log,context,None
 Search.choose=choose;return state
