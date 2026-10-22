"""Common ordinary forecasts and controller timers, online feature/inference charged."""
from pathlib import Path
import importlib.util,time
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid104_policy',ROOT/'work/grid104_learning_closedloop/policies.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def install(Search,item):
    from common import flags,graph_features
    policy=P.CompletionPolicy(item['method'],item.get('model_seed'))
    def rank(self,obs,base,eligible,reps,aliases):
        started=time.perf_counter();return sorted(reps),None,None,dict(feature_s=0.,prior_s=0.,forward_rank_s=time.perf_counter()-started)
    def choose(self,obs,base,reward,done=False,**kwargs):
        start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter()
        order,prior,features,times=rank(self,obs,base,eligible,reps,aliases)
        limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())))
        cache={};forecast_s=0.
        def reveal(i,phase):
            nonlocal forecast_s
            assert i not in cache;t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base)
            elapsed=time.perf_counter()-t;forecast_s+=elapsed;f=flags(info);rho=float(future.rho.max())
            valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous'])
            row=dict(pool_id=i,rho=rho,reward=float(rr),done=bool(ended),**f,strict_admissible=valid,forecast_s=elapsed,phase=phase)
            cache[i]=row;return row
        t=time.perf_counter();xx,ee,gg=graph_features(obs,self.env,dict(offered=False,predictions=[None,None]));graph_s=time.perf_counter()-t
        context=dict(x=xx,edges=ee,glob=gg)
        result=policy.run(order,reveal,self.subids,aliases,limit,context)
        prefix=result['shortlist_choice'];safe=.9
        reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=safe else 'shortlist_below_author_safe'
        times['feature_s']=graph_s+result['local_feature_s']
        log=dict(rule=self.rule,policy=self.policy,budget=128,eligible=eligible,representatives=reps,aliases=aliases,order=order,
            order_mode='STATIC_ID',shortlist=order[:128],masked_ids=changed,rho_limit=limit,guard_rho=safe,guard_reason=reason,
            **result,source_s=time.perf_counter()-start,prepare_s=prepared-start,forecast_s=forecast_s,**times)
        selected=result['chosen_pool_id'];return None if selected is None else actions[selected],log,context,prior
    Search.rank=rank;Search.choose=choose
