"""Bind qualified completion to the unchanged Search/common controller interface."""
from pathlib import Path
import importlib.util,time
R=Path('D:/Codex/GRL研究')
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
N=load('grid184_qualified_completion','work/grid183_online_contract/runtime.py')
M=load('grid184_deployment_models','work/grid184_search_adapter/models.py')
def install(Search,item,control_api=None,models=None):
    if control_api is None:
        from common import flags,graph_features
    else:flags,graph_features=control_api
    mode=item.get('completion_mode',item['method']);assert mode in ['FRAME_GUARD','LOCAL_BATCH1','FACTOR','CONSENSUS']
    policy=N.CompletionPolicy(mode,models or (M.FrozenModels(item['model_seed']) if mode in ['FACTOR','CONSENSUS'] else None))
    def choose(self,obs,base,reward,done=False,**kwargs):
        start=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter();order=sorted(reps)
        limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())));cache={};forecast_s=0.
        def reveal(i,phase):
            nonlocal forecast_s
            assert i not in cache;t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base);elapsed=time.perf_counter()-t;forecast_s+=elapsed;f=flags(info);rho=float(future.rho.max())
            valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous'])
            row=dict(pool_id=i,rho=rho,reward=float(rr),done=bool(ended),**f,strict_admissible=valid,forecast_s=elapsed,phase=phase);cache[i]=row;return row
        t=time.perf_counter();xx,ee,gg=graph_features(obs,self.env,dict(offered=False,predictions=[None,None]));graph_s=time.perf_counter()-t;context=dict(x=xx,edges=ee,glob=gg)
        t=time.perf_counter();provider=N.PublicState(obs,base,actions,self.env) if mode in ['FACTOR','CONSENSUS'] else None;provider_s=time.perf_counter()-t
        result=policy.run(order,reveal,self.subids,aliases,limit=limit,provider=provider,context=context);result['local_feature_s']+=provider_s
        prefix=result['shortlist_choice'];reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=.9 else 'shortlist_below_author_safe'
        # Exclusive search components; model input packing/normalization is charged
        # to local_inference_s, public input building to local_feature_s.
        log=dict(rule=self.rule,policy=self.policy,budget=128,eligible=eligible,representatives=reps,aliases=aliases,order=order,order_mode='STATIC_ID',shortlist=order[:128],masked_ids=changed,
            rho_limit=limit,guard_rho=.9,guard_reason=reason,**result,source_s=time.perf_counter()-start,prepare_s=prepared-start,forecast_s=forecast_s,
            feature_s=graph_s+result['local_feature_s'],prior_s=0.,forward_rank_s=0.,common_graph_s=graph_s,completion_mode=mode,model_seed=item.get('model_seed'))
        selected=result['chosen_pool_id'];return None if selected is None else actions[selected],log,context,None
    Search.choose=choose
    return policy
