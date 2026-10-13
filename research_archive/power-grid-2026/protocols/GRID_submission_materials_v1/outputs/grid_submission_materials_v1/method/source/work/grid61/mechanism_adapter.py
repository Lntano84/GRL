"""One search order, four stopping/completion policies; unchanged control."""
import importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid61_mechanism_policies',ROOT/'work/grid60/policies.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
spec=importlib.util.spec_from_file_location('grid61_frequency_rank',ROOT/'work/grid36/cheap_rank.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
from common import flags

def install_mechanism_adapter(Search):
    old_choose,old_rank=Search.choose,Search.rank
    def rank(self,obs,base,eligible,reps,aliases):
        if self.rule!='STATIC_ID':return old_rank(self,obs,base,eligible,reps,aliases)
        return R.rank(self,obs,base,eligible,reps,aliases)
    def choose(self,obs,base,reward,done=False,**kwargs):
        if self.policy not in P.POLICIES:return old_choose(self,obs,base,reward,done,**kwargs)
        assert self.rule=='STATIC_ID'
        started=time.perf_counter();eligible,reps,aliases,actions,changed=self.prepare(obs,base);prepared=time.perf_counter()
        order,prior,features,times=self.rank(obs,base,eligible,reps,aliases)
        assert order==sorted(reps) and prior is features is None
        limit=min(float(obs.rho.max()),float(kwargs.get('rho_threshold',obs.rho.max())))
        cache={};forecast_s=0.
        def dispatch(i,phase):
            nonlocal forecast_s
            assert i not in cache
            stamp=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base)
            elapsed=time.perf_counter()-stamp;forecast_s+=elapsed
            f=flags(info);rho=float(future.rho.max())
            valid=bool(0<rho<limit and not ended and not f['exceptions'] and not f['illegal'] and not f['ambiguous'])
            row={'pool_id':i,'rho':rho,'reward':float(rr),'done':bool(ended),**f,
                'strict_admissible':valid,'forecast_s':elapsed,'phase':phase}
            cache[i]=row;return row
        safe=float(self.ctl.base.rho_safe);assert safe==.9
        result=P.run(order,dispatch,self.policy,self.subids,aliases,safe)
        # Metadata normalization only: exhaustive FULL has no fallback; the
        # sequential reference falls back to best admissible after exhaustion.
        if self.policy=='FULL':result['fallback']=False
        if self.policy=='FIRST_SAFE_BEST':result['fallback']=result['stop_reason']=='full_exhaustion'
        prefix=result['shortlist_choice']
        reason='no_admissible_shortlist' if prefix is None else 'shortlist_above_author_safe' if cache[prefix]['rho']>=safe else 'shortlist_below_author_safe'
        log={'rule':self.rule,'policy':self.policy,'budget':128,'eligible':eligible,'representatives':reps,'aliases':aliases,
            'order':order,'shortlist':order[:128],'masked_ids':changed,'rho_limit':limit,'guard_rho':safe,'guard_reason':reason,
            **result,'source_s':time.perf_counter()-started,'prepare_s':prepared-started,'forecast_s':forecast_s,**times}
        chosen=result['chosen_pool_id']
        return None if chosen is None else actions[chosen],log,features,prior
    Search.rank,Search.choose=rank,choose
