"""Experimental staged search with semantic completion; old guard retained separately."""
import importlib.util
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("grid26_engine",ROOT / "work/grid26/engine.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location("grid28_semantic_policy",ROOT / "work/grid28/policy.py")
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
from common import np, flags


class Search(old.Search):
    policy="PROTECTED"
    def choose(self,obs,base,reward,done=False,**kwargs):
        if self.policy=="PROTECTED":return super().choose(obs,base,reward,done,**kwargs)
        started=time.perf_counter()
        eligible,reps,aliases,actions,changed=self.prepare(obs,base)
        prepared=time.perf_counter()
        order,prior,features,times=self.rank(obs,base,eligible,reps,aliases)
        limit=min(float(obs.rho.max()),float(kwargs.get("rho_threshold",obs.rho.max())))
        cache={};forecast_s=0.
        def dispatch(i,phase):
            nonlocal forecast_s
            assert i not in cache
            t=time.perf_counter();future,rr,ended,info=obs.simulate(actions[i]+base)
            elapsed=time.perf_counter()-t;forecast_s+=elapsed
            f=flags(info);rho=float(future.rho.max())
            valid=bool(0<rho<limit and not ended and not f["exceptions"] and not f["illegal"] and not f["ambiguous"])
            r={"pool_id":i,"rho":rho,"reward":float(rr),"done":bool(ended),**f,
               "strict_admissible":valid,"forecast_s":elapsed,"phase":phase}
            cache[i]=r;return r
        safe=float(self.ctl.base.rho_safe);assert safe==.9
        result=P.local_complete(order,dispatch,self.policy,self.subids,aliases,safe)
        elapsed=time.perf_counter()-started
        prefix=result["shortlist_choice"]
        reason="no_admissible_shortlist" if prefix is None else ("shortlist_above_author_safe" if cache[prefix]["rho"]>=safe else "shortlist_below_author_safe")
        log={"rule":self.rule,"policy":self.policy,"budget":128,"eligible":eligible,"representatives":reps,
             "aliases":aliases,"order":order,"shortlist":order[:128],"masked_ids":changed,
             "rho_limit":limit,"guard_rho":safe,"guard_reason":reason,**result,
             "source_s":elapsed,"prepare_s":prepared-started,"forecast_s":forecast_s,**times}
        chosen=result["chosen_pool_id"]
        return None if chosen is None else actions[chosen],log,features,prior
