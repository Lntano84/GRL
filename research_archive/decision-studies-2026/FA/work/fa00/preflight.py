import copy
import json
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from core import (Ledger, ReplayEnvironment, ObservationCache, paid_difference,
                  Selector, PARAMS, RefitLearner)

OUT = Path(__file__).resolve().parent.parents[1]/"outputs"/"fa00"

def main():
    checks = {}
    # No solver query is needed for algebraically zero differences.
    log=[]; led=Ledger(10000,log.append)
    env=ReplayEnvironment([[150,500],[30,600]],[[True,True],[True,False]],led)
    cache=ObservationCache((2,2))
    delta,n=paid_difference(env,cache,np.array([0,1]),np.array([0,1]),[0,1])
    assert delta==0 and n==0 and not log
    checks["identical_choices_no_queries"]=True
    env.observe(cache,0,0,100,"train")
    assert cache.kind[0,0]==1 and cache.value[0,0]==100
    env.observe(cache,0,0,200,"train")
    assert led.train==250 and cache.kind[0,0]==2 and cache.value[0,0]==150
    checks["restart_100_plus_150_equals_250"]=True
    env.observe(cache,0,0,600,"train")
    assert led.train==250 and log[-1]["cached"]
    checks["successful_cache_not_recharged"]=True
    env.observe(cache,0,1,100,"train")
    env.observe(cache,0,1,50,"train")
    assert cache.value[0,1]==100 and log[-1]["charged_s"]==0
    checks["lower_bound_monotone_no_shorter_repeat"]=True
    env.observe(cache,1,1,600,"train")
    assert cache.score(1,1)==6000 and log[-1]["charged_s"]==600
    checks["native_timeout_par10_is_not_charge"]=True
    # Two worlds reveal the same history; unobserved outcomes cannot enter the selector.
    X=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.]])
    outcomes=np.array([[5.,30.],[40.,7.],[9.,50.],[20.,15.]])
    k=ObservationCache((4,2)); l=Ledger(10000); e=ReplayEnvironment(outcomes,np.ones((4,2),bool),l)
    for i in [0,1]:
        for a in [0,1]: e.observe(k,i,a,100,"train")
    s=Selector(X,["a","b"],[0,1,2,3],k,7)
    s.synchronize_labels();s.fit(0)
    before=s.candidates(100)
    altered=outcomes.copy();altered[2:,0]=590;altered[2:,1]=599
    e2=ReplayEnvironment(altered,np.ones((4,2),bool),Ledger(10000))
    after=copy.deepcopy(s).candidates(100)
    assert np.array_equal(before,after)
    picked1=s.select_batch(before,1,1)
    picked2=copy.deepcopy(s).select_batch(after,1,1)
    assert np.array_equal(picked1,picked2)
    # Dispatch the shared first action through both actual environments; feedback matches.
    i,a=int(picked1[0,0]),0
    # Use cap=1, below both hidden runtimes in either world.
    c1,c2=copy.deepcopy(k),copy.deepcopy(k)
    e.observe(c1,i,a,1,"train");e2.observe(c2,i,a,1,"train")
    assert c1.snapshot()==c2.snapshot()
    checks["two_world_actual_feedback_action_invariance"]=True
    # Adapter parity with a direct sklearn fit, rather than a self-comparison.
    x=np.array([[0.],[1.],[2.],[3.]]);y=np.array([0,0,1,1]);w=np.array([1.,2.,4.,3.])
    direct=RandomForestClassifier(**PARAMS,random_state=5).fit(x,y,sample_weight=w)
    adapter=RefitLearner(RandomForestClassifier(**PARAMS,random_state=5))
    adapter.teach(x,y,sample_weight=w,only_new=True)
    assert np.array_equal(direct.predict_proba(x),adapter.predict_proba(x))
    checks["upstream_refit_adapter_sklearn_parity"]=True
    # A budget-limited execution reveals only the dispatched shorter lower bound.
    l=Ledger(3);e=ReplayEnvironment([[5.]],[[True]],l);c=ObservationCache((1,1))
    e.observe(c,0,0,10,"validation")
    assert l.validation==3 and c.kind[0,0]==1 and c.value[0,0]==3 and l.remaining==0
    checks["budget_clipped_dispatch_no_hidden_completion"]=True
    # Exact paid comparison on a hand example (30 replaces 5 => deterioration +25).
    l=Ledger(100);e=ReplayEnvironment([[5.,30.]],[[True,True]],l);c=ObservationCache((1,2))
    d,n=paid_difference(e,c,np.array([0]),np.array([1]),[0])
    assert d==25 and n==1 and l.validation==35
    checks["paid_difference_sign_and_charge_hand_example"]=True
    # Real-data initial labels: no private runtime enters label derivation.
    p=json.loads((OUT/"FA00_protocol.json").read_text());d=np.load(OUT/"FA00_data.npz")
    counts={}
    for seed in p["seeds"]:
        l=Ledger(p["budget_s"]);e=ReplayEnvironment(d["runtimes"],d["ok"],l)
        c=ObservationCache(d["runtimes"].shape)
        for i in p["initial_rows"][str(seed)]:
            for a in range(len(p["algorithms"])): e.observe(c,i,a,100,"train")
        s=Selector(d["X"],p["algorithms"],p["train"],c,seed);s.synchronize_labels()
        informative=[sum(z[3]>0 for z in pair.values()) for pair in s.label_rows]
        assert min(informative)>0 and l.train<p["budget_s"]
        counts[str(seed)]={"initial_cost_s":l.train,"min_pair_labels":min(informative),
                           "max_pair_labels":max(informative)}
    checks["all_real_initial_pair_models_have_information"]=True
    result={"checks":checks,"passed":len(checks),"hard_failures":0,"initials":counts}
    (OUT/"FA00_preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
