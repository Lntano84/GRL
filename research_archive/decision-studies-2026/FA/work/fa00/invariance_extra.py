"""Supplemental full-state two-world check, added without changing the policy."""
import copy
import json
from pathlib import Path
import numpy as np
from core import ObservationCache, Ledger, ReplayEnvironment, Selector

def main():
    log1=[];log2=[]
    x=np.array([[0.,1.],[1.,0.],[2.,1.],[3.,0.]])
    first=np.array([[5.,30.],[40.,7.],[200.,250.],[210.,260.]])
    second=first.copy();second[2:]=[[500.,550.],[510.,560.]]
    l1,l2=Ledger(10000,log1.append),Ledger(10000,log2.append)
    e1,e2=ReplayEnvironment(first,np.ones((4,2),bool),l1),ReplayEnvironment(second,np.ones((4,2),bool),l2)
    c1,c2=ObservationCache((4,2)),ObservationCache((4,2))
    for i in [0,1]:
        for a in [0,1]:e1.observe(c1,i,a,100,"train");e2.observe(c2,i,a,100,"train")
    assert log1==log2 and c1.snapshot()==c2.snapshot() and l1.remaining==l2.remaining
    s1=Selector(x,["a","b"],[0,1,2,3],c1,7)
    s2=Selector(x,["a","b"],[0,1,2,3],c2,7)
    s1.synchronize_labels();s2.synchronize_labels();s1.fit(0);s2.fit(0)
    for step in range(2):
        z1=s1.select_batch(s1.candidates(100),1,step+1)
        z2=s2.select_batch(s2.candidates(100),1,step+1)
        assert np.array_equal(z1,z2) and l1.remaining==l2.remaining
        for z,env,c in [(z1,e1,c1),(z2,e2,c2)]:
            i,pi=int(z[0,0]),int(z[0,1]);a,b=s1.pairs[pi]
            env.observe(c,i,a,100,"train");env.observe(c,i,b,100,"train")
        assert c1.snapshot()==c2.snapshot() and log1==log2
        assert l1.train==l2.train and l1.validation==l2.validation and l1.remaining==l2.remaining
        assert (l1.remaining<=0)==(l2.remaining<=0)
    result=dict(passed=True,steps=2,events_compared=len(log1),cache_equal=True,
                paid_cost_equal=True,remaining_budget_equal=True,stop_rule_equal=True,
                phase="supplemental post-freeze verification; policy unchanged; no test outcomes read")
    out=Path(__file__).resolve().parent.parents[1]/"outputs"/"fa00"/"FA00_invariance_extra.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
