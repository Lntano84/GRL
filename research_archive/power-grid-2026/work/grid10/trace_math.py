"""Physical-step versus decision-linked GAE, with unchanged physical discounting."""
import numpy as np

def advantages_trace(rewards,values,terminals,offered,gamma,lam,mode):
    n=len(rewards)
    assert len(values)==n+1 and len(terminals)==n==len(offered)
    assert mode in ['physical','decision']
    result=np.zeros(n,dtype=np.float32);carry=0.
    for i in range(n-1,-1,-1):
        mask=0. if terminals[i] else 1.
        delta=rewards[i]+gamma*values[i+1]*mask-values[i]
        link_lambda=lam
        if mode=='decision' and i+1<n and not offered[i+1]:link_lambda=1.
        carry=delta+gamma*link_lambda*mask*carry;result[i]=carry
    assert np.isfinite(result).all()
    return result

def preflight():
    rng=np.random.default_rng(20261009);maximum=0.;cases=0
    for n in [1,2,6,32,256]:
        for _ in range(40):
            rewards=rng.normal(size=n);values=rng.normal(size=n+1)
            terminals=rng.random(n)<.05;offered=rng.random(n)<.2
            reference=np.zeros(n,dtype=np.float32);carry=0.
            for i in reversed(range(n)):
                mask=1.-float(terminals[i])
                delta=rewards[i]+.995*values[i+1]*mask-values[i]
                carry=delta+.995*.95*mask*carry;reference[i]=carry
            actual=advantages_trace(rewards,values,terminals,offered,.995,.95,'physical')
            maximum=max(maximum,float(np.max(np.abs(actual-reference))))
            assert np.array_equal(actual,reference);cases+=1
    # Independent interval expansion: exact discounted rewards within each interval,
    # and lambda applied only at the next actual offered state.
    rewards=np.array([1.,2.,3.,4.,5.,6.]);values=np.array([.2,-.8,1.1,3.,-.5,.7,2.])
    offered=[True,False,True,False,True,False];gamma=.9;lam=.7
    result=advantages_trace(rewards,values,[False]*6,offered,gamma,lam,'decision')
    events=[0,2,4,6];a=0.
    for j in reversed(range(3)):
        i,k=events[j:j+2]
        total=sum(gamma**t*rewards[i+t] for t in range(k-i))
        delta=total+gamma**(k-i)*values[k]-values[i]
        a=delta+gamma**(k-i)*lam*a
        assert abs(float(result[i])-a)<1e-5
    # A terminal stops the trace; rewards on the next episode cannot flow backwards.
    before=advantages_trace([2.,3.],[0.,0.,0.],[True,False],[True,True],.9,.7,'decision')
    changed=advantages_trace([2.,3000.],[0.,0.,0.],[True,False],[True,True],.9,.7,'decision')
    assert before[0]==changed[0]==2.
    # If every step is a choice, both definitions reduce to physical-step GAE.
    for _ in range(40):
        r=rng.normal(size=32);v=rng.normal(size=33)
        assert np.array_equal(advantages_trace(r,v,[False]*32,[True]*32,.995,.95,'physical'),
                              advantages_trace(r,v,[False]*32,[True]*32,.995,.95,'decision'))
    return {'passed':True,'physical_equivalence_cases':cases,'max_abs_error':maximum,
            'interval_expansion_cases':3,'terminal_isolation':True,'all_steps_offered_cases':40,
            'note':'Mathematical/control-interface tests, not evidence of better learned performance.'}

if __name__=='__main__':
    import json
    print(json.dumps(preflight(),indent=2))
