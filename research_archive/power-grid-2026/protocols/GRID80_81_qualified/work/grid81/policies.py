"""Frozen gate vs unconditional verification vs calibrated-rate random skipping."""
from pathlib import Path
import hashlib,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid81_source',ROOT/'work/grid76/policies.py')
C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
POLICIES=('FULL','CONTEXT_GATE','FRAME_GUARD','RANDOM_GATE0','RANDOM_GATE1')
key=C.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    if policy in ('FULL','CONTEXT_GATE','FRAME_GUARD'):
        return C.run(order,observe,policy,subids,aliases,safe,limit,context=context)
    assert policy in ('RANDOM_GATE0','RANDOM_GATE1') and context is not None
    initial=[]
    for i in order[:128]:
        r=observe(i,'shortlist');assert r['pool_id']==i;initial.append(r)
    best=C.G.M.best_row(initial);x=None;decision=None
    if best is not None and best['rho']<safe:
        x=C.G.features(initial,order,subids,aliases,limit,context)
        ood=C.C.outside(C.G.load('plain'),x)
        # No model prediction, no date or teacher suffix: coin from visible float32 prefix/context only.
        token=hashlib.sha256(b'GRID81_RANDOM_GATE_V1|'+policy.encode()+np.asarray(x[:24],dtype='<f4').tobytes()).hexdigest()
        coin=int(token[:16],16)/2**64
        skip=not ood and coin<4/13
        decision=dict(outside_support=ood,coin=coin,skip_probability=4/13,random_seed=int(policy[-1]),defer=not skip)
        if skip:
            return dict(outcomes=initial,chosen_pool_id=best['pool_id'],shortlist_choice=best['pool_id'],
                fallback=False,stop_reason='calibrated_random_early_return',events=[dict(queries=len(initial),phase='shortlist',chosen=best['pool_id'],improved=True)],
                public_queries=len(initial),local_substation=None,local_queries=0,completion_budget=0,global_extra_queries=0,
                pre_extra_choice=best['pool_id'],completion_candidate_ids=[],extra_selected_ids=[],verification_guard=False,prefix_safe_at_entry=True,
                gate_input=x.tolist(),gate_decision=decision,gate_model='RANDOM_calibration_only_rate_4_of_13')
    cached={r['pool_id']:r for r in initial}
    def reuse(i,phase):
        if i in cached:
            assert phase=='shortlist';return cached.pop(i)
        return observe(i,phase)
    result=C.old.run(order,reuse,'FRAME_GUARD',subids,aliases,safe,limit);assert not cached
    result.update(gate_input=None if x is None else x.tolist(),gate_decision=decision,gate_model='RANDOM_calibration_only_rate_4_of_13')
    return result
