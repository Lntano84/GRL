"""Qualified cheap operating-context baseline; no structural novelty claim."""
from pathlib import Path
import importlib.util,sys
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
old=module('grid76_guard','work/grid70/policies.py')
G=module('grid76_context','work/grid75/feature_gate.py')
C=module('grid76_correct_precision','work/grid75_review/runtime_corrected.py')
POLICIES=old.POLICIES+('CONTEXT_GATE',)
key=old.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    if policy!='CONTEXT_GATE':return old.run(order,observe,policy,subids,aliases,safe,limit)
    assert context is not None and limit is not None
    initial=[]
    for i in order[:128]:
        r=observe(i,'shortlist');assert r['pool_id']==i;initial.append(r)
    best=G.M.best_row(initial);x=None;decision=None
    if best is not None and best['rho']<safe:
        x=G.features(initial,order,subids,aliases,limit,context)
        defer,decision=C.decide(G.load('plain'),x)
        if not defer:
            return dict(outcomes=initial,chosen_pool_id=best['pool_id'],shortlist_choice=best['pool_id'],
                fallback=False,stop_reason='operating_context_early_return',events=[dict(queries=len(initial),phase='shortlist',chosen=best['pool_id'],improved=True)],
                public_queries=len(initial),local_substation=None,local_queries=0,completion_budget=0,global_extra_queries=0,
                pre_extra_choice=best['pool_id'],completion_candidate_ids=[],extra_selected_ids=[],verification_guard=False,prefix_safe_at_entry=True,
                gate_input=x.tolist(),gate_decision=decision,gate_model='GRID75_plain_qualified_cheap_baseline')
    cached={r['pool_id']:r for r in initial}
    def reuse(i,phase):
        if i in cached:
            assert phase=='shortlist';return cached.pop(i)
        return observe(i,phase)
    result=old.run(order,reuse,'FRAME_GUARD',subids,aliases,safe,limit);assert not cached
    result.update(gate_input=None if x is None else x.tolist(),gate_decision=decision,gate_model='GRID75_plain_qualified_cheap_baseline')
    return result
