"""Narrow repair v2: guard only initial-safe returns; preserve old fallback."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid70_reference',ROOT/'work/grid67/policies.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
key=old.key
POLICIES=old.POLICIES+('FRAME_GUARD',)
def run(order,observe,policy,subids,aliases,safe=.9,limit=None):
    if policy!='FRAME_GUARD':return old.run(order,observe,policy,subids,aliases,safe,limit)
    assert len(order)==len(set(order))
    rows=[];events=[];best=None
    def purchase(ids,phase):
        nonlocal best
        before=None if best is None else key(best)
        for i in ids:
            r=observe(i,phase);assert r['pool_id']==i;rows.append(r)
            if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
        after=None if best is None else key(best)
        improved=after is not None and (before is None or after>before)
        events.append(dict(queries=len(rows),phase=phase,chosen=None if best is None else best['pool_id'],improved=improved))
        return improved
    n=min(128,len(order));purchase(order[:n],'shortlist')
    prefix=None if best is None else best['pool_id']
    prefix_safe=best is not None and best['rho']<safe
    if not prefix_safe:
        # Feed cached prefix back into the unchanged reference policy without
        # dispatching or paying for any prefix observation twice.
        cached={r['pool_id']:r for r in rows}
        def reuse(i,phase):
            if i in cached:
                assert phase=='shortlist';return cached.pop(i)
            return observe(i,phase)
        result=old.run(order,reuse,'FRAME',subids,aliases,safe,limit)
        assert not cached
        result.update(verification_guard=False,prefix_safe_at_entry=False)
        return result
    reason='full_exhaustion';stale=0
    for offset in range(n,len(order),64):
        improved=purchase(order[offset:offset+64],'verification')
        stale=0 if improved else stale+1
        if len(rows)>=256 and stale>=2 and len(rows)<len(order):
            reason='guard_verified_plateau';break
    chosen=best['pool_id'];local=[];sub=None
    # An initially safe incumbent unchanged by verification retains the
    # original action.  Newly discovered incumbents receive usual completion.
    # This isolates the March regression: v1 activated completion even when
    # the additional verification had not changed the old safe incumbent.
    if chosen!=prefix and len(rows)<len(order):
        sub=int(subids[chosen]);targets={int(rep) for alias,rep in aliases.items() if int(subids[int(alias)])==sub}
        seen={r['pool_id'] for r in rows};local=[i for i in order if i not in seen and i in targets]
        purchase(local,'local_completion');reason+='_local_complete'
    elif len(rows)<len(order):reason+='_unchanged_incumbent'
    return dict(outcomes=rows,chosen_pool_id=best['pool_id'],shortlist_choice=prefix,
        fallback=n<len(order),stop_reason=reason,events=events,public_queries=len(rows),
        local_substation=sub,local_queries=len(local),completion_budget=len(local),global_extra_queries=0,
        pre_extra_choice=chosen,completion_candidate_ids=local,extra_selected_ids=local,
        initial_prefix=128,minimum_verification=256,stale_batches=2,
        verification_guard=True,prefix_safe_at_entry=True)
