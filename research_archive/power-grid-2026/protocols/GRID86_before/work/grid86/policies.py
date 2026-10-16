"""Frozen guard versus literal fixed-prefix query budgets, common selection key."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid86_guard',ROOT/'work/grid70/policies.py')
G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
POLICIES=('FRAME_GUARD','FIXED_256','FIXED_512')
key=G.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES and safe==.9
    if policy=='FRAME_GUARD':return G.run(order,observe,policy,subids,aliases,safe,limit)
    cap=int(policy.split('_')[1]);assert cap in (256,512) and len(order)==len(set(order))
    rows=[];best=None;prefix=None
    for j,i in enumerate(order[:cap]):
        r=observe(i,'fixed_prefix');assert r['pool_id']==i;rows.append(r)
        if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
        if j<128:prefix=None if best is None else best['pool_id']
    chosen=None if best is None else best['pool_id']
    prefix_row=next((r for r in rows if r['pool_id']==prefix),None)
    return dict(outcomes=rows,chosen_pool_id=chosen,shortlist_choice=prefix,
                fallback=len(rows)>min(128,len(order)),stop_reason='fixed_prefix_exhausted' if len(rows)==len(order) else 'fixed_budget_reached',
                events=[dict(queries=len(rows),phase='fixed_prefix',chosen=chosen,improved=chosen is not None)],
                public_queries=len(rows),local_substation=None,local_queries=0,completion_budget=0,global_extra_queries=0,
                pre_extra_choice=chosen,completion_candidate_ids=[],extra_selected_ids=[],verification_guard=False,
                prefix_safe_at_entry=prefix_row is not None and prefix_row['rho']<safe,fixed_budget=cap)
