"""Frozen equal-per-state extra-query comparison; public metadata only."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid63_original',ROOT/'work/grid60/policies.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
key=old.key
POLICIES=('FULL','FRAME','GLOBAL_MATCH')

def run(order,observe,policy,subids,aliases,safe=.9):
    assert policy in POLICIES and len(order)==len(set(order))
    if policy=='FULL':
        result=old.run(order,observe,'FULL',subids,aliases,safe)
        result.update(completion_budget=0,global_extra_queries=0,pre_extra_choice=result['chosen_pool_id'],completion_candidate_ids=[],extra_selected_ids=[])
        return result
    result=old.run(order,observe,'NO_LOCAL',subids,aliases,safe)
    chosen=result['chosen_pool_id'];seen={r['pool_id'] for r in result['outcomes']}
    result.update(completion_budget=0,global_extra_queries=0,pre_extra_choice=chosen,completion_candidate_ids=[],extra_selected_ids=[])
    if chosen is None or not result['fallback'] or len(seen)==len(order):return result
    sub=int(subids[chosen])
    targets={int(rep) for alias,rep in aliases.items() if int(subids[int(alias)])==sub}
    remaining=[i for i in order if i not in seen]
    local=[i for i in remaining if i in targets]
    q=len(local)
    selected=local if policy=='FRAME' else remaining[:q]
    assert len(selected)==q
    result.update(completion_budget=q,completion_candidate_ids=local,extra_selected_ids=selected,local_substation=sub)
    incumbent=next(r for r in result['outcomes'] if r['pool_id']==chosen)
    phase='local_completion' if policy=='FRAME' else 'global_budget_extension'
    for i in selected:
        assert i not in seen
        row=observe(i,phase);assert row['pool_id']==i
        result['outcomes'].append(row);seen.add(i)
        if row['strict_admissible'] and key(row)>key(incumbent):incumbent=row
    result['local_queries']=q if policy=='FRAME' else 0
    result['global_extra_queries']=q if policy=='GLOBAL_MATCH' else 0
    result.update(chosen_pool_id=incumbent['pool_id'],public_queries=len(seen),stop_reason=result['stop_reason']+('_local_complete' if policy=='FRAME' else '_global_budget_complete'))
    result['events'].append({'queries':len(seen),'phase':phase,'chosen':incumbent['pool_id'],'improved':incumbent['pool_id']!=chosen})
    return result
