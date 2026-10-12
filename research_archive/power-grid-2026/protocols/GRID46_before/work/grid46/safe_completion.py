"""One common restoration of the author's existing rho-safe fallback condition.

This is a conservative quality/speed endpoint, not a new safety certificate.
Keep the original semantic completion, then exhaust remaining candidates only
when its incumbent still fails the preexisting .9 author buffer.
"""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid46_original_semantic',ROOT/'work/grid28/policy.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
key=original.key

def safe_complete(order,observe,policy,subids,aliases,safe=.9):
    assert safe==.9
    result=original.local_complete(order,observe,policy,subids,aliases,safe)
    chosen=result['chosen_pool_id'];seen={r['pool_id'] for r in result['outcomes']}
    incumbent=next((r for r in result['outcomes'] if r['pool_id']==chosen),None)
    added=0
    if incumbent is not None and incumbent['rho']>=safe and len(seen)<len(order):
        for i in order:
            if i in seen:continue
            r=observe(i,'author_threshold_completion');assert r['pool_id']==i
            result['outcomes'].append(r);seen.add(i);added+=1
            if r['strict_admissible'] and key(r)>key(incumbent):incumbent=r
        result['chosen_pool_id']=incumbent['pool_id']
        result['public_queries']=len(seen)
        result['stop_reason']+='_author_threshold_complete'
        result['events'].append({'queries':len(seen),'phase':'author_threshold_completion','chosen':incumbent['pool_id']})
    result['author_completion_queries']=added
    return result

def hand_checks():
    order=list(range(400));aliases={i:i for i in order};subs=[0]*399+[1]
    def response(i,p):return {'pool_id':i,'reward':2. if i==399 else 1.,'rho':.94 if i==399 else .95,'strict_admissible':True,'phase':p}
    r=safe_complete(order,response,'PATIENCE64',subs,aliases)
    assert r['chosen_pool_id']==399 and r['public_queries']==400 and r['author_completion_queries']==1
    def already_safe(i,p):return {**response(i,p),'rho':.8}
    r=safe_complete(order,already_safe,'PATIENCE64',subs,aliases)
    assert r['chosen_pool_id']==0 and r['public_queries']==128 and r['author_completion_queries']==0
    def none(i,p):return {**response(i,p),'strict_admissible':False}
    r=safe_complete(order,none,'PATIENCE64',subs,aliases)
    assert r['chosen_pool_id'] is None and r['public_queries']==400 and r['author_completion_queries']==0
    assert len({x['pool_id'] for x in r['outcomes']})==r['public_queries']
    return 4
