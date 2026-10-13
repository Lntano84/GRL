"""Four public-feedback policies, with one common author-frequency order."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('grid60_staged',ROOT/'work/grid27/policy.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
key=old.key
POLICIES=('FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME')

def run(order, observe, policy, subids, aliases, safe=.9):
    assert policy in POLICIES and len(order)==len(set(order))
    if policy in ('NO_LOCAL','FRAME'):
        result=old.staged(order,observe,'PATIENCE64',safe)
        result.update(local_substation=None,local_queries=0)
        chosen=result['chosen_pool_id'];seen={r['pool_id'] for r in result['outcomes']}
        if policy=='NO_LOCAL' or chosen is None or not result['fallback'] or len(seen)==len(order):return result
        sub=int(subids[chosen]);targets={int(rep) for i,rep in aliases.items() if int(subids[int(i)])==sub}
        incumbent=next(r for r in result['outcomes'] if r['pool_id']==chosen)
        for i in order:
            if i not in targets or i in seen:continue
            row=observe(i,'local_completion');assert row['pool_id']==i
            result['outcomes'].append(row);seen.add(i);result['local_queries']+=1
            if row['strict_admissible'] and key(row)>key(incumbent):incumbent=row
        result.update(chosen_pool_id=incumbent['pool_id'],public_queries=len(seen),local_substation=sub,
            stop_reason=result['stop_reason']+'_local_complete')
        result['events'].append({'queries':len(seen),'phase':'local_completion','chosen':incumbent['pool_id'],'improved':incumbent['pool_id']!=chosen})
        return result
    results=[];incumbent=None;prefix=None;reason='full_exhaustion';events=[]
    for j,i in enumerate(order):
        row=observe(i,'full_search' if policy=='FULL' else 'first_safe_search');assert row['pool_id']==i
        results.append(row)
        if row['strict_admissible'] and (incumbent is None or key(row)>key(incumbent)):incumbent=row
        if j<128:prefix=None if incumbent is None else incumbent['pool_id']
        if policy=='FIRST_SAFE_BEST' and row['strict_admissible'] and row['rho']<safe:
            incumbent=row;reason='first_verified_below_author_safe';break
    chosen=None if incumbent is None else incumbent['pool_id']
    events.append({'queries':len(results),'phase':'full_search' if policy=='FULL' else 'first_safe_search','chosen':chosen,'improved':chosen is not None})
    return {'outcomes':results,'chosen_pool_id':chosen,'shortlist_choice':prefix,'fallback':len(results)>min(128,len(order)),
        'stop_reason':reason,'events':events,'public_queries':len(results),'local_substation':None,'local_queries':0}

def hand_checks():
    def observe(i,phase):return {'pool_id':i,'reward':2. if i==399 else 1.,'rho':1.,'strict_admissible':True,'phase':phase}
    order=list(range(400));aliases={i:i for i in order};subs=[0]*400
    no=run(order,observe,'NO_LOCAL',subs,aliases);yes=run(order,observe,'FRAME',subs,aliases)
    assert no['public_queries']==256 and no['chosen_pool_id']==0
    assert yes['public_queries']==400 and yes['chosen_pool_id']==399
    subs[-1]=1;remote=run(order,observe,'FRAME',subs,aliases)
    assert remote['chosen_pool_id']==0 and remote['public_queries']==399
    # First-safe comparison is strengthened: when no safe action exists, keep
    # the best admissible action rather than the paper's DoNothing fallback.
    first=run(order,observe,'FIRST_SAFE_BEST',subs,aliases)
    assert first['public_queries']==400 and first['chosen_pool_id']==399
    def safe(i,phase):return {**observe(i,phase),'rho':.8}
    assert run(order,safe,'FIRST_SAFE_BEST',subs,aliases)['public_queries']==1
    for policy in POLICIES:
        result=run(order,lambda i,p:{**observe(i,p),'strict_admissible':False},policy,subs,aliases)
        assert result['chosen_pool_id'] is None and result['public_queries']==400
    assert key({'reward':1.+1e-9,'pool_id':4})<key({'reward':1.,'pool_id':3})
    return 10
