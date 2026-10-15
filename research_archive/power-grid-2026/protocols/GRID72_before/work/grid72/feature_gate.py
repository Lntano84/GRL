"""Legal prefix features and portable fixed-forest conditional verification."""
from pathlib import Path
import functools,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid72_base_features',ROOT/'work/grid64/metacontrol.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
NAMES=M.NAMES+['represented_station_fraction','incumbent_station_coverage',
    'unseen_station_candidate_fraction','remaining_incumbent_station_fraction',
    'best_other_station_gap','top8_incumbent_station_fraction','top8_station_entropy','station_best_std']
SUPPORT=[1,3,8,9,10,13]
def features(rows,order,subids,aliases,limit):
    assert len({r['pool_id'] for r in rows})==len(rows)
    members={int(i):{int(subids[int(i)])} for i in order}
    for alias,rep in aliases.items():
        if int(rep) in members:members[int(rep)].add(int(subids[int(alias)]))
    groups=set().union(*members.values()) if members else set()
    seen={r['pool_id'] for r in rows};visited=set().union(*(members[i] for i in seen)) if seen else set()
    remain=[i for i in order if i not in seen]
    ranked=sorted((r for r in rows if r['strict_admissible']),key=M.key,reverse=True)
    best=ranked[0] if ranked else None;station=None if best is None else int(subids[best['pool_id']])
    current={i for i in order if station in members[i]} if station is not None else set()
    q=M.quality(best);others=[M.quality(r) for r in ranked if int(subids[r['pool_id']])!=station]
    top=ranked[:8];counts={}
    for r in top:
        s=int(subids[r['pool_id']]);counts[s]=counts.get(s,0)+1
    entropy=-sum((n/len(top))*math.log(n/len(top)) for n in counts.values())/math.log(8) if top else 0.
    groupbest={}
    for r in ranked:
        s=int(subids[r['pool_id']]);groupbest[s]=max(groupbest.get(s,0.),M.quality(r))
    extra=[len(visited)/max(1,len(groups)),len(seen&current)/max(1,len(current)),
        sum(not (members[i]&visited) for i in remain)/max(1,len(remain)),len(set(remain)&current)/max(1,len(remain)),
        q-max(others) if others else 0.,sum(int(subids[r['pool_id']])==station for r in top)/max(1,len(top)),
        entropy,float(np.std(list(groupbest.values()))) if groupbest else 0.]
    x=np.concatenate((M.features(rows,len(order),limit),np.asarray(extra,dtype=np.float32)))
    assert x.shape==(20,) and np.isfinite(x).all()
    return x
def predict(model,x):
    x=np.asarray(x);prob=[]
    for t in model['trees']:
        node=0
        while t['left'][node]!=-1:
            node=t['left'][node] if x[t['feature'][node]]<=t['threshold'][node] else t['right'][node]
        prob.append(t['positive'][node])
    return math.fsum(prob)/len(prob)
def out_of_support(model,x):
    return any(float(x[i])<lo-1e-6 or float(x[i])>hi+1e-6 for i,lo,hi in model['support'])
def decide(model,x):
    p=predict(model,x);ood=out_of_support(model,x)
    return bool(ood or p>=model['threshold']),dict(probability=p,threshold=model['threshold'],outside_support=ood)
@functools.lru_cache(maxsize=4)
def load(name):
    assert name in ('plain','struct')
    return json.loads((ROOT/f'outputs/grid72/models/{name}.json').read_text(encoding='utf-8'))
