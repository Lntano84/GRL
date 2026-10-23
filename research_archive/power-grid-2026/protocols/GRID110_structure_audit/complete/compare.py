"""Descriptive train-neighbour comparison; no predictor, threshold or validation selection."""
import collections,gzip,json,pickle
import numpy as np
from extract import R,O,read,save,sha
from structure import NAMES
def main():
    assert not (O/'comparison.json').exists()
    assert read(O/'extraction_receipt.json')['passed']
    rows=read(O/'rows.json'); Z=np.load(O/'diagnostic_descriptors.npz')['X']
    C=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());D=read(R/'outputs/grid102_search_rl/design.json')
    raw=np.concatenate([c['states'] for c in C]);oldrows=[(c['sample_id'],j) for c in C for j in range(len(c['states']))]
    assert oldrows==[(r['sample_id'],r['position']) for r in rows]
    Y=np.concatenate([c['gains']-D['cost_per_query']*c['queries'] for c in C])
    with np.load(R/'outputs/grid102_search_rl/normalization.npz') as n:X=np.clip((raw-n['mean'])/n['scale'],-20,20).astype(np.float32).astype(float)
    train=np.array([i for i,r in enumerate(rows) if r['partition']=='train']); val=np.array([i for i,r in enumerate(rows) if r['partition']=='validation'])
    counts=collections.Counter(rows[i]['date'] for i in train);w=np.array([1/counts[rows[i]['date']] for i in train]);w/=w.sum()
    mu=np.sum(w[:,None]*Z[train],axis=0);sd=np.sqrt(np.sum(w[:,None]*(Z[train]-mu)**2,axis=0));sd[sd<1e-8]=1
    B=np.clip((Z-mu)/sd,-20,20).astype(np.float32).astype(float)
    np.savez_compressed(O/'diagnostic_normalization.npz',mean=mu,scale=sd)
    candidates={'original41':X,'structure11':B,'combined52':np.c_[X,B]}
    groupix=collections.defaultdict(list)
    for t,i in enumerate(train):groupix[rows[i]['physical_state_group']].append(t)
    groups=[np.array(groupix[g]) for g in sorted(groupix)]
    lookup={(r['sample_id'],r['position']):i for i,r in enumerate(rows)}
    flips=read(R/'outputs/grid109_input_coverage/positive_flip_details.json');newkeys={(r['sample_id'],r['position']) for r in flips if any(e['event']=='positive_newly_missed' for e in r['events'])}
    reckeys={(r['sample_id'],r['position']) for r in flips if any(e['event']=='positive_recovered' for e in r['events'])}
    results=[]
    def nearest(v,Ztrain):
        d=np.linalg.norm(Ztrain-v,axis=1)/np.sqrt(len(v)); reps=np.array([ix[np.argmin(d[ix])] for ix in groups]);reps=reps[np.argsort(d[reps],kind='stable')]
        return dict(nearest_distance=float(d[reps[0]]),k={str(k):dict(positive_calls=int(np.sum(Y[train[reps[:k]]]>0)),
            nearest_groups=[rows[train[j]]['physical_state_group'] for j in reps[:k]],
            normalized_targets=(Y[train[reps[:k]]]/D['gain_scale']).tolist()) for k in [1,5,20]})
    for i in val:
        key=oldrows[i];results.append(dict(sample_id=key[0],position=key[1],physical_state_group=rows[i]['physical_state_group'],date=rows[i]['date'],
            normalized_target=float(Y[i]/D['gain_scale']),newly_missed=key in newkeys,recovered=key in reckeys,
            neighbours={m:nearest(A[i],A[train]) for m,A in candidates.items()}))
    save('validation_comparison.json',results)
    summary={}
    for cohort,rs in [('all_validation',results),('positive_validation',[r for r in results if r['normalized_target']>0]),
        ('newly_missed',[r for r in results if r['newly_missed']]),('recovered',[r for r in results if r['recovered']])]:
        summary[cohort]={m:dict(states=len(rs),zero_positive_among5=sum(r['neighbours'][m]['k']['5']['positive_calls']==0 for r in rs),
            mean_positive_among5=float(np.mean([r['neighbours'][m]['k']['5']['positive_calls'] for r in rs])),
            physical_calls=len({r['physical_state_group'] for r in rs})) for m in candidates}
    profiles=[]
    for j,name in enumerate(NAMES):
        z=Z[:,j];matches=[D['input']['features'][k] for k in range(41) if np.array_equal(z,raw[:,k])]
        profiles.append(dict(descriptor=name,train_unique=int(len(np.unique(z[train]))),validation_unique=int(len(np.unique(z[val]))),
            train_min=float(z[train].min()),train_max=float(z[train].max()),exact_old_channel_matches=matches,
            train_constant=bool(np.ptp(z[train])==0),all_constant=bool(np.ptp(z)==0)))
    save('descriptor_profile.json',profiles)
    # Read every original neighbour context with both labels preserved. No model reruns.
    details=[];oldlookup={k:i for i,k in enumerate(oldrows)}
    for r in flips:
        i=oldlookup[r['sample_id'],r['position']];pair={}
        for m in ['standardized41','semantic_balanced']:
            neighbours=r['neighbours'][m]['nearest20'][:5]
            pair[m]=[dict(physical_state_group=n['physical_state_group'],sample_id=n['sample_id'],position=n['position'],normalized_target=n['normalized_target'],
                descriptor_values=dict(zip(NAMES,Z[oldlookup[n['sample_id'],n['position']]].tolist()))) for n in neighbours]
        details.append(dict(sample_id=r['sample_id'],position=r['position'],physical_state_group=r['physical_state_group'],step=r['step'],events=r['events'],normalized_target=r['normalized_target'],
            descriptor_values=dict(zip(NAMES,Z[i].tolist())),old_nearest5=pair,
            new_neighbour_counts=next(s['neighbours'] for s in results if s['sample_id']==r['sample_id'] and s['position']==r['position'])))
    save('flip_structure_details.json',details)
    # Reproduce original41 nearest group sets/counts against GRID109 before interpreting new descriptors.
    old=read(R/'outputs/grid109_input_coverage/validation_coverage.json');prior={(r['sample_id'],r['position']):r for r in old}
    for r in results:
        p=prior[r['sample_id'],r['position']]['neighbours']['standardized41']
        for k in [1,5,20]:assert p['k'][str(k)]['positive_calls']==r['neighbours']['original41']['k'][str(k)]['positive_calls']
        assert abs(p['k']['1']['distance']['min']-r['neighbours']['original41']['nearest_distance'])<1e-12
    save('comparison.json',dict(cohorts=summary,all_original41_neighbours_reproduced=True,all11_descriptors_retained=True,
        historical_negative_result_preserved=True,feature_selection=False,model_fits=0,model_inference=0,
        note='Posthoc neighbourhood description on exposed validation dates. More favourable label neighbourhoods do not establish predictive improvement. All three metrics retained, no selection.'))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
