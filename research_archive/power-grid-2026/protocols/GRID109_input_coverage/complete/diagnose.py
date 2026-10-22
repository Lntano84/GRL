"""Posthoc, saved-data-only input and training-coverage diagnosis. No model fitting."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json, pickle
import numpy as np

ROOT=Path('D:/Codex/GRL研究'); OUT=ROOT/'outputs/grid109_input_coverage'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf8')
def summary(a):
    a=np.asarray(a,dtype=float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max())) if len(a) else None

def main():
    assert not (OUT/'diagnosis.json').exists()
    D=read(ROOT/'outputs/grid102_search_rl/design.json'); names=D['input']['features']
    # Partition all existing channels by meaning before computing either distance.
    groups={
        'budget_and_group_size':[0,1,2,3,4,5,6,8,36,37,38,39,40],
        'observed_quality':[7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,34,35],
        'feedback_history':[25,26,27,28,29,30,31,32,33]}
    assert sorted(sum(groups.values(),[]))==list(range(41))
    sources=[ROOT/'outputs/grid102_search_rl/cases.pkl',ROOT/'outputs/grid102_search_rl/normalization.npz',
        ROOT/'outputs/grid102_search_rl/design.json',ROOT/'outputs/grid107_direct_value/train.npz',
        ROOT/'outputs/grid107_direct_value/training_rows.json',ROOT/'outputs/grid108_loss_budget/state_predictions.jsonl.gz',
        ROOT/'outputs/grid108_loss_budget/error_transitions.json',ROOT/'outputs/grid108_loss_budget/critical_probes.json',
        ROOT/'outputs/grid108_loss_budget/probe_sources.json',ROOT/'outputs/grid108_loss_budget/validation_trajectories.json',
        ROOT/'outputs/grid100_data_foundation/bulk_dataset_manifest.json',ROOT/'work/grid101_gain_learning/features.py',
        ROOT/'work/grid102_search_rl/observations.py']
    for seed in range(3): sources += [ROOT/f'outputs/grid108_loss_budget/draws_HUBER_29000_{seed}.npz']
    probes=read(ROOT/'outputs/grid108_loss_budget/probe_sources.json')
    sources += [ROOT/p for p in probes.values()]
    frozen={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    save('plan.json',dict(id='GRID109_INPUT_COVERAGE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read-only posthoc diagnosis of all Huber positive action flips and March probes; no new fits, inference, features, simulation or final test.',
        selection='All newly missed/recovered positive validation states, deduplicate sample_id+position; preserve seeds and physical-call groups. Compare with all validation controls.',
        distances={'standardized41':'RMS of frozen clipped training-standardized41 input.',
        'semantic_balanced':'Square root of mean of within-group mean squared standardized differences; all41 channels retained, three groups equal weight.'},
        groups={k:[names[i] for i in ix] for k,ix in groups.items()},
        neighbours='Nearest1/5/20 DISTINCT training physical calls, represented by closest reachable prefix in that call. Descriptive only, not a learned rule or proof of missing information.',
        exposure='Existing per-state training draws, no altered sampling.',
        limits='Overlapping prefixes/date replicates are correlated. Close opposite labels do not prove feature insufficiency; no new feature or method selected. Fixed-teacher search utilities are not physical week costs.'))
    save('input_manifest.json',frozen)
    cases=pickle.loads((ROOT/'outputs/grid102_search_rl/cases.pkl').read_bytes())
    norm=np.load(ROOT/'outputs/grid102_search_rl/normalization.npz'); mean,scale=norm['mean'],norm['scale']
    def normalize(a): return np.clip((a-mean)/scale,-20,20).astype(np.float32)
    meta=[]; raw=[]; target=[]
    for c in cases:
        for j,s in enumerate(c['states']):
            meta.append(dict(sample_id=c['sample_id'],position=j,date=c['date'],partition=c['partition'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],next_queries=int(c['queries'][j])))
            raw.append(s); target.append(float(c['gains'][j]-D['cost_per_query']*c['queries'][j]))
    raw=np.stack(raw); X=normalize(raw); y=np.array(target); lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)}
    train=np.array([i for i,r in enumerate(meta) if r['partition']=='train']); val=np.array([i for i,r in enumerate(meta) if r['partition']=='validation'])
    Z=X[train].astype(float); ty=y[train]; tm=[meta[i] for i in train]
    oldtrain=np.load(ROOT/'outputs/grid107_direct_value/train.npz'); assert np.array_equal(Z.astype(np.float32),oldtrain['X'])
    assert np.array_equal((ty/D['gain_scale']).astype(np.float32),oldtrain['target'])
    trainrows=read(ROOT/'outputs/grid107_direct_value/training_rows.json')
    assert all(a['sample_id']==b['sample_id'] and a['position']==b['position'] for a,b in zip(tm,trainrows))
    predictions={}
    with gzip.open(ROOT/'outputs/grid108_loss_budget/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['method'].startswith('HUBER_'):
                key=(r['method'],r['sample_id'],r['position']); predictions[key]=r
                assert y[lookup[r['sample_id'],r['position']]]==r['target_net']
    events=read(ROOT/'outputs/grid108_loss_budget/error_transitions.json')['events']
    selected=collections.defaultdict(list)
    for e in events:
        if e['event'].startswith('positive'): selected[e['sample_id'],e['position']].append(e)
    exposure=[]
    for seed in range(3):
        with np.load(ROOT/f'outputs/grid108_loss_budget/draws_HUBER_29000_{seed}.npz') as z: exposure.append(z['state_draws'])
    exposure=np.stack(exposure)
    call_rows=collections.defaultdict(list)
    for i,r in enumerate(tm): call_rows[r['physical_state_group']].append(i)
    callids=sorted(call_rows); callix=[np.array(call_rows[g]) for g in callids]
    def neighbourhood(x,metric):
        delta=(Z-x)**2
        d=np.sqrt(delta.mean(axis=1)) if metric=='standardized41' else np.sqrt(np.mean([delta[:,ix].mean(axis=1) for ix in groups.values()],axis=0))
        reps=np.array([ix[np.argmin(d[ix])] for ix in callix]); reps=reps[np.argsort(d[reps],kind='stable')]
        result={}
        for k in [1,5,20]:
            chosen=reps[:k]
            result[str(k)]=dict(positive_calls=int(np.sum(ty[chosen]>0)),dates=len({tm[i]['date'] for i in chosen}),
                distance=summary(d[chosen]),normalized_targets=summary(ty[chosen]/D['gain_scale']))
        # Keep nearest20 fully traceable, not just an average over adjacent prefixes.
        details=[]
        for i in reps[:20]:
            r=tm[i]; q={}
            for seed in range(3):
                q[str(seed)]={b:predictions[f'HUBER_{b}_{seed}',r['sample_id'],r['position']]['action'] for b in [7250,29000]}
            details.append(dict(**r,distance=float(d[i]),normalized_target=float(ty[i]/D['gain_scale']),training_draws=exposure[:,i].tolist(),actions=q))
        pos=np.flatnonzero(ty>0); neg=np.flatnonzero(ty<=0)
        return dict(k=result,nearest_positive_distance=float(d[pos].min()),nearest_nonpositive_distance=float(d[neg].min()),nearest20=details)
    # Controls include ALL validation states; no threshold chosen from the outcomes.
    control=[]
    for i in val:
        ns={m:neighbourhood(X[i],m) for m in ['standardized41','semantic_balanced']}
        control.append(dict(**meta[i],normalized_target=float(y[i]/D['gain_scale']),neighbours={m:{k:v for k,v in n.items() if k!='nearest20'} for m,n in ns.items()}))
    save('validation_coverage.json',control)
    ctrl={(r['sample_id'],r['position']):r for r in control}
    samples=read(ROOT/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']; sm={s['sample_id']:s for s in samples}
    details=[]
    for key,es in sorted(selected.items()):
        i=lookup[key]; r=meta[i]; n={m:neighbourhood(X[i],m) for m in ['standardized41','semantic_balanced']}
        details.append(dict(**r,source_call=sm[r['sample_id']]['source_call'],step=sm[r['sample_id']]['step'],
            normalized_target=float(y[i]/D['gain_scale']),events=es,
            raw_features=dict(zip(names,raw[i].tolist())),neighbours=n,
            actions={str(s):{str(b):predictions[f'HUBER_{b}_{s}',*key]['action'] for b in [7250,29000]} for s in range(3)}))
    critical=[]
    for probe in read(ROOT/'outputs/grid108_loss_budget/critical_probes.json'):
        call=read(ROOT/probes[probe['context']]); dec=call['learned_decisions'][0]
        a=np.asarray(dec['raw_features']);x=np.asarray(dec['normalized_features'],dtype=np.float32);assert np.array_equal(x,normalize(a))
        critical.append(dict(context=probe['context'],source=probe['source'],normalized_target=probe['actual_next_normalized_net'],
            raw_features=dict(zip(names,a.tolist())),scores=probe['scores'],neighbours={m:neighbourhood(x,m) for m in ['standardized41','semantic_balanced']}))
    save('positive_flip_details.json',details); save('critical_coverage.json',critical)
    def cohort(rows):
        return {m:dict(nearest_distance=summary([r['neighbours'][m]['k']['1']['distance']['min'] for r in rows]),
            nearest5_positive_calls=summary([r['neighbours'][m]['k']['5']['positive_calls'] for r in rows]),
            zero_positive_among_nearest5=sum(r['neighbours'][m]['k']['5']['positive_calls']==0 for r in rows),
            positive_closer_than_nonpositive=sum(r['neighbours'][m]['nearest_positive_distance']<r['neighbours'][m]['nearest_nonpositive_distance'] for r in rows)) for m in ['standardized41','semantic_balanced']}
    newkeys={k for k,es in selected.items() if any(e['event']=='positive_newly_missed' for e in es)}
    reckeys={k for k,es in selected.items() if any(e['event']=='positive_recovered' for e in es)}
    counts={}
    for part,indices in [('train',train),('validation',val)]:
        pos=[i for i in indices if y[i]>0]
        counts[part]=dict(states=len(indices),positive=len(pos),physical_calls=len({meta[i]['physical_state_group'] for i in indices}),
            positive_physical_calls=len({meta[i]['physical_state_group'] for i in pos}),positive_dates=len({meta[i]['date'] for i in pos}))
    # Exact collisions: frozen model input, not an approximate threshold.
    keys=[x.tobytes() for x in X]; labels=collections.defaultdict(set)
    for k,t in zip(keys,y): labels[k].add(t>0)
    cross=set(keys[i] for i in train)&set(keys[i] for i in val)
    diagnosis=dict(counts=counts,newly_missed_unique_states=len(newkeys),newly_missed_physical_calls=len({meta[lookup[k]]['physical_state_group'] for k in newkeys}),
        recovered_unique_states=len(reckeys),recovered_physical_calls=len({meta[lookup[k]]['physical_state_group'] for k in reckeys}),
        new_misses_calls=sorted({meta[lookup[k]]['physical_state_group'] for k in newkeys}),
        positive_training_sampling_mass=float(oldtrain['sampling_probability'][ty>0].sum()),
        training_state_draws={str(s):dict(all=summary(exposure[s]),positive=summary(exposure[s,ty>0]),never_drawn=int(np.sum(exposure[s]==0))) for s in range(3)},
        cohorts={'all_validation':cohort(control),'positive_validation':cohort([r for r in control if r['normalized_target']>0]),
            'newly_missed':cohort([ctrl[k] for k in sorted(newkeys)]),'recovered':cohort([ctrl[k] for k in sorted(reckeys)])},
        exact_inputs=dict(all_unique=len(labels),sign_conflicting_vectors=sum(len(s)>1 for s in labels.values()),cross_partition_matches=len(cross)),
        raw_constant_channels=[names[j] for j in range(41) if np.ptp(raw[train,j])==0],
        new_fits=0,new_inference=0,new_physical_calls=0,new_features=0,final_test_materialized=False)
    # Teacher-path utility loss is decomposed by ALL physical calls, keeping seed and date normalization.
    tr=read(ROOT/'outputs/grid108_loss_budget/validation_trajectories.json'); old={}
    for r in tr:
        if r['method'].startswith('HUBER_'): old[r['method'],r['sample_id']]=r
    decomposed=[]
    date_counts=collections.Counter(c['date'] for c in cases if c['partition']=='validation')
    for seed in range(3):
        bycall=collections.defaultdict(list)
        for c in cases:
            if c['partition']!='validation': continue
            a=old[f'HUBER_7250_{seed}',c['sample_id']]; b=old[f'HUBER_29000_{seed}',c['sample_id']]
            bycall[c['physical_state_group']].append((c['date'],b['net_utility']-a['net_utility']))
        for g,rs in sorted(bycall.items()):
            decomposed.append(dict(seed=seed,physical_state_group=g,date=rs[0][0],prefixes=len(rs),mean_prefix_utility_delta=float(np.mean([v for _,v in rs])),
                contribution_to_date_macro=sum(v/date_counts[d]/4 for d,v in rs)))
    save('utility_decomposition.json',dict(all_calls=decomposed,note='Exact fixed-teacher-prefix metric decomposition; not independent physical-week causal effects.'))
    save('diagnosis.json',diagnosis)
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    save('analysis_receipt.json',dict(passed=True,source_hashes_unchanged=True,states=len(X),validation_controls=len(control),positive_flip_states=len(details),new_fits=0,new_inference=0,new_physical_calls=0))
    print(json.dumps(diagnosis,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
