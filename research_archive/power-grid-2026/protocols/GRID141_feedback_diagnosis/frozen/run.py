"""Read-only diagnosis of the frozen GRID140 model and public search feedback."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,pickle,shutil,statistics,sys
sys.path.insert(0,'D:/Codex/GRL研究/work/grid140_global_compression')
import common as C
import numpy as np
R=C.R;W=Path(__file__).parent;O=R/'outputs/grid141_feedback_diagnosis';S=C.O
REPO=C.REPO;A=REPO/'research_archive/power-grid-2026/protocols/GRID141_feedback_diagnosis'
read=C.read;sha=C.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rowsfile(n):
    with gzip.open(S/n,'rt',encoding='utf8') as f:return json.load(f)
def verify():
    C.verify_trained()
    for filename in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/filename).items():assert sha(R/p)==h,p
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def extra(c,n):
    # Only four public fields: full candidate order/aliases, limit and paid prefix.
    order=c['order'];visible=c['outcomes'][:n];m=len(order);subs=C.subids();b=C.P.best(visible);initial=C.P.best(visible[:min(128,n)])
    memberships={int(rep):set() for rep in order}
    for alias,rep in c['aliases'].items():
        if int(rep) in memberships:memberships[int(rep)].add(subs[int(alias)])
    assert all(memberships.values())
    allstations=set().union(*memberships.values());seenstations=set().union(*(memberships[i] for i in order[:n]))
    def q(values,p):return float(np.quantile(values,p)) if values else None
    def physical(rows):return [float(r['rho']) for r in rows if r['rho']>0 and np.isfinite(r['rho'])]
    tail=visible[-64:];prior=visible[-128:-64];v=physical(visible);t=physical(tail);pr=physical(prior)
    f=dict(has_followup_batch=n>128,followup_batches=(n-128)//64,
        tail64_admissible_fraction=sum(r['strict_admissible'] for r in tail)/len(tail),
        tail64_bad_feedback_fraction=sum(r['done'] or r['illegal'] or r['ambiguous'] or bool(r['exceptions']) for r in tail)/len(tail))
    for label,values in [('visible_rho',v),('tail64_rho',t)]:
        for suffix,p in [('q25',.25),('q50',.5),('q75',.75),('q90',.9)]:f[label+'_'+suffix]=q(values,p)
    f['tail64_vs_prior64_median_rho']=q(t,.5)-q(pr,.5) if t and pr else None
    tailbest=C.P.best(tail);priorbest=C.P.best(prior)
    f['tail64_best_reward']=C.P.key(tailbest)[0] if tailbest else None
    f['tail64_vs_prior64_best_reward']=C.P.key(tailbest)[0]-C.P.key(priorbest)[0] if tailbest and priorbest else None
    f.update(stations_total=len(allstations),seen_station_fraction=len(seenstations)/len(allstations),unseen_station_count=len(allstations-seenstations))
    station=subs[b['pool_id']] if b else None;stationids={i for i in order if station is not None and station in memberships[i]}
    seenstationids=stationids&set(order[:n]);remainingstationids=stationids-set(order[:n]);prefixsafe=initial is not None and initial['rho']<.9
    trigger=b is not None and n<m and (not prefixsafe or C.P.bid(b)!=C.P.bid(initial))
    f.update(incumbent_station_candidates=len(stationids) if b else None,
        incumbent_station_remaining=len(remainingstationids) if b else None,
        incumbent_station_seen_fraction=len(seenstationids)/len(stationids) if stationids else None,
        completion_would_trigger=trigger,prospective_completion_queries=len(remainingstationids) if trigger else 0,
        stop_total_queries_known_now=n+(len(remainingstationids) if trigger else 0))
    nextids=order[n:min(n+64,m)]
    f['next_batch_new_station_candidate_fraction']=sum(bool(memberships[i]-seenstations) for i in nextids)/len(nextids) if nextids else None
    f['next_batch_incumbent_station_fraction']=sum(i in stationids for i in nextids)/len(nextids) if nextids and b else None
    other=C.P.best([r for r in visible if station is not None and subs[r['pool_id']]!=station])
    f['best_other_station_reward_gap']=C.P.key(b)[0]-C.P.key(other)[0] if b and other else None
    valid=sorted((r for r in visible if r['strict_admissible']),key=C.P.key,reverse=True)
    f['top_two_same_station']=subs[valid[0]['pool_id']]==subs[valid[1]['pool_id']] if len(valid)>1 else None
    return f

def prepare():
    assert not O.exists();O.mkdir(parents=True);C.verify_trained();assert read(S/'review.json')['passed']
    c=C.call_at(C.refs()[0]['source']);feature_names=list(extra(c,128))
    save('design.json',dict(id='GRID141_FEEDBACK_DIAGNOSIS',utc=stamp(),authorization='User continues GRID140 diagnosis;no new fit or physical replay.',
        population='All726 existing prefixes:386fit/198calibration/142development validation;all labels/scores remain fixed.',
        analyses=['Reconstruct all64 bootstrap leaf supports and saved scores without fit/predict calls.','Exact original-feature and full64-leaf-signature label collisions.','All negative-prefix opposite-label nearest-fit matches within same checkpoint and original entry-safe branch;exclude same date;fit-only median/IQR normalization;diagnostic not classifier.','All fixed supplemental public summaries with hidden-suffix poisoning.','Exact action mismatch versus signed immediate reward difference;do not relabel.'],
        supplemental_features=feature_names,public_fields=['order','aliases','rho_limit','outcomes[:n]'],
        support='Bootstrap multiplicities and original date/call weights reconstructed exactly;number of unique fit dates/calls reported per leaf. Training purity is not held-out confidence.',
        matching='Original18 features only. IQR fit normalization,range fallback if zero IQR,unit fallback if constant;append missing flags. Same n and prefix_safe,opposite stop label,exclude same calendar date. Save all pairs;do not deploy or tune by pair outcomes.',
        cases='Prespecified two threshold-raising calibration errors plus all four calibrated validation stops;all other negative rows retained in matching and summaries.',
        boundaries='Already exposed development data. Supplemental features not fed into any model;no feature selection,threshold sweep,new predictions,refit,RL,NN,scenarios,final8tests or paper edit.',
        prohibited=['fit','predict','new model or threshold','new physical simulation','final test materialization','label changes','automatic scale']))
    names=['design.json','model_export.json','model.pkl','preprocessing.json','fit_sample_weights.json','fit_receipt.json','trained_freeze.json','train_scores.npz','validation_scores.npz','fit_rows.json.gz','calibration_rows.json.gz','cutoff.json','calibration_history.json','all_search_replays.json','review.json','decision.json','post_validation_diagnosis.json','final_verification.json']
    paths=[S/n for n in names]+[R/'work/grid140_global_compression/common.py',R/'work/grid139_global_feasibility/run.py',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))});save('state.json',dict(stage='frozen'));verify();archive('frozen');print('GRID141_FROZEN',len(feature_names))

def leaf(t,x):
    node=0
    while t['left'][node]!=-1:node=t['left'][node] if x[t['feature'][node]]<=t['threshold'][node] else t['right'][node]
    return node
def summary(values):return dict(minimum=min(values),median=float(np.median(values)),maximum=max(values)) if values else None

def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(utc=stamp(),new_fits=0,new_physical=0))
    fit=rowsfile('fit_rows.json.gz');cal=rowsfile('calibration_rows.json.gz');val=C.eligible_data('validation');rows=fit+cal+val;parts=['fit']*386+['calibration']*198+['validation']*142
    z=np.load(S/'train_scores.npz');scores=np.concatenate([z['fit'],z['calibration'],np.load(S/'validation_scores.npz')['scores']]);assert len(rows)==len(scores)==726
    pre=read(S/'preprocessing.json');xx=C.transform(rows,pre['features'],pre['medians']).astype(np.float32);model=read(S/'model_export.json');weights=np.asarray([r['weight'] for r in read(S/'fit_sample_weights.json')]);y=np.asarray([r['target']['stop_label'] for r in fit],bool)
    with (S/'model.pkl').open('rb') as f:est=pickle.load(f)
    bootstrap=est.estimators_samples_;routes=np.asarray([[leaf(t,x) for t in model['trees']] for x in xx],int);assert routes.shape==(726,64)
    leafinfo={};maxerr=0.
    for ti,t in enumerate(model['trees']):
        idx=bootstrap[ti]
        for node in set(routes[:386,ti]):
            bootidx=idx[routes[idx,ti]==node]
            if len(bootidx)==0:continue
            unique=np.unique(bootidx);w=weights[bootidx];p=float(w[y[bootidx]].sum()/w.sum());stored=t['value'][node][1]/sum(t['value'][node]);maxerr=max(maxerr,abs(p-stored));assert abs(p-stored)<1e-10
            leafinfo[(ti,node)]=dict(tree=ti,node=int(node),stop_fraction=stored,bootstrap_rows=len(bootidx),unique_rows=len(unique),dates=len({fit[j]['meta']['date'] for j in unique}),calls=len({fit[j]['meta']['call_id'] for j in unique}),continue_unique_rows=sum(not y[j] for j in unique),fit_dates=sorted({fit[j]['meta']['date'] for j in unique}))
    reconstructed=np.asarray([np.mean([model['trees'][ti]['value'][node][1]/sum(model['trees'][ti]['value'][node]) for ti,node in enumerate(nodes)]) for nodes in routes]);assert np.max(abs(reconstructed-scores))<1e-10
    save('bootstrap_leaf_supports.json',list(leafinfo.values()))
    supplemental=[];hiddenchecks=0
    for r in rows:
        c=C.call_at(r['meta']['source']);n=r['meta']['visible_count'];public=dict(order=c['order'],aliases=c['aliases'],rho_limit=c['rho_limit'],outcomes=c['outcomes'][:n]);f=extra(public,n)
        poisoned=dict(c,outcomes=c['outcomes'][:n]+[dict(v,reward=-999.,rho=999.,strict_admissible=False,pool_id=-999) for v in c['outcomes'][n:]],chosen_pool_id=-999,step=-999)
        assert extra(poisoned,n)==f;assert f['stop_total_queries_known_now']==r['target']['stop_total_queries_known_now'];hiddenchecks+=1
        supplemental.append(dict(sample_id=r['meta']['sample_id'],features=f))
    with gzip.open(O/'public_supplemental_inputs.jsonl.gz','wt',encoding='utf8') as f:
        for r in supplemental:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    save('supplemental_field_definitions.json',dict(fields=read(O/'design.json')['supplemental_features'],public_only=True,unused_by_models=True,notes='Alias-aware public coverage;rho quantiles use finite positive revealed rho;last64 slices refer to observed prefix even atn128,has_followup_batch distinguishes later full batches.'))
    def group_collisions(keys):
        groups=collections.defaultdict(list)
        for i,k in enumerate(keys):groups[k].append(i)
        mixed=[]
        for ii in groups.values():
            if len({rows[i]['target']['stop_label'] for i in ii})>1:mixed.append([dict(sample_id=rows[i]['meta']['sample_id'],partition=parts[i],date=rows[i]['meta']['date'],call_id=rows[i]['meta']['call_id'],visible_count=rows[i]['meta']['visible_count'],stop_label=rows[i]['target']['stop_label'],score=float(scores[i])) for i in ii])
        return dict(groups=len(groups),mixed_groups=len(mixed),rows_in_mixed_groups=sum(map(len,mixed)),groups_detail=mixed)
    collisions=dict(original_features=group_collisions([json.dumps(r['features'],sort_keys=True) for r in rows]),forest_leaf_signature=group_collisions([tuple(v) for v in routes]));save('representation_collisions.json',collisions)
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in pre['features']] for r in fit]);scales=[]
    for j in range(18):
        vv=raw[np.isfinite(raw[:,j]),j];scale=float(np.quantile(vv,.75)-np.quantile(vv,.25)) if len(vv) else 0.
        if scale==0:scale=float(np.max(vv)-np.min(vv)) if len(vv) else 0.
        scales.append(scale or 1.)
    norm=xx.astype(float).copy();norm[:,:18]=(norm[:,:18]-np.asarray(pre['medians']))/scales
    pairs=[]
    for i,r in enumerate(rows):
        if r['target']['stop_label']:continue
        jj=[j for j,s in enumerate(fit) if s['target']['stop_label'] and s['meta']['date']!=r['meta']['date'] and s['meta']['visible_count']==r['meta']['visible_count'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
        if not jj:pairs.append(dict(sample_id=r['meta']['sample_id'],partition=parts[i],matched=False));continue
        j=min(jj,key=lambda j:(float(np.mean((norm[i]-norm[j])**2)),fit[j]['meta']['sample_id']))
        pairs.append(dict(sample_id=r['meta']['sample_id'],partition=parts[i],matched=True,call_id=r['meta']['call_id'],date=r['meta']['date'],visible_count=r['meta']['visible_count'],score=float(scores[i]),nearest_positive_sample_id=fit[j]['meta']['sample_id'],nearest_positive_call_id=fit[j]['meta']['call_id'],nearest_positive_date=fit[j]['meta']['date'],nearest_positive_score=float(scores[j]),distance=float(np.sqrt(np.mean((norm[i]-norm[j])**2))),original_features_negative=r['features'],original_features_positive=fit[j]['features'],supplemental_negative=supplemental[i]['features'],supplemental_positive=supplemental[j]['features']))
    save('all_negative_matched_pairs.json',pairs);save('matching_normalization.json',dict(medians=pre['medians'],scales=scales,fit_only=True,not_a_classifier=True))
    cases=[];prior=read(S/'post_validation_diagnosis.json');identities=[(v['call_id'],v['global_queries'],'calibration_error') for v in prior['calibration_threshold_raising_examples']]+[(v['call_id'],v['visible_count'],'validation_success') for v in prior['validation_calibrated_stops']]
    for identity,n,kind in identities:
        i=next(i for i,r in enumerate(rows) if r['meta']['call_id']==identity and r['meta']['visible_count']==n);supports=[leafinfo[ti,int(node)] for ti,node in enumerate(routes[i])];r=rows[i]
        cases.append(dict(kind=kind,call_id=identity,date=r['meta']['date'],sample_id=r['meta']['sample_id'],visible_count=n,score=float(scores[i]),stop_label=r['target']['stop_label'],bootstrap_leaf_stop_fraction=summary([s['stop_fraction'] for s in supports]),bootstrap_leaf_unique_dates=summary([s['dates'] for s in supports]),bootstrap_leaf_unique_calls=summary([s['calls'] for s in supports]),bootstrap_leaf_unique_rows=summary([s['unique_rows'] for s in supports]),pure_stop_trees=sum(s['stop_fraction']>=1-1e-12 for s in supports),support_per_tree=supports,original_features=r['features'],supplemental=supplemental[i]['features'],matched_pair=next((p for p in pairs if p['sample_id']==r['meta']['sample_id']),None)))
    save('prespecified_cases.json',cases)
    labelrows=[]
    for i,r in enumerate(rows):
        if r['target']['stop_label']:continue
        c=C.call_at(r['meta']['source']);truth={t['pool_id']:t for t in c['outcomes']};a=r['target']['after_stop_choice_label'];b=r['target']['original_frame_choice_label'];delta=C.P.key(truth[b])[0]-C.P.key(truth[a])[0] if a is not None and b is not None else None
        reason='same_action_no_strict_query_saving' if a==b else ('original_reward_higher' if delta is not None and delta>0 else 'equal_reward_different_action' if delta==0 else 'stop_reward_higher' if delta is not None and delta<0 else 'availability_differs')
        labelrows.append(dict(sample_id=r['meta']['sample_id'],partition=parts[i],call_id=r['meta']['call_id'],date=r['meta']['date'],visible_count=r['meta']['visible_count'],score=float(scores[i]),reason=reason,signed_reward_gap=delta,queries_saved=r['target']['queries_saved'],original_action=b,early_stop_action=a))
    save('all_negative_label_components.json',labelrows)
    labelsummary=[]
    for part in ['fit','calibration','validation']:
        rr=[r for r in labelrows if r['partition']==part];labelsummary.append(dict(partition=part,n=len(rr),reasons=dict(collections.Counter(r['reason'] for r in rr)),positive_reward_gap=summary([r['signed_reward_gap'] for r in rr if r['signed_reward_gap'] is not None and r['signed_reward_gap']>0])))
    save('negative_label_summary.json',labelsummary)
    res=dict(prefixes=726,fit_prefixes=386,calibration_prefixes=198,development_validation_prefixes=142,bootstrap_leaves=len(leafinfo),max_leaf_reconstruction_error=maxerr,max_score_reconstruction_error=float(np.max(abs(reconstructed-scores))),weighted_fit_stop_prior=float(np.sum(weights*y)/weights.sum()),supplemental_feature_count=len(supplemental[0]['features']),public_suffix_checks=hiddenchecks,known_stop_query_checks=hiddenchecks,collisions={k:{x:y for x,y in v.items() if x!='groups_detail'} for k,v in collisions.items()},matching=dict(negative_rows=len(pairs),matched=sum(p['matched'] for p in pairs),unmatched=sum(not p['matched'] for p in pairs)),negative_label_summary=labelsummary,new_fits=0,new_predict_calls=0,new_physical=0,final_tests_absent=True)
    save('analysis_summary.json',res);verify();save('state.json',dict(stage='analysed_pending_review'));print(json.dumps(res,ensure_ascii=False))

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
