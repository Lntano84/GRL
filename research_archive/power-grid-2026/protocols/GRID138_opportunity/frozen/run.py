"""Saved-log timing opportunity and predecision overlap; no fit or simulation."""
from pathlib import Path
import collections,datetime,hashlib,json,math,shutil,statistics,sys
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid138_opportunity';W=Path(__file__).parent
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID138_opportunity'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sealed():
    d=read(R/'outputs/grid100_data_foundation/design.json')
    for c in d['scenario_plan']:
        if c['partition']=='test':
            assert not (R/d['environment']/'chronics'/c['scenario']).exists()
            assert not (R/'outputs/grid100_data_foundation/datasets'/c['scenario']).exists()
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    for p,h in read(R/'outputs/grid137_quality_mapping/source_hashes.json').items():assert sha(R/p)==h,p
    sealed()
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
def folder(s):return R/'outputs/grid104_learning_closedloop/runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed()
    assert read(R/'outputs/grid104_learning_closedloop/review.json')['passed_saved_data']
    assert read(R/'outputs/grid137_quality_mapping/final_verification.json')['passed']
    save('design.json',dict(id='GRID138_SAVED_OPPORTUNITY_AND_PUBLIC_OVERLAP',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='User continues agreed steps1/2: saved timing opportunity and17 trajectory-changing predecision cases. No new training, physical collection or rules.',
        status='Exploratory already-exposed development records;source freeze is reproducibility,not prospective confirmation.',
        timing='All48 GRID104 summaries/calls;mutually exclusive optimizer,all forecasts,search overhead,other. Primary opportunity on8 originalFRAME repetitions and4 dates.',
        timing_components=['all recorded local forecast seconds','local after first64 on same original path','local seconds in calls returning same pool_id as before completion'],
        timing_limits='Recorded forecast components only;excluding unmeasured removable bookkeeping. Not an attainable speedup guarantee,whole-closed-loop upper bound,or a policy rollout. Simple baseline comparison uses original same-batch GRID104 only.',
        public_features='All15 original GRID96 features,unchanged. Metadata step/budget/station displayed separately. No outcome-based feature selection.',
        neighborhood='For each of17 changed trajectories,all nonpilot known-quality cases from other calendar dates;each feature scaled by reference max-min (constant span=1),unweighted mean squared distance;report first3 and all distance rows. Labels do not affect distance;not a classifier,policy,calibrated risk or generalization test.',
        descriptive_statistics='All15 feature Spearman correlations with continuous wholeweek cost delta over16 completed changed trajectories;all coefficients retained,no p values,best feature selection or trigger thresholds. Exact public-feature collisions over56 also recorded.',
        inherited_strata='Only original rho safe0.9 and physical overload1.0 used for descriptive grouping;no new cutoff.',
        prohibited=['fit','model forward','threshold search','new control rule','new physical/scenario/final tests','cost-headroom or future outcome as input','automatic scale','manuscript changes']))
    paths=[R/p for p in ['outputs/grid104_learning_closedloop/design.json','outputs/grid104_learning_closedloop/finished.json','outputs/grid104_learning_closedloop/review.json','outputs/grid104_learning_closedloop/decision.json','work/grid104_learning_closedloop/audit.py',
        'outputs/grid96_value_learning/design.json','outputs/grid96_value_learning/labels.json','outputs/grid96_value_learning/learning_review.json','work/grid96_value_learning/learn.py',
        'outputs/grid137_quality_mapping/quality_training_evidence_v2.json','outputs/grid137_quality_mapping/final_verification.json','outputs/grid137_quality_mapping/source_hashes.json','outputs/grid137_quality_mapping/decision.json']]
    summaries=read(R/'outputs/grid104_learning_closedloop/finished.json')['summaries'];assert len(summaries)==48
    for s in summaries:paths += [folder(s)/'summary.json',folder(s)/'components.json',*sorted(folder(s).glob('call[0-9][0-9][0-9].json'))]
    for c in read(R/'outputs/grid96_value_learning/design.json')['cases']:paths.append(R/c['source'])
    paths += [O/'design.json'];save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sorted(set(paths))})
    save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))});save('state.json',dict(stage='frozen'));verify();archive('frozen');print('GRID138_FROZEN')
def ranks(xx):
    ix=sorted(range(len(xx)),key=lambda j:xx[j]);out=[None]*len(xx);j=0
    while j<len(ix):
        k=j+1
        while k<len(ix) and xx[ix[k]]==xx[ix[j]]:k+=1
        for t in range(j,k):out[ix[t]]=(j+k-1)/2
        j=k
    return out
def corr(x,y):
    x,y=ranks(x),ranks(y);xm,ym=statistics.mean(x),statistics.mean(y)
    a=math.fsum((v-xm)*(w-ym) for v,w in zip(x,y));b=math.fsum((v-xm)**2 for v in x)*math.fsum((w-ym)**2 for w in y)
    return a/math.sqrt(b) if b else None
def run():
    verify();assert not (O/'analysis_started.json').exists();save('analysis_started.json',dict(fits=0,forwards=0,physical=0))
    audit=read(R/'outputs/grid104_learning_closedloop/review.json');ar={(r['repeat'],r['week'],r['method']):r for r in audit['descriptive_runs']};runs=[];calls=[]
    for s in read(R/'outputs/grid104_learning_closedloop/finished.json')['summaries']:
        p=folder(s);assert read(p/'summary.json')==s;cc=[read(x) for x in sorted(p.glob('call[0-9][0-9][0-9].json'))]
        assert len(cc)==s['n1_calls'] and sum(c['public_queries'] for c in cc)==s['n1_forecasts']
        local=tail=same=0.0;localq=tailq=sameq=0
        for j,c in enumerate(cc):
            outcomes=c['outcomes'];ll=[r for r in outcomes if r['phase']=='local_completion'];tt=ll[64:]
            assert len(outcomes)==len({r['pool_id'] for r in outcomes})==c['public_queries'];assert len(ll)==c['local_queries']
            assert abs(math.fsum(r['forecast_s'] for r in outcomes)-c['forecast_s'])<1e-7
            if s['method']=='FRAME_GUARD':assert len(ll)==c['completion_budget']
            if s['method']=='LOCAL_BATCH1':assert len(ll)==min(64,c['completion_budget'])
            l=math.fsum(r['forecast_s'] for r in ll);t=math.fsum(r['forecast_s'] for r in tt);samechoice=c['pre_extra_choice']==c['chosen_pool_id']
            local+=l;tail+=t;localq+=len(ll);tailq+=len(tt)
            if samechoice:same+=l;sameq+=len(ll)
            calls.append(dict(method=s['method'],date=s['date'],repeat=s['repeat'],index=j,step=c['step'],local_queries=len(ll),local_s=l,after64_s=t,after64_queries=len(tt),same_choice=samechoice,pre_choice=c['pre_extra_choice'],post_choice=c['chosen_pool_id']))
        opt=math.fsum(c['wall_s'] for c in read(p/'components.json')['calls'] if c['in_control']);over=s['n1_source_s']-s['n1_forecast_s'];other=s['controller_s']-opt-s['all_forecast_s']-over
        old=ar[(s['repeat'],s['week'],s['method'])]
        assert abs(opt-old['optimizer_s'])<1e-7 and abs(other-old['other_s'])<1e-7 and min(other,over,opt)>=-1e-6
        assert abs(math.fsum(c['forecast_s'] for c in cc)-s['n1_forecast_s'])<1e-7 and local<=s['n1_forecast_s']+1e-7
        runs.append(dict(method=s['method'],date=s['date'],week=s['week'],repeat=s['repeat'],steps=s['steps'],complete=s['complete'],cost=s['cost'],controller_s=s['controller_s'],optimizer_s=opt,all_forecast_s=s['all_forecast_s'],search_overhead_s=over,other_s=other,
            local_s=local,local_queries=localq,local_after64_s=tail,local_after64_queries=tailq,unchanged_choice_local_s=same,unchanged_choice_queries=sameq,
            local_share=local/s['controller_s'],after64_share=tail/s['controller_s'],unchanged_choice_share=same/s['controller_s'],feature_s=old['local_feature_s'],inference_s=old['local_inference_s']))
    save('timing_runs.json',runs);save('timing_calls.json',calls)
    frame=[r for r in runs if r['method']=='FRAME_GUARD'];assert len(frame)==8
    bydate=[]
    for date in sorted({r['date'] for r in frame}):
        rr=[r for r in frame if r['date']==date];assert len(rr)==2
        fields=['controller_s','optimizer_s','all_forecast_s','search_overhead_s','other_s','local_s','local_queries','local_after64_s','local_after64_queries','unchanged_choice_local_s','unchanged_choice_queries','local_share','after64_share','unchanged_choice_share']
        bydate.append(dict(date=date,complete=rr[0]['complete'],steps=rr[0]['steps'],**{k:statistics.mean(r[k] for r in rr) for k in fields}))
    timing=dict(primary_dates=bydate,all4_date_macro={k:statistics.mean(r[k] for r in bydate) for k in ['controller_s','local_s','local_share','after64_share','unchanged_choice_share']},
        completed3_date_macro={k:statistics.mean(r[k] for r in bydate if r['complete']) for k in ['controller_s','local_s','local_share','after64_share','unchanged_choice_share']},
        boundary='Same recorded FRAME path;does not include trajectory changes,removable bookkeeping or online model overhead. Ratios calculated within runs then average repetitions and dates;incomplete date separately retained.')
    save('timing_opportunity.json',timing)
    D=read(R/'outputs/grid96_value_learning/design.json');byid={c['id']:c for c in D['cases']};features=D['learning']['features'];assert len(features)==15
    quality=read(R/'outputs/grid137_quality_mapping/quality_training_evidence_v2.json');allrows=[]
    for q in quality:
        c=byid[q['case_id']];s=read(R/c['source']);assert s['step']==c['step'] and s['state_hash']==c['state_hash'] and s['pre_extra_choice']==c['before_choice'] and s['chosen_pool_id']==c['after_choice']
        assert s['completion_budget']==c['completion_budget'] and all(math.isfinite(c['features'][f]) for f in features)
        truth={r['pool_id']:r for r in s['outcomes']};before=truth[c['before_choice']];after=truth[c['after_choice']]
        assert before['rho']==c['features']['incumbent_rho']
        gain=after['reward']-before['reward'];assert gain>=0
        assert (c['before_choice']==c['after_choice'])==q['physical_path_identical']
        allrows.append(dict(case_id=c['id'],date=c['calendar_date'],week=c['week'],cohort=c['cohort'],step=c['step'],pilot_known=c['pilot_known'],station=s['local_substation'],completion_budget=c['completion_budget'],
            features=c['features'],physical_path_identical=q['physical_path_identical'],quality_evidence_label=q['quality_evidence_label'],quality_outcome_known=q['quality_outcome_known'],
            cost_delta=q['skip_cost_minus_completion'],queries_saved=q['queries_saved'],instantaneous_reward_gain=gain,
            cost_direction=('same' if q['skip_cost_minus_completion']==0 else 'higher' if q['skip_cost_minus_completion']>0 else 'lower') if q['skip_cost_minus_completion'] is not None else 'not_comparable',
            skip_complete=q['skip_complete'],completion_complete=q['completion_complete']))
    changed=[r for r in allrows if not r['physical_path_identical']];assert len(changed)==17
    completed=[r for r in changed if r['cost_delta'] is not None];assert len(completed)==16
    save('public_feature_outcome_ledger.json',allrows)
    correlations=[]
    for f in features:
        correlations.append(dict(feature=f,n=16,spearman= corr([r['features'][f] for r in completed],[r['cost_delta'] for r in completed]),
            range=[min(r['features'][f] for r in completed),max(r['features'][f] for r in completed)]))
    save('all_feature_descriptive_correlations.json',dict(exploratory_only=True,rows=correlations,instantaneous_gain_vs_cost_delta=corr([r['instantaneous_reward_gain'] for r in completed],[r['cost_delta'] for r in completed]),
        limits='16 rows,correlated dates,15 inspected features,known outcomes. No significance,selection,causal or out-of-date predictive claim. Survival failure excluded only from undefined cost delta,retained in ledger.'))
    neighborhoods=[]
    for q in changed:
        ref=[r for r in allrows if not r['pilot_known'] and r['quality_outcome_known'] and r['date']!=q['date']];assert ref
        spans={f:max(r['features'][f] for r in ref)-min(r['features'][f] for r in ref) for f in features}
        bounds={f:[min(r['features'][f] for r in ref),max(r['features'][f] for r in ref)] for f in features}
        distances=[]
        for r in ref:
            components={f:((q['features'][f]-r['features'][f])/(spans[f] or 1.0))**2 for f in features}
            distances.append(dict(case_id=r['case_id'],date=r['date'],distance=math.fsum(components.values())/len(features),quality=r['quality_evidence_label'],cost_direction=r['cost_direction'],cost_delta=r['cost_delta'],physical_path_identical=r['physical_path_identical'],feature_contributions=components))
        distances.sort(key=lambda r:(r['distance'],r['case_id']))
        neighborhoods.append(dict(case_id=q['case_id'],date=q['date'],query_is_known_pilot=q['pilot_known'],query_quality=q['quality_evidence_label'],reference_cases=len(ref),
            public_reference_bounds=bounds,outside_reference_features=[f for f in features if not bounds[f][0]<=q['features'][f]<=bounds[f][1]],
            nearest3=distances[:3],all_distances=distances,descriptive_not_predictions=True))
    save('date_excluded_feature_neighbors.json',neighborhoods)
    duplicates=[]
    buckets=collections.defaultdict(list)
    for r in allrows:buckets[tuple(r['features'][f] for f in features)].append(r)
    for rr in buckets.values():
        if len(rr)>1:duplicates.append(dict(ids=[r['case_id'] for r in rr],quality=[r['quality_evidence_label'] for r in rr],cost_direction=[r['cost_direction'] for r in rr]))
    strata=[]
    for scope,rr in [('all56',allrows),('changed17',changed)]:
        for name,lo,hi in [('below_existing_safe',-math.inf,.9),('safe_to_overload',.9,1.0),('overload',1.0,math.inf)]:
            ss=[r for r in rr if lo<=r['features']['incumbent_rho']<hi]
            strata.append(dict(scope=scope,stratum=name,case_ids=[r['case_id'] for r in ss],quality=dict(collections.Counter(r['quality_evidence_label'] for r in ss)),cost_direction=dict(collections.Counter(r['cost_direction'] for r in ss))))
    public=dict(cases=56,changed=17,completed_changed=16,features=features,exact_feature_duplicate_groups=duplicates,
        changed_cost_counts=dict(collections.Counter(r['cost_direction'] for r in changed)),inherited_rho_strata=strata,
        failure_neighborhoods=[r for r in neighborhoods if r['query_quality'].startswith('FAIL_')],
        conclusion='Descriptive overlap does not establish predictability or impossibility;no trigger threshold/model selected. After-completion reward gain and branch outcome are audit-only.')
    save('public_information_summary.json',public)
    verify();save('review.json',dict(passed=True,runs=48,calls=len(calls),frame_runs=8,timing_dates=4,changed_cases=17,cost_comparable_changed=16,
        forecast_times_and_counts_recomputed=True,mutually_exclusive_components_match_old_audit=True,old15features_unchanged=True,all_date_excluded_neighbors=True,
        all15correlations_retained=True,source_hashes_unchanged=True,final_tests_absent=True,new_fits=0,new_model_forwards=0,new_physical=0,new_rules=0))
    save('complete.json',dict(passed=True,new_fits=0,new_physical=0,adopted=False,ready_to_scale=False));save('state.json',dict(stage='analysis_complete'));archive('complete')
    print(json.dumps(dict(timing=timing,changed_cost=public['changed_cost_counts'],failure_neighbors=[dict(id=r['case_id'],outside=r['outside_reference_features'],nearest=[(v['case_id'],v['quality'],v['cost_direction'],v['distance']) for v in r['nearest3']]) for r in public['failure_neighborhoods']],duplicates=duplicates),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
