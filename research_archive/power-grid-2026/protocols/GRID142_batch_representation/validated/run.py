"""One fixed forest fit; only input representation changes from GRID140."""
from base import *
def rowfile(n):
    with gzip.open(O/n,'rt',encoding='utf8') as f:return json.load(f)
def score_map(rows,scores):return {(r['meta']['call_id'],r['meta']['visible_count']):float(p) for r,p in zip(rows,scores)}
def prepare():
    import sklearn
    assert not O.exists();O.mkdir(parents=True);B.verify_trained();assert read(PREV/'review.json')['passed'];assert read(R/'outputs/grid141_feedback_diagnosis/publication_verification.json')['passed']
    old=read(PREV/'design.json');train=eligible_data('train');fit=[r for r in train if r['meta']['date'] in old['fit_dates']];cal=[r for r in train if r['meta']['date'] in old['calibration_dates']]
    assert len(fit)==386 and len(cal)==198;names=old['features']+ADDED;assert list(fit[0]['features'])==names and len(names)==24
    design={k:old[k] for k in ['task','model','model_parameters','fit_wall_cap_s','fit_dates','calibration_dates','validation_dates','sample_weight','score','calibration','calibration_caveat','reference_scope','development_screen','scope']}
    design.update(id='GRID142_BATCH_REPRESENTATION_RF',utc=stamp(),authorization='User continues after GRID141 six-feature draft and explicit pre-training notice.',
        intervention='Only append the exact six draft batch features;old18 inputs,labels,date split,weights,forest settings,seed,search and calibration unchanged.',
        new_fits=1,fit_prefixes=386,calibration_prefixes=198,features=names,added_features=ADDED,input_dimensions=48,
        feature_semantics='Last64 and previous64 are entirely revealed. At128 both are halves of the initial prefix. Best rewards use original float32/admissibility;rho summaries use positive finite returned rho;empty valid sets yield null. IQR=q75-q25,linear quantiles.',
        preprocessing='Fit12-only per-feature median;all-missing median0;append24missing flags. No scaling/selection/IDs/future fields.',
        reference_methods=['FRAME','CAP128','CAP192','CAP256','CAP512','RF140_FIXED_0.5','RF140_CALIBRATED','RF142_FIXED_0.5','RF142_CALIBRATED'],
        primary='RF142_CALIBRATED',order='Freeze protocol/code and push;one fit and four-date callwise calibration;freeze and push model/preprocessing/cutoff before new validation scoring.',
        evaluation='Same161fit/82calibration/62development validation complete calls,all9methods. Both fixed0.5 and calibrated readouts retained;primary fixed in advance.',
        development_exposure='Features motivated by already exposed GRID141 development data. Reusing four validation dates is development evaluation,not a fresh independent test.',
        prohibited=['second fit','parameter/seed/feature/class-weight/threshold search','validation-selected model/cutoff','new physics/scenarios/labels','RL/NN','final8tests','paper edit','automatic scale'])
    save('design.json',design)
    for name,rows in [('fit_rows.json.gz',fit),('calibration_rows.json.gz',cal)]:
        with gzip.open(O/name,'wt',encoding='utf8') as f:json.dump(rows,f,ensure_ascii=False,allow_nan=False)
    oldfit=read(PREV/'fit_sample_weights.json');assert [r['meta']['sample_id'] for r in fit]==[r['sample_id'] for r in oldfit]
    assert [r['target'] for r in fit+cal]==[r['target'] for r in B.eligible_data('train') if r['meta']['date'] in old['fit_dates']]+[r['target'] for r in B.eligible_data('train') if r['meta']['date'] in old['calibration_dates']]
    checks=public_checks(fit+cal);assert checks==584
    save('preflight.json',dict(passed=True,public_prefix_and_suffix_checks=checks,old18_features_and_labels_unchanged=True,fit_order_same=True,features=24,model_dimensions=48,validation_scored=False))
    sources=[PREV/n for n in ['design.json','model.pkl','model_export.json','trained_freeze.json','preprocessing.json','fit_sample_weights.json','fit_rows.json.gz','calibration_rows.json.gz','train_scores.npz','validation_scores.npz','all_search_replays.json','summary.json','cutoff.json','review.json','final_verification.json']]
    sources += [R/n for n in ['work/grid140_global_compression/common.py','work/grid140_global_compression/run.py','outputs/grid141_feedback_diagnosis/全量核查后的结论与训练草案.md','outputs/grid141_feedback_diagnosis/public_supplemental_inputs.jsonl.gz','outputs/grid141_feedback_diagnosis/publication_verification.json']]
    sources += [O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz']]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('runtime.json',dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__))
    verify();save('state.json',dict(stage='frozen_before_one_fit'));archive('frozen');print('GRID142_FROZEN',checks)

def fit_calibrate():
    verify();assert not (O/'fit_started.json').exists();D=read(O/'design.json');fit=rowfile('fit_rows.json.gz');cal=rowfile('calibration_rows.json.gz');names=D['features']
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in fit],float)
    medians=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(len(names))]
    x=transform(fit,names,medians);xc=transform(cal,names,medians);y=np.asarray([r['target']['stop_label'] for r in fit],int);assert set(y)=={0,1}
    dates=sorted({r['meta']['date'] for r in fit});callsbydate={d:len({r['meta']['call_id'] for r in fit if r['meta']['date']==d}) for d in dates};counts=collections.Counter(r['meta']['call_id'] for r in fit)
    weights=np.asarray([1/(len(dates)*callsbydate[r['meta']['date']]*counts[r['meta']['call_id']]) for r in fit]);weights*=len(fit)/weights.sum()
    for d in dates:assert abs(sum(w for r,w in zip(fit,weights) if r['meta']['date']==d)-len(fit)/len(dates))<1e-10
    save('preprocessing.json',dict(features=names,medians=medians,append_missing_flags=True,dimensions=x.shape[1]));save('fit_sample_weights.json',[dict(sample_id=r['meta']['sample_id'],date=r['meta']['date'],call_id=r['meta']['call_id'],weight=float(w)) for r,w in zip(fit,weights)])
    from sklearn.ensemble import RandomForestClassifier
    est=RandomForestClassifier(**D['model_parameters']);save('fit_started.json',dict(utc=stamp(),fits_permitted=1,refit_forbidden=True));t=time.perf_counter();est.fit(x,y,sample_weight=weights);elapsed=time.perf_counter()-t
    with (O/'model.pkl').open('wb') as f:pickle.dump(est,f,protocol=5)
    save('model_export.json',export_model(est));save('fit_receipt.json',dict(fits=1,elapsed_s=elapsed,rows=len(fit),class_counts=dict(collections.Counter(int(v) for v in y)),trees=len(est.estimators_),features=x.shape[1],validation_scored=False))
    assert elapsed<=D['fit_wall_cap_s'],'Preserve fitted artifact,no refit after time cap'
    pf=est.predict_proba(x)[:,list(est.classes_).index(1)];pc=est.predict_proba(xc)[:,list(est.classes_).index(1)];model=read(O/'model_export.json')
    assert np.max(np.abs(exported_scores(model,x)-pf))<1e-10 and np.max(np.abs(exported_scores(model,xc)-pc))<1e-10
    np.savez_compressed(O/'train_scores.npz',fit=pf,calibration=pc);calmap=score_map(cal,pc);calrefs=refs('train',set(D['calibration_dates']));history=[];tau=.5;initial=None
    for iteration in range(len(cal)+2):
        rr=[replay(r,'MODEL',tau,calmap) for r in calrefs];bad=[r for r in rr if r['bad_learned_stop']]
        if initial is None:initial=rr
        history.append(dict(iteration=iteration,threshold=tau,summary=summarize(rr),bad_stops=[dict(call_id=r['call_id'],score=r['stopped_score'],selected=r['selected'],original=r['original_selected'],global_queries=r['global_queries'],queries=r['queries'],original_queries=r['original_queries']) for r in bad]))
        if not bad:break
        new=max(r['stopped_score'] for r in bad);assert new>tau and new<=1;tau=new
    else:raise AssertionError('Finite monotone calibration did not terminate;no refit allowed')
    assert summarize(rr)['action_differences']==0
    save('calibration_history.json',history);save('calibration_fixed_calls.json',initial);save('calibration_final_calls.json',rr)
    save('cutoff.json',dict(primary_threshold=tau,fixed_threshold=.5,strict_greater=True,iterations=len(history),calibration_calls=len(calrefs),dates=D['calibration_dates'],validation_scored=False,empirical_only=True))
    save('training_node_metrics.json',dict(fit_fixed=probability_metrics(fit,pf,.5),fit_calibrated=probability_metrics(fit,pf,tau),calibration_fixed=probability_metrics(cal,pc,.5),calibration_calibrated=probability_metrics(cal,pc,tau)))
    names_to_freeze=['model.pkl','model_export.json','preprocessing.json','fit_sample_weights.json','fit_receipt.json','train_scores.npz','calibration_history.json','calibration_fixed_calls.json','calibration_final_calls.json','cutoff.json','training_node_metrics.json']
    save('trained_freeze.json',{n:sha(O/n) for n in names_to_freeze});save('state.json',dict(stage='trained_calibrated_before_validation'));verify_trained();archive('trained');print('GRID142_TRAINED',elapsed,'cutoff',tau,'iterations',len(history))


def validate():
    verify_trained();publication=read(O/'prevalidation_publication.json');assert publication['passed'] and publication['trained_freeze_sha256']==sha(O/'trained_freeze.json')
    assert not (O/'validation_started.json').exists();save('validation_started.json',dict(utc=stamp(),trained_freeze_sha256=sha(O/'trained_freeze.json'),published_sha=publication['local_sha'],fits_so_far=1))
    D=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json');val=eligible_data('validation');assert len(val)==142;assert public_checks(val)==142
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    xx=transform(val,pre['features'],pre['medians']);pv=est.predict_proba(xx)[:,list(est.classes_).index(1)];assert max(abs(exported_scores(read(O/'model_export.json'),xx)-pv))<1e-10
    np.savez_compressed(O/'validation_scores.npz',scores=pv);save('validation_node_metrics.json',dict(fixed=probability_metrics(val,pv,.5),calibrated=probability_metrics(val,pv,cut['primary_threshold'])))
    fit=rowfile('fit_rows.json.gz');cal=rowfile('calibration_rows.json.gz');z=np.load(O/'train_scores.npz');allmap=score_map(fit,z['fit']);allmap.update(score_map(cal,z['calibration']));allmap.update(score_map(val,pv))
    old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(PREV/'all_search_replays.json')};allrows=[];sums=[]
    for part,callrefs in [('fit',refs('train',set(D['fit_dates']))),('calibration',refs('train',set(D['calibration_dates']))),('validation',refs('validation'))]:
        for name in D['reference_methods']:
            rr=[]
            for ref in callrefs:
                if name in ['RF142_FIXED_0.5','RF142_CALIBRATED']:r=replay(ref,'MODEL',.5 if name.endswith('0.5') else cut['primary_threshold'],allmap)
                else:
                    original_name=name.replace('RF140_','RF_');r=dict(old[part,original_name,ref['call_id']])
                rr.append(dict(r,method=name,evaluation_partition=part))
            allrows+=rr;sums.append(dict(evaluation_partition=part,method=name,**summarize(rr)))
    save('all_search_replays.json',allrows);save('summary.json',sums)
    s=lambda name:next(r for r in sums if r['evaluation_partition']=='validation' and r['method']==name)
    a=s('RF142_CALIBRATED');b=s('CAP256');previous=s('RF140_CALIBRATED');bd={r['date']:r for r in b['by_date']};pd={r['date']:r for r in previous['by_date']}
    dd=[dict(date=r['date'],new_model_queries=r['queries'],old_model_queries=pd[r['date']]['queries'],cap256_queries=bd[r['date']]['queries'],new_action_differences=r['action_differences'],not_more_than_cap256=r['queries']<=bd[r['date']]['queries'],strictly_fewer_than_cap256=r['queries']<bd[r['date']]['queries']) for r in a['by_date']]
    qualifies=a['action_differences']==0 and a['date_macro_query_reduction']>b['date_macro_query_reduction'] and all(r['not_more_than_cap256'] for r in dd) and any(r['strictly_fewer_than_cap256'] for r in dd)
    save('decision.json',dict(decision='PASS_DEVELOPMENT_COMPRESSION_SCREEN' if qualifies else 'NO_STABLE_INCREMENT_OVER_STRONG_SIMPLE_CONTROL',primary='RF142_CALIBRATED',passes_development_screen=qualifies,ready_for_frame_state_coverage_check=qualifies,ready_to_scale=False,adopted=False,
        action_differences=a['action_differences'],query_reduction_macro=a['date_macro_query_reduction'],old_model_query_reduction_macro=previous['date_macro_query_reduction'],cap256_query_reduction_macro=b['date_macro_query_reduction'],per_date=dd,fits=1,new_physical=0,final_tests_absent=True,
        scope='Already exposed development FULL-teacher states;not independent test,weekly quality or measured time.'))
    verify_trained();save('state.json',dict(stage='validated_pending_audit'));archive('validated');print(json.dumps(read(O/'decision.json'),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'fit':fit_calibrate,'validate':validate}[sys.argv[1]]()
