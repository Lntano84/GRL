"""Exactly one inherited shallow forest; call-wise calibration before validation scoring."""
from common import *
def prepare():
    import sklearn,sklearn.ensemble._forest,sklearn.tree._classes
    assert not O.exists();O.mkdir(parents=True);P.verify();assert read(SOURCE/'final_verification.json')['passed']
    prior=read(R/'outputs/grid130_calibrated_gate/design.json');forest=read(R/'outputs/grid133_nonlinear_gate/design.json');fitdates=prior['fit_dates'];caldates=prior['calibration_dates'];assert len(fitdates)==12 and len(caldates)==4 and not set(fitdates)&set(caldates)
    train=eligible_data('train');fit=[r for r in train if r['meta']['date'] in fitdates];cal=[r for r in train if r['meta']['date'] in caldates]
    assert len(train)==584 and len(fit)==386 and len(cal)==198;names=list(fit[0]['features']);assert len(names)==18
    params=dict(forest['forest_parameters'],**forest['classifier_defaults'])
    save('design.json',dict(id='GRID140_GLOBAL_COMPRESSION_RF',utc=stamp(),authorization='User says continue after explicit pre-training notice;one simple-model fit,offline replay only.',
        task='Earlier STOP on originalFRAME global path then unchanged conditional completion. Original forced stops stay;no admissible incumbent => continue. Action equality and candidate query saving target fromGRID139.',
        model='Single inherited RandomForestClassifier,not RL',model_parameters=params,new_fits=1,fit_wall_cap_s=120,
        model_provenance='Exact forest structure/defaults/seed fromGRID133;no new model/seed/parameter grid.',fit_dates=fitdates,calibration_dates=caldates,fit_prefixes=386,calibration_prefixes=198,
        validation_dates=['2037-03-02','2037-05-04','2037-08-03','2037-10-19'],features=names,input_dimensions=36,
        preprocessing='Fit12-only per-feature median,all-missing median0;append one missing flag per original18features. No scaling/feature selection/IDs/future fields.',
        sample_weight='Equal fit date total;within date equal physical call total;within call equal prefix weight. Normalize to mean1;no class weighting beyond inherited null.',
        score='P(stop_label=True),strict greater threshold. Fixed0.5 readout auxiliary;sequential-calibrated readout primary,never choose using validation.',
        calibration='Start threshold0.5. Replay all original4cal-date complete search calls. If learned STOP changes original action or fails strictly to reduce total queries,raise threshold to max score among currently bad stops;ties continue. Repeat whole calls until no bad learned stop. Only increasing thresholds;no query-optimization grid or validation use. If cutoff reaches1,baseline fallback.',
        calibration_caveat='Empirical zero call mismatch on4known calibration dates,not formal risk or safety guarantee. Bad unvisited rows may remain;evaluate whole sequential calls.',
        order='Freeze protocol/source;one fit;export/manual-probability audit;freeze weights+preprocessing+cutoff+calibration receipt and push;then new validation model scoring.',
        reference_methods=['FRAME','CAP128','CAP192','CAP256','CAP512','RF_FIXED_0.5','RF_CALIBRATED'],
        reference_scope='All caps retain original forced stops,incumbent mask and same conditional completion,asGRID139. All retained,not budget selection.',
        development_screen='Primary calibrated gate:zero action differences all62validation calls;strictly higher date-macro query reduction thanCAP256;no larger query total on any4date and strictly fewer on at least1. Offline-only qualification for next FRAME-state coverage check,not scale/closedloop readiness.',
        evaluation='All train calls descriptive,fit and calibration separately;all4known validation dates both scenario replicas. Report all readouts,all simple controls,node ROC/AP/majority and whole-call actions/queries/date results.',
        scope='FULL-teacher physical states,not actualFRAME trajectory;query count not whole-control time or economic cost. Final8test not generated.',
        prohibited=['refit','hyperparameter/seed/feature/threshold grid','validation-selected cutoff/model','new physical/scenario/labels','RL/NN expansion','final8test','paper edit','automatic scale']))
    for name,rows in [('fit_rows.json.gz',fit),('calibration_rows.json.gz',cal)]:
        with gzip.open(O/name,'wt',encoding='utf8') as f:json.dump(rows,f,ensure_ascii=False,allow_nan=False)
    paths=[R/p for p in ['outputs/grid130_calibrated_gate/design.json','outputs/grid133_nonlinear_gate/design.json','outputs/grid139_global_feasibility/final_verification.json','outputs/grid139_global_feasibility/source_hashes.json',
        'outputs/grid139_global_feasibility/public_inputs.jsonl.gz','outputs/grid139_global_feasibility/metadata.jsonl.gz','outputs/grid139_global_feasibility/compression_task_targets.jsonl.gz','outputs/grid139_global_feasibility/call_ledger.json',
        'outputs/grid139_global_feasibility/compression_baseline_calls.json','outputs/grid139_global_feasibility/compression_baseline_summary.json','work/grid139_global_feasibility/run.py']]
    paths += [O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz']]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')})
    save('runtime.json',dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,forest_source=sha(Path(sklearn.ensemble._forest.__file__)),tree_source=sha(Path(sklearn.tree._classes.__file__))))
    save('state.json',dict(stage='frozen_before_one_fit'));verify();archive('frozen');print('GRID140_FROZEN',len(fit),len(cal))
def rowfile(n):
    with gzip.open(O/n,'rt',encoding='utf8') as f:return json.load(f)
def score_map(rows,scores):return {(r['meta']['call_id'],r['meta']['visible_count']):float(p) for r,p in zip(rows,scores)}
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
    save('trained_freeze.json',{n:sha(O/n) for n in names_to_freeze});save('state.json',dict(stage='trained_calibrated_before_validation'));verify_trained();archive('trained');print('GRID140_TRAINED',elapsed,'cutoff',tau,'iterations',len(history))
def validate():
    verify_trained();assert not (O/'validation_started.json').exists();save('validation_started.json',dict(utc=stamp(),trained_freeze_sha256=sha(O/'trained_freeze.json'),fits_so_far=1));D=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json');rows=eligible_data('validation');assert len(rows)==142
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    xv=transform(rows,pre['features'],pre['medians']);pv=est.predict_proba(xv)[:,list(est.classes_).index(1)];assert np.max(np.abs(exported_scores(read(O/'model_export.json'),xv)-pv))<1e-10
    np.savez_compressed(O/'validation_scores.npz',scores=pv);save('validation_node_metrics.json',dict(fixed=probability_metrics(rows,pv,.5),calibrated=probability_metrics(rows,pv,cut['primary_threshold'])))
    train=eligible_data('train');fit=rowfile('fit_rows.json.gz');cal=rowfile('calibration_rows.json.gz');z=np.load(O/'train_scores.npz');allmap=score_map(fit,z['fit']);allmap.update(score_map(cal,z['calibration']));allmap.update(score_map(rows,pv))
    allrows=[];summaries=[];old={(r['call_id'],r['cap']):r for r in read(SOURCE/'compression_baseline_calls.json')}
    for partition,callrefs in [('fit',refs('train',set(D['fit_dates']))),('calibration',refs('train',set(D['calibration_dates']))),('validation',refs('validation'))]:
        for name,mode,tau in [('FRAME','FRAME',None),('CAP128','CAP128',None),('CAP192','CAP192',None),('CAP256','CAP256',None),('CAP512','CAP512',None),('RF_FIXED_0.5','MODEL',.5),('RF_CALIBRATED','MODEL',cut['primary_threshold'])]:
            rr=[]
            for ref in callrefs:
                r=replay(ref,mode,tau,allmap)
                if mode.startswith('CAP'):
                    original=old[(r['call_id'],int(mode[3:]))];assert r['queries']==original['queries'] and r['selected']==original['selected']
                rr.append(dict(r,method=name,evaluation_partition=partition))
            allrows+=rr;summaries.append(dict(evaluation_partition=partition,method=name,**summarize(rr)))
    save('all_search_replays.json',allrows);save('summary.json',summaries)
    a=next(r for r in summaries if r['evaluation_partition']=='validation' and r['method']=='RF_CALIBRATED');b=next(r for r in summaries if r['evaluation_partition']=='validation' and r['method']=='CAP256');byb={r['date']:r for r in b['by_date']}
    per_date=[dict(date=r['date'],model_queries=r['queries'],cap256_queries=byb[r['date']]['queries'],model_action_differences=r['action_differences'],model_strictly_fewer=r['queries']<byb[r['date']]['queries'],model_not_more=r['queries']<=byb[r['date']]['queries']) for r in a['by_date']]
    qualifies=a['action_differences']==0 and a['date_macro_query_reduction']>b['date_macro_query_reduction'] and all(r['model_not_more'] for r in per_date) and any(r['model_strictly_fewer'] for r in per_date)
    save('decision.json',dict(decision='PASS_OFFLINE_COMPRESSION_SCREEN' if qualifies else 'NO_STABLE_INCREMENT_OVER_STRONG_SIMPLE_CONTROL',primary='RF_CALIBRATED',passes_offline_screen=qualifies,ready_for_frame_state_coverage_check=qualifies,
        ready_to_scale=False,adopted=False,validation_action_differences=a['action_differences'],date_macro_query_reduction=a['date_macro_query_reduction'],cap256_date_macro_query_reduction=b['date_macro_query_reduction'],per_date=per_date,
        scope='Known development FULL-teacher states;not wholeweek quality or actual timing. No model chosen based on validation.',fits=1,new_physical=0,final_tests_absent=True))
    verify_trained();save('state.json',dict(stage='validated_pending_audit'));archive('validated');print(json.dumps(dict(cutoff=cut,decision=read(O/'decision.json'),summaries=[{k:v for k,v in r.items() if k not in ['by_date','by_scenario']} for r in summaries]),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'fit':fit_calibrate,'validate':validate}[sys.argv[1]]()
