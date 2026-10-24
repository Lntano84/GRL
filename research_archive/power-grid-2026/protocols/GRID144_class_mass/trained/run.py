"""Exactly one fit; only external class loss weights change relative to RF140."""
from support import *
def score_map(rows,scores):return {(r['meta']['call_id'],r['meta']['visible_count']):float(p) for r,p in zip(rows,scores)}
def prepare():
    import sklearn
    assert not O.exists();O.mkdir(parents=True);B.verify_trained();assert read(DIAG/'publication_verification.json')['passed']
    old=read(PREV/'design.json');fit=rowfile(PREV/'fit_rows.json.gz');cal=rowfile(PREV/'calibration_rows.json.gz');assert len(fit)==386 and len(cal)==198
    base,weights,receipt=planned_weights(fit);expected=read(DIAG/'fit_class_mass_diagnosis.json');assert receipt['stop_multiplier']==expected['planning_only_equal_mass_stop_multiplier'] and receipt['continue_multiplier']==expected['planning_only_equal_mass_continue_multiplier']
    design={k:old[k] for k in ['task','model','model_parameters','fit_wall_cap_s','fit_dates','calibration_dates','validation_dates','features','preprocessing','calibration','calibration_caveat','reference_scope','development_screen','scope']}
    design.update(id='GRID144_EQUAL_CLASS_MASS_RF',utc=stamp(),authorization='User continues GRID143 predeclared equal-class-mass draft;explicit notice before training.',
        intervention='Only external fit loss weights change versusRF140. Original18features+18missing flags;noRF142added6. Labels,dates,seed,forest/search/calibration unchanged.',new_fits=1,input_dimensions=36,fit_prefixes=386,calibration_prefixes=198,
        base_sample_weight=old['sample_weight'],sample_weight='Originalfit date/call/prefix base weights times total_base_mass/(2*base_class_mass),computed only from386fit rows. Preserve within-class relative weights;overall date/call totals change. No class-weight sweep;class_weight parameter remains null.',
        class_normalization=receipt,score='Weighted tree stop score;not calibrated natural event probability. Strict score>cutoff stops;ties continue.',
        primary='RF144_CALIBRATED',reference_methods=['FRAME','CAP128','CAP192','CAP256','CAP512','RF140_FIXED_0.5','RF140_CALIBRATED','RF142_FIXED_0.5','RF142_CALIBRATED','RF144_FIXED_0.5','RF144_CALIBRATED'],
        order='Freeze protocol/code and push;one fit and original4-date callwise calibration;freeze/push model,weights,preprocessing,cutoff before new validation scoring.',
        auxiliary='All parts/date continue-positive AP,prevalence,continue and stop recall,same-query-depth/entry-safe pair AUC.4 mechanical leave-one-calibration-date diagnostics,never threshold/model selection.',
        development_exposure='All20dates and validation data already exposed. Repeat evaluation is development,not fresh independent validation.',
        prohibited=['refit','parameter/seed/class-weight/feature/threshold grid','validation-selected cutoff/model','label/cost/margin change','new physical/scenario','final8tests','RL/NN','paper edit','automatic scale'])
    save('design.json',design);save('planned_weighting.json',receipt)
    save('planned_new_fit_sample_weights.json',[dict(sample_id=r['meta']['sample_id'],date=r['meta']['date'],call_id=r['meta']['call_id'],weight=float(w)) for r,w in zip(fit,weights)])
    for n in ['fit_rows.json.gz','calibration_rows.json.gz']:shutil.copy2(PREV/n,O/n)
    assert public_checks(fit+cal)==584
    save('preflight.json',dict(passed=True,old18_features_labels_rows_bytes_same=True,public_suffix_checks=584,class_normalization_fit_only=True,feature_dimensions=36,model_parameters_same=True,validation_scored=False))
    sources=[PREV/n for n in ['design.json','model.pkl','model_export.json','trained_freeze.json','preprocessing.json','fit_sample_weights.json','fit_rows.json.gz','calibration_rows.json.gz','train_scores.npz','validation_scores.npz','all_search_replays.json','summary.json','cutoff.json','review.json','final_verification.json']]
    sources += [OTHER/n for n in ['all_search_replays.json','summary.json','cutoff.json','train_scores.npz','validation_scores.npz','review.json','trained_freeze.json']]
    sources += [DIAG/n for n in ['下一轮只改变类别损失的训练草案.md','fit_class_mass_diagnosis.json','ranking_metrics.json','conditional_ranking_summary.json','all_calibration_date_folds.json','review.json','publication_verification.json']]
    sources += [R/n for n in ['work/grid140_global_compression/common.py','work/grid140_global_compression/run.py','work/grid143_continue_risk/run.py']]
    sources += [O/n for n in ['design.json','fit_rows.json.gz','calibration_rows.json.gz','planned_weighting.json','planned_new_fit_sample_weights.json']]
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in sources});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});save('runtime.json',dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__));verify();save('state.json',dict(stage='frozen_before_one_fit'));archive('frozen');print('GRID144_FROZEN',receipt['continue_multiplier']/receipt['stop_multiplier'])

def fit_calibrate():
    verify();assert not (O/'fit_started.json').exists();D=read(O/'design.json');fit=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');names=D['features']
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in names] for r in fit],float)
    medians=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(len(names))]
    x=transform(fit,names,medians);xc=transform(cal,names,medians);y=np.asarray([r['target']['stop_label'] for r in fit],int);assert set(y)=={0,1}
    dates=sorted({r['meta']['date'] for r in fit});callsbydate={d:len({r['meta']['call_id'] for r in fit if r['meta']['date']==d}) for d in dates};counts=collections.Counter(r['meta']['call_id'] for r in fit)
    weights=np.asarray([1/(len(dates)*callsbydate[r['meta']['date']]*counts[r['meta']['call_id']]) for r in fit]);weights*=len(fit)/weights.sum()
    for d in dates:assert abs(sum(w for r,w in zip(fit,weights) if r['meta']['date']==d)-len(fit)/len(dates))<1e-10
    base_weights,new_weights,norm=planned_weights(fit);assert np.array_equal(weights,base_weights);weights=new_weights
    planned=read(O/'planned_new_fit_sample_weights.json');assert np.array_equal(weights,np.asarray([r['weight'] for r in planned]))
    save('fit_base_sample_weights.json',read(PREV/'fit_sample_weights.json'))
    norm['fit_dates_base_totals']={d:float(sum(base_weights[i] for i,r in enumerate(fit) if r['meta']['date']==d)) for d in dates}
    norm['fit_dates_new_totals']={d:float(sum(weights[i] for i,r in enumerate(fit) if r['meta']['date']==d)) for d in dates}
    save('weighting_receipt.json',norm)
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
    names_to_freeze=['weighting_receipt.json','fit_base_sample_weights.json','model.pkl','model_export.json','preprocessing.json','fit_sample_weights.json','fit_receipt.json','train_scores.npz','calibration_history.json','calibration_fixed_calls.json','calibration_final_calls.json','cutoff.json','training_node_metrics.json']
    save('trained_freeze.json',{n:sha(O/n) for n in names_to_freeze});save('state.json',dict(stage='trained_calibrated_before_validation'));verify_trained();archive('trained');print('GRID144_TRAINED',elapsed,'cutoff',tau,'iterations',len(history))


def validate():
    verify_trained();publication=read(O/'prevalidation_publication.json');assert publication['passed'] and publication['trained_freeze_sha256']==sha(O/'trained_freeze.json')
    assert not (O/'validation_started.json').exists();save('validation_started.json',dict(utc=stamp(),trained_freeze_sha256=sha(O/'trained_freeze.json'),published_sha=publication['local_sha'],fits_so_far=1))
    D=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json');val=eligible_data('validation');assert len(val)==142 and public_checks(val)==142
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    xv=transform(val,pre['features'],pre['medians']);pv=est.predict_proba(xv)[:,list(est.classes_).index(1)];assert max(abs(exported_scores(read(O/'model_export.json'),xv)-pv))<1e-10
    np.savez_compressed(O/'validation_scores.npz',scores=pv);save('validation_node_metrics.json',dict(fixed=probability_metrics(val,pv,.5),calibrated=probability_metrics(val,pv,cut['primary_threshold'])))
    fit=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');z=np.load(O/'train_scores.npz');allmap=score_map(fit,z['fit']);allmap.update(score_map(cal,z['calibration']));allmap.update(score_map(val,pv))
    old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(OTHER/'all_search_replays.json')};allrows=[];sums=[]
    for part,callrefs in [('fit',refs('train',set(D['fit_dates']))),('calibration',refs('train',set(D['calibration_dates']))),('validation',refs('validation'))]:
        for name in D['reference_methods']:
            rr=[]
            for ref in callrefs:
                if name.startswith('RF144'):r=replay(ref,'MODEL',.5 if name.endswith('0.5') else cut['primary_threshold'],allmap)
                else:r=dict(old[part,name,ref['call_id']])
                rr.append(dict(r,method=name,evaluation_partition=part))
            allrows+=rr;sums.append(dict(evaluation_partition=part,method=name,**summarize(rr)))
    save('all_search_replays.json',allrows);save('summary.json',sums);metric=lambda name:next(r for r in sums if r['evaluation_partition']=='validation' and r['method']==name)
    a=metric('RF144_CALIBRATED');b=metric('CAP256');previous=metric('RF140_CALIBRATED');batch=metric('RF142_CALIBRATED');bd={r['date']:r for r in b['by_date']};pd={r['date']:r for r in previous['by_date']}
    dd=[dict(date=r['date'],new_model_queries=r['queries'],original_model_queries=pd[r['date']]['queries'],cap256_queries=bd[r['date']]['queries'],new_action_differences=r['action_differences'],not_more_than_cap256=r['queries']<=bd[r['date']]['queries'],strictly_fewer_than_cap256=r['queries']<bd[r['date']]['queries']) for r in a['by_date']]
    qualifies=a['action_differences']==0 and a['date_macro_query_reduction']>b['date_macro_query_reduction'] and all(r['not_more_than_cap256'] for r in dd) and any(r['strictly_fewer_than_cap256'] for r in dd)
    save('decision.json',dict(decision='PASS_DEVELOPMENT_COMPRESSION_SCREEN' if qualifies else 'NO_STABLE_INCREMENT_OVER_STRONG_SIMPLE_CONTROL',primary='RF144_CALIBRATED',passes_development_screen=qualifies,ready_for_frame_state_coverage_check=qualifies,ready_to_scale=False,adopted=False,
        action_differences=a['action_differences'],query_reduction_macro=a['date_macro_query_reduction'],old_model_query_reduction_macro=previous['date_macro_query_reduction'],batch_model_query_reduction_macro=batch['date_macro_query_reduction'],cap256_query_reduction_macro=b['date_macro_query_reduction'],per_date=dd,new_fits=1,old_fits=0,new_physical=0,final_tests_absent=True,
        scope='Known development FULL-teacher states;not independent test,weekly economics or real compute time.'))
    aux=diagnostic_api();rows=fit+cal+val;parts=['fit']*386+['calibration']*198+['validation']*142;pp=np.concatenate([z['fit'],z['calibration'],pv]);risk=[];date_risk=[];conditional=[]
    for part in ['fit','calibration','validation']:
        ii=[i for i,p in enumerate(parts) if p==part];risk.append(dict(model='RF144',partition=part,**aux.rank_metrics([rows[i] for i in ii],pp[ii],cut['primary_threshold'])))
        for date in sorted({rows[i]['meta']['date'] for i in ii}):
            jj=[i for i in ii if rows[i]['meta']['date']==date];date_risk.append(dict(model='RF144',partition=part,date=date,**aux.rank_metrics([rows[i] for i in jj],pp[jj],cut['primary_threshold'])))
        groups=collections.defaultdict(list)
        for i in ii:groups[rows[i]['meta']['visible_count'],rows[i]['features']['prefix_safe']].append(i)
        for (n,safe),jj in groups.items():
            pos=[float(pp[i]) for i in jj if rows[i]['target']['stop_label']];neg=[float(pp[i]) for i in jj if not rows[i]['target']['stop_label']];count=len(pos)*len(neg);correct=sum(p>q for p in pos for q in neg);ties=sum(p==q for p in pos for q in neg)
            conditional.append(dict(partition=part,visible_count=n,prefix_safe=safe,pairs=count,correct=correct,ties=ties,auc=(correct+.5*ties)/count if count else None))
    save('continue_risk_metrics.json',risk);save('continue_risk_per_date.json',date_risk);save('same_depth_branch_ranking.json',conditional)
    folds=[];histories=[];omitted_calls=[]
    for omit in D['calibration_dates']:
        dates=set(D['calibration_dates'])-{omit};donors={k:v for k,v in allmap.items() if any(r['meta']['call_id']==k[0] and r['meta']['date'] in dates for r in cal)};tests={k:v for k,v in allmap.items() if any(r['meta']['call_id']==k[0] and r['meta']['date']==omit for r in cal)}
        tau,h,donor_rr=aux.calibrate(refs('train',dates),donors);rr=[replay(r,'MODEL',tau,tests) for r in refs('train',{omit})]
        folds.append(dict(omitted_date=omit,donor_dates=sorted(dates),diagnostic_cutoff=tau,donor_summary=summarize(donor_rr),heldout_summary=summarize(rr),not_adopted=True));histories.append(dict(omitted_date=omit,history=h));omitted_calls += [dict(r,omitted_date=omit) for r in rr]
    save('all_calibration_date_folds.json',folds);save('all_calibration_date_histories.json',histories);save('all_omitted_calibration_call_replays.json',omitted_calls)
    verify_trained();save('state.json',dict(stage='validated_pending_audit'));archive('validated');print(json.dumps(read(O/'decision.json'),ensure_ascii=False))
if __name__=='__main__':{'prepare':prepare,'fit':fit_calibrate,'validate':validate}[sys.argv[1]]()
