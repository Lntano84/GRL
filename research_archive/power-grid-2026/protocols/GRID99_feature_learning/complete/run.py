from common import *
import joblib,gc,traceback
def main():
    verify_freeze();assert not (OUT/'finished.json').exists()
    d=read(OUT/'design.json');lookup={r['case_id']:r for r in read(OUT/'dataset.json')};fits=[];predictions=[]
    for fi,f in enumerate(d['folds']):
        models=list(d['models']);models=models[fi%3:]+models[:fi%3]
        arms=list(d['feature_sets']);arms=arms if fi%2==0 else arms[::-1]
        for name in models:
            for arm in arms:
                key=f'{fi:02d}__{arm}__{name}';file=OUT/'folds'/(key+'.json')
                if file.exists():
                    saved=read(file);assert saved['complete'] and saved['freeze_sha256']==sha(OUT/'freeze.json');fits.append(saved['fit']);predictions.extend(saved['predictions']);continue
                features=d['feature_sets'][arm];x,y=arrays(lookup,f['train_ids'],features);xt,yt=arrays(lookup,f['test_ids'],features);assert set(y)=={0,1}
                t=time.perf_counter();m=estimator(name,d);m.fit(x,y);fit_s=time.perf_counter()-t
                if name=='LOGISTIC':assert np.max(m['model'].n_iter_)<2000
                t=time.perf_counter();p=m.predict_proba(xt)[:,list(m.classes_).index(1)];first_s=time.perf_counter()-t
                warm=[]
                for _ in range(3):
                    t=time.perf_counter();again=m.predict_proba(xt)[:,list(m.classes_).index(1)];warm.append(time.perf_counter()-t);assert np.allclose(p,again,atol=1e-6,rtol=0)
                single=[];ps=[]
                for row in xt:
                    t=time.perf_counter();ps.append(float(m.predict_proba(row[None,:])[0,list(m.classes_).index(1)]));single.append(time.perf_counter()-t)
                assert np.allclose(p,ps,atol=1e-5,rtol=0)
                if name=='LOGISTIC':export=dict(mean=m['scale'].mean_.tolist(),scale=m['scale'].scale_.tolist(),coef=m['model'].coef_[0].tolist(),intercept=float(m['model'].intercept_[0]))
                elif name=='FOREST64':export=dict(trees=[dict(left=e.tree_.children_left.tolist(),right=e.tree_.children_right.tolist(),feature=e.tree_.feature.tolist(),threshold=e.tree_.threshold.tolist(),value=e.tree_.value[:,0,:].tolist()) for e in m.estimators_])
                else:export=None
                if name!='TABPFN_V2':(OUT/'folds').mkdir(exist_ok=True);joblib.dump(m,OUT/'folds'/(key+'.joblib'))
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                fit=dict(key=key,arm=arm,model=name,scheme=f['scheme'],fold=f['fold'],features=features,train_ids=f['train_ids'],test_ids=f['test_ids'],fit_or_context_seconds=fit_s,first_batch_seconds=first_s,warm_batch_seconds=warm,single_row_seconds=single,export=export)
                pp=[]
                for i,prob,one in zip(f['test_ids'],p,ps):
                    r=lookup[i];pp.append(dict(case_id=i,arm=arm,model=name,scheme=f['scheme'],fold=f['fold'],fit_key=key,y=r['y_same_path'],p_same_path=float(prob),single_p_same_path=float(one),pilot_known=r['pilot_known'],calendar_date=r['calendar_date'],cohort=r['cohort'],original_label=r['original_label'],both_complete=r['both_complete']))
                save('folds/'+key+'.json',dict(complete=True,freeze_sha256=sha(OUT/'freeze.json'),fit=fit,predictions=pp));fits.append(fit);predictions.extend(pp);del m;gc.collect()
                save('pipeline_state.json',dict(stage='offline_evaluation',utc=utc(),completed_contexts=len(fits),expected_contexts=108,new_physical_replays=0));print(key,'complete',flush=True)
    for p in read(ROOT/'outputs/grid97_effect_learning/predictions.json'):
        if p['model'] not in d['models']:predictions.append(dict(p,arm='REUSED_BASELINE',source='GRID97 frozen baseline prediction'))
    verify_freeze();save('fits.json',fits);save('predictions.json',predictions)
    save('finished.json',dict(utc=utc(),fitted_contexts=len(fits),prediction_rows=len(predictions),reused_baseline_rows=237,new_physical_replays=0))
if __name__=='__main__':
    try:main()
    except Exception:save('error.json',dict(utc=utc(),traceback=traceback.format_exc()));raise
