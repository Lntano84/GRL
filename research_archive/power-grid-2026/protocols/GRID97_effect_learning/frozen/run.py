from common import *
import joblib,gc,traceback
def main():
    verify_freeze();assert not (OUT/'finished.json').exists()
    d=read(OUT/'design.json');data=read(OUT/'dataset.json');lookup={r['case_id']:r for r in data};all_predictions=[];fit_records=[]
    (OUT/'folds').mkdir(exist_ok=True)
    for fi,f in enumerate(d['folds']):
        x,y=arrays(lookup,f['train_ids'],d['features']);xt,yt=arrays(lookup,f['test_ids'],d['features'])
        assert set(y)=={0,1}
        names=list(d['models']);names=names[fi%3:]+names[:fi%3]
        for name in names+['ALWAYS_RETAIN','ALWAYS_OMIT','TRAIN_PRIOR']:
            key=f'{fi:02d}__{name}';receipt=OUT/'folds'/(key+'.json')
            if receipt.exists():
                saved=read(receipt);assert saved['complete'] and saved['freeze_sha256']==sha(OUT/'freeze.json')
                all_predictions.extend(saved['predictions']);fit_records.append(saved['fit']);continue
            t=time.perf_counter()
            if name in d['models']:
                m=estimator(name,d);m.fit(x,y)
                if name=='LOGISTIC':assert np.max(m['model'].n_iter_)<2000
                fit_s=time.perf_counter()-t
                t=time.perf_counter();probs=m.predict_proba(xt)[:,list(m.classes_).index(1)];first_s=time.perf_counter()-t
                warm=[]
                for _ in range(d['timing']['warm_batch_repeats']):
                    t=time.perf_counter();again=m.predict_proba(xt)[:,list(m.classes_).index(1)];warm.append(time.perf_counter()-t);assert np.allclose(probs,again,atol=1e-6,rtol=0)
                single=[];sp=[]
                for row in xt:
                    t=time.perf_counter();sp.append(float(m.predict_proba(row[None,:])[0,list(m.classes_).index(1)]));single.append(time.perf_counter()-t)
                assert np.allclose(probs,sp,atol=1e-5,rtol=0),('batch_single_disagreement',name,np.max(np.abs(probs-sp)))
                if name!='TABPFN_V2':joblib.dump(m,OUT/'folds'/(key+'.joblib'))
                export=None
                if name=='LOGISTIC':export=dict(mean=m['scale'].mean_.tolist(),scale=m['scale'].scale_.tolist(),coef=m['model'].coef_[0].tolist(),intercept=float(m['model'].intercept_[0]))
                elif name=='FOREST64':export=dict(trees=[dict(left=e.tree_.children_left.tolist(),right=e.tree_.children_right.tolist(),feature=e.tree_.feature.tolist(),threshold=e.tree_.threshold.tolist(),value=e.tree_.value[:,0,:].tolist()) for e in m.estimators_])
                del m;gc.collect()
            else:
                prior={'ALWAYS_RETAIN':0.,'ALWAYS_OMIT':1.,'TRAIN_PRIOR':float(y.mean())}[name]
                probs=np.full(len(xt),prior);sp=probs.tolist();fit_s=first_s=0.;warm=[];single=[];export=dict(prior=prior)
            assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
            fit=dict(key=key,model=name,fold=f['fold'],scheme=f['scheme'],train_ids=f['train_ids'],test_ids=f['test_ids'],fit_or_context_seconds=fit_s,first_batch_seconds=first_s,warm_batch_seconds=warm,single_row_seconds=single,export=export)
            preds=[]
            for i,p,ps in zip(f['test_ids'],probs,sp):
                r=lookup[i];preds.append(dict(case_id=i,model=name,scheme=f['scheme'],fold=f['fold'],fit_key=key,y=r['y_same_path'],p_same_path=float(p),single_p_same_path=float(ps),pilot_known=r['pilot_known'],calendar_date=r['calendar_date'],cohort=r['cohort'],original_label=r['original_label'],both_complete=r['both_complete']))
            save('folds/'+key+'.json',dict(complete=True,freeze_sha256=sha(OUT/'freeze.json'),fit=fit,predictions=preds));all_predictions.extend(preds);fit_records.append(fit)
            save('pipeline_state.json',dict(stage='offline_evaluation',utc=utc(),completed_contexts=len(fit_records),total_contexts=108,new_physical_replays=0))
            print(key,'complete',flush=True)
    verify_freeze();save('predictions.json',all_predictions);save('fits.json',fit_records)
    save('finished.json',dict(utc=utc(),contexts=len(fit_records),estimator_contexts=sum(f['model'] in d['models'] for f in fit_records),prediction_rows=len(all_predictions),new_physical_replays=0))
if __name__=='__main__':
    try:main()
    except Exception:
        save('error.json',dict(utc=utc(),traceback=traceback.format_exc()));raise
