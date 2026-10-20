"""Recalculate labels/splits, independent LR/RF probabilities, reconstruct TabPFN."""
from common import *
import gc
def independent(name,ex,x):
    if name=='LOGISTIC':
        z=(x-np.array(ex['mean']))/np.array(ex['scale']);s=z@np.array(ex['coef'])+ex['intercept'];return 1/(1+np.exp(-s))
    ans=[]
    for row in x.astype(np.float32):
        vals=[]
        for t in ex['trees']:
            n=0
            while t['left'][n]!=-1:n=t['left'][n] if row[t['feature'][n]]<=t['threshold'][n] else t['right'][n]
            v=t['value'][n];vals.append(v[1]/sum(v))
        ans.append(float(np.mean(vals)))
    return np.array(ans)
def main():
    n=verify_freeze();d=read(OUT/'design.json');rr=read(OUT/'dataset.json');lookup={r['case_id']:r for r in rr};pred=read(OUT/'predictions.json');fits=read(OUT/'fits.json')
    raw=read(ROOT/'outputs/grid96_value_learning/labels.json');assert len(raw)==len(rr)==56
    for r in raw:
        z=lookup[r['case_id']];assert z['features']==r['features'] and z['y_same_path']==int(r['physical_path_identical'])
    assert len(fits)==108 and len({f['key'] for f in fits})==108
    checks={'frozen_items':n,'labels_crosschecked':56,'independent_lr_rf_rows':0,'tabpfn_reconstructed_rows':0,'split_checks':0,'single_row_checks':0}
    for f in fits:
        expected=next(s for s in d['folds'] if s['scheme']==f['scheme'] and s['fold']==f['fold']);assert expected['train_ids']==f['train_ids'] and expected['test_ids']==f['test_ids']
        assert not set(f['train_ids'])&set(f['test_ids']);checks['split_checks']+=1
        x,y=arrays(lookup,f['train_ids'],d['features']);xt,yt=arrays(lookup,f['test_ids'],d['features'])
        pp=[p for p in pred if p['fit_key']==f['key']];assert [p['case_id'] for p in pp]==f['test_ids'];assert [p['y'] for p in pp]==yt.tolist()
        expectedp=np.array([p['p_same_path'] for p in pp])
        if f['model'] in ['LOGISTIC','FOREST64']:
            p=independent(f['model'],f['export'],xt);checks['independent_lr_rf_rows']+=len(pp)
        elif f['model']=='TABPFN_V2':
            m=estimator(f['model'],d);m.fit(x,y);p=m.predict_proba(xt)[:,list(m.classes_).index(1)];del m;gc.collect();checks['tabpfn_reconstructed_rows']+=len(pp)
        else:p=np.full(len(pp),f['export']['prior'])
        assert np.allclose(p,expectedp,atol=1e-5 if f['model']=='TABPFN_V2' else 1e-10,rtol=0),f['key']
        assert np.allclose(expectedp,[p['single_p_same_path'] for p in pp],atol=1e-5,rtol=0);checks['single_row_checks']+=len(pp)
        print('audited',f['key'],flush=True)
    results=[]
    for scheme in ['calendar_loo','old_to_generated']:
        for name in list(d['models'])+['ALWAYS_RETAIN','ALWAYS_OMIT','TRAIN_PRIOR']:
            pp=[p for p in pred if p['model']==name and p['scheme']==scheme]
            dates=sorted({p['calendar_date'] for p in pp if not p['pilot_known']})
            bydate={date:score([p for p in pp if p['calendar_date']==date and not p['pilot_known']]) for date in dates}
            relevant=[f for f in fits if f['model']==name and f['scheme']==scheme];single=[s for f in relevant for s in f['single_row_seconds']]
            results.append(dict(model=name,scheme=scheme,all56_or_transfer=score(pp),primary_nonpilot=score([p for p in pp if not p['pilot_known']]),known_pilots=score([p for p in pp if p['pilot_known']]),same_path_incomplete=score([p for p in pp if p['y']==1 and not p['both_complete']]),by_date=bydate,
                date_macro_accuracy=float(np.mean([z['accuracy'] for z in bydate.values()])),date_macro_balanced_accuracy=float(np.mean([z['balanced_accuracy'] for z in bydate.values()])),
                by_cohort={c:score([p for p in pp if p['cohort']==c and not p['pilot_known']]) for c in ['old13','G1','G2']},
                timing=dict(fit_context_total_s=sum(f['fit_or_context_seconds'] for f in relevant),single_row_mean_ms=float(np.mean(single)*1000) if single else None,single_row_p95_ms=float(np.quantile(single,.95)*1000) if single else None,single_row_count=len(single),warm_batch_per_row_median_ms=float(np.median([s/len(f['test_ids'])*1000 for f in relevant for s in f['warm_batch_seconds']])) if single else None)))
    save('results.json',results);verify_freeze()
    save('review.json',dict(passed=True,utc=utc(),checks=checks,target='physical path identity only',no_new_physical_replays=True,no_gate_deployment=True,all_historical_hashes_unchanged=True,tabpfn_audit='Same frozen support context and checkpoint reconstructed once per fold, no gradient updates, not independent predictive evidence'))
if __name__=='__main__':main()
