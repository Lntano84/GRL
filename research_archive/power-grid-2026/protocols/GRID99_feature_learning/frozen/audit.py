from common import *
import gc,math
def independent(name,e,x):
    if name=='LOGISTIC':return 1/(1+np.exp(-((x-np.array(e['mean']))/np.array(e['scale'])@np.array(e['coef'])+e['intercept'])))
    out=[]
    for row in x.astype(np.float32):
        values=[]
        for t in e['trees']:
            n=0
            while t['left'][n]!=-1:n=t['left'][n] if row[t['feature'][n]]<=t['threshold'][n] else t['right'][n]
            v=t['value'][n];values.append(v[1]/sum(v))
        out.append(float(np.mean(values)))
    return np.array(out)
def arithmetic(m,pp):
    a=[p['p_same_path'] for p in pp if p['y']==1];b=[p['p_same_path'] for p in pp if p['y']==0]
    if a and b:
        auc=sum((x>y)+.5*(x==y) for x in a for y in b)/(len(a)*len(b));ba=.5*(sum(x>=.5 for x in a)/len(a)+sum(x<.5 for x in b)/len(b))
        assert math.isclose(auc,m['auc'],abs_tol=1e-12) and math.isclose(ba,m['balanced_accuracy'],abs_tol=1e-12)
    assert math.isclose(sum((p['p_same_path']-p['y'])**2 for p in pp)/len(pp),m['brier'],abs_tol=1e-12)
    for t in [.5,.95]:
        ss=[p for p in pp if p['p_same_path']>=t];z=m['thresholds'][str(t)]
        assert z['proposed_omissions']==len(ss) and z['changed_path_misclassified']==sum(p['y']==0 for p in ss)
def main():
    nf=verify_freeze();d=read(OUT/'design.json');data=read(OUT/'dataset.json');lookup={r['case_id']:r for r in data};fits=read(OUT/'fits.json');pp=read(OUT/'predictions.json')
    old=read(ROOT/'outputs/grid97_effect_learning/dataset.json');new={r['case_id']:r for r in read(ROOT/'outputs/grid98_feature_enrichment/features.json')}
    for r in old:
        now=lookup[r['case_id']];assert {k:v for k,v in now.items() if k!='features'}=={k:v for k,v in r.items() if k!='features'};assert now['features']==new[r['case_id']]['all35'];assert {k:now['features'][k] for k in d['feature_sets']['ORIGINAL15']}==r['features']
    assert len(fits)==108 and len({f['key'] for f in fits})==108 and len(pp)==711
    previous={(p['model'],p['scheme'],p['fold'],p['case_id']):p for p in read(ROOT/'outputs/grid97_effect_learning/predictions.json')}
    checks=dict(frozen_items=nf,labels_and_old_features=56,split_checks=0,independent_lr_rf_rows=0,reconstructed_tabpfn_rows=0,original15_repeat_matches=0,reused_baseline_matches=0,single_batch_checks=0)
    max_repeat_error=0.
    for f in fits:
        split=next(z for z in d['folds'] if z['scheme']==f['scheme'] and z['fold']==f['fold']);assert split['train_ids']==f['train_ids'] and split['test_ids']==f['test_ids'];assert not set(f['train_ids'])&set(f['test_ids'])
        assert not {lookup[i]['calendar_date'] for i in f['train_ids']}&{lookup[i]['calendar_date'] for i in f['test_ids']};checks['split_checks']+=1
        assert f['features']==d['feature_sets'][f['arm']]
        x,y=arrays(lookup,f['train_ids'],f['features']);xt,yt=arrays(lookup,f['test_ids'],f['features']);p=[z for z in pp if z['fit_key']==f['key'] and z['arm']==f['arm']]
        assert [z['case_id'] for z in p]==f['test_ids'] and [z['y'] for z in p]==yt.tolist()
        expected=np.array([z['p_same_path'] for z in p])
        if f['model']=='TABPFN_V2':m=estimator(f['model'],d);m.fit(x,y);actual=m.predict_proba(xt)[:,list(m.classes_).index(1)];checks['reconstructed_tabpfn_rows']+=len(p);del m;gc.collect()
        else:actual=independent(f['model'],f['export'],xt);checks['independent_lr_rf_rows']+=len(p)
        assert np.allclose(actual,expected,atol=1e-5 if f['model']=='TABPFN_V2' else 1e-10,rtol=0)
        assert np.allclose(expected,[z['single_p_same_path'] for z in p],atol=1e-5,rtol=0);checks['single_batch_checks']+=len(p)
        if f['arm']=='ORIGINAL15':
            for z in p:
                prior=previous[z['model'],z['scheme'],z['fold'],z['case_id']];error=abs(z['p_same_path']-prior['p_same_path']);max_repeat_error=max(max_repeat_error,error);assert error<=(1e-5 if z['model']=='TABPFN_V2' else 1e-10);checks['original15_repeat_matches']+=1
        print('audited',f['key'],flush=True)
    for p in pp:
        if p['arm']=='REUSED_BASELINE':
            prior=previous[p['model'],p['scheme'],p['fold'],p['case_id']];assert all(p[k]==v for k,v in prior.items());checks['reused_baseline_matches']+=1
    result=[]
    for scheme in ['calendar_loo','old_to_generated']:
        configurations=[(a,m) for a in d['feature_sets'] for m in d['models']]+[('REUSED_BASELINE',m) for m in ['ALWAYS_RETAIN','ALWAYS_OMIT','TRAIN_PRIOR']]
        for arm,name in configurations:
            p=[z for z in pp if z['arm']==arm and z['model']==name and z['scheme']==scheme];primary=[z for z in p if not z['pilot_known']]
            allscore=score(p);prscore=score(primary);arithmetic(allscore,p);arithmetic(prscore,primary)
            ff=[f for f in fits if f['arm']==arm and f['model']==name and f['scheme']==scheme];single=[v for f in ff for v in f['single_row_seconds']]
            dates=sorted({z['calendar_date'] for z in primary});bydate={date:score([z for z in primary if z['calendar_date']==date]) for date in dates}
            result.append(dict(arm=arm,model=name,scheme=scheme,primary_nonpilot=prscore,all56_or_transfer=allscore,known_pilots=score([z for z in p if z['pilot_known']]),by_date=bydate,date_macro_accuracy=float(np.mean([v['accuracy'] for v in bydate.values()])),by_cohort={c:score([z for z in primary if z['cohort']==c]) for c in ['old13','G1','G2']},same_path_incomplete=score([z for z in p if z['y']==1 and not z['both_complete']]),timing=dict(fit_context_total_s=sum(f['fit_or_context_seconds'] for f in ff),single_row_mean_ms=float(np.mean(single)*1000) if single else None,single_row_p95_ms=float(np.quantile(single,.95)*1000) if single else None,single_row_count=len(single))))
    changes=[]
    for scheme in ['calendar_loo','old_to_generated']:
        for name in d['models']:
            a=next(z for z in result if z['arm']=='ORIGINAL15' and z['model']==name and z['scheme']==scheme)['primary_nonpilot'];b=next(z for z in result if z['arm']=='ENRICHED35' and z['model']==name and z['scheme']==scheme)['primary_nonpilot']
            changes.append(dict(model=name,scheme=scheme,auc_delta=b['auc']-a['auc'],balanced_accuracy_delta=b['balanced_accuracy']-a['balanced_accuracy'],brier_delta=b['brier']-a['brier'],fixed95_original=a['thresholds']['0.95'],fixed95_enriched=b['thresholds']['0.95']))
    verify_freeze();save('results.json',result);save('paired_changes.json',changes)
    save('review.json',dict(passed=True,utc=utc(),checks=checks,max_original15_probability_error=max_repeat_error,metrics_independently_recomputed=True,all_historical_hashes_unchanged=True,no_new_physical_replays=True,no_deployment=True))
if __name__=='__main__':main()
