"""Fixed small models; grouped evaluation; no deployable final all-data model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from collections import Counter
import json, math, warnings
import numpy as np
import scipy,sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from data import ROOT,W,OUT,read,save,csvsave,sha


def fit_one(name,x,y,groups):
    counts=Counter(groups);sw=np.array([1/counts[g] for g in groups]);sw*=len(sw)/sw.sum()
    if len(np.unique(y))<2:
        return None,None,dict(type='CONSTANT',probability=float(y[0])),sw
    scaler=None
    if name=='LOGISTIC':
        scaler=StandardScaler().fit(x,sample_weight=sw)
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,class_weight=None)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(scaler.transform(x),y,sample_weight=sw)
        detail=dict(type=name,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
            coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),n_iter=model.n_iter_.tolist())
    else:
        model=DecisionTreeClassifier(max_depth=2,min_samples_leaf=10,criterion='gini',random_state=20261020)
        model.fit(x,y,sample_weight=sw);t=model.tree_
        detail=dict(type=name,children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),
            feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value.tolist(),classes=model.classes_.tolist(),
            rules=export_text(model,feature_names=read(W/'design.json')['features']))
    return model,scaler,detail,sw


def manual_predict(detail,x):
    if detail['type']=='CONSTANT':return np.full(len(x),detail['probability'])
    if detail['type']=='LOGISTIC':
        z=(x-np.array(detail['mean']))/np.array(detail['scale'])
        z=z@np.array(detail['coefficient'])+detail['intercept']
        return np.array([1/(1+math.exp(-float(v))) for v in np.clip(z,-700,700)])
    out=[]
    for row in x.astype(np.float32):
        n=0
        while detail['children_left'][n]!=-1:
            n=detail['children_left'][n] if row[detail['feature'][n]]<=detail['threshold'][n] else detail['children_right'][n]
        values=detail['value'][n][0];out.append(values[detail['classes'].index(1)]/sum(values))
    return np.array(out)


def metrics(pred,threshold):
    y=np.array([r['label_skip_same_choice'] for r in pred]);p=np.array([r['probability'] for r in pred]);skip=p>=threshold
    q=np.array([r['completion_queries'] for r in pred]);gain=np.array([r['reward_gain'] for r in pred])
    bad=skip & (y==0);correct=skip & (y==1)
    changed=int(sum(y==0));skipped=int(skip.sum());n=len(y)
    return dict(calls=n,calendar_groups=len({r['calendar_date'] for r in pred}),same_choice=int(y.sum()),choice_changes=changed,
        prevalence=float(y.mean()),roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(brier_score_loss(y,p)),threshold=threshold,accuracy=float(np.mean(skip==y)),
        skipped_calls=skipped,correct_same_choice_skips=int(correct.sum()),missed_choice_changes=int(bad.sum()),
        missed_change_scenarios=len({r['week'] for r,b in zip(pred,bad) if b}),
        skip_precision=float(correct.sum()/skipped) if skipped else None,
        retained_choice_change_fraction=float(1-bad.sum()/changed) if changed else None,
        nominal_queries_saved=int(q[skip].sum()),same_choice_queries_saved=int(q[correct].sum()),
        total_completion_queries=int(q.sum()),nominal_query_saving_fraction=float(q[skip].sum()/q.sum()),
        same_choice_query_saving_fraction=float(q[correct].sum()/q.sum()),
        missed_reward_gain=float(gain[bad].sum()),total_reward_gain=float(gain.sum()),
        missed_rows=[r['row_id'] for r,b in zip(pred,bad) if b],
        interpretation='Teacher-forced original-state screen only; missed changes are not automatically long-term harm. Nominal query savings are not closed-loop totals.')


def main():
    d=read(W/'design.json');pre=read(OUT/'preflight.json');assert pre['passed']
    freeze=read(OUT/'code_freeze.json')
    assert all(sha(ROOT/p)==h for p,h in freeze['files'].items())
    rows=read(OUT/'dataset.json');assert sha(OUT/'dataset.json')==pre['dataset_sha256']
    keys=d['features'];x=np.array([[r[k] for k in keys] for r in rows]);y=np.array([r['label_skip_same_choice'] for r in rows]);groups=np.array([r['calendar_date'] for r in rows])
    assert np.isfinite(x).all()
    splits=[('GROUPED_OOF',g,np.flatnonzero(groups!=g),np.flatnonzero(groups==g)) for g in sorted(set(groups))]
    splits.append(('OLD_TO_GENERATED','G1+G2',np.array([i for i,r in enumerate(rows) if r['cohort']=='old13']),np.array([i for i,r in enumerate(rows) if r['cohort']!='old13'])))
    predictions=[];fits=[];checks=dict(split_isolation=0,manual_probabilities=0,train_only_preprocessing=0)
    for evaluation,fold,train,test in splits:
        assert set(groups[train]).isdisjoint(groups[test]);checks['split_isolation']+=1
        prior=float(y[train].mean())
        for name in ['LOGISTIC','TREE2']:
            model,scaler,detail,sw=fit_one(name,x[train],y[train],groups[train])
            if model is None:p=manual_predict(detail,x[test])
            else:
                xx=x[test] if scaler is None else scaler.transform(x[test])
                p=model.predict_proba(xx)[:,list(model.classes_).index(1)]
                assert np.allclose(p,manual_predict(detail,x[test]),rtol=0,atol=1e-12)
            checks['manual_probabilities']+=len(test)
            if scaler is not None:
                assert np.allclose(scaler.mean_,np.average(x[train],axis=0,weights=sw));checks['train_only_preprocessing']+=1
            fits.append(dict(evaluation=evaluation,fold=fold,model=name,train_rows=[rows[i]['row_id'] for i in train],
                test_rows=[rows[i]['row_id'] for i in test],train_groups=sorted(set(groups[train])),test_groups=sorted(set(groups[test])),
                details=detail))
            for i,prob in zip(test,p):predictions.append(dict(evaluation=evaluation,fold=fold,model=name,probability=float(prob),**rows[i]))
        for name,prob in [('TRAIN_PRIOR',prior),('ALWAYS_COMPLETE',0.),('ALWAYS_SKIP',1.)]:
            for i in test:predictions.append(dict(evaluation=evaluation,fold=fold,model=name,probability=prob,**rows[i]))
    summaries=[]
    for evaluation in ['GROUPED_OOF','OLD_TO_GENERATED']:
        for model in ['LOGISTIC','TREE2','TRAIN_PRIOR','ALWAYS_COMPLETE','ALWAYS_SKIP']:
            rr=[r for r in predictions if r['evaluation']==evaluation and r['model']==model]
            for cohort in ['ALL','old13','G1','G2']:
                rrr=[r for r in rr if cohort=='ALL' or r['cohort']==cohort]
                if not rrr:continue
                for threshold in [.95,.5]:summaries.append(dict(evaluation=evaluation,model=model,cohort=cohort,**metrics(rrr,threshold)))
    # Calendar aggregation preserves all failures rather than averaging them away.
    by_calendar=[]
    for evaluation in ['GROUPED_OOF','OLD_TO_GENERATED']:
        for model in ['LOGISTIC','TREE2']:
            rr=[r for r in predictions if r['evaluation']==evaluation and r['model']==model]
            for g in sorted({r['calendar_date'] for r in rr}):
                for threshold in [.95,.5]:
                    by_calendar.append(dict(evaluation=evaluation,model=model,calendar_date=g,**metrics([r for r in rr if r['calendar_date']==g],threshold)))
    candidates=[]
    for model in ['LOGISTIC','TREE2']:
        pair=[next(r for r in summaries if r['evaluation']==e and r['model']==model and r['cohort']=='ALL' and r['threshold']==.95) for e in ['GROUPED_OOF','OLD_TO_GENERATED']]
        candidates.append(dict(model=model,limited_signal=all(r['skipped_calls']>0 and r['missed_choice_changes']==0 for r in pair)))
    save(OUT/'decision.json',dict(stage=d['stage'],result='LIMITED_OFFLINE_SKIP_SIGNAL' if any(r['limited_signal'] for r in candidates) else 'NO_DEMONSTRATED_CONSERVATIVE_SKIP_SIGNAL',
        fixed_models=candidates,deployment_authorized=False,closed_loop_quality_established=False,
        harmful_completion_predictability_established=False,thresholds_selected_after_results=False))
    save(OUT/'predictions.json',predictions);csvsave(OUT/'predictions.csv',predictions)
    save(OUT/'fits.json',fits);save(OUT/'metrics.json',summaries);save(OUT/'calendar_metrics.json',by_calendar)
    save(OUT/'fit_verification.json',dict(passed=True,checks=checks,model_fits=len(fits),
        packages=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        design_sha256=sha(W/'design.json'),code_freeze_sha256=sha(OUT/'code_freeze.json')))
    print(json.dumps(dict(decision=read(OUT/'decision.json'),all_metrics=[r for r in summaries if r['cohort']=='ALL']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
