"""Two train-only fits; all validation predictions retained."""
from pathlib import Path
import collections,json,math,time
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score
from prepare import ROOT,W,OUT,read,save,sha


def verify_freeze():
    for freeze in ['code_freeze.json','data_freeze.json']:
        for name,expected in read(OUT/freeze).items():assert sha(ROOT/name)==expected,name


def budget_diagnostic(score,gain,queries,fraction):
    n=len(score);left=math.ceil(fraction*n);allocation=np.zeros(n)
    for value in np.unique(score)[::-1]:
        block=np.flatnonzero(score==value)
        amount=min(left,len(block));allocation[block]=amount/len(block);left-=amount
        if left==0:break
    positive=gain>0
    return dict(selected_fraction=float(allocation.mean()),
        positive_recall=float(allocation[positive].sum()/positive.sum()) if positive.any() else None,
        gain_capture=float(np.dot(allocation,gain)/gain.sum()) if gain.sum()>0 else None,
        known_query_fraction=float(np.dot(allocation,queries)/queries.sum()))


def metrics(gain,score,probability=None,predicted_gain=None):
    y=gain>0;two=len(np.unique(y))==2
    result=dict(n=len(y),positive=int(y.sum()),positive_rate=float(y.mean()),
        auc=float(roc_auc_score(y,score)) if two else None,
        ap=float(average_precision_score(y,score)) if y.any() else None,
        brier=None,balanced_accuracy=None,recall_at_half=None,precision_at_half=None,
        mae=None,rmse=None,positive_mae=None)
    if probability is not None:
        selected=probability>=.5
        result.update(brier=float(brier_score_loss(y,probability)),
            balanced_accuracy=float(balanced_accuracy_score(y,selected)) if two else None,
            recall_at_half=float(selected[y].mean()) if y.any() else None,
            precision_at_half=float(y[selected].mean()) if selected.any() else None)
    if predicted_gain is not None:
        error=predicted_gain-gain
        result.update(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
                      positive_mae=float(np.abs(error[y]).mean()) if y.any() else None)
    return result


def main():
    assert not (OUT/'fits.json').exists(),'Do not repeat completed fits'
    verify_freeze();D=read(OUT/'design.json');rows=read(OUT/'rows.json')
    with np.load(OUT/'data.npz') as data:X=data['X'];gain=data['gain']
    train=np.array([r['partition']=='train' for r in rows]);val=~train
    dates=np.array([r['calendar_group'] for r in rows]);assert not set(dates[train])&set(dates[val])
    counts=collections.Counter(dates[train]);weight=np.array([1/counts[d] for d in dates[train]])
    weight/=weight.mean();target=gain[train]>0
    prior=float(np.average(target,weights=weight));mean=float(np.average(gain[train],weights=weight))
    (OUT/'models').mkdir(exist_ok=False)
    started=time.perf_counter();scaler=StandardScaler().fit(X[train],sample_weight=weight)
    clf=LogisticRegression(**D['logistic']).fit(scaler.transform(X[train]),target,sample_weight=weight)
    logistic_fit_s=time.perf_counter()-started
    joblib.dump(dict(scaler=scaler,classifier=clf),OUT/'models/logistic.joblib')
    started=time.perf_counter();forest=RandomForestRegressor(**D['forest']).fit(X[train],gain[train],sample_weight=weight)
    forest_fit_s=time.perf_counter()-started
    joblib.dump(forest,OUT/'models/forest.joblib')
    logistic=clf.predict_proba(scaler.transform(X))[:,1];regression=forest.predict(X)
    outputs={'ZERO':dict(score=np.zeros(len(rows)),prob=np.zeros(len(rows)),gain=np.zeros(len(rows))),
             'TRAIN_CONSTANT':dict(score=np.full(len(rows),prior),prob=np.full(len(rows),prior),gain=np.full(len(rows),mean)),
             'LOGISTIC':dict(score=logistic,prob=logistic,gain=None),
             'FOREST_REG':dict(score=regression,prob=None,gain=regression)}
    query_column=D['feature_names'].index('log_completion_queries');queries=np.rint(np.expm1(X[:,query_column]))
    results={};predictions=[]
    for name,pred in outputs.items():
        per_date={}
        for date in sorted(set(dates[val])):
            mask=val&(dates==date)
            prob=None if pred['prob'] is None else pred['prob'][mask]
            estimate=None if pred['gain'] is None else pred['gain'][mask]
            record=metrics(gain[mask],pred['score'][mask],prob,estimate)
            record['top20']=budget_diagnostic(pred['score'][mask],gain[mask],queries[mask],D['top_fraction'])
            per_date[date]=record
        macro={}
        for key in ['auc','ap','brier','balanced_accuracy','recall_at_half','precision_at_half','mae','rmse','positive_mae']:
            values=[r[key] for r in per_date.values() if r[key] is not None]
            macro[key]=float(np.mean(values)) if values else None
        macro['top20']={k:float(np.mean([r['top20'][k] for r in per_date.values() if r['top20'][k] is not None]))
                        for k in next(iter(per_date.values()))['top20']}
        pooled=metrics(gain[val],pred['score'][val],None if pred['prob'] is None else pred['prob'][val],
                       None if pred['gain'] is None else pred['gain'][val])
        training=metrics(gain[train],pred['score'][train],None if pred['prob'] is None else pred['prob'][train],
                         None if pred['gain'] is None else pred['gain'][train])
        results[name]=dict(validation_date_macro=macro,validation_pooled=pooled,per_validation_date=per_date,training_in_sample=training)
        for i,row in enumerate(rows):
            predictions.append(dict(**row,model=name,true_gain=float(gain[i]),positive=bool(gain[i]>0),score=float(pred['score'][i]),
                probability=None if pred['prob'] is None else float(pred['prob'][i]),
                predicted_gain=None if pred['gain'] is None else float(pred['gain'][i]),known_queries=int(queries[i])))
    save(OUT/'fits.json',dict(model_fits=2,logistic_fit_s=logistic_fit_s,forest_fit_s=forest_fit_s,
        logistic_iterations=int(clf.n_iter_[0]),logistic_converged=bool(clf.n_iter_[0]<D['logistic']['max_iter']),
        train_ids=[r['sample_id'] for r in rows if r['partition']=='train'],train_date_weights={d:float(weight[dates[train]==d].sum()) for d in counts},
        train_positive_prior=prior,train_mean_gain=mean,validation_used_for_fit=False,final_test_materialized=False,
        model_sha256={p.name:sha(p) for p in (OUT/'models').iterdir()},scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
        logistic_coefficients=clf.coef_[0].tolist(),logistic_intercept=float(clf.intercept_[0]),forest_importances=forest.feature_importances_.tolist()))
    save(OUT/'predictions.json',predictions);save(OUT/'results.json',results)
    print(json.dumps({name:record['validation_date_macro'] for name,record in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
