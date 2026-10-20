"""Fixed simple models; calendar-isolated evaluation with known pilots separate."""
from pathlib import Path
from collections import Counter
import json,os,warnings
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
CLASSES=['KEEP','SKIP','MIXED']
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def metrics(rr):
    if not rr:return dict(n=0)
    y=[r['label'] for r in rr];yp=[r['prediction'] for r in rr];p=[r['probabilities'][1] for r in rr];binary=[v=='SKIP' for v in y]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');ba=balanced_accuracy_score(y,yp)
    out=dict(n=len(rr),calendar_dates=len({r['calendar_date'] for r in rr}),class_counts=dict(Counter(y)),confusion=confusion_matrix(y,yp,labels=CLASSES).tolist(),balanced_accuracy=float(ba),macro_f1=float(f1_score(y,yp,labels=CLASSES,average='macro',zero_division=0)),
        skip_auc=float(roc_auc_score(binary,p)) if len(set(binary))==2 else None,skip_ap=float(average_precision_score(binary,p)) if any(binary) else None,skip_brier=float(brier_score_loss(binary,p)))
    for threshold in [.5,.95]:
        selected=[r for r in rr if r['probabilities'][1]>=threshold and r['prediction']=='SKIP']
        out[str(threshold)]=dict(proposed_skips=len(selected),coverage=len(selected)/len(rr),correct_skips=sum(r['label']=='SKIP' for r in selected),missed_keep=sum(r['label']=='KEEP' for r in selected),disputed_mixed=sum(r['label']=='MIXED' for r in selected),
            precision=sum(r['label']=='SKIP' for r in selected)/len(selected) if selected else None,case_ids=[r['case_id'] for r in selected])
    return out
def exported_predict(model,xx):
    classes=model['classes'];x=np.asarray(xx,float)
    if model['kind']=='CONSTANT':raw=np.ones((len(x),1))
    elif model['kind']=='LOGISTIC':
        z=(x-np.asarray(model['mean']))/np.asarray(model['scale']);s=z@np.asarray(model['coef']).T+np.asarray(model['intercept'])
        if len(classes)==2:
            v=1/(1+np.exp(-s[:,0]));raw=np.column_stack([1-v,v])
        else:
            s-=s.max(axis=1,keepdims=True);e=np.exp(s);raw=e/e.sum(axis=1,keepdims=True)
    else:
        raw=np.zeros((len(x),len(classes)))
        for tree in model['trees']:
            for j,row in enumerate(x.astype(np.float32)):
                n=0
                while tree['left'][n]!=-1:n=tree['left'][n] if row[tree['feature'][n]]<=tree['threshold'][n] else tree['right'][n]
                val=np.asarray(tree['value'][n],float);raw[j]+=val/val.sum()
        raw/=len(model['trees'])
    padded=np.zeros((len(x),3))
    for j,c in enumerate(classes):padded[:,CLASSES.index(c)]=raw[:,j]
    return padded
def main():
    assert read(OUT/'finished.json')['runs']==74 and read(OUT/'review.json')['passed_saved_data']
    D=read(OUT/'design.json');data=read(OUT/'labels.json');valid=[r for r in data if r['label'] in CLASSES];names=D['learning']['features']
    splits=[('calendar_loo',date,[r for r in valid if r['calendar_date']!=date],[r for r in valid if r['calendar_date']==date]) for date in sorted({r['calendar_date'] for r in valid})]
    splits += [('old_to_generated','old13',[r for r in valid if r['cohort']=='old13'],[r for r in valid if r['cohort'] in ['G1','G2']])]
    fits=[];predictions=[];checks=Counter()
    for scheme,fold,train,test in splits:
        assert train and test
        if scheme=='calendar_loo':assert not ({r['calendar_date'] for r in train}&{r['calendar_date'] for r in test})
        assert not ({r['case_id'] for r in train}&{r['case_id'] for r in test})
        x=np.asarray([[r['features'][n] for n in names] for r in train],float);xt=np.asarray([[r['features'][n] for n in names] for r in test],float);y=np.asarray([r['label'] for r in train]);counts=Counter(r['calendar_date'] for r in train)
        weights=np.asarray([1/counts[r['calendar_date']] for r in train]);weights/=weights.mean()
        mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights));scale[scale<1e-12]=1
        assert np.isfinite(mean).all() and np.isfinite(scale).all()
        for name in ['LOGISTIC','FOREST64','ALWAYS_KEEP','ALWAYS_SKIP','TRAIN_PRIOR']:
            if name.startswith('ALWAYS_'):
                m=dict(kind='CONSTANT',classes=[name.removeprefix('ALWAYS_')]);probs=exported_predict(m,xt)
            elif name=='TRAIN_PRIOR':
                totals=np.asarray([weights[y==c].sum() for c in CLASSES]);probs=np.tile(totals/totals.sum(),(len(test),1));m=dict(kind='TRAIN_PRIOR',classes=CLASSES,prior=(totals/totals.sum()).tolist())
            elif len(set(y))==1:
                m=dict(kind='CONSTANT',classes=[y[0]]);probs=exported_predict(m,xt)
            else:
                if name=='LOGISTIC':
                    estimator=LogisticRegression(**D['learning']['models'][name]);estimator.fit((x-mean)/scale,y,sample_weight=weights)
                    assert np.max(estimator.n_iter_)<2000,'Logistic failed to converge'
                    m=dict(kind=name,classes=estimator.classes_.tolist(),mean=mean.tolist(),scale=scale.tolist(),coef=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist(),iterations=estimator.n_iter_.tolist())
                    raw=estimator.predict_proba((xt-mean)/scale)
                else:
                    estimator=RandomForestClassifier(**D['learning']['models'][name]);estimator.fit(x,y,sample_weight=weights)
                    m=dict(kind=name,classes=estimator.classes_.tolist(),trees=[dict(left=e.tree_.children_left.tolist(),right=e.tree_.children_right.tolist(),feature=e.tree_.feature.tolist(),threshold=e.tree_.threshold.tolist(),value=e.tree_.value[:,0,:].tolist()) for e in estimator.estimators_],feature_importances=estimator.feature_importances_.tolist())
                    raw=estimator.predict_proba(xt)
                probs=exported_predict(m,xt);expected=np.zeros_like(probs)
                for j,c in enumerate(estimator.classes_):expected[:,CLASSES.index(c)]=raw[:,j]
                assert np.allclose(probs,expected,atol=1e-10,rtol=0);checks['independent_probability_checks']+=len(test)
            assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1)
            fitid=f'{scheme}|{fold}|{name}';fits.append(dict(id=fitid,model=name,scheme=scheme,fold=fold,train_ids=[r['case_id'] for r in train],test_ids=[r['case_id'] for r in test],train_calendar_counts=dict(counts),weights=weights.tolist(),export=m))
            for r,p in zip(test,probs):predictions.append(dict(case_id=r['case_id'],week=r['week'],calendar_date=r['calendar_date'],cohort=r['cohort'],label=r['label'],pilot_known=r['pilot_known'],scheme=scheme,fold=fold,model=name,fit_id=fitid,probabilities=p.tolist(),prediction=CLASSES[int(np.argmax(p))]))
            checks['fits_or_baselines']+=1
    results=[]
    for scheme in ['calendar_loo','old_to_generated']:
        for model in ['LOGISTIC','FOREST64','ALWAYS_KEEP','ALWAYS_SKIP','TRAIN_PRIOR']:
            pp=[r for r in predictions if r['scheme']==scheme and r['model']==model]
            results.append(dict(scheme=scheme,model=model,headline=metrics([r for r in pp if not r['pilot_known']]),known_pilots_diagnostic=metrics([r for r in pp if r['pilot_known']]),by_cohort={c:metrics([r for r in pp if r['cohort']==c and not r['pilot_known']]) for c in ['old13','G1','G2']}))
    save('fits.json',fits);save('predictions.json',predictions);save('learning_results.json',results)
    save('learning_review.json',dict(passed=True,checks=dict(checks),valid_states=len(valid),unresolved_states=len(data)-len(valid),features=names,classes=CLASSES,group_isolation_passed=True,thresholds=[.5,.95],deployment=False))
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
