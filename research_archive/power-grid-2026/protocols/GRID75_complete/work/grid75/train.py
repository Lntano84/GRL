"""Fixed forest teacher gate and feature ablation; no neural/RL training."""
from pathlib import Path
import hashlib,json,math,sys,time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid75';sys.path.insert(0,str(Path(__file__).parent));import feature_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'training_finished.json').exists()
    D=read(OUT/'design.json')
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    with np.load(OUT/'inputs.npz') as z:X=z['X'];y=z['y'];train=z['train']
    items=read(OUT/'input_receipts.json')['items'];models=OUT/'models';models.mkdir(exist_ok=False)
    start=time.perf_counter();results=[]
    for name in D['models']:
        k=D[name+'_features'];fit=RandomForestClassifier(**D['forest']);t=time.perf_counter();fit.fit(X[train,:k],y[train])
        exported=dict(name=name,feature_count=k,trees=[],support=[[i,float(X[train,i].min()),float(X[train,i].max())] for i in G.SUPPORT if i<k])
        for estimator in fit.estimators_:
            tree=estimator.tree_;values=tree.value[:,0,:]
            exported['trees'].append(dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),feature=tree.feature.tolist(),
                threshold=tree.threshold.tolist(),positive=(values[:,1]/values.sum(axis=1)).tolist()))
        parity=max(abs(G.predict(exported,x)-p) for x,p in zip(X,fit.predict_proba(X[:,:k])[:,1]));assert parity<1e-12
        positive=[G.predict(exported,x) for x,v,istrain in zip(X,y,train) if not istrain and v and not G.out_of_support(exported,x)]
        exported['threshold']=min(positive) if positive else 0.
        rows=[]
        for j,item in enumerate(items):
            defer,info=G.decide(exported,X[j]);queries=item['guard_queries'] if defer else item['initial_queries']
            rows.append(dict(id=item['id'],family=item['family'],split=item['split'],changed_action=item['changed_action'],
                defer=defer,queries=queries,guard_queries=item['guard_queries'],false_stop=bool(item['changed_action'] and not defer),**info))
        val=[r for r in rows if r['split']=='calibration'];assert not any(r['false_stop'] for r in val)
        saving=1-sum(r['queries'] for r in val)/sum(r['guard_queries'] for r in val)
        summary=dict(name=name,fit_s=time.perf_counter()-t,portable_max_error=parity,threshold=exported['threshold'],
            calibration_false_stop=sum(r['false_stop'] for r in val),calibration_skipped=sum(not r['defer'] for r in val),
            calibration_outside_support=sum(r['outside_support'] for r in val),calibration_query_saving=saving,
            qualified=saving>=.10,rows=rows)
        save(models/f'{name}.json',exported);save(models/f'{name}_review.json',summary);results.append(summary)
    result=dict(passed_engineering=True,fits=2,wall_s=time.perf_counter()-start,models=results,
        structural_qualified=results[1]['qualified'],structural_increment_over_plain=results[1]['calibration_query_saving']-results[0]['calibration_query_saving'],
        scope='Final fixed public-grid-context qualification;14 safe calibration states across4 families. Threshold fitted here, no confidence/quality guarantee. Target is teacher action change, not long-term harm. No evaluation feedback, new queries, or physical replay.')
    assert result['wall_s']<D['caps']['wall_s']
    save(OUT/'training_finished.json',result)
    (OUT/'RUN_STATE.md').write_text('FITS COMPLETE: structural qualification '+str(result['structural_qualified'])+'; closed-loop not started.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='models'}));print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in results]))
if __name__=='__main__':main()
