from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid74';sys.path.insert(0,str(ROOT/'work/grid72'));import feature_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'training_finished.json').exists();D=read(OUT/'design.json')
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    with np.load(OUT/'inputs.npz') as z:X=z['X'];y=z['y'];train=z['train'];stable=z['stable']
    items=read(OUT/'input_receipts.json')['items'];models=OUT/'models';models.mkdir(exist_ok=False);results=[];start=time.perf_counter()
    for name in D['models']:
        k=D[name+'_features'];fit=RandomForestClassifier(**D['forest']);fit.fit(X[train&stable,:k],y[train&stable]);assert list(fit.classes_)==[0,1]
        m=dict(name=name,feature_count=k,trees=[],support=[[i,float(X[train,i].min()),float(X[train,i].max())] for i in G.SUPPORT if i<k])
        for estimator in fit.estimators_:
            t=estimator.tree_;values=t.value[:,0,:];m['trees'].append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),positive=(values[:,1]/values.sum(axis=1)).tolist()))
        parity=max(abs(G.predict(m,x)-p) for x,p in zip(X,fit.predict_proba(X[:,:k])[:,1]));assert parity<1e-12
        positive=[G.predict(m,x) for x,v,istrain,st in zip(X,y,train,stable) if not istrain and st and v and not G.out_of_support(m,x)]
        m['threshold']=min(positive) if positive else 0.;rows=[]
        for j,item in enumerate(items):
            defer,info=G.decide(m,X[j]);defer=bool(item['probe_changes_incumbent'] or defer)
            rows.append(dict(id=item['id'],family=item['family'],split=item['split'],changed_action=item['changed_action'],
                forced_probe_defer=item['probe_changes_incumbent'],defer=defer,queries=item['guard_queries'] if defer else item['probe_queries'],guard_queries=item['guard_queries'],
                false_stop=bool(item['changed_action'] and not defer),**info))
        val=[r for r in rows if r['split']=='calibration'];assert not any(r['false_stop'] for r in val)
        saving=1-sum(r['queries'] for r in val)/sum(r['guard_queries'] for r in val)
        result=dict(name=name,portable_error=parity,threshold=m['threshold'],calibration_query_saving=saving,
            calibration_skipped=sum(not r['defer'] for r in val),calibration_false_stop=sum(r['false_stop'] for r in val),qualified=saving>=.10,rows=rows)
        save(models/f'{name}.json',m);save(models/f'{name}_review.json',result);results.append(result)
    result=dict(passed_engineering=True,fits=2,wall_s=time.perf_counter()-start,models=results,structural_qualified=results[1]['qualified'],
        structural_increment_over_plain=results[1]['calibration_query_saving']-results[0]['calibration_query_saving'],new_physical_queries=0,
        scope='Teacher-action imitation qualification after paid64 probe. No evaluation data, closed-loop quality guarantee or new physical query.')
    save(OUT/'training_finished.json',result);(OUT/'RUN_STATE.md').write_text('FITS COMPLETE:structural qualification '+str(result['structural_qualified'])+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='models'}));print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results]))
if __name__=='__main__':main()
