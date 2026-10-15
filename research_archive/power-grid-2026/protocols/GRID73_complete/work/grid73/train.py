from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid73';sys.path.insert(0,str(Path(__file__).parent));import leaf_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not (OUT/'training_finished.json').exists();D=read(OUT/'design.json')
    for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    with np.load(ROOT/'outputs/grid72/inputs.npz') as z:X=z['X'];y=z['y'];train=z['train']
    items=read(ROOT/'outputs/grid72/input_receipts.json')['items'];models=OUT/'models';models.mkdir(exist_ok=False)
    results=[];start=time.perf_counter()
    for name in D['models']:
        k=D[name+'_features'];fit=DecisionTreeClassifier(**D['tree']);fit.fit(X[train,:k],y[train]);t=fit.tree_;values=t.value[:,0,:]
        model=dict(name=name,feature_count=k,trees=[dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),
            threshold=t.threshold.tolist(),positive=(values[:,1]/values.sum(axis=1)).tolist())],
            support=[[i,float(X[train,i].min()),float(X[train,i].max())] for i in G.G.SUPPORT if i<k],stop_leaves=[],leaf_support={})
        pred=fit.apply(X[:,:k]);assert all(G.leaf(model,x)==p for x,p in zip(X,pred))
        parity=max(abs(G.G.predict(model,x)-p) for x,p in zip(X,fit.predict_proba(X[:,:k])[:,1]));assert parity<1e-12
        for node in set(int(v) for v in pred):
            tr=[j for j,p in enumerate(pred) if p==node and train[j]];va=[j for j,p in enumerate(pred) if p==node and not train[j]]
            sup=dict(train_cases=len(tr),train_positive=sum(int(y[j]) for j in tr),train_families=len({items[j]['family'] for j in tr}),
                calibration_cases=len(va),calibration_positive=sum(int(y[j]) for j in va))
            model['leaf_support'][str(node)]=sup
            if len(tr)>=4 and sup['train_positive']==0 and sup['train_families']>=3 and len(va)>=1 and sup['calibration_positive']==0:model['stop_leaves'].append(node)
        model['stop_leaves'].sort();rows=[]
        for j,item in enumerate(items):
            defer,info=G.decide(model,X[j]);rows.append(dict(id=item['id'],family=item['family'],split=item['split'],changed_action=item['changed_action'],
                defer=defer,queries=item['guard_queries'] if defer else item['initial_queries'],guard_queries=item['guard_queries'],
                false_stop=bool(item['changed_action'] and not defer),**info))
        val=[r for r in rows if r['split']=='calibration'];assert not any(r['false_stop'] for r in val)
        saving=1-sum(r['queries'] for r in val)/sum(r['guard_queries'] for r in val)
        result=dict(name=name,portable_error=parity,calibration_query_saving=saving,calibration_skipped=sum(not r['defer'] for r in val),
            calibration_false_stop=sum(r['false_stop'] for r in val),stop_leaves=model['stop_leaves'],qualified=saving>=.10,rows=rows)
        save(models/f'{name}.json',model);save(models/f'{name}_review.json',result);results.append(result)
    result=dict(passed_engineering=True,fits=2,wall_s=time.perf_counter()-start,models=results,structural_qualified=results[1]['qualified'],
        structural_increment_over_plain=results[1]['calibration_query_saving']-results[0]['calibration_query_saving'],new_physical_queries=0,
        scope='Local calibration uses same14 safe states; not independent validation, confidence or closed-loop quality guarantee. All13 evaluation families excluded.')
    save(OUT/'training_finished.json',result)
    (OUT/'RUN_STATE.md').write_text('FITS COMPLETE:structural qualification '+str(result['structural_qualified'])+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='models'}));print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results]))
if __name__=='__main__':main()
