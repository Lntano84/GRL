"""Read-only train/held-out diagnosis from frozen GRID99 coefficients and trees."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/frame_learning_route_20261021';SOURCE=ROOT/'outputs/grid99_feature_learning'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def auc(y,p):
    a=[v for t,v in zip(y,p) if t];b=[v for t,v in zip(y,p) if not t]
    return float(sum((x>z)+.5*(x==z) for x in a for z in b)/(len(a)*len(b)))
def main():
    for n,h in read(SOURCE/'freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
    data={r['case_id']:r for r in read(SOURCE/'dataset.json')};results=read(SOURCE/'results.json');fits=read(SOURCE/'fits.json');details=[];summary=[]
    for arm in ['ORIGINAL15','ENRICHED35']:
        for name in ['LOGISTIC','FOREST64']:
            scores=[]
            for f in fits:
                if f['arm']!=arm or f['model']!=name or f['scheme']!='calendar_loo':continue
                x=np.array([[data[i]['features'][k] for k in f['features']] for i in f['train_ids']]);y=[data[i]['y_same_path'] for i in f['train_ids']];e=f['export']
                if name=='LOGISTIC':p=1/(1+np.exp(-((x-np.array(e['mean']))/np.array(e['scale'])@np.array(e['coef'])+e['intercept'])))
                else:
                    p=[]
                    for row in x.astype(np.float32):
                        values=[]
                        for t in e['trees']:
                            n=0
                            while t['left'][n]!=-1:n=t['left'][n] if row[t['feature'][n]]<=t['threshold'][n] else t['right'][n]
                            v=t['value'][n];values.append(v[1]/sum(v))
                        p.append(sum(values)/len(values))
                value=auc(y,p);scores.append(value);details.append(dict(arm=arm,model=name,fold=f['fold'],train_ids=f['train_ids'],train_y=y,train_probability=[float(v) for v in p],train_resubstitution_auc=value))
            assert len(scores)==17
            test=next(z for z in results if z['arm']==arm and z['model']==name and z['scheme']=='calendar_loo')['primary_nonpilot']
            summary.append(dict(arm=arm,model=name,folds=17,train_auc_median=statistics.median(scores),train_auc_range=[min(scores),max(scores)],heldout_53_auc=test['auc'],heldout_53_ba=test['balanced_accuracy']))
    receipt=dict(scope='Post-hoc diagnosis, no fitting or new physical episodes. Training resubstitution AUC is not a generalization score.',summary=summary,folds=details,source_sha256={n:sha(SOURCE/n) for n in ['dataset.json','fits.json','results.json','freeze.json']},script_sha256=sha(Path(__file__)),new_fits=0,new_physical_replays=0,unchanged_parameters=True,
        interpretation='Current models can fit training states but fail on date-held-out states. Consistent with overfitting and/or date distribution shift; not proof that sample size alone explains failure. Do not compare the median of17 training scores to pooled OOF as a statistical significance test.',
        tabpfn='No in-context training-row evaluation: its own support labels would make that a different and potentially misleading diagnostic.')
    (OUT/'train_heldout_diagnosis.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
