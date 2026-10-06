"""Independent raw-matrix fee/feedback/label audit, then terminal-only scoring."""
import csv, gzip, hashlib, itertools, json, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/fa00'))
from core import Selector, keyed_seed, PARAMS
OUT=ROOT/'outputs/fa02_end'

def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def audit():
    started=time.perf_counter()
    p=json.loads((ROOT/'outputs/fa00/FA00_protocol.json').read_text())
    data=np.load(ROOT/'outputs/fa00/FA00_data.npz');X=data['X'];rt=data['runtimes'];ok=data['ok']
    freeze=json.loads((OUT/'FA02_freeze.json').read_text())
    for r in freeze['files']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    results=json.loads((OUT/'FA02_training_results.json').read_text());assert len(results)==18
    pairs=list(itertools.combinations(range(len(p['algorithms'])),2));test=np.asarray(p['test'])
    rows=[];scores=[];paid_total=label_total=identity_checks=0;max_fee_error=0.
    for r in results:
        dest=Path(r['artifact']);saved=np.load(dest/'visible_cache.npz')
        paid=json.loads((dest/'paid_prefix.json').read_text());labels=json.loads((dest/'labels.json').read_text())
        stage='fa00' if r['arm']=='FIXED-100' else 'fa01'
        src=ROOT/f'outputs/{stage}/runs/{r["arm"]}_s{r["seed"]}'
        native={}
        with gzip.open(src/'actions.jsonl.gz','rt') as f:
            for line in f:
                e=json.loads(line)
                if e['type']=='observation':native[e['event_id']]=e
                if e['type']=='observation' and e['train_cost_s']>r['limit_s']+1e-8:break
        kinds=np.zeros_like(saved['kind']);values=np.zeros_like(saved['value']);cost=0.;seen=set()
        for q in paid:
            e=native[q['event_id']];i,a=q['i'],q['a'];assert (i,a) not in seen
            seen.add((i,a));assert i in p['train'] and not e['cached']
            cap=min(100.,r['limit_s']-cost)
            # Exact complete checkpoints tolerate accumulated float roundoff only.
            completed=bool(ok[i,a] and rt[i,a]<=cap+1e-8)
            fee=float(rt[i,a]) if completed else cap
            kind=2 if completed else 1;value=float(rt[i,a]) if completed else cap
            max_fee_error=max(max_fee_error,abs(fee-q['charged_s']))
            assert abs(fee-q['charged_s'])<1e-7 and kind==q['kind'] and abs(value-q['value'])<1e-7
            kinds[i,a]=kind;values[i,a]=value;cost+=fee
        assert abs(cost-r['acquisition_cost_s'])<1e-7 and cost<=r['limit_s']+1e-7
        assert np.array_equal(kinds,saved['kind']) and np.allclose(values,saved['value'],rtol=0,atol=1e-7)
        # Full canonical labels calculated from legal cache; no core synchronizer.
        expected=[]
        for a,b in pairs:
            z=[]
            for i in sorted(p['train']):
                ka,kb=int(kinds[i,a]),int(kinds[i,b]);va,vb=values[i,a],values[i,b]
                if not ka or not kb or (ka!=2 and kb!=2):continue
                if ka!=2 and va<vb-1e-10:continue
                if kb!=2 and vb<va-1e-10:continue
                if ka!=2 and va==vb:va=np.nextafter(va,np.inf)
                if kb!=2 and vb==va:vb=np.nextafter(vb,np.inf)
                z.append([i,float(va),float(vb),0 if va<=vb else 1,
                          abs((va if ka==2 else 10*va)-(vb if kb==2 else 10*vb))])
            expected.append(z)
        assert expected==labels
        with (dest/'selector.pkl').open('rb') as f:s=pickle.load(f)
        assert np.array_equal(s.cache.kind,kinds) and np.allclose(s.cache.value,values,rtol=0,atol=1e-7)
        assert s.seed==r['seed'] and np.array_equal(s.train_rows,p['train'])
        assert np.array_equal(s.X,X)
        for pi,m in enumerate(s.pair_models):
            if m is None:continue
            assert m.learner.estimator.random_state==keyed_seed(r['seed'],'pair',0,pi)
            for k,v in PARAMS.items():assert m.learner.estimator.get_params()[k]==v
            lab=[e for e in expected[pi] if e[4]>0];a,b=pairs[pi]
            assert np.array_equal(m.initial_train_data[s.features].to_numpy(),X[[e[0] for e in lab]])
            assert np.array_equal(m.initial_train_data[p['algorithms'][a]].to_numpy(),[e[1] for e in lab])
            assert np.array_equal(m.initial_train_data[p['algorithms'][b]].to_numpy(),[e[2] for e in lab])
            assert np.array_equal(m.sample_weight,[e[4] for e in lab])
        for a,m in enumerate(s.regressors):
            assert m.model.random_state==keyed_seed(r['seed'],'reg',0,a)
            inds=s.train_rows[kinds[s.train_rows,a]>0]
            assert np.array_equal(m.data[s.features].to_numpy(),X[inds])
            assert np.allclose(m.labels,values[inds,a],rtol=0,atol=1e-7)
        pred=np.load(dest/'predictions.npy');assert np.array_equal(s.predict(),pred)
        # Same fitted model and visible input yield identical output for each arm
        # adapter; no extra fitting or model selection in this guard.
        for arm in ['FIXED-100','RANDOM-100','CHEAP-100']:
            alias=Selector(X,p['algorithms'],p['train'],s.cache,s.seed)
            alias.label_rows=s.label_rows;alias.pair_models=s.pair_models
            assert np.array_equal(alias.predict(),pred);identity_checks+=1
        selected=pred[test];values_test=np.where(ok[test,selected],rt[test,selected],6000.)
        score=float(values_test.mean());n_timeout=int((~ok[test,selected]).sum())
        rows.append(dict(r,par10=score,test_timeouts=n_timeout))
        for i,a,v in zip(test,selected,values_test):
            scores.append(dict(arm=r['arm'],seed=r['seed'],mode=r['mode'],i=int(i),
                               algorithm=int(a),par10=float(v)))
        paid_total+=len(paid);label_total+=sum(map(len,labels))
    means={mode:{arm:float(np.mean([r['par10'] for r in rows if r['mode']==mode and r['arm']==arm]))
                 for arm in ['FIXED-100','RANDOM-100','CHEAP-100']} for mode in ['COMPLETE','ALL']}
    comparisons={}
    for mode in means:
        for arm in ['RANDOM-100','CHEAP-100']:
            direction=sum(next(r['par10'] for r in rows if r['mode']==mode and r['arm']=='FIXED-100' and r['seed']==seed)
                          <next(r['par10'] for r in rows if r['mode']==mode and r['arm']==arm and r['seed']==seed) for seed in [7,42,99])
            gain=(means[mode][arm]-means[mode]['FIXED-100'])/means[mode][arm]
            comparisons[f'{mode}_vs_{arm}']=dict(relative_improvement=gain,better_seed_count=direction)
    active=means['ALL']['FIXED-100']
    close=any(means['ALL'][arm]<=1.02*active for arm in ['RANDOM-100','CHEAP-100'])
    cont=all(comparisons[f'ALL_vs_{arm}']['relative_improvement']>=.05 and comparisons[f'ALL_vs_{arm}']['better_seed_count']>=2 for arm in ['RANDOM-100','CHEAP-100'])
    verdict='ACCEPT_CHEAP_BASELINE' if close else 'RETAIN_LOW_BUDGET_HYPOTHESIS' if cont else 'UNDETERMINED'
    for name,items in [('FA02_per_fit.csv',rows),('FA02_test_predictions.csv',scores)]:
        with (OUT/name).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    save(OUT/'FA02_analysis.json',dict(means=means,comparisons=comparisons,verdict=verdict,results=rows))
    save(OUT/'FA02_audit.json',dict(passed=True,models=18,raw_matrix_paid_items=paid_total,
        independently_derived_labels=label_total,prediction_identity_checks=identity_checks,
        scored_predictions=len(scores),max_charge_error_s=max_fee_error,freeze_unchanged=True,
        wall_s=time.perf_counter()-started))
    state=json.loads((OUT/'RUN_STATE.json').read_text());state['status']='complete_audited';save(OUT/'RUN_STATE.json',state)
    print(json.dumps(dict(means=means,comparisons=comparisons,verdict=verdict),indent=2))

if __name__=='__main__':audit()
