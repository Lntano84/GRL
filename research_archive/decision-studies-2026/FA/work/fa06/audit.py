"""Independent feedback, weights, votes and one-time scoring after the seal.

Does not import FA06 common/runner or refit RF models.
"""
import ast
import csv
import hashlib
import itertools
import json
import math
import pickle
import statistics
import time
from pathlib import Path
import arff
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fa06'
DATA=ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO'
PAIRS=list(itertools.combinations(range(11),2));ARMS=['ORIGINAL','UNIT','NATIVE-UPPER']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)
def key(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,parts)).encode()).digest()[:4],'big')
def source(name):
    with (DATA/f'{name}.arff').open(encoding='utf-8') as f:
        for line in f:
            if line.strip().lower()=='@data':break
        yield from (r for r in csv.reader(f) if r and not r[0].lstrip().startswith('%'))

def main():
    start=time.perf_counter();d=read(ROOT/'outputs/fa06_design/FA06_design_freeze.json')
    seal=read(OUT/'FA06_predictions_seal.json')
    assert seal['fits']==9 and seal['evaluation_revealed'] is False
    assert all(sha(ROOT/p)==h for p,h in seal['files'].items())
    write('FA06_evaluator_freeze.json',dict(frozen_unix_s=time.time(),evaluator_sha256=sha(__file__),
        design_sha256=sha(ROOT/'outputs/fa06_design/FA06_design_freeze.json'),predictions_seal_sha256=sha(OUT/'FA06_predictions_seal.json')))
    frozen=read(OUT/'FA06_run_freeze.json');assert all(sha(ROOT/p)==h for p,h in frozen['code_hashes'].items())
    assert all(sha(ROOT/p)==h for p,h in d['input_hashes'].items())
    # Confirm the copied random-row policy is syntactically the same function.
    def row_ast(p):
        tree=ast.parse(Path(p).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='row_batch'),include_attributes=False)
    assert row_ast(ROOT/'work/fa06/common.py')==row_ast(ROOT/'work/fa03b/policy.py')
    train=set(d['train']);test=d['test'];reserved=set(d['reserved_unused'])
    assert len(train)==1048 and len(test)==129 and len(reserved)==117
    assert train.isdisjoint(test) and train.isdisjoint(reserved) and reserved.isdisjoint(test)
    by_id={name:i for i,name in enumerate(d['ids'])};by_algorithm={name:a for a,name in enumerate(d['algorithms'])}
    truth={}
    for r in source('algorithm_runs'):
        if int(r[1])!=1 or by_id[r[0]] not in train:continue
        truth[(by_id[r[0]],by_algorithm[r[2]])]=(float(r[3]),r[4])
    assert len(truth)==1048*11
    with (DATA/'feature_values.arff').open() as f:tab=arff.load(f)
    raw_by_id={r[0]:r[2:] for r in tab['data'] if r[1]==1}
    raw=np.array([[np.nan if v is None else float(v) for v in raw_by_id[i]] for i in d['ids']])
    with np.load(OUT/'features.npz') as z:
        x=z['X'];keep=z['keep'];med=z['medians'];mean=z['mean'];scale=z['scale']
        assert np.array_equal(z['raw_features'],raw,equal_nan=True)
    ix=np.array(sorted(train));expected_keep=~np.isnan(raw[ix]).all(axis=0)
    assert np.array_equal(keep,expected_keep)
    assert np.allclose(med,np.nanmedian(raw[ix][:,keep],axis=0),rtol=1e-12,atol=1e-12)
    imputed=np.where(np.isnan(raw[:,keep]),med,raw[:,keep])
    assert np.allclose(mean,imputed[ix].mean(axis=0),rtol=1e-10,atol=1e-8)
    std=imputed[ix].std(axis=0)
    assert np.allclose(scale[std>1e-7],std[std>1e-7],rtol=1e-8,atol=1e-8)
    assert np.allclose(x,(imputed-mean)/scale,rtol=1e-12,atol=1e-12)
    audit_runs=[];saved_predictions={};label_events=0;paid_total=0;models_checked=0
    for seed in d['seeds']:
        dest=OUT/'collections'/f's{seed}';events=[json.loads(line) for line in (dest/'actions.jsonl').open()]
        visible={};cost=0.;paid=0;initial_seen=False;next_round=0;pending=[];cursor=0
        for count,e in enumerate(events):
            assert e['event_id']==count
            if e['type']=='batch':
                if e['initial']:
                    assert not initial_seen;initial_seen=True
                    expected=[[int(i),a] for i in d['initial_rows'][str(seed)] for a in range(11)]
                else:
                    next_round+=1;assert e['round']==next_round
                    pool=np.array([i for i in sorted(train) if any(visible.get((i,a),(0,0))[0]==0 for a in range(11))])
                    order=np.random.default_rng(key(seed,'row_acquire',next_round)).permutation(len(pool))
                    chosen=pool[order[:11]].copy()
                    np.random.default_rng(key(seed,'row_execute',next_round)).shuffle(chosen)
                    assert e['selected_rows']==chosen.tolist()
                    expected=[]
                    for i0 in chosen:
                        i=int(i0);alg=np.arange(11)
                        np.random.default_rng(key(seed,'row_alg',next_round,i)).shuffle(alg)
                        expected.extend([[i,int(a)] for a in alg if visible.get((i,int(a)),(0,0))[0]==0])
                assert e['work']==expected;pending=expected;cursor=0
            elif e['type']=='observation':
                i,a=e['i'],e['a'];assert [i,a]==pending[cursor] and e['work_index']==cursor;cursor+=1
                assert i in train and e['role']=='train' and not e['cached']
                k,v=visible.get((i,a),(0,0));assert e['before_kind']==k;close(e['before_value'],v)
                assert k==0
                cap=min(100,max(0,86460-cost));close(e['actual_cap_s'],cap)
                runtime,status=truth[(i,a)];completed=status=='ok' and runtime<=cap
                fee=runtime if completed else cap;kind=2 if completed else 1;value=runtime if completed else cap
                close(e['charged_s'],fee);assert e['kind']==kind;close(e['value'],value)
                cost+=fee;assert cost<=86460+1e-8;visible[(i,a)]=(kind,value);paid+=1
            elif e['type']=='batch_complete':assert cursor==len(pending)
            elif e['type']=='collection_end':assert e['reason'] in ['budget','pool_exhausted']
            else:raise AssertionError(e['type'])
            close(e['train_cost_s'],cost);close(e['validation_cost_s'],0)
        close(cost,86460)
        with np.load(dest/'visible_cache.npz') as z:
            for i in range(len(d['ids'])):
                for a in range(11):
                    k,v=visible.get((i,a),(0,0));assert z['kind'][i,a]==k;close(z['value'][i,a],v)
        tables=read(dest/'labels.json');assert len(tables)==55
        for pi,((a,b),table) in enumerate(zip(PAIRS,tables)):
            expected={}
            for i in train:
                ka,va=visible.get((i,a),(0,0));kb,vb=visible.get((i,b),(0,0))
                if not ka or not kb or 2 not in (ka,kb):continue
                if ka==kb==2:
                    if va==vb:continue
                    y=int(vb<va);w=abs(va-vb);upper=w
                else:
                    l,v=(va,vb) if ka==1 else (vb,va)
                    if l<v:continue
                    y=1 if ka==1 else 0;w=abs(10*l-v);upper=6000-v
                expected[i]=(y,w,upper,ka,kb,va,vb)
            assert {r['i'] for r in table}==set(expected)
            for r in table:
                y,w,upper,ka,kb,va,vb=expected[r['i']]
                assert r['label']==y and (r['ka'],r['kb'])==(ka,kb)
                close(r['ORIGINAL'],w);close(r['UNIT'],1);close(r['NATIVE-UPPER'],upper)
                close(r['va_visible'],va);close(r['vb_visible'],vb)
        label_events+=sum(map(len,tables));paid_total+=paid
        all_inputs=[];support_hashes=set();input_arrays=[]
        for arm in ARMS:
            fit=OUT/'fits'/f'{arm}_s{seed}';meta=read(fit/'input.json');support_hashes.add(meta['support_hash'])
            with np.load(fit/'input.npz') as z:
                rows=z['rows'];y=z['y'];w=z['weights'];offsets=z['offsets'];model_seeds=z['model_seeds']
                input_arrays.append((rows.copy(),y.copy(),model_seeds.copy()))
                assert rows.tolist()==[r['i'] for table in tables for r in table]
                assert y.tolist()==[r['label'] for table in tables for r in table]
                assert np.array_equal(w,np.array([r[arm] for table in tables for r in table]))
                assert offsets.tolist()==np.r_[0,np.cumsum(list(map(len,tables)))].tolist()
                assert model_seeds.tolist()==[key(seed,'pair','FA06-terminal',pi) for pi in range(55)]
            votes=[]
            for pi,((a,b),table) in enumerate(zip(PAIRS,tables)):
                with (fit/f'pair_{pi:02d}.pkl').open('rb') as f:model=pickle.load(f)
                if model is not None:
                    assert model.random_state==int(model_seeds[pi])
                    for k,v in d['model'].items():assert model.get_params()[k]==v
                    p=model.predict(x[test]).astype(int);votes.append(np.where(p==0,a,b));models_checked+=1
            assert np.array_equal(np.asarray(votes),np.load(fit/'pair_votes.npy'))
            manual=[]
            for vote_row in np.asarray(votes).T:
                hist=np.bincount(vote_row,minlength=11);maximum=hist.max()
                manual.append(next(int(a) for a in vote_row if hist[a]==maximum))
            pred=np.load(fit/'predictions.npy')
            assert np.array_equal(pred,manual) and pred.shape==(129,)
            saved_predictions[(arm,seed)]=pred
        assert len(support_hashes)==1
        for other in input_arrays[1:]:
            assert all(np.array_equal(a,b) for a,b in zip(input_arrays[0],other))
        audit_runs.append(dict(seed=seed,paid_executions=paid,charged_s=cost,labels=sum(map(len,tables)),
            common_support_verified=True,all_votes_verified=True))
    before=read(OUT/'FA06_historical_before.json')
    assert all(sha(ROOT/p)==h for p,h in before.items())
    training_check=dict(status='passed',runs=audit_runs,paid_executions=paid_total,label_entries=label_events,
        fit_inputs_verified=9,models_repredicted=models_checked,historical_unchanged=len(before),
        independent_rf_refits=0,scoring_started=False,wall_s=time.perf_counter()-start)
    write('FA06_training_audit.json',training_check)
    # Numeric evaluation outcomes enter only here, after the entire training audit.
    if (OUT/'FA06_scores.json').exists():
        scores=read(OUT/'FA06_scores.json')
        print(json.dumps({'status':'existing_scores_preserved','training_audit':'passed'}));return
    scores_by_row={};testset=set(test)
    for r in source('algorithm_runs'):
        if int(r[1])!=1 or by_id[r[0]] not in testset:continue
        i,a=by_id[r[0]],by_algorithm[r[2]];runtime=float(r[3]);status=r[4]
        assert status in ['ok','timeout'] and runtime<=600
        scores_by_row[(i,a)]=(runtime if status=='ok' else 6000,status=='timeout')
    assert len(scores_by_row)==129*11
    per_run=[];per_instance=[]
    for seed in d['seeds']:
        baseline=saved_predictions[('ORIGINAL',seed)]
        for arm in ARMS:
            pred=saved_predictions[(arm,seed)]
            losses=[scores_by_row[(i,int(a))][0] for i,a in zip(test,pred)]
            timeout_count=sum(scores_by_row[(i,int(a))][1] for i,a in zip(test,pred))
            result=read(OUT/'fits'/f'{arm}_s{seed}'/'result.json')
            per_run.append(dict(arm=arm,seed=seed,par10_s=math.fsum(losses)/129,
                native_timeouts=timeout_count,choice_disagreements_vs_original=int(np.count_nonzero(pred!=baseline)),
                fit_s=result['fit_s'],predict_s=result['predict_s']))
            for i,a,loss in zip(test,pred,losses):
                per_instance.append(dict(arm=arm,seed=seed,row=i,algorithm=int(a),native_par10_s=loss))
    means={arm:statistics.mean(r['par10_s'] for r in per_run if r['arm']==arm) for arm in ARMS}
    improvements={arm:1-means[arm]/means['ORIGINAL'] for arm in ARMS[1:]}
    baseline={r['seed']:r['par10_s'] for r in per_run if r['arm']=='ORIGINAL'}
    better={arm:sum(r['par10_s']<baseline[r['seed']]-1e-9 for r in per_run if r['arm']==arm) for arm in ARMS[1:]}
    simple=[arm for arm in ARMS[1:] if improvements[arm]>=.05 and better[arm]>=2]
    original_robust=all(1-means['ORIGINAL']/means[arm]>=.05 and
        sum(baseline[r['seed']]<r['par10_s']-1e-9 for r in per_run if r['arm']==arm)>=2 for arm in ARMS[1:])
    close_gate=all(abs(means[arm]/means['ORIGINAL']-1)<=.02 for arm in ARMS[1:])
    verdict=('simple_rule_development_signal' if simple else 'original_robust' if original_robust
             else 'close_in_this_protocol' if close_gate else 'undetermined')
    scores=dict(stage='FA06',per_run=per_run,means=means,improvements_vs_original=improvements,
        better_seeds=better,simple_signals=simple,original_robust=original_robust,close_gate=close_gate,
        verdict=verdict,scored_unix_s=time.time(),predictions_seal_unix_s=seal['sealed_unix_s'],
        independent_refits=0,wall_s=time.perf_counter()-start)
    for filename,rows in [('FA06_per_run.csv',per_run),('FA06_per_instance.csv',per_instance)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('FA06_scores.json',scores)
    write('RUN_STATE.json',dict(status='complete_pending_report',completed_fits=9,active_jobs=False,
        verdict=verdict,scoring_started=True))
    print(json.dumps(scores))

if __name__=='__main__':main()
