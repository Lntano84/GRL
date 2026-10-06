"""Three formal shared caches, nine terminal fits; no scoring in this process."""
from common import *
import time
from sklearn.ensemble import RandomForestClassifier

def fit_one(d,x,tables,seed,arm,dest,deadline):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    support=support_hash(tables,x)
    model_seeds=[keyed_seed(seed,'pair','FA06-terminal',pi) for pi in range(55)]
    flat_rows=np.array([r['i'] for t in tables for r in t],dtype=int)
    flat_y=np.array([r['label'] for t in tables for r in t],dtype=np.int8)
    flat_w=np.array([r[arm] for t in tables for r in t],dtype=float)
    offsets=np.r_[0,np.cumsum([len(t) for t in tables])]
    h=hashlib.sha256(support.encode()+flat_w.tobytes()+np.array(model_seeds,dtype=np.uint32).tobytes())
    input_signature=h.hexdigest()
    result_path=dest/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['input_signature']==input_signature
        return result
    if (dest/'input.json').exists():
        assert json.loads((dest/'input.json').read_text())['input_signature']==input_signature
    else:
        np.savez_compressed(dest/'input.npz',rows=flat_rows,y=flat_y,weights=flat_w,offsets=offsets,model_seeds=model_seeds)
        save(dest/'input.json',dict(seed=seed,arm=arm,support_hash=support,input_signature=input_signature,
            rows=len(flat_rows),feature_sha256=sha(OUT/'features.npz'),params=d['model'],model_seeds=model_seeds))
    votes=[];fit_wall=0.;predict_wall=0.;models=[]
    for pi,((a,b),table) in enumerate(zip(PAIRS,tables)):
        if time.perf_counter()>=deadline:raise TimeoutError('FA06 wall limit; completed models retained')
        path=dest/f'pair_{pi:02d}.pkl';meta=dest/f'pair_{pi:02d}.json'
        if path.exists() and meta.exists():
            m=json.loads(meta.read_text());assert m['input_signature']==input_signature
            assert m['model_sha256']==sha(path)
            with path.open('rb') as f:model=pickle.load(f)
        else:
            model=None;begin=time.perf_counter()
            if table:
                ix=np.array([r['i'] for r in table]);y=np.array([r['label'] for r in table])
                weights=np.array([r[arm] for r in table])
                assert np.all(weights>0) and set(ix)<=set(d['train'])
                model=RandomForestClassifier(**d['model'],random_state=model_seeds[pi])
                model.fit(x[ix],y,sample_weight=weights)
            elapsed=time.perf_counter()-begin
            dump(path,model)
            m=dict(input_signature=input_signature,pair=pi,fit_s=elapsed,
                random_state=model_seeds[pi],samples=len(table),model_sha256=sha(path))
            save(meta,m)
        if model is not None:
            assert model.random_state==model_seeds[pi]
            begin=time.perf_counter();z=model.predict(x[d['test']]).astype(int)
            votes.append(np.where(z==0,a,b));predict_wall+=time.perf_counter()-begin
        fit_wall+=m['fit_s'];models.append(m)
        save(dest/'progress.json',dict(completed_pairs=pi+1,total_pairs=55,fit_s=fit_wall))
    assert votes
    pred=hard_vote(votes)
    np.save(dest/'pair_votes.npy',np.asarray(votes,dtype=np.int16))
    np.save(dest/'predictions.npy',pred)
    result=dict(seed=seed,arm=arm,support_hash=support,input_signature=input_signature,
        label_count=len(flat_rows),pair_models=sum(bool(t) for t in tables),
        fit_s=fit_wall,predict_s=predict_wall,predictions_sha256=sha(dest/'predictions.npy'),
        input_sha256=sha(dest/'input.npz'),model_hashes=[m['model_sha256'] for m in models])
    save(result_path,result);return result

def main():
    start=time.perf_counter();d=design();deadline=start+d['wall_limit_s']
    assert json.loads((OUT/'FA06_preflight.json').read_text())['status']=='passed'
    if (OUT/'FA06_scores.json').exists():raise RuntimeError('Scores already revealed; no rerun')
    if (OUT/'FA06_run_freeze.json').exists():
        freeze=json.loads((OUT/'FA06_run_freeze.json').read_text())
        assert all(sha(ROOT/p)==h for p,h in freeze['code_hashes'].items())
    else:
        save(OUT/'FA06_historical_before.json',historical())
        save(OUT/'FA06_run_freeze.json',dict(design_sha256=sha(ROOT/'outputs/fa06_design/FA06_design_freeze.json'),
            code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')},
            stage='formal caches and terminal fits; no scores',frozen_unix_s=time.time()))
    x,pre=features(d)
    if (OUT/'features.npz').exists():
        with np.load(OUT/'features.npz') as z:assert np.array_equal(z['X'],x)
    else:np.savez_compressed(OUT/'features.npz',X=x,**pre)
    runtime,ok=train_truth(d);collections=[]
    try:
        for seed in d['seeds']:
            dest=OUT/'collections'/f's{seed}'
            cache,tables,events,result=collect(d,runtime,ok,seed,dest,deadline)
            collections.append(result)
            print('collection',seed,'executions',result['executions'],'labels',result['labels'],flush=True)
        save(OUT/'FA06_collection_results.json',collections)
        results=[]
        for seed in d['seeds']:
            tables=json.loads((OUT/'collections'/f's{seed}'/'labels.json').read_text())
            for arm in ARMS:
                result=fit_one(d,x,tables,seed,arm,OUT/'fits'/f'{arm}_s{seed}',deadline)
                results.append(result)
                save(OUT/'RUN_STATE.json',dict(status='fitting',completed_fits=len(results),total_fits=9))
                print('fit',len(results),arm,seed,'fit_s',round(result['fit_s'],3),flush=True)
        save(OUT/'FA06_training_results.json',results)
        paths=[p for p in (OUT/'fits').rglob('*') if p.is_file()]
        paths += [OUT/'features.npz',OUT/'FA06_collection_results.json',OUT/'FA06_training_results.json']
        paths += [p for p in (OUT/'collections').rglob('*') if p.is_file()]
        save(OUT/'FA06_predictions_seal.json',dict(sealed_unix_s=time.time(),fits=9,
            files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
            evaluation_revealed=False,wall_s=time.perf_counter()-start))
        before=json.loads((OUT/'FA06_historical_before.json').read_text())
        assert historical()==before
        save(OUT/'RUN_STATE.json',dict(status='predictions_sealed_pending_independent_audit',completed_fits=9,
            scoring_started=False,wall_s=time.perf_counter()-start))
        print('all predictions sealed; no scoring',round(time.perf_counter()-start,3),flush=True)
    except BaseException as e:
        save(OUT/'RUN_STATE.json',dict(status='interrupted_prefix_retained',error=str(e),wall_s=time.perf_counter()-start))
        raise

if __name__=='__main__':main()
