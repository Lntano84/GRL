"""Independent fee, feedback, sampling and first-label reconstruction."""
from common import *
import arff, csv, itertools, time

def derive(kind,value,train,pairs,labels):
    new={}
    for pi,(a,b) in enumerate(pairs):
        for i in train:
            if i in labels[pi]:continue
            ka,kb=int(kind[i,a]),int(kind[i,b]);va,vb=float(value[i,a]),float(value[i,b])
            if ka==0 or kb==0 or (ka!=2 and kb!=2):continue
            if ka!=2 and va<vb-1e-10:continue
            if kb!=2 and vb<va-1e-10:continue
            if ka!=2 and va==vb:va=float(np.nextafter(va,np.inf))
            if kb!=2 and vb==va:vb=float(np.nextafter(vb,np.inf))
            label=int(va>vb)
            weight=abs((va if ka==2 else 10*va)-(vb if kb==2 else 10*vb))
            new[(pi,i)]=(va,vb,label,weight,ka,kb,a,b)
    return new

def main():
    started=time.perf_counter();verify_design()
    freeze=json.loads((OUT/'FA04_execution_freeze.json').read_text())
    for path,value in freeze['files'].items():assert sha(ROOT/path)==value,path
    assert json.loads((OUT/'RUN_STATE.json').read_text())['completed']==3
    training=json.loads((OUT/'FA04_training_results.json').read_text())
    assert len(training)==3 and all(r['status']=='complete' for r in training)
    # Freeze every final predictor BEFORE reading the performance table.
    manifest={}
    for r in training:
        dest=OUT/'runs'/f'{r["arm"]}_s{r["seed"]}'
        for name,value in json.loads((dest/'artifacts.json').read_text()).items():
            assert sha(dest/name)==value,(dest,name)
        for name in ['predictions.npy','visible_cache.npz','labels.json','selector.pkl','actions.jsonl','result.json']:
            path=dest/name;manifest[str(path.relative_to(ROOT))]=sha(path)
    save(OUT/'FA04_prediction_freeze.json',dict(files=manifest,predictors=3,performance_table_read_after_freeze=True))
    p,d=load_data()
    source=ROOT/'work/fa00/upstream/DATASETS/ASP-POTASSCO/algorithm_runs.arff'
    raw=arff.load(source.open())['data']
    records={(r[0],r[2]):r for r in raw if r[1]==1}
    assert len(records)==len(p['ids'])*len(p['algorithms'])
    runtime=np.array([[records[(i,a)][3] for a in p['algorithms']] for i in p['ids']],float)
    ok=np.array([[records[(i,a)][4]=='ok' for a in p['algorithms']] for i in p['ids']],bool)
    assert np.array_equal(runtime,d['runtimes']) and np.array_equal(ok,d['ok'])
    pairs=list(itertools.combinations(range(len(p['algorithms'])),2))
    train=set(p['train']);totals=dict(observations=0,paid_observations=0,label_events=0,batches=0,test_predictions=0)
    reports=[];prediction_rows=[];maxerr=0.
    for result in training:
        arm,seed=result['arm'],result['seed'];dest=OUT/'runs'/f'{arm}_s{seed}'
        kind=np.zeros(runtime.shape,np.int8);value=np.zeros(runtime.shape,float)
        labels=[{} for _ in pairs];queue={};spent=0.;pending=None;cursor=0
        counts=dict(executions=0,cache_hits=0,completions=0,cancellations=0)
        batches=0;label_events=0;initial_obs=[]
        overlaps=[];cost_ties=[];late_key_changes=[];complete_round_costs=[]
        with (dest/'actions.jsonl').open() as f:
            for event_id,line in enumerate(f):
                e=json.loads(line);assert e['event_id']==event_id
                if e['type']=='batch':
                    assert not queue
                    pending=e;cursor=0;batches+=int(not e['initial'])
                    if e['initial']:
                        expected=[[int(i),a] for i in p['initial_rows'][str(seed)] for a in range(len(p['algorithms']))]
                    else:
                        assert arm=='COARSE-PC-100'
                        pool=np.array(sorted(i for i in train if any(kind[i,a]<2 and (kind[i,a]==0 or value[i,a]<100-1e-10)
                                                                   for a in range(len(p['algorithms'])))),int)
                        archive=np.load(dest/f'candidates_r{e["round"]:04d}.npz')
                        assert np.array_equal(pool,archive['pool'])
                        maxima=archive['max_probabilities'];missing=archive['missing_models']
                        assert maxima.shape==(len(pool),55) and missing.shape==(55,)
                        assert np.all(np.isfinite(maxima)) and np.all(maxima>=.5-1e-12) and np.all(maxima<=1+1e-12)
                        expected_missing=np.array([not any(v[3]>0 for v in z.values()) for z in labels])
                        assert np.array_equal(missing,expected_missing) and np.all(maxima[:,missing]==.5)
                        # Independent explicit column accumulation versus runner's np.mean(axis=1).
                        mean=np.zeros(len(pool))
                        for j in range(55):mean+=1-maxima[:,j]
                        mean/=55
                        assert np.allclose(mean,archive['mean_uncertainty'],rtol=0,atol=1e-14)
                        # Use recorded formula values for exact tie handling, after validating them above.
                        score=archive['mean_uncertainty']
                        shuffled=np.random.default_rng(keyed_seed(seed,'row_acquire',e['round'])).permutation(len(pool))
                        rank=np.empty(len(pool),int);rank[shuffled]=np.arange(len(pool))
                        order=np.lexsort((rank,-score))
                        selected=pool[order[:min(11,len(pool))]].copy()
                        np.random.default_rng(keyed_seed(seed,'row_execute',e['round'])).shuffle(selected)
                        assert selected.tolist()==e['selected_rows']==archive['selected'].tolist()
                        serial_now=[[[int(i),*list(v)] for i,v in sorted(z.items())] for z in labels]
                        digest=hashlib.sha256(kind.tobytes()+value.tobytes())
                        digest.update(json.dumps(serial_now,separators=(',',':')).encode())
                        digest.update(d['X'][p['train']].tobytes())
                        digest.update(json.dumps(PARAMS,sort_keys=True).encode());digest.update(str(seed).encode())
                        assert str(archive['input_sha256'])==digest.hexdigest()
                        expected=[]
                        for i in selected:
                            algs=np.arange(len(p['algorithms']))
                            np.random.default_rng(keyed_seed(seed,'row_alg',e['round'],int(i))).shuffle(algs)
                            expected.extend([[int(i),int(a)] for a in algs if kind[i,a]<2 and (kind[i,a]==0 or value[i,a]<100-1e-10)])
                    assert e['work']==expected
                elif e['type']=='observation':
                    assert pending and e['work_index']==cursor
                    assert pending['work'][cursor]==[e['i'],e['a']];cursor+=1
                    i,a=e['i'],e['a'];assert i in train and e['role']=='train'
                    assert e['before_kind']==kind[i,a] and e['before_value']==value[i,a]
                    query=kind[i,a]<2 and (kind[i,a]==0 or value[i,a]<100-1e-10)
                    if not query:
                        assert e['cached'] and e['charged_s']==0 and e['actual_cap_s']==0
                        counts['cache_hits']+=1
                    else:
                        assert not e['cached']
                        cap=min(100,BUDGET-spent);assert cap>1e-9
                        success=bool(ok[i,a] and runtime[i,a]<=cap)
                        charge=float(runtime[i,a]) if success else cap
                        target_kind=2 if success else 1
                        target_value=float(runtime[i,a]) if success else max(value[i,a],cap)
                        for k,expected in [('charged_s',charge),('actual_cap_s',cap),('value',target_value)]:
                            err=abs(e[k]-expected);maxerr=max(maxerr,err);assert err<1e-7,(arm,seed,k)
                        assert e['kind']==target_kind
                        spent+=charge;counts['executions']+=1
                        counts['completions' if success else 'cancellations']+=1
                        kind[i,a]=target_kind;value[i,a]=target_value
                    assert e['kind']==kind[i,a] and e['value']==value[i,a]
                    totals['observations']+=1;totals['paid_observations']+=int(not e['cached'])
                    if pending['initial']:initial_obs.append(e)
                    elif arm=='COARSE-PC-100':
                        assert not queue
                        queue=derive(kind,value,p['train'],pairs,labels)
                elif e['type']=='pair_label':
                    if not queue:queue=derive(kind,value,p['train'],pairs,labels)
                    key=(e['pair'],e['i']);assert key in queue,(arm,seed,key)
                    va,vb,label,weight,ka,kb,a,b=queue.pop(key)
                    assert [e['ka'],e['kb'],e['a'],e['b'],e['label']]==[ka,kb,a,b,label]
                    assert e['va']==va and e['vb']==vb and abs(e['weight']-weight)<1e-9
                    labels[e['pair']][e['i']]=(va,vb,label,weight);label_events+=1
                elif e['type']=='batch_complete':
                    assert cursor==len(pending['work']) and not queue
                    assert not derive(kind,value,p['train'],pairs,labels)
                    complete_round_costs.append(spent)
                elif e['type']=='terminal_start':
                    assert not queue
                    queue=derive(kind,value,p['train'],pairs,labels)
                else:raise AssertionError(e['type'])
                err=abs(e['train_cost_s']-spent);maxerr=max(maxerr,err);assert err<1e-7
                assert e['validation_cost_s']==0
                assert abs(e['remaining_s']-max(0,BUDGET-spent))<1e-7
        assert not queue and not derive(kind,value,p['train'],pairs,labels)
        assert spent<=BUDGET+1e-7
        stored=np.load(dest/'visible_cache.npz')
        assert np.array_equal(kind,stored['kind']) and np.array_equal(value,stored['value'])
        serial=[[[int(i),*list(v)] for i,v in sorted(z.items())] for z in labels]
        assert serial==json.loads((dest/'labels.json').read_text())
        assert label_events==sum(map(len,labels))==result['labels']
        assert result['acquisition_s']==spent
        for k,v in counts.items():assert result[k]==v,(k,v,result[k])
        # Compare formal initialization to original logs with the new budget field excluded.
        import gzip
        old=[]
        with gzip.open(ROOT/f'outputs/fa00/runs/FIXED-100_s{seed}/actions.jsonl.gz','rt') as f:
            for line in f:
                e=json.loads(line)
                if e['phase']!='init':break
                if e['type']=='observation':old.append(e)
        assert len(initial_obs)==len(old)==220
        for a,b in zip(initial_obs,old):
            for k in ['i','a','kind','value','charged_s','before_kind','before_value','cached']:assert a[k]==b[k]
        with (dest/'selector.pkl').open('rb') as f:s=pickle.load(f)
        assert np.array_equal(s.cache.kind,kind) and np.array_equal(s.cache.value,value)
        assert labels_json(s)==serial and signature(s)==result['input_sha256']
        assert s.algorithms==p['algorithms'] and s.train_rows.tolist()==p['train']
        for pi,m in enumerate(s.pair_models):
            if m is not None:
                assert m.learner.estimator.random_state==keyed_seed(seed,'pair',0,pi)
                assert m.learner.estimator.n_estimators==100
        for a,m in enumerate(s.regressors):assert m.model.random_state==keyed_seed(seed,'reg',0,a)
        pred=np.load(dest/'predictions.npy')
        assert np.array_equal(s.predict(),pred)
        test=p['test'];score=np.where(ok[np.arange(len(pred)),pred],runtime[np.arange(len(pred)),pred],6000.)
        report=dict(result,par10=float(score[test].mean()),test_timeouts=int((~ok[test,pred[test]]).sum()),
            batch_count=batches,cheap_overlap_mean=float(np.mean(overlaps)) if overlaps else None,
            secondary_keys_change_batches=sum(late_key_changes),cost_tied_candidates_total=sum(cost_ties),
            independent_labels=label_events,fee_error_s=maxerr)
        reports.append(report)
        totals['label_events']+=label_events;totals['batches']+=batches
        for i in test:prediction_rows.append(dict(arm=arm,seed=seed,i=i,algorithm=int(pred[i]),par10=float(score[i])))
    # Re-score six inherited final predictions, without rerunning collection or fitting.
    old_results=json.loads((ROOT/'outputs/fa03b/FA03B_analysis.json').read_text())['results']
    for r in old_results:
        if r['arm'] not in ['FIXED-100','RANDOM-ROW']:continue
        artifact=(ROOT/f'outputs/fa02_end/models/FIXED-100_s{r["seed"]}_ALL') if r['arm']=='FIXED-100' else (ROOT/f'outputs/fa03b/runs/RANDOM-ROW_s{r["seed"]}')
        pred=np.load(artifact/'predictions.npy');test=p['test']
        score=np.where(ok[np.arange(len(pred)),pred],runtime[np.arange(len(pred)),pred],6000.)
        assert abs(float(score[test].mean())-r['par10'])<1e-9
        reports.append(dict(arm=r['arm'],seed=r['seed'],par10=float(score[test].mean()),inherited=True,
                            test_timeouts=int((~ok[test,pred[test]]).sum())))
        for i in test:prediction_rows.append(dict(arm=r['arm'],seed=r['seed'],i=i,algorithm=int(pred[i]),par10=float(score[i])))
    totals['test_predictions']=len(prediction_rows)
    means={arm:float(np.mean([r['par10'] for r in reports if r['arm']==arm])) for arm in ['COARSE-PC-100','RANDOM-ROW','FIXED-100']}
    coarse={r['seed']:r['par10'] for r in reports if r['arm']=='COARSE-PC-100'}
    comparisons={}
    for arm in ['RANDOM-ROW','FIXED-100']:
        vals={r['seed']:r['par10'] for r in reports if r['arm']==arm}
        comparisons[arm]=dict(improvement=(means[arm]-means['COARSE-PC-100'])/means[arm],
            better_seeds=sum(coarse[s]<vals[s]-1e-9 for s in SEEDS))
    primary=comparisons['RANDOM-ROW']
    verdict=('RETAIN_MATURE_ACTIVE_SIGNAL' if primary['improvement']>=.05 and primary['better_seeds']>=2
             else 'NO_PRACTICAL_MEAN_GAIN' if primary['improvement']<=.02 else 'UNDETERMINED')
    secondary=means['COARSE-PC-100']<=1.02*means['FIXED-100']
    save(OUT/'FA04_analysis.json',dict(means=means,comparisons=comparisons,verdict=verdict,
        secondary_competitive_vs_pareto=secondary,results=reports,FA03B_verdict_unchanged='UNDETERMINED'))
    save(OUT/'FA04_audit.json',dict(passed=True,counts=totals,max_fee_error_s=maxerr,
        wall_s=time.perf_counter()-started,source_sha256=sha(source),
        scope='Independent raw-table fee/feedback, explicit 55-column uncertainty average, one-pass sort, visible row pool, labels, final inputs and score reconstruction. Recorded online probabilities are used, not independently re-predicted for every round. Saved final models re-predict; all online models are not independently refitted.'))
    with (OUT/'FA04_test_predictions.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['arm','seed','i','algorithm','par10']);w.writeheader();w.writerows(prediction_rows)
    save(OUT/'RUN_STATE.json',dict(status='complete_audited',completed=3,target=3,test_scored=True,active_jobs=False))
    print(json.dumps(dict(verdict=verdict,means=means,comparisons=comparisons,audit=totals,wall_s=time.perf_counter()-started),indent=2))

if __name__=='__main__':main()
