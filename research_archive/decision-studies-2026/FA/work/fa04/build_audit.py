"""Adapt the independent FA03B ledger audit; no inherited files are edited."""
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
src=(BASE/'work/fa03b/audit.py').read_text(encoding='utf8')
src=src.replace('FA03B','FA04').replace("['completed']==6", "['completed']==3").replace('len(training)==6','len(training)==3').replace('predictors=6','predictors=3')
start=src.index("                    elif arm=='RANDOM-ROW':")
end=src.index("                    assert e['work']==expected",start)
src=src[:start]+'''                    else:
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
''' + src[end:]
src=src.replace("elif arm=='RANDOM-ROW':", "elif arm=='COARSE-PC-100':")
start=src.index('    # Re-score all nine inherited terminal predictions')
end=src.index("    totals['test_predictions']=len(prediction_rows)",start)
src=src[:start]+'''    # Re-score six inherited final predictions, without rerunning collection or fitting.
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
''' + src[end:]
start=src.index('    means={arm:')
end=src.index("    save(OUT/'FA04_audit.json'",start)
src=src[:start]+'''    means={arm:float(np.mean([r['par10'] for r in reports if r['arm']==arm])) for arm in ['COARSE-PC-100','RANDOM-ROW','FIXED-100']}
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
''' + src[end:]
src=src.replace("source_sha256=sha(source),", "source_sha256=sha(source),")
src=src.replace("scope='Independent raw-table fee/feedback, one-pass sort, visible row pool, labels, final inputs and score reconstruction. Saved final models re-predict; all online models are not independently refitted.'", "scope='Independent raw-table fee/feedback, explicit 55-column uncertainty average, one-pass sort, visible row pool, labels, final inputs and score reconstruction. Recorded online probabilities are used, not independently re-predicted for every round. Saved final models re-predict; all online models are not independently refitted.'")
src=src.replace('completed=6,target=6', 'completed=3,target=3')
assert not (HERE/'audit.py').exists()
(HERE/'audit.py').write_text(src,encoding='utf8')
print('Created independent FA04 ledger/formula/ranking/score audit.')
