"""Freeze comparison and qualify train-only next-batch regression labels."""
import collections,copy,importlib.metadata,json,pickle,shutil
import numpy as np
from common import ROOT,W,OLD,OUT,ARCHIVE,read,save,sha,load,normalized,test_sealed

def main():
    assert not OUT.exists(),'Preserve prior outputs'
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(OLD/name).items():assert sha(ROOT/p)==h,p
    test_sealed();assert read(OLD/'review.json')['passed']
    OUT.mkdir();cases,mean,scale,D=load()
    train=[c for c in cases if c['partition']=='train'];val=[c for c in cases if c['partition']=='validation']
    assert len(train)==2378 and len(val)==590
    assert not {c['date'] for c in train}&{c['date'] for c in val}
    X=[];y=[];dates=[];ids=[];counts=[]
    for c in train:
        for j,state in enumerate(c['states']):
            X.append(normalized(state,mean,scale));y.append(float(c['gains'][j])/D['gain_scale'])
            dates.append(c['date']);ids.append([c['sample_id'],j]);counts.append(int(c['queries'][j]))
    counter=collections.Counter(dates);weight=np.array([1/counter[d] for d in dates]);weight/=weight.mean()
    np.savez_compressed(OUT/'regression_train.npz',X=np.stack(X),y=np.asarray(y),weight=weight,next_queries=np.asarray(counts))
    save(OUT/'training_rows.json',ids)
    # Rebuild every reachable observation using ONLY purchased raw feedback.
    from observations import observe,best,reward
    meta={r['sample_id']:r for r in read(ROOT/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']}
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];cache={};checks=0
    for c in cases:
        m=meta[c['sample_id']];path=ROOT/m['source_call']
        if str(path) not in cache:cache[str(path)]=read(path)
        call=cache[str(path)];order=call['order'];truth={r['pool_id']:r for r in call['outcomes']}
        visible=[copy.deepcopy(truth[i]) for i in order[:m['visible_count']]]
        station=subs[best(visible)['pool_id']]
        locked={int(rep) for original,rep in call['aliases'].items() if subs[int(original)]==station}
        pending=[i for i in order[m['visible_count']:] if i in locked];budget=len(pending)
        assert len(pending)==int(c['queries'].sum())
        for j,state in enumerate(c['states']):
            rebuilt=observe(visible,order,call['aliases'],subs,call['rho_limit'],locked,pending,budget,j)
            assert np.array_equal(state,rebuilt)
            # Replace every unpurchased feedback row, then reconstruct the purchased prefix again.
            seen={v['pool_id'] for v in visible}
            poisoned={i:(copy.deepcopy(r) if i in seen else dict(r,reward=-1e12,rho=1e12,strict_admissible=False)) for i,r in truth.items()}
            visible2=[poisoned[r['pool_id']] for r in visible]
            assert np.array_equal(state,observe(visible2,order,call['aliases'],subs,call['rho_limit'],locked,pending,budget,j))
            q=int(c['queries'][j]);assert q==min(64,len(pending))
            before=reward(best(visible));visible.extend(copy.deepcopy(truth[i]) for i in pending[:q]);pending=pending[q:]
            assert reward(best(visible))-before==float(c['gains'][j])
            assert best(visible)['pool_id']==c['chosen'][j+1]
            checks+=1
    # Boundary toy: a temporarily unprofitable batch can unlock positive net utility.
    assert np.argmax(np.r_[0.,np.cumsum([-1.,3.])])==2
    assert np.argmax(np.r_[0.,np.cumsum([-1.,.5])])==0
    design=dict(id='GRID103_LEARNING_MECHANISM',scope='Saved-feedback development mechanism probe; no physical rollout or final test.',
        source='GRID102 frozen transitions and all3 final DQN models; original artifacts unchanged.',
        split=dict(train_cases=2378,train_dates=16,validation_cases=590,validation_dates=4),
        inputs='Same41 purchased-feedback/public-structure features and frozen training normalization as GRID102.',
        cost_per_query=D['cost_per_query'],gain_scale=D['gain_scale'],
        regression=dict(target='Next batch nonnegative reward gain divided by GRID102 gain_scale; no future-optimal or whole-week label.',
            model='RandomForestRegressor',n_estimators=64,max_depth=4,min_samples_leaf=20,max_features=1.,random_state=20261022,n_jobs=1,
            weights='Equal total weight per training date across all reachable states; mean weight1.',
            decision='Continue iff predicted normalized next-batch gain > fixed query price * known next-batch count / gain_scale. Strict tie stops.'),
        temporal_ablation=dict(model='SB3 vanilla DQN, identical architecture/seed/training budget to GRID102; gamma0 replaces gamma1 only.',
            seeds=D['seeds'],timesteps=D['timesteps'],dqn={**D['dqn'],'gamma':0.},
            purpose='Test value of future-reward training under matched implementation; trajectories differ after policies diverge, not identical replay buffers.'),
        fitting='Exactly1 forest and3 gamma0 DQN fits; retain all final30000 checkpoints and seeds. Reuse gamma1 models without fitting.',
        inference='Frozen41-dim observation and known next-batch count only; no case ID/date/remaining outcomes or labels.',
        baselines=['STOP','FULL_LOCAL','BATCH1','COST_LOGISTIC_existing','MYOPIC_RF64','DQN_G1_seed0/1/2_existing','DQN_G0_seed0/1/2',
                   'ORACLE_OPTIMAL_hindsight_only','ORACLE_MYOPIC_next_actual_gain_hindsight_only'],
        analyses=['all/single-batch/multi-batch/horizon2/3/4 date macros','paired gamma1-gamma0 all3 seeds and each date',
            'prefix-weighted and physical-call-balanced within-date utility','next-batch regression MAE/RMSE versus zero/training-mean',
            'negative-next-batch positive-later hindsight opportunities','October negative-utility decomposition and missed gain',
            'all training curves, visited decisions, prediction scores, and adverse cases'],
        interpretation='Gamma1 superiority requires positive mean and at least2/3 seed differences on multi-batch states; report dates and opposite cases. This is exploratory,4 exposed validation dates, not statistical confirmation. If absent, do not claim sequential RL benefit.',
        limits='Fixed train-calibrated proxy price; no cost sweep, threshold tuning, extra features/data, larger net, checkpoint/seed selection, installation, new physical rollout or final-test generation. Prefixes correlated; hindsight diagnostics cannot be deployed.')
    save(OUT/'design.json',design)
    save(OUT/'preflight.json',dict(passed=True,reconstructed_states=checks,hidden_suffix_perturbations=checks,
        independent_next_batch_gain_checks=checks,train_states=len(y),train_positive_batches=int(np.sum(np.asarray(y)>0)),
        fitting0=True,final_test_materialized=False,new_physical_calls=0))
    save(OUT/'environment.json',dict(packages={n:importlib.metadata.version(n) for n in ['stable-baselines3','gymnasium','torch','numpy','scikit-learn']},
        sources=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
                 'https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html'],note='Web docs current; actual locally installed versions recorded above, no upgrade.'))
    seal()
    print(json.dumps(read(OUT/'preflight.json')),flush=True)

def seal():
    assert read(OUT/'preflight.json')['passed']
    assert not (OUT/'freeze.json').exists() and not (OUT/'models').exists()
    D=read(OLD/'design.json')
    files=list(W.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file()]
    files += [OLD/'cases.pkl',OLD/'normalization.npz',OLD/'design.json',OLD/'validation_trajectories.json',OLD/'validation_results.json',OLD/'models/cost_logistic.joblib']
    files += [OLD/'models'/f'dqn_seed{s}_final.zip' for s in D['seeds']]
    import sklearn
    files += list(__import__('pathlib').Path(sklearn.__file__).parent.rglob('*.py'))
    def key(p):
        try:return p.relative_to(ROOT).as_posix()
        except ValueError:return p.as_posix()
    save(OUT/'freeze.json',{key(p):sha(p) for p in files})
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
if __name__=='__main__':
    import sys
    if '--seal-existing' in sys.argv:seal();print('Qualified existing inputs sealed, no repeated preflight or fit.',flush=True)
    else:main()
