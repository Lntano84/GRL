"""Construct saved-search transitions, qualify then freeze before model fitting."""
import copy,collections,gzip,importlib.metadata,json,pickle,platform,shutil,sys
import numpy as np
from common import ROOT,W,OUT,SOURCE,ARCHIVE,read,save,sha,test_sealed
from observations import NAMES,observe,best,reward


def main():
    assert not OUT.exists(),'Preserve prior run'
    assert read(SOURCE/'bulk_complete.json')['passed']
    test_sealed();OUT.mkdir()
    metadata=read(SOURCE/'bulk_dataset_manifest.json')['samples'];assert len(metadata)==2968
    source101=ROOT/'outputs/grid101_gain_learning'
    with np.load(source101/'data.npz') as z:previous=z['X']
    previous_rows=read(source101/'rows.json');assert [r['sample_id'] for r in metadata]==[r['sample_id'] for r in previous_rows]
    cache={};cases=[];hashes={};checks=collections.Counter()
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    for i,m in enumerate(metadata):
        path=ROOT/m['source_call']
        if str(path) not in cache:
            call=read(path);assert sha(path)==m['source_sha256'];cache[str(path)]=call
            hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        call=cache[str(path)];order=call['order'];truth={r['pool_id']:r for r in call['outcomes']}
        n=m['visible_count'];visible=[dict(truth[j]) for j in order[:n]];inc=best(visible)
        station=subs[inc['pool_id']]
        targets={int(rep) for original,rep in call['aliases'].items() if subs[int(original)]==station}
        pending=[j for j in order[n:] if j in targets];assert pending
        original_pending=pending.copy();initial=reward(inc);states=[];gains=[];queries=[];chosen=[inc['pool_id']]
        chunks=[pending[j:j+64] for j in range(0,len(pending),64)]
        for step,chunk in enumerate(chunks):
            state=observe(visible,order,call['aliases'],subs,call['rho_limit'],targets,pending,len(original_pending),step)
            if step==0:assert np.allclose(state[:37],previous[i],rtol=1e-6,atol=1e-6)
            # Future rows never enter observe(); poisoning the private suffix cannot affect current state.
            poisoned={j:dict(truth[j],reward=1e6,rho=-1.) for j in pending}
            assert not set(poisoned)&{r['pool_id'] for r in visible}
            assert np.array_equal(state,observe(copy.deepcopy(visible),order,call['aliases'],subs,call['rho_limit'],targets,pending,len(original_pending),step))
            states.append(state);before=reward(best(visible));visible.extend(dict(truth[j]) for j in chunk)
            current=best(visible);after=reward(current);assert after>=before
            gains.append(after-before);queries.append(len(chunk));chosen.append(current['pool_id']);pending=pending[len(chunk):]
            checks['prefix_observation_checks']+=1
        target_path=ROOT/m['dataset_path']/m['target_file'];target=read(target_path)
        assert sum(gains)==target['local_reward_gain'] and sum(queries)==target['known_completion_queries']
        assert chosen[-1]==target['after_id'] and len(set(order[:n]+original_pending))==n+sum(queries)
        hashes[target_path.relative_to(ROOT).as_posix()]=sha(target_path)
        cases.append(dict(sample_id=m['sample_id'],date=m['calendar_group'],partition=m['partition'],scenario=m['scenario'],
            physical_state_group=m['physical_state_group'],states=np.stack(states),gains=np.asarray(gains,dtype=np.float64),
            queries=np.asarray(queries,dtype=np.int64),chosen=chosen))
        checks['completion_reproduction_checks']+=1
    training=[c for c in cases if c['partition']=='train'];validation=[c for c in cases if c['partition']=='validation']
    train_dates={c['date'] for c in training};assert len(train_dates)==16 and not train_dates&{c['date'] for c in validation}
    # Cost scale uses only training dates; no performance-based calibration on validation.
    date_gain=[np.mean([c['gains'].sum() for c in training if c['date']==d]) for d in sorted(train_dates)]
    date_queries=[np.mean([c['queries'].sum() for c in training if c['date']==d]) for d in sorted(train_dates)]
    cost=.5*np.mean(date_gain)/np.mean(date_queries)
    positive=[c['gains'].sum() for c in training if c['gains'].sum()>0];gain_scale=float(np.mean(positive))
    train_states=[];weights=[];labels=[]
    for c in training:
        k=len(c['gains']);utilities=np.r_[0.,np.cumsum(c['gains']-cost*c['queries'])]
        for j,state in enumerate(c['states']):
            train_states.append(state);weights.append(c['date']);labels.append(bool(np.max(utilities[j+1:])-utilities[j]>0))
    count=collections.Counter(weights);w=np.array([1/count[d] for d in weights]);w/=w.mean()
    X=np.stack(train_states).astype(np.float64);mean=np.average(X,axis=0,weights=w)
    scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.
    np.savez_compressed(OUT/'normalization.npz',mean=mean,scale=scale)
    with (OUT/'cases.pkl').open('xb') as f:pickle.dump(cases,f,protocol=5)
    np.savez_compressed(OUT/'supervised_train.npz',X=np.clip((X-mean)/scale,-20,20).astype(np.float32),y=np.asarray(labels),weight=w)
    design=dict(id='GRID102_SEARCH_RL',scope='Exploratory saved-feedback local-search decision learning; no physical grid rollout or final test.',
        cases=dict(train=len(training),validation=len(validation),train_dates=16,validation_dates=4,physical_calls=len(cache)),
        input=dict(features=NAMES,normalization='Training reachable states only, each date equal total weight; clipped standardized features in[-20,20].',
            station='Group locked to initial incumbent station, aliases respected. Learned policy does not change global search or station choice.'),
        actions=dict(stop=0,continue_next_at_most64=1),
        reward='(increment in best-admissible float32 reward - cost_per_query*paid_queries)/gain_scale; gamma1 finite horizon.',
        cost_per_query=float(cost),gain_scale=gain_scale,cost_rule='0.5 * train-date macro full-completion mean gain / train-date macro full-completion mean queries; heuristic proxy price, not actual seconds or currency.',
        seeds=[0,1,2],timesteps=30000,
        dqn=dict(learning_rate=.0003,buffer_size=20000,learning_starts=1000,batch_size=64,tau=1.,gamma=1.,train_freq=4,gradient_steps=1,
            target_update_interval=500,exploration_fraction=.3,exploration_initial_eps=1.,exploration_final_eps=.05,max_grad_norm=10.,policy_kwargs={'net_arch':[64,64]},device='cpu',verbose=0),
        logistic=dict(C=1.,solver='lbfgs',max_iter=2000,random_state=20261022),
        logistic_teacher='Train-state hindsight label: does any later stopping point have strictly larger net utility than stopping now? Not validation labels; same observation/actions/cost as DQN.',
        baselines=['STOP','FULL_LOCAL','COST_LOGISTIC','ORACLE_hindsight_upper_bound'],
        fitting='One cost-logistic fit and three independent DQN fits, fixed final30000-step checkpoints. No validation-selected checkpoint, seed, cost or hyperparameter.',
        evaluation='Macro of four validation dates. Report reward gain, paid-query fraction, gain capture and net utility; all three seeds and per-date results.',
        caveats='Initial prefixes and local trajectories correlated within teacher states/date; FULL teacher state distribution. Saved-feedback environment training is not whole-week grid control, no latency claim. Final8 test scenarios remain ungenerated.')
    save(OUT/'design.json',design)
    save(OUT/'environment.json',dict(python=platform.python_version(),runtime=sys.executable,packages={n:importlib.metadata.version(n) for n in ['stable-baselines3','gymnasium','torch','numpy','scikit-learn']}))
    # Transition API and telescope qualification before any model fit.
    from environment import LocalSearchEnv
    from gymnasium.utils.env_checker import check_env
    env=LocalSearchEnv(training,mean,scale,cost,gain_scale);check_env(env,skip_render_check=True)
    for index,c in enumerate(cases):
        current=LocalSearchEnv([c],mean,scale,cost,gain_scale);obs,_=current.reset(seed=0,options={'index':0})
        obs0=obs.copy();_,r,done,_,info=current.step(0);assert r==0 and done and info['paid_queries']==0
        current.reset(seed=0,options={'index':0});total=0.;done=False
        while not done:_,r,done,_,info=current.step(1);total+=r
        expected=(c['gains'].sum()-cost*c['queries'].sum())/gain_scale
        assert abs(total-expected)<1e-9 and info['paid_queries']==c['queries'].sum()
        checks['environment_full_vs_stop_checks']+=1
    save(OUT/'preflight.json',dict(passed=True,checks=dict(checks),train_states=len(X),train_continue_labels=int(np.sum(labels)),gym_env_check=True,
         model_fits=0,new_physical_calls=0,final_test_materialized=False))
    for p in [SOURCE/'bulk_dataset_manifest.json',ROOT/'work/grid101_gain_learning/features.py',source101/'data.npz']:
        hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'data_freeze.json',hashes)
    import stable_baselines3,gymnasium
    library_sources=[]
    for module in [stable_baselines3,gymnasium]:
        library_sources.extend(__import__('pathlib').Path(module.__file__).parent.rglob('*.py'))
    files=list(W.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.name!='data_freeze.json']+library_sources
    save(OUT/'code_freeze.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
    print(json.dumps(dict(passed=True,cases=len(cases),checks=dict(checks),cost_per_query=cost,gain_scale=gain_scale)),flush=True)


if __name__=='__main__':main()
