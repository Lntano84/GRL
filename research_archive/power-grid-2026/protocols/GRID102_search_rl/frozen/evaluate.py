"""All validation search trajectories, with cost and hindsight upper bound."""
import collections,json,pickle
import joblib,numpy as np,torch
from stable_baselines3 import DQN
from common import OUT,read,save,verify,test_sealed
from environment import LocalSearchEnv


def rollout(case,mean,scale,D,method,policy=None):
    env=LocalSearchEnv([case],mean,scale,D['cost_per_query'],D['gain_scale'])
    obs,_=env.reset(seed=0,options={'index':0});done=False;actions=[];returned=0.
    utility=np.r_[0.,np.cumsum(case['gains']-D['cost_per_query']*case['queries'])]/D['gain_scale']
    oracle_position=int(np.argmax(utility))
    while not done:
        if method=='STOP':action=0
        elif method=='FULL_LOCAL':action=1
        elif method=='ORACLE':action=int(env.position<oracle_position)
        elif method=='COST_LOGISTIC':action=int(policy.predict(obs[None,:])[0])
        else:action=int(policy.predict(obs,deterministic=True)[0])
        actions.append(action);obs,r,done,_,info=env.step(action);returned+=r
    position=env.position;gain=float(case['gains'][:position].sum());queries=int(case['queries'][:position].sum())
    net=(gain-D['cost_per_query']*queries)/D['gain_scale']
    assert abs(returned-net)<1e-9 and net<=float(utility.max())+1e-9
    return dict(sample_id=case['sample_id'],date=case['date'],scenario=case['scenario'],method=method,
        physical_state_group=case['physical_state_group'],batches=position,actions=actions,gain=gain,queries=queries,
        full_gain=float(case['gains'].sum()),full_queries=int(case['queries'].sum()),net_utility=net,
        oracle_utility=float(utility.max()),chosen_pool_id=case['chosen'][position],initial_pool_id=case['chosen'][0])


def summarize(records):
    per_date={}
    for date in sorted({r['date'] for r in records}):
        rows=[r for r in records if r['date']==date]
        full_gain=sum(r['full_gain'] for r in rows);full_queries=sum(r['full_queries'] for r in rows)
        per_date[date]=dict(cases=len(rows),mean_net_utility=float(np.mean([r['net_utility'] for r in rows])),
            mean_gain=float(np.mean([r['gain'] for r in rows])),mean_queries=float(np.mean([r['queries'] for r in rows])),
            query_fraction=sum(r['queries'] for r in rows)/full_queries,
            gain_capture=sum(r['gain'] for r in rows)/full_gain if full_gain>0 else None,
            stop_immediately=float(np.mean([r['queries']==0 for r in rows])),
            changed_action=float(np.mean([r['chosen_pool_id']!=r['initial_pool_id'] for r in rows])))
    macro={}
    for key in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','stop_immediately','changed_action']:
        values=[r[key] for r in per_date.values() if r[key] is not None]
        macro[key]=float(np.mean(values)) if values else None
    return dict(macro=macro,per_date=per_date)


def main():
    assert not (OUT/'validation_results.json').exists(),'Do not repeat delivered evaluation'
    verify();test_sealed();assert read(OUT/'fits.json')['model_fits']==4
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    D=read(OUT/'design.json')
    with (OUT/'cases.pkl').open('rb') as f:validation=[c for c in pickle.load(f) if c['partition']=='validation']
    with np.load(OUT/'normalization.npz') as z:mean=z['mean'];scale=z['scale']
    methods={'STOP':None,'FULL_LOCAL':None,'ORACLE':None,'COST_LOGISTIC':joblib.load(OUT/'models/cost_logistic.joblib')}
    for seed in D['seeds']:methods[f'DQN{seed}']=DQN.load(OUT/'models'/f'dqn_seed{seed}_final.zip',device='cpu')
    trajectories=[];results={}
    for method,policy in methods.items():
        records=[rollout(c,mean,scale,D,method,policy) for c in validation]
        trajectories.extend(records);results[method]=summarize(records)
    results['DQN_SEED_MEAN']=dict(macro={k:float(np.mean([results[f'DQN{s}']['macro'][k] for s in D['seeds']])) for k in results['DQN0']['macro']},
        note='Mean over all3 seeds, not independent scenarios; no selected best seed.')
    save(OUT/'validation_trajectories.json',trajectories);save(OUT/'validation_results.json',results)
    print(json.dumps({k:r['macro'] for k,r in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
