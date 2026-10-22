"""Locked evaluation: whole validation, horizon strata, dates, and hindsight diagnostics."""
import collections,json,time
import joblib,numpy as np,torch
from stable_baselines3 import DQN
from common import OLD,OUT,read,save,verify,load,normalized,LocalSearchEnv,summarize

def main():
    assert not (OUT/'validation_trajectories.json').exists(),'Preserve prior evaluation'
    verify();assert read(OUT/'fits.json')['model_fits']==4
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases,mean,scale,D=load();cases=[c for c in cases if c['partition']=='validation']
    rf=joblib.load(OUT/'models/next_batch_rf.joblib');lr=joblib.load(OLD/'models/cost_logistic.joblib')
    X=np.stack([normalized(state,mean,scale) for c in cases for state in c['states']])
    # Predictions are functions of X alone. IDs and future outcomes are attached only for saved diagnostics.
    start=time.perf_counter();gain_pred=rf.predict(X);rf_elapsed=time.perf_counter()-start
    actions={'MYOPIC_RF64':[], 'COST_LOGISTIC':lr.predict(X).astype(int).tolist()};scores={}
    query_vector=np.array([q for c in cases for q in c['queries']])
    actions['MYOPIC_RF64']=(gain_pred>D['cost_per_query']*query_vector/D['gain_scale']).astype(int).tolist()
    inference={'MYOPIC_RF64':dict(states=len(X),wall_s=rf_elapsed,scope='Batched cached-observation inference only, not end-to-end latency.')}
    for gamma in [1,0]:
        for seed in [0,1,2]:
            name=f'DQN_G{gamma}_{seed}'
            path=(OLD/'models'/f'dqn_seed{seed}_final.zip') if gamma else OUT/'models'/f'dqn_gamma0_seed{seed}_final.zip'
            model=DQN.load(path,device='cpu');assert model.gamma==gamma
            start=time.perf_counter()
            with torch.no_grad():q=model.q_net(torch.as_tensor(X)).cpu().numpy()
            elapsed=time.perf_counter()-start
            predicted=model.predict(X,deterministic=True)[0]
            assert np.array_equal(predicted,np.argmax(q,axis=1))
            actions[name]=predicted.astype(int).tolist();scores[name]=q.tolist()
            inference[name]=dict(states=len(X),wall_s=elapsed,scope='Batched cached-observation forward only, not end-to-end latency.')
    methods=['STOP','FULL_LOCAL','BATCH1','COST_LOGISTIC','MYOPIC_RF64']+[f'DQN_G{g}_{s}' for g in [1,0] for s in [0,1,2]]+['ORACLE_OPTIMAL','ORACLE_MYOPIC']
    trajectories=[];decisions=[];state_rows=[];offset=0
    for c in cases:
        horizon=len(c['queries']);netsteps=(c['gains']-D['cost_per_query']*c['queries'])/D['gain_scale']
        cumulative=np.r_[0.,np.cumsum(netsteps)];oracle=int(np.argmax(cumulative))
        for j in range(horizon):
            idx=offset+j
            state_rows.append(dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],position=j,
                horizon=horizon,actual_gain=float(c['gains'][j]/D['gain_scale']),next_queries=int(c['queries'][j]),
                actual_next_net=float(netsteps[j]),predicted_gain=float(gain_pred[idx]),
                best_remaining_net=float(max(0.,max(np.cumsum(netsteps[j:])))),
                delayed_opportunity=bool(netsteps[j]<=0 and max(np.cumsum(netsteps[j:]))>0),
                actions={k:v[idx] for k,v in actions.items()},q_values={k:v[idx] for k,v in scores.items()}))
        for method in methods:
            env=LocalSearchEnv([c],mean,scale,D['cost_per_query'],D['gain_scale']);obs,_=env.reset(seed=0,options={'index':0})
            done=False;seq=[];total=0.
            while not done:
                j=env.position;idx=offset+j
                assert np.array_equal(obs,X[idx])
                if method=='STOP':a=0
                elif method=='FULL_LOCAL':a=1
                elif method=='BATCH1':a=int(j==0)
                elif method=='ORACLE_OPTIMAL':a=int(j<oracle)
                elif method=='ORACLE_MYOPIC':a=int(netsteps[j]>0)
                else:a=actions[method][idx]
                seq.append(a)
                decisions.append(dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],
                    method=method,position=j,action=a,actual_next_net=float(netsteps[j]),
                    best_remaining_net=float(max(0.,max(np.cumsum(netsteps[j:])))),
                    delayed_opportunity=bool(netsteps[j]<=0 and max(np.cumsum(netsteps[j:]))>0)))
                obs,r,done,_,_=env.step(a);total+=r
            j=env.position;gain=float(c['gains'][:j].sum());queries=int(c['queries'][:j].sum())
            net=(gain-D['cost_per_query']*queries)/D['gain_scale'];assert abs(total-net)<1e-9
            trajectories.append(dict(sample_id=c['sample_id'],date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],
                method=method,horizon=horizon,stratum='single' if horizon==1 else 'multi',actions=seq,batches=j,
                gain=gain,queries=queries,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),
                net_utility=net,normalized_gain=gain/D['gain_scale'],normalized_query_cost=D['cost_per_query']*queries/D['gain_scale'],
                oracle_utility=float(cumulative.max()),chosen_pool_id=c['chosen'][j]))
        offset+=horizon
    oldrows={(r['method'],r['sample_id']):r for r in read(OLD/'validation_trajectories.json')};reproduced=0
    for r in trajectories:
        oldname=r['method'].replace('DQN_G1_','DQN')
        if oldname not in ['STOP','FULL_LOCAL','COST_LOGISTIC','DQN0','DQN1','DQN2']:continue
        old=oldrows[oldname,r['sample_id']]
        for key in ['actions','batches','gain','queries','chosen_pool_id']:assert r[key]==old[key],(oldname,key)
        assert abs(r['net_utility']-old['net_utility'])<1e-12;reproduced+=1
    results={}
    for method in methods:
        rows=[r for r in trajectories if r['method']==method]
        results[method]={}
        strata=([('all',lambda r:True),('single',lambda r:r['horizon']==1),('multi',lambda r:r['horizon']>1)]
                + [(f'horizon{k}',lambda r,k=k:r['horizon']==k) for k in [2,3,4]])
        for label,predicate in strata:
            subset=[r for r in rows if predicate(r)]
            if subset:results[method][label]=summarize(subset)
    save(OUT/'validation_trajectories.json',trajectories);save(OUT/'visited_decisions.json',decisions)
    save(OUT/'state_predictions.json',state_rows);save(OUT/'validation_results.json',results)
    save(OUT/'evaluation_receipt.json',dict(methods=methods,trajectories=len(trajectories),validation_states=len(X),
        old_trajectory_reproductions=reproduced,inference=inference,new_physical_calls=0,final_test_materialized=False))
    print(json.dumps({m:r['all']['macro'] for m,r in results.items()},indent=2),flush=True)
if __name__=='__main__':main()
