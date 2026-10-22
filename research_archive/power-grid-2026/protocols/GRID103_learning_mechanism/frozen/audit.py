"""Check saved actions and summaries independently; analyze sequential value without fitting."""
import collections,csv,json
import joblib,numpy as np,torch
from stable_baselines3 import DQN
from common import ROOT,OLD,OUT,read,save,verify,load,normalized

def grouped_utility(rows):
    dates=sorted({r['date'] for r in rows})
    return float(np.mean([sum(r['net_utility'] for r in rows if r['date']==d)/sum(r['date']==d for r in rows) for d in dates]))

def main():
    verify();cases,mean,scale,D=load();cases={c['sample_id']:c for c in cases if c['partition']=='validation'}
    rows=read(OUT/'validation_trajectories.json');results=read(OUT/'validation_results.json');states=read(OUT/'state_predictions.json')
    decisions=read(OUT/'visited_decisions.json');checks=0;methods=read(OUT/'evaluation_receipt.json')['methods']
    assert len(rows)==len(cases)*len(methods) and len({(r['method'],r['sample_id']) for r in rows})==len(rows)
    for r in rows:
        c=cases[r['sample_id']];j=0;terminal=False;visited=[]
        for a in r['actions']:
            assert not terminal and a in [0,1];visited.append((j,a))
            if a==0:terminal=True
            else:j+=1;terminal=j==len(c['queries'])
        assert terminal and j==r['batches']
        assert r['queries']==sum(int(q) for q in c['queries'][:j])
        assert abs(r['gain']-sum(float(g) for g in c['gains'][:j]))<1e-15
        expected=(r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale']
        assert abs(r['net_utility']-expected)<1e-12 and r['chosen_pool_id']==c['chosen'][j]
        assert r['net_utility']<=r['oracle_utility']+1e-10
        checks+=1
    macro_checks=0
    for method,entry in results.items():
        for stratum,s in entry.items():
            rs=[r for r in rows if r['method']==method and (stratum=='all' or stratum==r['stratum'] or stratum==f"horizon{r['horizon']}")]
            assert len(rs)==s['cases'] and abs(grouped_utility(rs)-s['macro']['mean_net_utility'])<1e-12
            for date,d in s['per_date'].items():
                rr=[r for r in rs if r['date']==date]
                assert d['cases']==len(rr)
                assert abs(d['query_fraction']-sum(r['queries'] for r in rr)/sum(r['full_queries'] for r in rr))<1e-12
                g=sum(r['full_gain'] for r in rr)
                assert d['gain_capture'] is None if not g else abs(d['gain_capture']-sum(r['gain'] for r in rr)/g)<1e-12
            macro_checks+=1
    # Load final policies and reproduce every reachable-state prediction in batches.
    X=np.stack([normalized(cases[r['sample_id']]['states'][r['position']],mean,scale) for r in states])
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rf=joblib.load(OUT/'models/next_batch_rf.joblib');pred=rf.predict(X)
    assert np.allclose(pred,[r['predicted_gain'] for r in states],rtol=0,atol=1e-12)
    cost=np.array([r['next_queries'] for r in states])*D['cost_per_query']/D['gain_scale']
    assert np.array_equal(pred>cost,[r['actions']['MYOPIC_RF64'] for r in states])
    for g in [0,1]:
        for seed in [0,1,2]:
            model=DQN.load((OLD/'models'/f'dqn_seed{seed}_final.zip') if g else (OUT/'models'/f'dqn_gamma0_seed{seed}_final.zip'),device='cpu')
            assert model.gamma==g
            assert np.array_equal(model.predict(X,deterministic=True)[0],[r['actions'][f'DQN_G{g}_{seed}'] for r in states])
    # Paired comparisons: seed/dates/horizon always all reported, no best-seed selection.
    contrasts={}
    for stratum in ['all','single','multi','horizon2','horizon3','horizon4']:
        records=[]
        for seed in [0,1,2]:
            one=results[f'DQN_G1_{seed}'].get(stratum);zero=results[f'DQN_G0_{seed}'].get(stratum)
            if one is None:continue
            records.append(dict(seed=seed,delta_net=one['macro']['mean_net_utility']-zero['macro']['mean_net_utility'],
                delta_group_balanced_net=one['macro']['group_balanced_utility']-zero['macro']['group_balanced_utility'],
                per_date={d:one['per_date'][d]['mean_net_utility']-zero['per_date'][d]['mean_net_utility'] for d in one['per_date']}))
        contrasts[stratum]=dict(seeds=records,mean_delta=float(np.mean([r['delta_net'] for r in records])),
            positive_seeds=sum(r['delta_net']>0 for r in records),total_seeds=len(records))
    summary={}
    for g in [0,1]:
        summary[f'DQN_G{g}_MEAN']={st:{k:float(np.mean([results[f'DQN_G{g}_{s}'][st]['macro'][k] for s in [0,1,2] if results[f'DQN_G{g}_{s}'][st]['macro'][k] is not None]))
            if any(results[f'DQN_G{g}_{s}'][st]['macro'][k] is not None for s in [0,1,2]) else None
            for k in results[f'DQN_G{g}_0'][st]['macro']} for st in results[f'DQN_G{g}_0']}
    # Regression skill at all reachable states (not only favorable visited ones), date macro.
    train=np.load(OUT/'regression_train.npz');constant=float(np.average(train['y'],weights=train['weight']))
    errors={}
    for name in ['RF64','ZERO','TRAIN_MEAN']:
        date_rows={}
        for date in sorted({r['date'] for r in states}):
            rs=[r for r in states if r['date']==date];y=np.array([r['actual_gain'] for r in rs])
            p=np.array([r['predicted_gain'] for r in rs]) if name=='RF64' else np.full(len(rs),0. if name=='ZERO' else constant)
            date_rows[date]=dict(states=len(rs),mae=float(np.mean(abs(p-y))),rmse=float(np.sqrt(np.mean((p-y)**2))))
        errors[name]=dict(per_date=date_rows,macro={k:float(np.mean([r[k] for r in date_rows.values()])) for k in ['mae','rmse']})
    # Delayed opportunities are hindsight statements, never training labels for the new regressor.
    delayed=[r for r in states if r['delayed_opportunity']]
    delayed_profile=dict(states=len(delayed),physical_groups=len({r['physical_state_group'] for r in delayed}),
        dates=sorted({r['date'] for r in delayed}),
        continuation_rates={m:float(np.mean([r['actions'][m] for r in delayed])) if delayed else None
            for m in ['MYOPIC_RF64']+[f'DQN_G{g}_{s}' for g in [0,1] for s in [0,1,2]]},
        note='Clairvoyant opportunities, not deployable observations or independent examples.')
    failure={}
    for method in methods:
        rs=[r for r in rows if r['method']==method and r['date']=='2037-10-19']
        failure[method]=dict(cases=len(rs),negative_cases=sum(r['net_utility'] < -1e-12 for r in rs),
            physical_groups_with_negative=len({r['physical_state_group'] for r in rs if r['net_utility'] < -1e-12}),
            mean_normalized_gain=float(np.mean([r['normalized_gain'] for r in rs])),
            mean_normalized_query_cost=float(np.mean([r['normalized_query_cost'] for r in rs])),mean_net=grouped_utility(rs),
            oracle_gap=float(np.mean([r['oracle_utility']-r['net_utility'] for r in rs])))
    visited={}
    for method in methods:
        ds=[d for d in decisions if d['method']==method]
        visited[method]=dict(decisions=len(ds),continued=sum(d['action']==1 for d in ds),
            continued_next_batch_nonpositive=sum(d['action']==1 and d['actual_next_net']<=0 for d in ds),
            continued_despite_nonpositive_best_remaining=sum(d['action']==1 and d['best_remaining_net']<=0 for d in ds),
            stopped_with_positive_hindsight_remaining=sum(d['action']==0 and d['best_remaining_net']>0 for d in ds))
    decision=dict(sequential_support_exploratory=bool(contrasts['multi']['mean_delta']>0 and contrasts['multi']['positive_seeds']>=2),
        gamma1_minus_gamma0=contrasts,seed_means=summary,regression_errors=errors,delayed_opportunities=delayed_profile,
        october_failure=failure,visited_hindsight_diagnostics=visited,
        scope='Fixed proxy price,4 exposed dates; no statistical, long-horizon, physical latency or week-level claim; gamma0 changes Bellman target and collected trajectories.')
    save(OUT/'decision.json',decision)
    with (OUT/'validation_trajectories.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'review.json',dict(passed=True,trajectory_checks=checks,summary_checks=macro_checks,
        prediction_checks=len(states)*7,old_trajectory_reproductions=read(OUT/'evaluation_receipt.json')['old_trajectory_reproductions'],
        model_fits=4,new_physical_calls=0,final_test_materialized=False,
        independent_physical_AC_resolve=False))
    print(json.dumps(dict(review=read(OUT/'review.json'),contrasts=contrasts['multi'],delayed=delayed_profile),indent=2),flush=True)
if __name__=='__main__':main()
