"""Fixed endpoint evaluation only after all six fits. Never physical replay."""
import gzip
import numpy as np,torch
from common import *
from model import RelationalValue
def trajectory(c,method,act,fee,scale):
    pos=0;seq=[];horizon=len(c['states'])
    while pos<horizon:
        a=int(act[pos]);seq.append(a)
        if not a:break
        pos+=1
    gain=float(c['gains'][:pos].sum());queries=int(c['queries'][:pos].sum())
    return dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],scenario=c['scenario'],method=method,
        horizon=horizon,actions=seq,batches=pos,gain=gain,queries=queries,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),
        net_utility=(gain-fee*queries)/scale,chosen_pool_id=c['chosen'][pos])
def main():
    assert not (O/'evaluation_receipt.json').exists();verify();configure();fits=read(O/'fits.json');assert fits['passed'] and len(fits['records'])==6
    D=read(O/'design.json');fee=D['target']['cost_per_query'];scale=D['target']['gain_scale']
    inputs={p:pickle.loads((O/f'{p}_public.pkl').read_bytes()) for p in ['train','validation']}
    meta={p:read(O/f'{p}_metadata.json') for p in inputs};targets={p:np.load(O/f'{p}_targets.npz')['target'] for p in inputs}
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());vc=[c for c in cases if c['partition']=='validation']
    lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta['validation'])}
    metrics={};Q={};trajectories=[];models={}
    for rec in fits['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";m=RelationalValue(rec['arm']);saved=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu')
        m.load_state_dict(saved['state_dict']);assert digest(m)==rec['final_hash'];models[name]=m;metrics[name]={}
        for part in inputs:
            q=predict(m,inputs[part]);Q[name+'__'+part]=q
            sm=state_metrics(targets[part]*scale,q,meta[part]);sm['continue_value_rmse']=float(np.sqrt(np.mean((q[:,1]-targets[part])**2)))
            sm['advantage_rmse']=float(np.sqrt(np.mean((q[:,1]-q[:,0]-targets[part])**2)));sm['two_output_huber']=float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(q),torch.from_numpy(np.column_stack([np.zeros_like(targets[part]),targets[part]]))))
            metrics[name][part]=sm
        actions=Q[name+'__validation'][:,1]>Q[name+'__validation'][:,0]
        for c in vc:
            ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))]
            trajectories.append(trajectory(c,name,actions[ids],fee,scale))
    # Recompute every historical baseline trajectory's payoff from original gains/query counts.
    old=read(R/'outputs/grid107_direct_value/validation_trajectories.json');bycase={c['sample_id']:c for c in vc}
    for r in old:
        c=bycase[r['sample_id']];a=r['actions']+[0]*len(c['states']);t=trajectory(c,r['method'],a,fee,scale)
        for k in ['actions','batches','gain','queries','chosen_pool_id']:assert t[k]==r[k],(r['method'],k)
        assert abs(t['net_utility']-r['net_utility'])<1e-12;trajectories.append(t)
    result={}
    for name in sorted({r['method'] for r in trajectories}):
        rs=[r for r in trajectories if r['method']==name];result[name]={}
        for label,sub in [('all',rs),('single',[r for r in rs if r['horizon']==1]),('multi',[r for r in rs if r['horizon']>1])]:
            if sub:result[name][label]=summarize(sub)
    critical=[]
    for r in read(R/'outputs/grid111_action_feedback/critical_details.json'):
        call=read(R/r['source']);g=np.asarray(call['learned_decisions'][0]['normalized_features'],np.float32)
        inp=pack(r['public_input'],g,fee,scale);scores={}
        for name,m in models.items():
            q=predict(m,[inp])[0];scores[name]=dict(q0=float(q[0]),q1=float(q[1]),action=int(q[1]>q[0]))
        critical.append(dict(context=r['context'],source=r['source'],known_batch_normalized_net=r['known_batch_normalized_net'],scores=scores,used_for_fit=False))
    dates=sorted(result['JOINT_seed0']['all']['per_date']);pairs=[];datedelta={}
    utility=lambda n:result[n]['all']['macro']['mean_net_utility']
    for seed in [0,1,2]:pairs.append(dict(seed=seed,joint=utility(f'JOINT_seed{seed}'),factor=utility(f'FACTOR_seed{seed}'),delta=utility(f'JOINT_seed{seed}')-utility(f'FACTOR_seed{seed}')))
    for d in dates:datedelta[d]=float(np.mean([result[f'JOINT_seed{s}']['all']['per_date'][d]['mean_net_utility']-result[f'FACTOR_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in [0,1,2]]))
    families={k:float(np.mean([utility(f'{k}_seed{s}' if k in ['JOINT','FACTOR'] else f'{k}{s}') for s in [0,1,2]])) for k in ['JOINT','FACTOR','DIRECT','DQN0_']}
    mechanism=families['JOINT']>families['FACTOR'] and all(r['delta']>0 for r in pairs) and sum(v>0 for v in datedelta.values())>=3
    competition=all(families['JOINT']>v for v in [utility('BATCH1'),families['DIRECT'],families['DQN0_']])
    decision=dict(mechanism_signal=mechanism,competitive_progression=mechanism and competition,all_baseline_anchors_exceeded=competition,
        paired_seeds=pairs,date_deltas=datedelta,family_mean_utility=families,batch1_utility=utility('BATCH1'),
        ready_to_scale=False,physical_rollout_authorized=False,limits='Offline immediate teacher-path value, exposed four-date development validation; not whole-week quality, latency, statistical significance or cross-grid generalization.')
    np.savez_compressed(O/'state_predictions.npz',**Q)
    save(O/'state_metrics.json',metrics);save(O/'validation_trajectories.json',trajectories);save(O/'validation_results.json',result)
    save(O/'critical_probes.json',critical);save(O/'decision.json',decision)
    save(O/'evaluation_receipt.json',dict(passed=True,six_final_fits_before_validation=True,new_trajectories=3540,historical_recomputed=len(old),validation_cases=590,
        no_checkpoint_or_seed_selection=True,new_physical_steps=0,final_test_materialized=False))
    save(O/'state.json',dict(stage='evaluated_awaiting_audit',fits=6));print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
