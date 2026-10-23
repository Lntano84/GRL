"""Fixed endpoint complete-search evaluation after all nine fits."""
import gzip,torch,numpy as np
from common119 import *
def trajectory(c,name,actions,fee,scale):
    p=0;seq=[]
    while p<len(c['states']):
        go=int(actions[p]);seq.append(go)
        if not go:break
        p+=1
    gain=float(c['gains'][:p].sum());queries=int(c['queries'][:p].sum())
    return dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],scenario=c['scenario'],method=name,
        horizon=len(c['states']),actions=seq,batches=p,gain=gain,queries=queries,full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),
        net_utility=(gain-fee*queries)/scale,chosen_pool_id=c['chosen'][p])
def summaries(rows):
    out={}
    for name in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==name];out[name]={}
        for label,ss in [('all',selected),('single',[r for r in selected if r['horizon']==1]),('multi',[r for r in selected if r['horizon']>1])]:
            if ss:out[name][label]=summarize(ss)
    return out
def main():
    assert not (O/'evaluation_receipt.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and F['new_fits']==9
    D=read(O/'design.json');fee=D['unchanged']['cost_per_query'];scale=D['unchanged']['gain_scale']
    rows={p:pickle.loads((S/f'{p}_public.pkl').read_bytes()) for p in ['train','validation']};meta={p:read(S/f'{p}_metadata.json') for p in rows};targets={p:np.load(S/f'{p}_targets.npz')['target'] for p in rows}
    trainprob=np.load(S/'train_targets.npz')['sampling_probability'];lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(meta[p])} for p in rows}
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());Q={};shifted={};metrics={};models={};sensitivity={};alltr={p:[] for p in rows}
    for rec in F['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";m=build(rec['seed'],rec['arm']);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');m.load_state_dict(cp['state_dict']);assert digest(m)==rec['final_hash'];models[name]=m;metrics[name]={};sensitivity[name]={}
        for part in rows:
            q=predict(m,rows[part]);Q[name+'__'+part]=q;target=targets[part];pos=target>0
            mm=state_metrics(target*scale,q,meta[part]);mm['continue_value_rmse']=float(np.sqrt(np.mean((q[:,1]-target)**2)));mm['advantage_rmse']=float(np.sqrt(np.mean((q[:,1]-q[:,0]-target)**2)))
            y=np.column_stack([np.zeros_like(target),target]);err=q-y;ll=np.where(abs(err)<1,.5*err**2,abs(err)-.5).mean(1);prob=trainprob if part=='train' else np.ones(len(target))/len(target)
            mm['sampling_weighted_huber']=float(prob@ll);mm['sampling_weighted_sign_regret']=float(prob@(np.maximum(target,0)-(q[:,1]>q[:,0])*target));metrics[name][part]=mm
            # One predeclared correspondence perturbation. Diagnostic only, no threshold/utility selection.
            changed=[dict(r,context_feedback=np.roll(r['context_feedback'],1,axis=0)) for r in rows[part]];sq=predict(m,changed);shifted[name+'__'+part]=sq
            delta=(sq[:,1]-sq[:,0])-(q[:,1]-q[:,0]);flips=(sq[:,1]>sq[:,0])!=(q[:,1]>q[:,0])
            sensitivity[name][part]=dict(mean_abs_advantage_change=float(abs(delta).mean()),max_abs_advantage_change=float(abs(delta).max()),
                positive_median_abs_change=float(np.median(abs(delta[pos]))),positive_mean_abs_change=float(abs(delta[pos]).mean()),decision_sign_changes=int(flips.sum()),positive_sign_changes=int((flips&pos).sum()))
            if rec['arm']=='FIXED_FACTOR':assert np.allclose(q,sq,atol=3e-5,rtol=3e-5)
        for c in cases:
            part=c['partition'];ids=[lookup[part][c['sample_id'],j] for j in range(len(c['states']))];q=Q[name+'__'+part]
            alltr[part].append(trajectory(c,name,q[ids,1]>q[ids,0],fee,scale))
    # All historical validation methods re-costed, not refitted or timed.
    historical=read(R/'outputs/grid107_direct_value/validation_trajectories.json');bycase={c['sample_id']:c for c in cases}
    for r in historical:
        c=bycase[r['sample_id']];a=r['actions']+[0]*len(c['states']);t=trajectory(c,r['method'],a,fee,scale)
        for k in ['actions','batches','gain','queries','chosen_pool_id']:assert t[k]==r[k]
        assert abs(t['net_utility']-r['net_utility'])<1e-12;alltr['validation'].append(t)
    # Train BASE/DQN families from their original saved predictions, no repeated fit.
    oldnames=[f'DIRECT{s}' for s in range(3)]+[f'DQN0_{s}' for s in range(3)]
    oldq={n:np.full((len(rows['train']),2),np.nan,np.float32) for n in oldnames}
    with gzip.open(R/'outputs/grid107_direct_value/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['partition']=='train' and r['method'] in oldq:oldq[r['method']][lookup['train'][r['sample_id'],r['position']]]=[r['q0'],r['q1']]
    for n,q in oldq.items():
        assert np.isfinite(q).all()
        for c in cases:
            if c['partition']!='train':continue
            ids=[lookup['train'][c['sample_id'],j] for j in range(len(c['states']))];alltr['train'].append(trajectory(c,n,q[ids,1]>q[ids,0],fee,scale))
    result={p:summaries(t) for p,t in alltr.items()};val=result['validation'];utility=lambda n:val[n]['all']['macro']['mean_net_utility'];dates=sorted(val['FIXED_JOINT_seed0']['all']['per_date']);pairs=[];dates_delta={}
    for seed in range(3):pairs.append(dict(seed=seed,joint=utility(f'FIXED_JOINT_seed{seed}'),factor=utility(f'FIXED_FACTOR_seed{seed}'),extra=utility(f'BASE_EXTRA_seed{seed}'),base=utility(f'DIRECT{seed}'),
        joint_minus_factor=utility(f'FIXED_JOINT_seed{seed}')-utility(f'FIXED_FACTOR_seed{seed}'),joint_minus_extra=utility(f'FIXED_JOINT_seed{seed}')-utility(f'BASE_EXTRA_seed{seed}'),joint_minus_base=utility(f'FIXED_JOINT_seed{seed}')-utility(f'DIRECT{seed}'),factor_minus_base=utility(f'FIXED_FACTOR_seed{seed}')-utility(f'DIRECT{seed}')))
    for date in dates:dates_delta[date]=float(np.mean([val[f'FIXED_JOINT_seed{s}']['all']['per_date'][date]['mean_net_utility']-val[f'FIXED_FACTOR_seed{s}']['all']['per_date'][date]['mean_net_utility'] for s in range(3)]))
    family={k:float(np.mean([utility(fmt.format(s)) for s in range(3)])) for k,fmt in [('JOINT','FIXED_JOINT_seed{}'),('FACTOR','FIXED_FACTOR_seed{}'),('BASE','DIRECT{}'),('EXTRA','BASE_EXTRA_seed{}'),('DQN','DQN0_{}')]}
    mechanism=family['JOINT']>family['FACTOR'] and all(r['joint_minus_factor']>0 for r in pairs) and sum(v>0 for v in dates_delta.values())>=3
    anchors=all(family['JOINT']>v for v in [family['BASE'],family['EXTRA'],family['DQN'],utility('BATCH1')])
    decision=dict(mechanism_signal=mechanism,incremental_over_base=family['JOINT']>family['BASE'],incremental_over_extra=family['JOINT']>family['EXTRA'],all_anchors_exceeded=anchors,competitive_candidate=mechanism and anchors,
        family_mean_utility=family,batch1_utility=utility('BATCH1'),paired_seeds=pairs,date_joint_minus_factor=dates_delta,
        ready_to_scale=False,physical_rollout_authorized=False,limits='Exposed four-date same-grid teacher-path offline evaluation; no wholeweek quality, statistical significance, latency or cross-grid claim.')
    probes=[]
    for r in read(R/'outputs/grid111_action_feedback/critical_details.json'):
        call=read(R/r['source']);inp=pack(r['public_input'],np.asarray(call['learned_decisions'][0]['normalized_features'],np.float32),fee,scale);scores={}
        for name,m in models.items():
            q=predict(m,[inp])[0];scores[name]=dict(q0=float(q[0]),q1=float(q[1]),action=int(q[1]>q[0]))
        probes.append(dict(context=r['context'],source=r['source'],known_batch_normalized_net=r['known_batch_normalized_net'],scores=scores,used_for_fit=False))
    np.savez_compressed(O/'state_predictions.npz',**Q);np.savez_compressed(O/'shifted_predictions.npz',**shifted)
    for part in rows:save(O/f'{part}_trajectories.json',alltr[part]);save(O/f'{part}_results.json',result[part])
    save(O/'state_metrics.json',metrics);save(O/'binding_sensitivity.json',sensitivity);save(O/'critical_probes.json',probes);save(O/'decision.json',decision)
    save(O/'evaluation_receipt.json',dict(passed=True,all_nine_fixed_endpoints_before_validation=True,old_validation_trajectories_reproduced=len(historical),
        new_validation_trajectories=5310,new_train_trajectories=21402,no_fit_or_threshold_selection=True,new_physical_steps=0,final_test_materialized=False))
    save(O/'state.json',dict(stage='evaluated_awaiting_audit',fits=9));print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
