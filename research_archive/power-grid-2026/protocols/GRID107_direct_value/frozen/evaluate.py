"""All fixed validation trajectories and train fit errors, no physical replay."""
import gzip,json
import joblib,numpy as np,torch
from stable_baselines3 import DQN
from common107 import ROOT,OUT,read,save,verify,loadcases,normalization,normalized,predict,state_metrics,summarize
def main():
    assert not (OUT/'validation_results.json').exists();verify();D=read(OUT/'design.json');assert read(OUT/'fits.json')['new_fits']==3
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases=loadcases();mean,std=normalization();metadata=[];X=[];target=[]
    for c in cases:
        for j,s in enumerate(c['states']):
            X.append(normalized(s,mean,std));target.append(float(c['gains'][j]-D['cost_per_query']*c['queries'][j]))
            metadata.append(dict(sample_id=c['sample_id'],position=j,date=c['date'],partition=c['partition'],physical_state_group=c['physical_state_group'],next_queries=int(c['queries'][j])))
    X=np.stack(X);target=np.array(target);queries=np.array([r['next_queries'] for r in metadata]);models={};Q={}
    for seed in D['seeds']:
        models[f'DIRECT{seed}']=DQN.load(OUT/f'models/direct_seed{seed}_final.zip',device='cpu')
        models[f'DQN0_{seed}']=DQN.load(ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{seed}_final.zip',device='cpu')
    for name,m in models.items():
        Q[name]=predict(m.q_net,X);assert np.array_equal(Q[name].argmax(axis=1),m.predict(X,deterministic=True)[0])
    rf=joblib.load(ROOT/'outputs/grid103_learning_mechanism/models/next_batch_rf.joblib')
    rfpred=rf.predict(X)-D['cost_per_query']*queries/D['gain_scale'];Q['RF']=np.column_stack([np.zeros_like(rfpred),rfpred])
    metrics={};predrows=[]
    for method,q in Q.items():
        metrics[method]={}
        for part in ['train','validation']:
            mask=np.array([r['partition']==part for r in metadata]);rs=[r for r in metadata if r['partition']==part]
            m=state_metrics(target[mask],q[mask],rs);m['continue_value_rmse']=float(np.sqrt(np.mean((q[mask,1]-target[mask]/D['gain_scale'])**2)))
            m['advantage_rmse']=float(np.sqrt(np.mean((q[mask,1]-q[mask,0]-target[mask]/D['gain_scale'])**2)));metrics[method][part]=m
        for i,r in enumerate(metadata):predrows.append(dict(r,method=method,target_net=target[i],q0=q[i,0],q1=q[i,1],action=int(q[i,1]>q[i,0])))
    with gzip.open(OUT/'state_predictions.jsonl.gz','wt',encoding='utf8') as f:
        for r in predrows:f.write(json.dumps(r,ensure_ascii=False,default=lambda v:v.item(),allow_nan=False)+'\n')
    # Freeze model predictions before attaching counterfactual targets and record IDs.
    actions={k:q[:,1]>q[:,0] for k,q in Q.items()};trajectories=[];offset=0
    for c in cases:
        horizon=len(c['states'])
        if c['partition']=='validation':
            for method in D['evaluation']['methods']:
                pos=0;seq=[]
                while pos<horizon:
                    a=0 if method=='STOP' else 1 if method=='FULL_LOCAL' else int(pos==0) if method=='BATCH1' else int(actions[method][offset+pos])
                    seq.append(a)
                    if not a:break
                    pos+=1
                gain=float(c['gains'][:pos].sum());count=int(c['queries'][:pos].sum());net=(gain-D['cost_per_query']*count)/D['gain_scale']
                assert abs(net-sum(float((c['gains'][j]-D['cost_per_query']*c['queries'][j])/D['gain_scale']) for j in range(pos)))<1e-9
                trajectories.append(dict(sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],scenario=c['scenario'],
                    method=method,horizon=horizon,actions=seq,batches=pos,gain=gain,queries=count,
                    full_gain=float(c['gains'].sum()),full_queries=int(c['queries'].sum()),net_utility=net,chosen_pool_id=c['chosen'][pos]))
        offset+=horizon
    assert len(trajectories)==5900
    old={(r['method'],r['sample_id']):r for r in read(ROOT/'outputs/grid103_learning_mechanism/validation_trajectories.json')};reproduced=0
    for r in trajectories:
        name='MYOPIC_RF64' if r['method']=='RF' else r['method'].replace('DQN0_','DQN_G0_')
        if r['method'].startswith('DIRECT'):continue
        prev=old[name,r['sample_id']]
        for k in ['actions','batches','gain','queries','chosen_pool_id']:assert r[k]==prev[k],(name,k)
        assert abs(r['net_utility']-prev['net_utility'])<1e-12;reproduced+=1
    results={}
    for method in D['evaluation']['methods']:
        rs=[r for r in trajectories if r['method']==method];results[method]={}
        for label,condition in [('all',lambda r:True),('single',lambda r:r['horizon']==1),('multi',lambda r:r['horizon']>1)]:
            selected=[r for r in rs if condition(r)]
            if selected:results[method][label]=summarize(selected)
    # The three known difficult contexts are development diagnostics only.
    prior=read(ROOT/'outputs/grid105_learning_diagnosis/proxy_value_diagnosis.json') if (ROOT/'outputs/grid105_learning_diagnosis/proxy_value_diagnosis.json').exists() else read(ROOT/'outputs/grid104_learning_closedloop/proxy_value_diagnosis.json')
    post=read(ROOT/'outputs/grid106_decision_branches/posthoc_interpretation.json')['rf_after_beneficial_108_omission']
    probes=[]
    for label,p in read(OUT/'probe_sources.json').items():
        c=read(ROOT/p);dec=c['learned_decisions'][0];obs=np.array(dec['normalized_features'],dtype=np.float32)
        assert np.array_equal(obs,normalized(np.array(dec['raw_features']),mean,std))
        if label=='RF_PATH552':net=post['actual_next_net']
        else:
            t=108 if label=='MARCH108' else 552
            a=next(r for r in prior['diagnoses'] if r['step']==t and r['method']==('LOCAL_RF' if t==108 else 'LOCAL_DQN0_0'))
            net=a['actual_first_batch_gain']-a['actual_first_batch_cost']
        scores={}
        for method,m in models.items():
            q=predict(m.q_net,obs[None])[0];scores[method]=dict(q0=float(q[0]),q1=float(q[1]),action=int(q[1]>q[0]))
        probes.append(dict(context=label,source=p,actual_next_normalized_net=net/D['gain_scale'],scores=scores,
            original_queries=len(dec['next_ids']),used_for_training=False))
    save(OUT/'state_metrics.json',metrics);save(OUT/'validation_trajectories.json',trajectories)
    save(OUT/'validation_results.json',results);save(OUT/'critical_probes.json',probes)
    save(OUT/'evaluation_receipt.json',dict(trajectories=5900,old_trajectory_reproductions=reproduced,state_predictions=len(predrows),
        model_functions_only_read_X=True,new_physical_steps=0,final_test_materialized=False))
    print(json.dumps({m:r['all']['macro'] for m,r in results.items()},indent=2),flush=True)
if __name__=='__main__':main()
