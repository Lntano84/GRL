"""Posthoc diagnostic only: fixed models, saved paid feedback, no fit or physics."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json, os, pickle, sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/grid105_learning_diagnosis'
sys.path.insert(0,str(ROOT/'work/grid102_search_rl'))
from observations import NAMES,best,reward
HASHES={}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(p):HASHES[p.relative_to(ROOT).as_posix()]=digest(p);return p
def read(p):return json.loads(record(p).read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def norm(x):return np.clip((x-MEAN)/SCALE,-20,20).astype(np.float32)
def qvalues(model,X):
    import torch
    with torch.no_grad():return model.q_net(torch.as_tensor(X,device='cpu')).cpu().numpy()
def metrics(rows):
    # Every sign error concerns realized saved target, not an observed conditional expectation.
    if not rows:return dict(states=0)
    t=np.array([r['target_net'] for r in rows]);a=np.array([r['action'] for r in rows])
    p=t>0; n=~p; fn=p&(a==0); fp=n&(a==1)
    return dict(states=len(rows),physical_groups=len({r['group'] for r in rows}),
        positive=int(p.sum()),false_stop=int(fn.sum()),unprofitable_continue=int(fp.sum()),
        positive_recall=float((a[p]==1).mean()) if p.any() else None,
        missed_positive_net=float(t[fn].sum()),wasted_negative_net=float(-t[fp].sum()),
        hindsight_sign_regret_mean=float(np.mean(np.maximum(t,0)-a*t)),
        qcontinue_rmse=float(np.sqrt(np.mean([(r['q1']-r['target_normalized'])**2 for r in rows])))
        if all('q1' in r for r in rows) else None)
def grouped_metrics(rows):
    dates=sorted({r['date'] for r in rows})
    d={x:metrics([r for r in rows if r['date']==x]) for x in dates}
    groupmeans=[metrics([r for r in rows if r['group']==g])['hindsight_sign_regret_mean'] for g in sorted({r['group'] for r in rows})]
    return dict(pooled=metrics(rows),per_date=d,date_macro_regret=float(np.mean([v['hindsight_sign_regret_mean'] for v in d.values()])),
        physical_group_macro_regret=float(np.mean(groupmeans)))
def state_file(p):return p.with_name(p.stem+'_state.npz')
def same_context(p,q,a,b):
    with np.load(record(state_file(p))) as x,np.load(record(state_file(q))) as y:
        same=np.array_equal(x['observation'],y['observation']) and np.array_equal(x['base_action'],y['base_action'])
    keys=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible','phase']
    glob=lambda c:[{k:r[k] for k in keys} for r in c['outcomes'] if r['phase']!='local_completion']
    return bool(same and a['order']==b['order'] and a['aliases']==b['aliases'] and a['rho_limit']==b['rho_limit'] and glob(a)==glob(b))

def main():
    global MEAN,SCALE
    assert not OUT.exists(),'Never overwrite diagnosis'
    OUT.mkdir(parents=True)
    protocol=dict(id='GRID105_TWO_ISSUE_DIAGNOSIS',posthoc=True,
        authorized='User prioritizes objective alignment and DQN decision error after GRID104.',
        fixed_scope=['All4438 saved GRID102 reachable states, train/validation separated; all3 frozen gamma0 models and existing RF.',
          'All repeat0 GRID104 local decisions; repeat1 used only for decision equality, never duplicate as new samples.',
          'Only exact matched physical observation/base action/global feedback can borrow paid FRAME local suffix labels.',
          'Nearest five distinct TRAIN physical-call groups for prespecified March108/552 cases; descriptive only.'],
        no_new_fits=True,no_new_forecasts=True,no_new_physical_steps=True,no_new_scenarios=True,
        no_price_or_threshold_changes=True,final_test_untouched=True,
        analysis=['Realized one-batch sign error, positive recall, missed gain, date and physical-group summaries.',
          'Replay Q values and RF scores; compare training errors and validation errors without selecting seeds.',
          'One-batch versus all-stopping-point hindsight net signs on matched saved calls.',
          'Source audit of physical max-rho reward versus passive operational cost; label identifiability.'],
        limits='Partial visible features need not uniquely determine realized gain; sign errors do not prove nonconvergence. No causal whole-week label from different policy trajectories.')
    save(OUT/'design.json',protocol)
    from stable_baselines3 import DQN
    import torch,joblib
    torch.set_num_threads(1)
    d=read(ROOT/'outputs/grid102_search_rl/design.json');cost=d['cost_per_query'];gs=d['gain_scale']
    with np.load(record(ROOT/'outputs/grid102_search_rl/normalization.npz')) as z:MEAN=z['mean'];SCALE=z['scale']
    with record(ROOT/'outputs/grid102_search_rl/cases.pkl').open('rb') as f:cases=pickle.load(f)
    assert len(cases)==2968 and not {c['date'] for c in cases if c['partition']=='train'}&{c['date'] for c in cases if c['partition']=='validation'}
    rows=[]
    for c in cases:
        for j,s in enumerate(c['states']):
            rows.append(dict(sample_id=c['sample_id'],date=c['date'],partition=c['partition'],group=c['physical_state_group'],
                batch=j,next_queries=int(c['queries'][j]),target_net=float(c['gains'][j]-cost*c['queries'][j]),
                target_normalized=float((c['gains'][j]-cost*c['queries'][j])/gs),raw=s))
    X=norm(np.stack([r['raw'] for r in rows]));assert X.shape==(4438,41)
    models={f'DQN0_{s}':DQN.load(record(ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{s}_final.zip'),device='cpu') for s in range(3)}
    rf=joblib.load(record(ROOT/'outputs/grid103_learning_mechanism/models/next_batch_rf.joblib'))
    all_predictions=[];summaries={};Q={}
    for name,m in models.items():
        assert m.gamma==0 and m.num_timesteps==30000
        Q[name]=qvalues(m,X);a=np.argmax(Q[name],axis=1)
        assert np.array_equal(a,m.predict(X,deterministic=True)[0])
        pred=[{k:v for k,v in r.items() if k!='raw'}|dict(model=name,action=int(a[i]),q0=float(Q[name][i,0]),q1=float(Q[name][i,1])) for i,r in enumerate(rows)]
        summaries[name]={p:grouped_metrics([r for r in pred if r['partition']==p]) for p in ['train','validation']}
        all_predictions.extend(pred)
    rfscore=rf.predict(X);actions=rfscore>cost*np.array([r['next_queries'] for r in rows])/gs
    pred=[{k:v for k,v in r.items() if k!='raw'}|dict(model='RF',action=int(actions[i]),normalized_gain=float(rfscore[i])) for i,r in enumerate(rows)]
    summaries['RF']={p:grouped_metrics([r for r in pred if r['partition']==p]) for p in ['train','validation']}
    all_predictions.extend(pred)
    with gzip.open(OUT/'offline_predictions.jsonl.gz','wt',encoding='utf8') as f:
        for r in all_predictions:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'offline_complete':True,'states':len(rows)},ensure_ascii=False),flush=True)
    # Closed-loop labels only where saved reference covers the identical context.
    base=ROOT/'outputs/grid104_learning_closedloop';design=read(base/'design.json')
    decisions=[];unknown=[];repeatchecks=0;callschecked=0;critical=[]
    methods=['LOCAL_RF']+[f'LOCAL_DQN0_{s}' for s in range(3)]
    for selected in design['selected']:
        week=selected['scenario'];refdir=base/'runs'/f'r0__{week}__FRAME_GUARD'
        refs={}
        for p in sorted(refdir.glob('call[0-9][0-9][0-9].json')):
            c=read(p);refs[c['step']]=(p,c)
        for method in methods:
            folder=base/'runs'/f'r0__{week}__{method}'
            for p in sorted(folder.glob('call[0-9][0-9][0-9].json')):
                c=read(p)
                p1=base/'runs'/f'r1__{week}__{method}'/p.name;c1=read(p1)
                assert c['learned_decisions']==c1['learned_decisions'];repeatchecks+=1
                ds=c['learned_decisions']
                if not ds:continue
                same=False
                if c['step'] in refs:
                    rp,rc=refs[c['step']];same=same_context(p,rp,c,rc);callschecked+=1
                for dec in ds:
                    item=dict(week=week,date=selected['date'],method=method,step=c['step'],batch=dec['batch'],action=dec['action'],
                        remaining=dec['remaining'],next_queries=len(dec['next_ids']),clipped_features=dec['clipped_features'],matched=same)
                    obs=np.array(dec['normalized_features'],dtype=np.float32)
                    assert np.array_equal(obs,norm(np.array(dec['raw_features'])))
                    if method=='LOCAL_RF':
                        score=float(rf.predict(obs[None])[0]);assert score==dec['predicted_normalized_gain']
                        assert int(score>cost*len(dec['next_ids'])/gs)==dec['action']
                        item['predicted_normalized_gain']=score
                    else:
                        q=qvalues(models[method.replace('LOCAL_','')],obs[None])[0]
                        assert int(np.argmax(q))==dec['action'];item.update(q0=float(q[0]),q1=float(q[1]))
                    if not same:
                        item['unknown_reason']='No identical FRAME observation/base/global context; do not borrow downstream labels.'
                        unknown.append(item);continue
                    truth={r['pool_id']:r for r in rc['outcomes']};visible=c['outcomes'][:dec['paid_queries']]
                    commonkeys=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
                    assert all(all(r[k]==truth[r['pool_id']][k] for k in commonkeys) for r in visible)
                    initial=reward(best(visible));pending=rc['completion_candidate_ids'][dec['batch']*64:]
                    assert pending[:64]==dec['next_ids'] and len(pending)==dec['remaining']
                    v=visible.copy();nets=[0.];gains=[]
                    for start in range(0,len(pending),64):
                        before=reward(best(v));v.extend(truth[i] for i in pending[start:start+64]);gains.append(reward(best(v))-before)
                        nets.append(reward(best(v))-initial-cost*min(start+64,len(pending)))
                    assert gains
                    t=gains[0]-cost*len(dec['next_ids'])
                    item.update(actual_next_gain=gains[0],target_net=t,target_normalized=t/gs,
                        best_future_net=max(nets),full_gain=reward(best(v))-initial,
                        hindsight_myopic_continue=int(t>0),hindsight_continue=int(max(nets[1:])>0),
                        realized_one_batch_sign_error=bool(int(t>0)!=dec['action']))
                    decisions.append(item)
                    if week=='2037-03-02_100' and c['step'] in [108,552] and dec['batch']==0:
                        tridx=[i for i,r in enumerate(rows) if r['partition']=='train']
                        dist=np.sqrt(np.mean((X[tridx]-obs)**2,axis=1));ix=np.argsort(dist,kind='stable');seen=set();near=[]
                        for k in ix:
                            r=rows[tridx[k]]
                            if r['group'] in seen:continue
                            seen.add(r['group']);near.append(dict(group=r['group'],date=r['date'],sample_id=r['sample_id'],batch=r['batch'],
                                standardized_rms_distance=float(dist[k]),target_normalized=r['target_normalized'],next_queries=r['next_queries']))
                            if len(near)==5:break
                        critical.append(item|dict(nearest_five_train_groups=near))
    def closed_summary(rs):
        return dict(decisions=len(rs),calls=len({(r['week'],r['step']) for r in rs}),
            positive_next_batch=sum(r['target_net']>0 for r in rs),missed_positive=sum(r['target_net']>0 and r['action']==0 for r in rs),
            unprofitable_continue=sum(r['target_net']<=0 and r['action']==1 for r in rs),
            stopped_myopic_nonpositive_but_later_profitable=sum(r['action']==0 and r['target_net']<=0 and r['best_future_net']>0 for r in rs),
            stopped_no_profitable_stopping_point=sum(r['action']==0 and r['best_future_net']==0 for r in rs),
            stopped_with_positive_full_gain=sum(r['action']==0 and r['full_gain']>0 for r in rs))
    closed={m:dict(matched=closed_summary([r for r in decisions if r['method']==m]),
        unknown_decisions=sum(r['method']==m for r in unknown)) for m in methods}
    train=[r for r in rows if r['partition']=='train']
    date_group={dt:dict(collections.Counter(r['group'] for r in train if r['date']==dt)) for dt in sorted({r['date'] for r in train})}
    sampling={dt:dict(states=sum(v.values()),groups=len(v),largest_group_state_share=max(v.values())/sum(v.values())) for dt,v in date_group.items()}
    result=dict(posthoc=True,cost_per_query=cost,gain_scale=gs,offline=summaries,closedloop=closed,critical_cases=critical,
        train_reachable_state_concentration=sampling,repeat_call_decision_checks=repeatchecks,matched_call_comparisons=callschecked,
        known_decisions=len(decisions),unknown_decisions=len(unknown),new_fits=0,new_forecasts=0,new_physical_steps=0,
        labels_do_not_establish_causal_week_value=True)
    save(OUT/'diagnosis.json',result);save(OUT/'closedloop_decisions.json',dict(known=decisions,unknown=unknown))
    for p in ['work/grid02/ljn_nn/modules/rewards.py','work/grid03/score_adapter.py','work/grid102_search_rl/environment.py',
        'work/grid102_search_rl/prepare.py','work/grid102_search_rl/observations.py','work/grid103_learning_mechanism/train.py',
        'work/grid101_gain_learning/features.py','work/grid104_learning_closedloop/policies.py',
        'outputs/grid104_learning_closedloop/review.json','outputs/grid104_learning_closedloop/decision.json']:
        record(ROOT/p)
    final=read(ROOT/'outputs/grid100_data_foundation/design.json')
    assert not any((ROOT/final['environment']/'chronics'/r['scenario']).exists() for r in final['scenario_plan'] if r['partition']=='test')
    assert all(digest(ROOT/p)==h for p,h in HASHES.items())
    save(OUT/'input_hashes.json',HASHES)
    save(OUT/'review.json',dict(passed=True,input_files_unchanged=len(HASHES),states=4438,predictions=len(all_predictions),
        known_online_decisions=len(decisions),unknown_online_decisions=len(unknown),repeated_calls_checked=repeatchecks,
        frozen_models_reproduced=True,new_fits=0,new_forecasts=0,new_physical_steps=0,final_test_materialized=False,
        note='Diagnostic arithmetic and fixed-model actions checked, not independent AC or whole-week counterfactual evaluation.'))
    print(json.dumps(dict(offline={m:{p:v['pooled'] for p,v in a.items()} for m,a in summaries.items()},closedloop=closed),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
