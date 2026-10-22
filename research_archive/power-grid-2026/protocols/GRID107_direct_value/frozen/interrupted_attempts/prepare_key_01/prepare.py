"""Freeze paired full-information targets and seed/budget before any fit."""
import collections,copy,datetime,inspect,shutil,sys
import numpy as np
from common107 import ROOT,W,OUT,ARCHIVE,read,save,sha,sealed,loadcases,normalization,normalized,carrier,weight_digest
def main():
    assert not OUT.exists();sealed()
    assert read(ROOT/'outputs/grid106_decision_branches/complete.json')['training_not_started']
    base=read(ROOT/'outputs/grid103_learning_mechanism/design.json');old=read(ROOT/'outputs/grid102_search_rl/design.json')
    inherited=read(ROOT/'outputs/grid104_learning_closedloop/code_freeze.json');prior_data=read(ROOT/'outputs/grid104_learning_closedloop/data_freeze.json')
    for p,h in {**inherited,**prior_data}.items():assert sha(ROOT/p)==h,p
    cases=loadcases();mean,std=normalization();train=[c for c in cases if c['partition']=='train'];val=[c for c in cases if c['partition']=='validation']
    assert len(train)==2378 and len(val)==590 and not {c['date'] for c in train}&{c['date'] for c in val}
    OUT.mkdir();metadata=[];X=[];target=[];prob=[]
    date_cases=collections.Counter(c['date'] for c in train)
    for c in train:
        for j,s in enumerate(c['states']):
            X.append(normalized(s,mean,std));target.append(float((c['gains'][j]-old['cost_per_query']*c['queries'][j])/old['gain_scale']))
            prob.append(1/len(date_cases)/date_cases[c['date']]/len(c['states']))
            metadata.append(dict(sample_id=c['sample_id'],position=j,date=c['date'],physical_state_group=c['physical_state_group'],
                next_queries=int(c['queries'][j]),partition='train'))
    p=np.array(prob);assert abs(p.sum()-1)<1e-12
    for date in date_cases:assert abs(sum(v for v,r in zip(p,metadata) if r['date']==date)-1/16)<1e-12
    y=np.array(target,dtype=np.float32);assert np.array_equal(np.sign(y),np.sign(target))
    np.savez_compressed(OUT/'train.npz',X=np.stack(X),target=y,sampling_probability=p)
    save(OUT/'training_rows.json',metadata)
    params=base['temporal_ablation']['dqn'];assert params['gamma']==0
    D=dict(id='GRID107_SAME_NETWORK_DIRECT_VALUE',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User authorizes the notified same-data/input/network direct regression versus existing DQN. Offline only.',
        source='Original GRID100 train prefixes, GRID102 states/normalization/price, GRID103 gamma0 DQN checkpoints. GRID106 cases diagnostic only.',
        split=dict(train_prefixes=2378,train_states=3582,train_dates=16,validation_prefixes=590,validation_states=856,validation_dates=4),
        features=old['features'],cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],
        seeds=[0,1,2],architecture='Exactly original SB3 QNetwork:41 ->64 ReLU ->64 ReLU ->2;6978 parameters, paired initial seed.',
        dqn_initialization=params,training=dict(updates=7250,states_per_batch=32,scalar_targets_per_update=64,
            optimizer='Original SB3 Adam parameter groups, learning_rate0.0003, max_grad_norm10.',
            loss='torch smooth_l1_loss beta1, mean over32 states x2 action values; same scalar target budget as DQN64 sampled action targets.',
            targets='STOP=0, CONTINUE=(next best-admissible float32 reward gain - unchanged per-query fee*next_queries)/old gain_scale.',
            sampling='Equal date, equal starting prefix within date, uniform reachable position within prefix. Same initial task masses as DQN; replaces policy-dependent visitation/replay with complete paid-feedback coverage.',
            fitting='Exactly3 fits; final7250 updates only, no validation early stop/checkpoint choice or seed selection.',
            sampler_seed_offset=20261022,trace_every_updates=500,
            intentionally_changed='Complete known-action labels and state coverage replace exploration/replay. This package cannot isolate sampling from full supervision separately.'),
        evaluation=dict(methods=['DIRECT0','DIRECT1','DIRECT2','DQN0_0','DQN0_1','DQN0_2','RF','STOP','FULL_LOCAL','BATCH1'],
            primary='Four-date macro search net utility versus paired/all3 old gamma0 DQN and BATCH1; report all dates and seeds.',
            secondary='Train/validation positive misses, negative continues, value RMSE, gain capture/query fraction, single/multi horizons and physical-call-balanced utility.',
            critical='Postfit fixed3 contexts: original March108, original March552, RF-path March552. No added labels, fit weights, feature changes or rules from those cases.',
            interpretation='Training mechanism evidence only if mean train date-macro sign regret falls and >=2/3 paired validation utility differences positive; no same-quality physical claim. BATCH1 remains a mandatory independent comparator.'),
        limits='No new observations/forecasts/physical steps/scenarios, price sweeps, positive-class oversampling, larger nets, DQN retraining, final-test materialization, or automatic closed-loop follow-up. Exposed development dates and correlated prefixes, not independent confirmation.')
    save(OUT/'design.json',D)
    import torch
    from stable_baselines3 import DQN
    sys.path.insert(0,str(ROOT/'work/grid102_search_rl'));from environment import LocalSearchEnv
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    initial={}
    for seed in D['seeds']:
        a=carrier(seed,params);b=DQN('MlpPolicy',LocalSearchEnv(train,mean,std,D['cost_per_query'],D['gain_scale']),seed=seed,**params)
        assert weight_digest(a.q_net)==weight_digest(b.q_net)
        assert sum(x.numel() for x in a.q_net.parameters())==6978
        initial[str(seed)]=weight_digest(a.q_net);a.env.close();b.env.close()
    # Verify the loss and gamma0 identity against the installed primary implementation.
    from stable_baselines3.dqn.policies import DQNPolicy
    source=inspect.getsource(DQN.train);assert 'F.smooth_l1_loss' in source and 'self.gamma * next_q_values' in source
    q=torch.tensor([[.2,-.4],[.1,.9]],requires_grad=True);t=torch.tensor([[0.,.6],[0.,-.2]])
    paired=torch.nn.functional.smooth_l1_loss(q,t)
    enumerated=torch.nn.functional.smooth_l1_loss(q.reshape(-1),t.reshape(-1));assert paired.item()==enumerated.item()
    save(OUT/'preflight.json',dict(passed=True,train_states=len(X),positive_targets=int(np.sum(y>0)),train_only_targets=True,
        date_mass_checked=True,scalar_action_targets_per_fit=7250*64,initial_q_hashes=initial,
        matched_original_initialization=True,paired_loss_equivalence=True,model_fits=0,new_physical_calls=0))
    paths=list(W.glob('*.py'))+[OUT/'design.json',OUT/'preflight.json']+[__import__('pathlib').Path(inspect.getsourcefile(DQN)),__import__('pathlib').Path(inspect.getsourcefile(DQNPolicy))]
    for p in paths:
        try:key=p.relative_to(ROOT).as_posix()
        except ValueError:key=p.as_posix()
        inherited[key]=sha(p)
    save(OUT/'code_freeze.json',inherited)
    paths=[OUT/'train.npz',OUT/'training_rows.json',ROOT/'outputs/grid102_search_rl/cases.pkl',ROOT/'outputs/grid103_learning_mechanism/validation_trajectories.json',ROOT/'outputs/grid105_learning_diagnosis/offline_predictions.jsonl.gz']
    # Critical probe inputs are evaluation-only: raw_features already saved before each decision.
    probe_sources={}
    runs=ROOT/'outputs/grid104_learning_closedloop/runs'
    for label,method,step in [('MARCH108','LOCAL_RF',108),('MARCH552','LOCAL_DQN0_0',552),('RF_PATH552','LOCAL_RF',552)]:
        for path in sorted((runs/f'r0__2037-03-02_100__{method}').glob('call[0-9][0-9][0-9].json')):
            if read(path)['step']==step:probe_sources[label]=path.relative_to(ROOT).as_posix();paths.append(path);break
    assert len(probe_sources)==3;save(OUT/'probe_sources.json',probe_sources);paths.append(OUT/'probe_sources.json')
    paths += [ROOT/'outputs/grid106_decision_branches/labels.json',ROOT/'outputs/grid106_decision_branches/posthoc_interpretation.json',
        ROOT/'outputs/grid104_learning_closedloop/proxy_value_diagnosis.json']
    save(OUT/'data_freeze.json',{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    print(read(OUT/'preflight.json'),flush=True)
if __name__=='__main__':main()
