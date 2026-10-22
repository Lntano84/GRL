"""Supplementary saved-data cross checks, no fitting or protocol changes."""
from pathlib import Path
import collections,json,sys
import numpy as np
root=Path('D:/Codex/GRL研究');sys.path.insert(0,str(root/'work/grid103_learning_mechanism'))
from common import OUT,read,save,load,verify
verify();cases,_,_,D=load()
states=read(OUT/'state_predictions.json');pred={(s['sample_id'],s['position']):s for s in states}
rows=read(OUT/'validation_trajectories.json');visited=read(OUT/'visited_decisions.json')
by_trajectory=collections.defaultdict(list)
for d in visited:by_trajectory[d['method'],d['sample_id']].append(d)
action_checks=0
for r in rows:
    ds=by_trajectory[r['method'],r['sample_id']]
    assert [d['action'] for d in ds]==r['actions']
    for j,d in enumerate(ds):
        assert d['position']==j
        if r['method'] in pred[r['sample_id'],j]['actions']:
            assert d['action']==pred[r['sample_id'],j]['actions'][r['method']]
            action_checks+=1
profiles={}
for partition in ['train','validation']:
    cs=[c for c in cases if c['partition']==partition];delayed=[];oracle_diff=[];gain_later=[];pos_later=[]
    for c in cs:
        net=(c['gains']-D['cost_per_query']*c['queries'])/D['gain_scale']
        optimum=max(np.r_[0.,np.cumsum(net)])
        greedy=0.
        for n in net:
            if n<=0:break
            greedy+=n
        oracle_diff.append(float(optimum-greedy))
        for j,n in enumerate(net):
            if n<=0 and max(np.cumsum(net[j:]))>0:delayed.append((c['sample_id'],j,c['physical_state_group'],c['date']))
        if len(net)>1 and any(c['gains'][1:]>0):gain_later.append(c)
        if len(net)>1 and any(net[1:]>0):pos_later.append(c)
    profiles[partition]=dict(cases=len(cs),states=sum(len(c['queries']) for c in cs),multi_cases=sum(len(c['queries'])>1 for c in cs),
        delayed_states=len(delayed),delayed_physical_groups=len({r[2] for r in delayed}),delayed_dates=sorted({r[3] for r in delayed}),
        oracle_better_than_next_batch_greedy_cases=sum(d>1e-12 for d in oracle_diff),
        max_oracle_difference=max(oracle_diff),cases_with_later_gross_gain=len(gain_later),cases_with_later_positive_net=len(pos_later),
        physical_groups_with_later_gain=len({c['physical_state_group'] for c in gain_later}))
    if partition=='validation':assert len(delayed)==read(OUT/'decision.json')['delayed_opportunities']['states']
receipt=dict(passed=True,visited_prediction_action_checks=action_checks,matched_visited_sequences=len(rows),profiles=profiles,
    scope='Post-evaluation diagnostic only. No fitting, configuration change, new simulation, or final-test access.')
save(OUT/'additional_checks.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
