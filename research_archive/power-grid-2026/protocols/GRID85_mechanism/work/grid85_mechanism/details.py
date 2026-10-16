"""Descriptive evidence extension; no strategy change or extra feedback."""
from pathlib import Path
from collections import Counter
import csv,json,statistics,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid85_mechanism'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def csvread(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    actual=csvread('actual_calls.csv');full=csvread('offline_full_calls.csv');matched=csvread('matched_actual_full.csv')
    split=[]
    for g in ['old13','new6']:
        rr=[r for r in actual if r['cohort']==g];none=[r for r in rr if not r['final_choice']];same_valid=[r for r in rr if r['final_choice'] and r['final_changed']=='False']
        split.append(dict(cohort=g,no_action_calls=len(none),no_action_queries=sum(int(r['queries']) for r in none),
           no_action_all_exhaustive=all(r['queries']==r['pool'] for r in none),same_valid_prefix_calls=len(same_valid),
           same_valid_prefix_extra_queries=sum(int(r['queries'])-int(r['prefix_queries']) for r in same_valid),
           matched_full_actual_calls=sum(r['cohort']==g and r['matched_full']=='True' for r in matched),
           material_local_gain_calls=sum(r['cohort']==g and float(r['local_reward_gain'] or 0)>1e-6 for r in actual)))
    cases=[]
    for r in read(OUT/'mechanism_results.json')['first_divergences']:
        r=dict(r)
        if r['guard_source']:
            c=read(ROOT/r['guard_source']);truth={x['pool_id']:x for x in c['outcomes']}
            old=truth[r['old_final_choice']];new=truth[r['guard_final_choice']]
            r.update(old_candidate_rho=old['rho'],new_candidate_rho=new['rho'],
                     quantized_one_step_reward_gain=float(np.float32(new['reward']))-float(np.float32(old['reward'])))
        else:r.update(old_candidate_rho=None,new_candidate_rho=None,quantized_one_step_reward_gain=None)
        cases.append(r)
    save('first_divergence_details.json',cases)
    with (OUT/'first_divergence_details.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(cases[0]));w.writeheader();w.writerows(cases)
    components=[]
    R=read(ROOT/'outputs/grid81/review.json')
    for method in ['FULL','FRAME_GUARD']:
        gs=[g for g in R['groups'] if g['method']==method]
        v=dict(method=method,source='GRID81 same-batch timing; equal six dates',date_count=len(gs))
        for k in ['controller_s','optimizer_s','forecast_s','n1_nonforecast_s','remainder_s','decision_p95_s']:
            v[k]=statistics.mean(g[k] for g in gs)
        v['forecast_share']=v['forecast_s']/v['controller_s'];components.append(v)
    original=(ROOT/'outputs/grid85_mechanism/source_receipt.json').read_bytes()
    sources=['work/grid85_mechanism/details.py','outputs/grid81/review.json']
    save('additional_details.json',dict(no_change_split=split,compute_components=components,
                initial_safe_first_divergences=sum(r['prefix_safe_at_divergence'] is True for r in cases),
                first_divergences_same_state_base=sum(r['divergence_same_observation_base'] for r in cases),
                first_divergences_with_positive_step_gain=sum((r['quantized_one_step_reward_gain'] or 0)>1e-6 for r in cases),
                first_divergences_old_and_guard_both_complete_better=sum(r['first_action_divergence_step'] is not None and r['old_cost_delta_full'] is not None and r['guard_cost_delta_full'] is not None and r['guard_cost_delta_full']<r['old_cost_delta_full']-1e-9 for r in cases),
                first_divergences_old_and_guard_both_complete_worse=sum(r['first_action_divergence_step'] is not None and r['old_cost_delta_full'] is not None and r['guard_cost_delta_full'] is not None and r['guard_cost_delta_full']>r['old_cost_delta_full']+1e-9 for r in cases),
                original_source_receipt_sha256=hashlib.sha256(original).hexdigest(),
                extra_sources=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sources]))
    print(json.dumps(read(OUT/'additional_details.json'),ensure_ascii=False))
if __name__=='__main__':main()
