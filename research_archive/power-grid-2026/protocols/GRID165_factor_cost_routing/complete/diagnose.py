"""Account for all fixed routing changes; no fitting or selection."""
import collections,math,pickle
import numpy as np
import run as D
def main():
 D.verify_trained();assert D.read(D.O/'review.json')['passed'];assert not (D.O/'diagnosis_receipt.json').exists();d=D.read(D.O/'design.json');fee=d['cost_per_query'];scale=d['gain_scale'];cases={c['sample_id']:c for c in pickle.loads((D.R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};tr=D.read(D.O/'validation_trajectories.json');v={(r['method'],r['sample_id']):r for r in tr};meta={p:D.read(D.R/f'outputs/grid113_relational_learning/{p}_metadata.json') for p in ['train','validation']};ix={p:{(r['sample_id'],r['position']):i for i,r in enumerate(rows)} for p,rows in meta.items()};scores=np.load(D.O/'state_scores.npz');old=np.load(D.R/'outputs/grid116_matched_module/state_predictions.npz');control=[]
 for seed in [0,1,2]:
  for part in ['train','validation']:
   a=scores[f'ZERO_MAIN_seed{seed}__{part}'];b=old[f'BASE_FACTOR_seed{seed}__{part}'];control.append(dict(seed=seed,partition=part,rows=len(a),max_score_error=float(np.max(abs(a-b))),decision_differences=int(np.sum((a[:,1]>a[:,0])!=(b[:,1]>b[:,0])))))
 paired=[];first=[];prior={(r['seed'],r['sample_id']):r['category'] for r in D.read(D.R/'outputs/grid162_positive_change_audit/factor_all1770_paired_sequences.json')};counts=collections.Counter(c['date'] for c in cases.values() if c['partition']=='validation')
 for seed in [0,1,2]:
  for c in cases.values():
   if c['partition']!='validation':continue
   sid=c['sample_id'];a=v[f'COST_MAIN_seed{seed}',sid];b=v[f'ZERO_MAIN_seed{seed}',sid];g=a['gain']-b['gain'];q=b['queries']-a['queries'];net=a['net_utility']-b['net_utility'];w=1/(3*4*counts[c['date']]);assert abs(net-(g+fee*q)/scale)<1e-12
   if g==0 and q==0:cat='UNCHANGED'
   elif q>0 and g>=0:cat='LESS_QUERY_NO_GAIN_LOSS'
   elif q>0 and net>0:cat='LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE'
   elif q<0 and net>0:cat='MORE_QUERY_GAIN_PAYS'
   elif q>0:cat='LESS_QUERY_GAIN_LOSS_TOO_LARGE'
   elif q<0:cat='MORE_QUERY_COST_NOT_REPAID'
   else:cat='SAME_QUERY_DIFFERENT_GAIN'
   r=dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],category=cat,previous_factor_event=prior[seed,sid],net_delta=net,gain_contribution=g/scale,query_contribution=fee*q/scale,macro_weight=w,saved_queries=q,same_action=a['chosen_pool_id']==b['chosen_pool_id']);paired.append(r)
   if a['batches']!=b['batches']:
    p=min(a['batches'],b['batches']);assert a['actions'][:p]==b['actions'][:p];i=ix['validation'][sid,p];aa=scores[f'COST_MAIN_seed{seed}__validation'][i];bb=scores[f'ZERO_MAIN_seed{seed}__validation'][i];assert int(aa[1]>aa[0])==a['actions'][p] and int(bb[1]>bb[0])==b['actions'][p];first.append(dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],category=cat,position=p,known_next_queries=int(c['queries'][p]),control_margin=float(bb[1]-bb[0]),treatment_margin=float(aa[1]-aa[0]),audit_next_gain=float(c['gains'][p]),audit_next_net=float((c['gains'][p]-fee*c['queries'][p])/scale),future_audit_only=True))
 summary=[]
 for cat in sorted({r['category'] for r in paired}):
  rr=[r for r in paired if r['category']==cat];summary.append(dict(category=cat,seed_start_pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),net_contribution=math.fsum(r['macro_weight']*r['net_delta'] for r in rr),gain_contribution=math.fsum(r['macro_weight']*r['gain_contribution'] for r in rr),query_contribution=math.fsum(r['macro_weight']*r['query_contribution'] for r in rr),same_action_pairs=sum(r['same_action'] for r in rr)))
 net=sum(r['net_contribution'] for r in summary);assert abs(net-D.read(D.O/'paired_summary.json')['mean_net_delta'])<1e-12;pay=dict(net=net,gain=sum(r['gain_contribution'] for r in summary),query=sum(r['query_contribution'] for r in summary));D.save('control_old_factor_reproduction.json',control);D.save('all1770_paired_events.json',paired);D.save('paired_event_summary.json',summary);D.save('first_difference_records.json',first);D.save('payoff_decomposition.json',pay);D.verify_trained();D.save('diagnosis_receipt.json',dict(passed=True,all1770_reconciled=True,no_new_fit=True,no_new_predictions=True,first_difference_events=len(first),all_seeds_dates_retained=True));print(__import__('json').dumps(dict(payoff=pay,events=summary,control=control),ensure_ascii=False))
if __name__=='__main__':main()
