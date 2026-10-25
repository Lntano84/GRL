"""Paired first differences and complementarity of existing final models; no fits."""
import collections,gzip,math,pickle
import numpy as np
import run as D
def main():
 D.verify_trained();assert D.read(D.O/'review.json')['passed'];assert not (D.O/'diagnosis_receipt.json').exists();plan=D.read(D.O/'design.json');fee=plan['cost_per_query'];scale=plan['gain_scale'];cases={c['sample_id']:c for c in pickle.loads((D.R/'outputs/grid102_search_rl/cases.pkl').read_bytes())};raw=D.read(D.O/'validation_trajectories.json');v={(r['method'],r['sample_id']):r for r in raw};data={p:D.data(p) for p in ['train','validation']};lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(data[p][2])} for p in data};scores=np.load(D.O/'state_scores.npz');checks=collections.Counter();oldmax=0.
 with gzip.open(D.R/'outputs/grid107_direct_value/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=__import__('json').loads(line)
   if r['method'] not in ['DIRECT0','DIRECT1','DIRECT2']:continue
   seed=int(r['method'][-1]);part=r['partition'];i=lookup[part][r['sample_id'],r['position']];q=scores[f'ZERO42_seed{seed}__{part}'][i];e=max(abs(float(q[0])-r['q0']),abs(float(q[1])-r['q1']));oldmax=max(oldmax,e);assert bool(q[1]>q[0])==bool(r['q1']>r['q0']);checks['old_base_state_decisions']+=1
 events=[];contributions=[];first=[];complement=[]
 for seed in [0,1,2]:
  datecounts=collections.Counter(c['date'] for c in cases.values() if c['partition']=='validation')
  for c in cases.values():
   if c['partition']!='validation':continue
   sid=c['sample_id'];a=v[f'COST42_seed{seed}',sid];b=v[f'ZERO42_seed{seed}',sid];f=v[f'BASE_FACTOR_seed{seed}',sid];dg=a['gain']-b['gain'];saved=b['queries']-a['queries'];net=a['net_utility']-b['net_utility'];weight=1/(3*4*datecounts[c['date']]);gc=dg/scale;qc=fee*saved/scale;assert abs(net-gc-qc)<1e-12
   if dg==0 and saved==0:cat='UNCHANGED'
   elif saved>0 and dg>=0:cat='LESS_QUERY_NO_GAIN_LOSS'
   elif saved>0 and net>0:cat='LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE'
   elif saved<0 and net>0:cat='MORE_QUERY_GAIN_PAYS'
   elif saved>0:cat='LESS_QUERY_GAIN_LOSS_TOO_LARGE'
   elif saved<0:cat='MORE_QUERY_COST_NOT_REPAID'
   else:cat='SAME_QUERY_DIFFERENT_GAIN'
   events.append(dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],category=cat,net_delta=net,gain_contribution=gc,query_contribution=qc,macro_weight=weight,saved_queries=saved,same_choice=a['chosen_pool_id']==b['chosen_pool_id']))
   if a['batches']!=b['batches']:
    p=min(a['batches'],b['batches']);assert a['actions'][:p]==b['actions'][:p];i=lookup['validation'][sid,p];qa=scores[f'COST42_seed{seed}__validation'][i];qb=scores[f'ZERO42_seed{seed}__validation'][i];assert int(qa[1]>qa[0])==a['actions'][p] and int(qb[1]>qb[0])==b['actions'][p];first.append(dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],category=cat,position=p,base_margin=float(qb[1]-qb[0]),cost_margin=float(qa[1]-qa[0]),continue_output_change=float(qa[1]-qb[1]),stop_output_contribution=float(qb[0]-qa[0]),known_next_queries=int(c['queries'][p]),audit_next_gain=float(c['gains'][p]),audit_next_net=float((c['gains'][p]-fee*c['queries'][p])/scale),future_audit_only=True))
   # Compare both new and old positive routes on the same teacher-state starting prefix.
   da=a['net_utility']-b['net_utility'];df=f['net_utility']-b['net_utility'];kind=('BOTH_GAIN' if da>0 and df>0 else ('COST_ONLY_GAIN' if da>0 else ('FACTOR_ONLY_GAIN' if df>0 else 'NEITHER_GAIN')));complement.append(dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],kind=kind,cost_vs_base=da,factor_vs_base=df,cost_vs_factor=a['net_utility']-f['net_utility'],macro_weight=weight,cost_same_base_choice=a['chosen_pool_id']==b['chosen_pool_id'],factor_same_base_choice=f['chosen_pool_id']==b['chosen_pool_id']))
 for cat in sorted({r['category'] for r in events}):
  rr=[r for r in events if r['category']==cat];contributions.append(dict(category=cat,seed_start_pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_state_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),net_macro_contribution=math.fsum(r['macro_weight']*r['net_delta'] for r in rr),gain_macro_contribution=math.fsum(r['macro_weight']*r['gain_contribution'] for r in rr),query_macro_contribution=math.fsum(r['macro_weight']*r['query_contribution'] for r in rr),same_choice_pairs=sum(r['same_choice'] for r in rr)))
 comp=[]
 for kind in sorted({r['kind'] for r in complement}):
  rr=[r for r in complement if r['kind']==kind];comp.append(dict(kind=kind,seed_start_pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_state_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),cost_contribution=math.fsum(r['macro_weight']*r['cost_vs_base'] for r in rr),factor_contribution=math.fsum(r['macro_weight']*r['factor_vs_base'] for r in rr),cost_vs_factor_contribution=math.fsum(r['macro_weight']*r['cost_vs_factor'] for r in rr)))
 net=math.fsum(r['net_macro_contribution'] for r in contributions);assert abs(net-D.read(D.O/'paired_summary.json')['mean_net_delta'])<1e-12;gain=math.fsum(r['gain_macro_contribution'] for r in contributions);query=math.fsum(r['query_macro_contribution'] for r in contributions);D.save('paired_event_decomposition.json',contributions);D.save('first_difference_records.json',first);D.save('complementarity_records.json',complement);D.save('complementarity_summary.json',comp);D.save('payoff_decomposition.json',dict(net=net,gain=gain,query=query,gain_fraction_of_net=gain/net,query_fraction_of_net=query/net,not_causal_percent_of_factor_improvement=True));D.verify_trained();D.save('diagnosis_receipt.json',dict(passed=True,checks=dict(checks),zero42_oldbase_decisions_same=True,max_zero42_oldbase_score_error=oldmax,all1770_paired_contributions_reconcile=True,first_difference_records=len(first),no_new_fits=True,no_new_predictions=True));print(__import__('json').dumps(dict(payoff=D.read(D.O/'payoff_decomposition.json'),events=contributions,complement=comp,oldmax=oldmax),ensure_ascii=False))
if __name__=='__main__':main()
