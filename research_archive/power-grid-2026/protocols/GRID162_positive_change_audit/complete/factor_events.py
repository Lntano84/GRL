"""All old local trajectories paired: where is FACTOR's net improvement coming from?"""
import collections,math
import run as D
def main():
 D.verify();assert not (D.O/'factor_event_summary.json').exists();path=D.R/'outputs/grid116_matched_module/validation_trajectories.json';source_sha=D.sha(path);raw=D.read(path);plan=D.read(D.R/'outputs/grid116_matched_module/design.json');fee=plan['unchanged']['cost_per_query'];scale=plan['unchanged']['gain_scale'];records=[];summary=[]
 methods={n:{r['sample_id']:r for r in raw if r['method']==n} for n in [*[f'DIRECT{s}' for s in range(3)],*[f'BASE_FACTOR_seed{s}' for s in range(3)]]}
 for seed in range(3):
  aa,bb=methods[f'BASE_FACTOR_seed{seed}'],methods[f'DIRECT{seed}'];assert aa.keys()==bb.keys() and len(aa)==590;datecounts=collections.Counter(r['date'] for r in aa.values());assert len(datecounts)==4
  for sid,a in aa.items():
   b=bb[sid];assert all(a[k]==b[k] for k in ['date','physical_state_group','scenario','horizon','full_gain','full_queries']);gain=a['gain']-b['gain'];saved=b['queries']-a['queries'];net=a['net_utility']-b['net_utility'];gc=gain/scale;qc=fee*saved/scale;assert abs(net-gc-qc)<1e-12
   if gain==0 and saved==0:cat='UNCHANGED'
   elif saved>0 and gain>=0:cat='LESS_QUERY_NO_GAIN_LOSS'
   elif saved>0 and net>0:cat='LESS_QUERY_SMALL_GAIN_SACRIFICE_NET_POSITIVE'
   elif saved<0 and net>0:cat='MORE_QUERY_GAIN_PAYS'
   elif saved>0:cat='LESS_QUERY_GAIN_LOSS_TOO_LARGE'
   elif saved<0:cat='MORE_QUERY_COST_NOT_REPAID'
   else:cat='SAME_QUERY_DIFFERENT_GAIN'
   weight=1/(3*4*datecounts[a['date']]);records.append(dict(seed=seed,sample_id=sid,date=a['date'],scenario=a['scenario'],physical_state_group=a['physical_state_group'],category=cat,base_batches=b['batches'],factor_batches=a['batches'],base_queries=b['queries'],factor_queries=a['queries'],saved_queries=saved,gain_delta=gain,gain_contribution=gc,query_charge_contribution=qc,net_delta=net,macro_weight=weight))
 for cat in sorted({r['category'] for r in records}):
  rr=[r for r in records if r['category']==cat];summary.append(dict(category=cat,seed_prefix_pairs=len(rr),distinct_start_prefixes=len({r['sample_id'] for r in rr}),physical_state_groups=len({r['physical_state_group'] for r in rr}),dates=sorted({r['date'] for r in rr}),net_macro_contribution=math.fsum(r['macro_weight']*r['net_delta'] for r in rr),gain_macro_contribution=math.fsum(r['macro_weight']*r['gain_contribution'] for r in rr),query_macro_contribution=math.fsum(r['macro_weight']*r['query_charge_contribution'] for r in rr)))
 overall=D.read(D.O/'factor_vs_base_summary.json');assert abs(sum(r['net_macro_contribution'] for r in summary)-overall['net_delta'])<1e-12;assert source_sha==D.sha(path);D.verify();D.save('factor_all1770_paired_sequences.json',records);D.save('factor_event_summary.json',summary);D.save('factor_event_receipt.json',dict(passed=True,source=path.relative_to(D.R).as_posix(),source_sha256=source_sha,post_hoc_descriptive=True,seed_prefix_pairs=1770,start_prefixes=590,physical_state_groups=len({r['physical_state_group'] for r in records}),macro_contributions_reconcile=True,new_fits=0,new_predictions=0,new_physics=0));print(summary)
if __name__=='__main__':main()
