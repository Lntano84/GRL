"""Describe all no-gain-loss saving events, without relabeling them as weekly quality."""
import collections
import run as D
def main():
 D.verify();assert not (D.O/'factor_equal_gain_action_check.json').exists();src=D.R/'outputs/grid116_matched_module/validation_trajectories.json';receipt=D.read(D.O/'factor_event_receipt.json');assert D.sha(src)==receipt['source_sha256'];lookup={(r['method'],r['sample_id']):r for r in D.read(src)};rr=[r for r in D.read(D.O/'factor_all1770_paired_sequences.json') if r['category']=='LESS_QUERY_NO_GAIN_LOSS'];out=[]
 for r in rr:
  a=lookup[f"BASE_FACTOR_seed{r['seed']}",r['sample_id']];b=lookup[f"DIRECT{r['seed']}",r['sample_id']];assert a['gain']==b['gain'] and a['queries']<b['queries'];out.append(dict(seed=r['seed'],sample_id=r['sample_id'],date=r['date'],physical_state_group=r['physical_state_group'],same_pool_id=a['chosen_pool_id']==b['chosen_pool_id'],both_gain_zero=a['gain']==b['gain']==0,base_queries=b['queries'],factor_queries=a['queries'],base_choice=b['chosen_pool_id'],factor_choice=a['chosen_pool_id']))
 assert len(out)==260;D.save('factor_equal_gain_action_records.json',out);D.save('factor_equal_gain_action_check.json',dict(passed=True,seed_prefix_pairs=260,distinct_start_prefixes=len({r['sample_id'] for r in out}),physical_state_groups=len({r['physical_state_group'] for r in out}),same_pool_id_pairs=sum(r['same_pool_id'] for r in out),both_gain_zero_pairs=sum(r['both_gain_zero'] for r in out),source_sha256=D.sha(src),not_original_frame_action_reference=True,not_weekly_quality=True));D.verify();print(D.read(D.O/'factor_equal_gain_action_check.json'))
if __name__=='__main__':main()
