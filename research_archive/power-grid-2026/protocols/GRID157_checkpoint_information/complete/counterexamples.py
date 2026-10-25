"""Post-analysis descriptive audit of ALL192 CONTINUE rows; never model input."""
import collections,json
import run as D
def main():
 D.verify();assert not (D.O/'negative192_audit.json').exists();rows=D.D.rowfile(D.R/'outputs/grid150_linear_control/original_fit_rows.json.gz');refs={r['call_id']:r for r in D.B.refs()};out=[]
 for r in rows:
  if r['meta']['visible_count']!=192 or r['target']['stop_label']:continue
  ref=refs[r['meta']['call_id']];c=D.read(D.R/ref['source']);truth={x['pool_id']:x for x in c['outcomes']};selected=r['target']['after_stop_choice_label'];orig=r['target']['original_frame_choice_label'];delta=D.B.P.key(truth[selected])[0]-D.B.P.key(truth[orig])[0] if selected is not None and orig is not None else None
  out.append(dict(call_id=ref['call_id'],date=ref['date'],visible_count=192,source=ref['source'],stop_choice=selected,original_choice=orig,stop_queries=r['target']['stop_total_queries_known_now'],original_queries=r['target']['original_future_total_queries_label'],reward_stop_minus_original=delta,prefix_safe=r['features']['prefix_safe'],incumbent_changed=r['features']['incumbent_changed_since_prefix'],last_batch_reward_gain=r['features']['last_batch_reward_gain'],stale_batches=r['features']['stale_batches'],original_choice_rank=c['order'].index(orig)+1 if orig is not None else None,source_sha256=D.sha(D.R/ref['source'])))
 assert len(out)==10;D.save('negative192_audit.json',out);D.save('negative192_summary.json',dict(post_analysis_descriptive=True,all10_retained=True,not_online_features=True,dates=dict(collections.Counter(x['date'] for x in out)),reward_sign=dict(collections.Counter('higher' if x['reward_stop_minus_original']>0 else ('lower' if x['reward_stop_minus_original']<0 else 'equal') for x in out if x['reward_stop_minus_original'] is not None)),stale=dict(collections.Counter(x['stale_batches'] for x in out)),new_fits=0,new_scores=0,no_label_or_policy_change=True));D.verify();print(json.dumps(D.read(D.O/'negative192_summary.json')))
if __name__=='__main__':main()
