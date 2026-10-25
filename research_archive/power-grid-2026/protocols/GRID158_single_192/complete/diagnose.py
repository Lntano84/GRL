"""All-partition192 scores and three pre-existing primary mistakes; audit only."""
import collections,json
import numpy as np
import run as D
def metrics(rows,p):
 from sklearn.metrics import average_precision_score,roc_auc_score
 y=np.array([not r['target']['stop_label'] for r in rows],int);q=1-np.array(p);return dict(rows=len(rows),continue_rows=int(y.sum()),continue_AP=float(average_precision_score(y,q)) if y.sum() else None,continue_AUC=float(roc_auc_score(y,q)) if len(set(y))==2 else None)
def main():
 D.verify_trained();assert not (D.O/'primary_counterexample_comparison.json').exists();fit=D.rowfile(D.O/'fit_rows.json.gz');cal=D.rowfile(D.O/'calibration_rows.json.gz');val=D.rowfile(D.O/'validation_rows.json.gz');z=np.load(D.O/'train_scores.npz');pv=np.load(D.O/'validation_scores.npz')['scores'];D.save('stage192_ranking.json',[dict(partition=k,**metrics(rr,pp)) for k,rr,pp in [('fit',fit,z['fit']),('calibration',cal,z['calibration']),('validation',val,pv)]])
 specs=[('RF146_ENTRY128_CALIBRATED','2037-03-02_100|call009',128),('LR150_CALIBRATED','2037-08-03_100|call001',192),('RF158_192_CALIBRATED','2037-10-19_100|call005',192)];refs={r['call_id']:r for r in D.B.refs()};coverage=[r for r in D.read(D.R/'outputs/grid153_path_coverage/call_coverage.json') if r['repeat']==0];out=[]
 for method,cid,n in specs:
  ref=refs[cid];c=D.B.call_at(ref['source']);truth={x['pool_id']:x for x in c['outcomes']};initial=D.B.P.best(c['outcomes'][:128]);b=D.B.P.best(c['outcomes'][:n]);safe=initial is not None and initial['rho']<.9;extra=[]
  if b is not None and n<len(c['order']) and (not safe or D.B.P.bid(b)!=D.B.P.bid(initial)):
   station=D.B.subids()[b['pool_id']];ids={int(rep) for a,rep in c['aliases'].items() if D.B.subids()[int(a)]==station};extra=[i for i in c['order'][n:] if i in ids]
  picked=D.B.P.best(c['outcomes'][:n]+[truth[i] for i in extra]);timeline=[]
  for k in range(128,ref['global_n']+1,64):
   f=D.B.P.input_at(c,k)['features'];bb=D.B.P.best(c['outcomes'][:k]);timeline.append(dict(n=k,incumbent=D.B.P.bid(bb),station=D.B.subids()[bb['pool_id']] if bb is not None else None,rho=f['incumbent_rho'],last_batch_reward_gain=f['last_batch_reward_gain'],stale=f['stale_batches']))
  hits=[x['source'] for x in coverage if cid in x['teacher_exact_hits']];assert hits
  out.append(dict(method=method,call_id=cid,step=c['step'],stop_n=n,stop_choice=D.B.P.bid(picked),original_choice=ref['frame_choice'],original_global_n=ref['global_n'],original_local_queries=ref['local_queries'],stop_extra_ids=extra,actual_frame_matches=hits,original_choice_rank=c['order'].index(ref['frame_choice'])+1,timeline=timeline,future_audit_only=True,source_sha256=D.sha(D.R/ref['source'])))
 D.save('primary_counterexample_comparison.json',out);D.verify_trained();D.save('diagnosis_receipt.json',dict(passed=True,cases=3,actual_path_exact_matches=3,new_fits=0,new_predictions=0,future_fields_audit_only=True,no_blacklist_or_trigger_change=True));print(json.dumps(dict(ranking=D.read(D.O/'stage192_ranking.json'),cases=out),ensure_ascii=False))
if __name__=='__main__':main()
