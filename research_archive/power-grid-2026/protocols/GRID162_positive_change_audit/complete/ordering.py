"""Post-hoc score ordering explanation; no new decision rule or model score."""
import numpy as np
import run as D
def main():
 D.verify();assert not (D.O/'ordering_explanation.json').exists();cross=D.read(D.O/'fixed_score_cutoff_crossing.json');d=D.read(D.B.O/'design.json');oz=np.load(D.B.O/'train_scores.npz');nr=D.M.rowfile(D.M.O/'calibration_rows.json.gz');op=D.M.rowfile(D.B.O/'calibration_rows.json.gz');np_=np.load(D.M.O/'train_scores.npz')['calibration'];old=D.M.scoremap(op,oz['calibration']);new=D.M.scoremap(nr,np_);cut=D.read(D.M.O/'cutoff.json');tau=cut['primary'];rescued=[]
 for ref in D.B.refs('train',dates=d['calibration_dates']):
  a=D.M.shadow(ref,tau,old);b=D.M.shadow(ref,tau,new)
  if a['bad_learned_stop'] and not b['bad_learned_stop']:
   row=next(x for x in nr if x['meta']['call_id']==ref['call_id']);rescued.append(dict(call_id=ref['call_id'],date=ref['date'],old_score=old[(ref['call_id'],192)],new_score=new[(ref['call_id'],192)],old_queries=a['queries'],new_queries=b['queries'],old_choice=a['selected'],original_choice=ref['frame_choice'],new_choice=b['selected'],features=row['features']))
 wins=[x for x in D.read(D.O/'all62_paired_calls.json') if x['category']=='ADDED_CORRECT_SAVING'];pairs=[]
 for p in wins:
  for n in rescued:pairs.append(dict(good_call=p['call_id'],calibration_bad_call=n['call_id'],old_correct_order=p['old_score192']>n['old_score'],new_correct_order=p['new_score192']>n['new_score']))
 assert len(rescued)==4 and len(pairs)==16;D.save('calibration_wrong_exits_suppressed.json',rescued);D.save('selected_pair_ordering.json',pairs);D.save('ordering_explanation.json',dict(post_hoc_selected_case_explanation=True,not_independent_pairs=True,not_new_validation=True,calibration_wrong_exits_suppressed=len(rescued),calibration_dates=sorted({x['date'] for x in rescued}),selected_good_bad_pairs=len(pairs),old_correct_order=sum(x['old_correct_order'] for x in pairs),new_correct_order=sum(x['new_correct_order'] for x in pairs),new_fits=0,new_predictions=0,new_thresholds_selected=0));D.verify();print(D.read(D.O/'ordering_explanation.json'))
if __name__=='__main__':main()
