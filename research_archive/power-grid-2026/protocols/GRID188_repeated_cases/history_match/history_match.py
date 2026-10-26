"""Match current saved states to old auxiliary-head OOF records, no forward."""
from pathlib import Path
import collections,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def main():
    C.verify();assert not (O/'history_match.json').exists();old=R/'outputs/grid123_event_auxiliary'
    sources=[old/n for n in ['design.json','predictions.npz','train_metadata.json','decision.json','complete.json','review.json']]+[R/'outputs/grid124_shared_gradient_audit/GRID124_report.md',R/'outputs/grid126_known_cost_gain/decision.json',R/'outputs/grid127_mean_gain_objective/decision.json']
    assert read(old/'complete.json')['passed_audit'] if 'passed_audit' in read(old/'complete.json') else read(old/'complete.json').get('passed',False)
    C.save('history_match_plan.json',dict(posthoc=True,source_code_sha256=C.sha(Path(__file__)),sources={p.relative_to(R).as_posix():C.sha(p) for p in sources},purpose='Check previously tried auxiliary/value objectives before proposing duplicate training. Same original state/date/seed keys; old6978summarymodel differs from current12980FACTOR and next-batch event differs from final-actionchange,so recurrence is not an auxiliary ablation.',no_new_model_forwards=True))
    meta=read(old/'train_metadata.json');lookup={(v['sample_id'],v['position']):i for i,v in enumerate(meta)};q=np.load(old/'predictions.npz');design=read(old/'design.json');events=read(O/'events_all_seeds.json');extra=read(O/'top3_loss_details.json')['all_action_changing_pairs'];allrows=[]
    for v in events:allrows.append(dict(sample_id=v['sample_id'],seed=v['seed'],position=v['position'],date=v['date'],group=v['physical_state_group'],category=v['category'],next_NET=v['outcome_only']['next_NET'],origin='all3seed_focus'))
    for v in extra:allrows.append(dict(sample_id=v['sample_id'],seed=v['seed'],position=v['position'],date=v['date'],group=v['physical_state_group'],category=v['category'],next_NET=v['next_NET'],origin='top3loss'))
    for v in allrows:
        f=next(i for i,x in enumerate(design['folds']) if v['date'] in x['held_dates']);idx=lookup[v['sample_id'],v['position']];s=v['seed'];na=q[f'f{f}__NET_ZERO_seed{s}__train'][idx];aa=q[f'f{f}__AUX_GAIN_seed{s}__train'][idx];ns=q[f'f{f}__AUX_NETSIGN_seed{s}__train'][idx]
        v.update(old_fold=f,old_NET_continue=bool(na[1]>0),old_AUX_GAIN_continue=bool(aa[1]>0),old_AUX_GAIN_event_positive=bool(aa[0]>0),old_AUX_GAIN_event_logit=float(aa[0]),old_AUX_GAIN_value=float(aa[1]),old_AUX_NETSIGN_continue=bool(ns[1]>0))
    summaries={}
    for key,rr in [('focus_restore',[v for v in allrows if v['origin']=='all3seed_focus' and v['category']=='RESTORED']),('focus_lost_profitable',[v for v in allrows if v['origin']=='all3seed_focus' and v['category']=='LOST' and v['next_NET']>0]),('top3loss', [v for v in allrows if v['origin']=='top3loss'])]:
        summaries[key]=dict(records=len(rr),groups=len({v['group'] for v in rr}),old_NET_continue=sum(v['old_NET_continue'] for v in rr),old_AUX_GAIN_continue=sum(v['old_AUX_GAIN_continue'] for v in rr),old_AUX_GAIN_event_positive=sum(v['old_AUX_GAIN_event_positive'] for v in rr),event_positive_but_value_STOP=sum(v['old_AUX_GAIN_event_positive'] and not v['old_AUX_GAIN_continue'] for v in rr),old_AUX_NETSIGN_continue=sum(v['old_AUX_NETSIGN_continue'] for v in rr))
    C.save('history_match.json',dict(records=allrows,summaries=summaries,not_pooled_with_current_scores=True));C.verify();C.save('history_match_review.json',dict(passed=True,matched_saved_records=len(allrows),same_sample_position_seed_and_held_date=True,new_fits=0,new_model_forwards=0,new_physics=0));C.archive('history_match');print('GRID188_HISTORY_MATCH',C.json.dumps(summaries))
if __name__=='__main__':main()
