"""Freeze diagnostic questions before computing feature associations."""
from common122 import *

def main():
    assert not O.exists();O.mkdir(parents=True);verify121();sealed();assert read(SOURCE/'review.json')['passed']
    plan=dict(id='GRID122_FALSE_STOP_VISIBLE_EVIDENCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User: 继续. Continue announced high-cost false-stop investigation; read-only, no new fits or data.',
        scope='All3582 OOF states and first actual common-prefix divergences of GRID121,3seeds; three previously identified worst calls for illustration, not selected test cases.',
        questions=['Are lost profitable continuations unsupported by similar training-date summaries?',
            'Do existing41 summaries and8 fixed public geometry/feedback descriptors differ from saved unprofitable continuations?',
            'Do zero/nonzero improvement probability and positive magnitude provide distinct potential supervision compared to net-sign classification?'],
        support=dict(representation='Original41, same fold-training-only standardization/clipping as models.',
            distance='Euclidean squared overall41. No chosen k, bandwidth, learned metric or threshold. Average all exact nearest ties.',
            references='Only12fitdates for each held4date fold. Training support baseline excludes entire same physical search call, not merely same prefix.',
            warning='Nearest positive/negative distances and nearest labels are descriptive training-label support, not proof of learnability or irreducibility; no deployed NN policy.',
            coverage='95th percentile of fit-only leave-call-out nearest distance under original restricted sampling probabilities, fixed before errors. This diagnostic threshold is not a new stop rule.'),
        groups=['utility0_huber1_nextpositive','utility0_huber1_nextnonpositive','utility1_huber0_nextpositive','utility1_huber0_nextnonpositive'],
        association='Describe ALL41 public summaries plus8 fixed descriptors. Call-balanced means/medians/ranges, all fields retained; no significance, feature selection for fitting or trigger optimization.',
        target_factorization='Describe original raw next gain/scale and known query charge separately, training partitions only; positive amount vs nonzero-event counts. No new labels from wholeweek results.',
        public_extra=EXTRA,
        guards=['No weight loading, forward prediction or training','No optimizer/backward/simulation/install','No final8test materialization','No change to models,data,fees,thresholds or folds','Do not infer wholeweek effects or generalization guarantees from nearest states or selected cases'])
    save(O/'plan.json',plan)
    code=list(W.glob('*.py'));save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code})
    files=[O/'plan.json']+[SOURCE/n for n in ['design.json','metadata.json','train_arrays.npz','train_cases.pkl','predictions.npz','oof_trajectories.json','decision.json','review.json','transfer_error_concentration.json','preflight.json']]
    files += [SOURCE/f'fold{f}_normalization.npz' for f in range(4)]
    files += [R/f'outputs/grid113_relational_learning/{n}' for n in ['train_public.pkl','train_metadata.json','train_targets.npz']]
    files += [R/'outputs/grid102_search_rl/design.json',R/'outputs/grid111_action_feedback/decision.json',R/'outputs/grid111_action_feedback/GRID111_report.md']
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'state.json',dict(stage='frozen_read_only_no_fits',new_fits=0));print('GRID122_PLAN_FROZEN',flush=True)
if __name__=='__main__':main()
