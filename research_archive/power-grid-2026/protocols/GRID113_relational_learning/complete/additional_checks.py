"""Postfit public/target boundary and independent decision/accounting checks."""
import inspect
import numpy as np,torch
from common import *
from model import RelationalValue
def main():
    verify();configure();assert read(O/'review.json')['passed']
    rows=pickle.loads((O/'train_public.pkl').read_bytes())[:2];clean=collate(rows)
    poisoned=collate([{**r,'target':float('nan'),'future_reward':1e30,'full_week_cost':-1e30} for r in rows])
    assert all(torch.equal(v,poisoned[k]) for k,v in clean.items())
    for arm in ['JOINT','FACTOR']:
        torch.manual_seed(0);m=RelationalValue(arm)
        with torch.no_grad():assert torch.equal(m(**clean),m(**poisoned))
        try:m(**clean,target=torch.ones(2))
        except TypeError:pass
        else:raise AssertionError('Target accepted by forward')
    accounting=read(O/'postfit_accounting.json');D=read(O/'decision.json')
    for date,r in accounting['net_decomposition']['JOINT']['FACTOR']['per_date'].items():assert abs(r['net_delta']-D['date_deltas'][date])<1e-12
    by={r['method']:r for r in read(O/'validation_trajectories.json')}
    families=D['family_mean_utility'];anchors=all(families['JOINT']>x for x in [families['DIRECT'],families['DQN0_'],D['batch1_utility']])
    assert anchors==D['all_baseline_anchors_exceeded'];assert (D['mechanism_signal'] and anchors)==D['competitive_progression']
    save(O/'additional_review.json',dict(passed=True,executed_after_all_fits=True,
        poisoned_extra_labels_do_not_change_collated_inputs_or_forward=True,forward_rejects_target_argument=True,
        model_signature=str(inspect.signature(RelationalValue.forward)),independent_per_date_decomposition_and_competition_gate=True,
        optimizer_steps=0,new_physical_steps=0,training_sources_unchanged=True,source_sha256=sha(W/'additional_checks.py')))
    print('ADDITIONAL_READONLY_CHECKS_PASSED')
if __name__=='__main__':main()
