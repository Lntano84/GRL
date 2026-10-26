"""Only repair boolean unary-minus in constant-prior BCE; original frozen source intact."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
import numpy as np
def fixed_stats(y,l,w,prior):
    p=np.exp(-np.logaddexp(0.,-l));go=l>0;tp=float(w[y&go].sum());pred=float(w[go].sum());positive=float(w[y].sum())
    return dict(rows=len(y),weighted_prevalence=positive,weighted_append_rate=pred,weighted_precision=tp/pred if pred else None,weighted_recall=tp/positive if positive else None,weighted_AUC=C.auc(y,l,w),weighted_BCE=float(w@(np.logaddexp(0.,l)-y*l)),weighted_Brier=float(w@((p-y)**2)),fit_prior=prior,constant_fitprior_BCE=float(w@(-np.log(prior)*y-np.log(1-prior)*(~y))),constant_fitprior_Brier=float(w@((prior-y)**2)),count_append=int(go.sum()),count_hit=int((go&y).sum()))
if __name__=='__main__':
    C.verify();assert not (C.O/'summary.json').exists()
    C.save('engineering_recovery01.json',dict(error='TypeError: numpy boolean negative in constant-fit-prior BCE',original_frozen_source_unchanged=True,repair='Same Bernoulli BCE formula with float log coefficient multiplied by boolean before subtraction. No scoring,labels,weights,selection,threshold or model changes.',wrapper_sha256=C.sha(Path(__file__)),first_prepare_permission_error='Default sandbox WinError5 creating D output, no output directory created. Same original prepare succeeded with escalated D permission.',completed_analysis_before_failure=False,new_fits=0,new_NET_forwards=0))
    C.stats=fixed_stats;C.analyze()
