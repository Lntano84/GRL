"""Fixed geometry/feedback arithmetic; no fitted parameters or future rewards in input."""
import numpy as np
from public import legal,signature
KEYS={'seen_ids','seen_plans','seen_feedback','next_ids','next_plans','incumbent_id','incumbent_reward','incumbent_plan'}
def describe(c):
    assert set(c)==KEYS and not set(c['seen_ids'])&set(c['next_ids'])
    use=[i for i,r in enumerate(c['seen_feedback']) if legal(r)];assert use
    S=np.stack([signature(c['seen_plans'][i]) for i in use]);T=np.stack([signature(p) for p in c['next_plans']]);inc=signature(c['incumbent_plan'])
    y=np.array([float(np.float32(c['seen_feedback'][i]['reward'])) for i in use]);n=len(y);assert y.max()<=c['incumbent_reward']+1e-12
    dist=T.sum(axis=1)[:,None]+S.sum(axis=1)[None,:]-2*(T@S.T);assert np.min(dist)>=0
    tied=dist==dist.min(axis=1)[:,None];count=tied.sum(axis=1)
    interpolation=(tied@y)/count;mean=np.mean(y);variance=np.mean((y-mean)**2)
    shuffle_variance=variance*(n-count)/(count*(n-1)) if n>1 else np.zeros(len(T))
    # Observed per-device-pair marginal contrasts, not fitted effects or causal estimates.
    n1=S.sum(axis=0);n0=n-n1;covered=(n1>0)&(n0>0)
    mu1=np.divide(S.T@y,n1,out=np.zeros(S.shape[1]),where=n1>0);mu0=np.divide((1-S).T@y,n0,out=np.zeros(S.shape[1]),where=n0>0)
    contrast=np.where(covered,mu1-mu0,0);change=T-inc;changed=change!=0;knownchange=changed&covered
    support=knownchange.sum(axis=1)
    direction=np.divide(change@contrast,support,out=np.zeros(len(T)),where=support>0)
    unknown=np.mean(np.where(T==1,n1==0,n0==0),axis=1)
    reversed_interpolation=(tied@y[::-1])/count
    return dict(legal_seen_indices=use,seen_rewards=y.tolist(),seen_bits=S.tolist(),next_bits=T.tolist(),incumbent_bits=inc.tolist(),
        closest_seen_indices=[np.flatnonzero(a).tolist() for a in tied],tie_counts=count.tolist(),nearest_distance=(dist.min(axis=1)/S.shape[1]).tolist(),
        interpolated_reward=interpolation.tolist(),unpaired_mean=float(mean),exact_permutation_prediction_variance=np.asarray(shuffle_variance).tolist(),
        pair_observed_zero=n0.tolist(),pair_observed_one=n1.tolist(),pair_reward_contrast=contrast.tolist(),
        pair_both_values_observed=covered.tolist(),next_contrast_direction=direction.tolist(),next_changed_pair_count=changed.sum(axis=1).tolist(),next_supported_changed_pair_count=support.tolist(),
        next_unseen_relation_fraction=unknown.tolist(),reversed_reward_assignment_changes_prediction=bool(np.any(np.abs(reversed_interpolation-interpolation)>1e-12)),
        interpolation_upper_bound=float(np.max(interpolation)),seen_max=float(y.max()))

def evaluate(c,d,future,fee,gain_scale):
    # Future outcomes enter only this diagnostic target comparison, never describe().
    assert len(future)==len(c['next_ids']);valid=np.array([legal(r) for r in future]);actual=np.array([float(np.float32(r['reward'])) for r in future])
    pred=np.array(d['interpolated_reward']);mean=d['unpaired_mean'];variance=np.array(d['exact_permutation_prediction_variance'])
    candidates=[c['incumbent_reward']]+[float(np.float32(r['reward'])) for r in future if r['strict_admissible']]
    gain=max(candidates)-c['incumbent_reward'];net=gain-fee*len(future)
    good=[i for i,r in enumerate(future) if r['strict_admissible']]
    winner=max(good,key=lambda i:(actual[i],-c['next_ids'][i])) if good else None
    return dict(legal_next=int(valid.sum()),paid_batch_size=len(future),legal_seen=len(d['seen_rewards']),next_gain=gain,target_net=net,normalized_net=net/gain_scale,
        nearest_mse=float(np.mean((pred[valid]-actual[valid])**2)) if valid.any() else None,
        mean_mse=float(np.mean((mean-actual[valid])**2)) if valid.any() else None,
        exact_unpaired_mse=float(np.mean((mean-actual[valid])**2+variance[valid])) if valid.any() else None,
        actual_positive=bool(net>0),positive_interpolation_gain_possible=bool(d['interpolation_upper_bound']>c['incumbent_reward']+1e-12),
        contrast_coverage=float(np.mean(d['pair_both_values_observed'])),mean_unseen_relation_fraction=float(np.mean(d['next_unseen_relation_fraction'])),
        best_next_contrast_direction=None if winner is None else d['next_contrast_direction'][winner],
        best_next_changed_pairs=None if winner is None else d['next_changed_pair_count'][winner],
        best_next_supported_changed_pairs=None if winner is None else d['next_supported_changed_pair_count'][winner],
        best_next_id_for_posthoc_audit_only=None if winner is None else c['next_ids'][winner],
        reversed_reward_assignment_changes_prediction=d['reversed_reward_assignment_changes_prediction'])
