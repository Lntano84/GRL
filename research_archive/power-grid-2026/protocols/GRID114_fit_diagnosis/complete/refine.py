"""Descriptive error-bin and actual decision-regret accounting. No model inference or fit."""
from analyze import R,O,S,read,save,sha,np,old_predictions,risk,stats
def main():
    assert not (O/'decision_loss_crosscheck.json').exists()
    meta=read(S/'train_metadata.json');z=np.load(S/'train_targets.npz');y=z['target'].astype(float);p=z['sampling_probability'];Q=np.load(S/'state_predictions.npz');old=old_predictions(meta);M=read(O/'endpoint_metrics.json');results={}
    for name in list(old)+[f'{a}_seed{s}' for a in ['JOINT','FACTOR'] for s in range(3)]:
        q=old[name] if name in old else Q[name+'__train'];go=q[:,1]>q[:,0];positive=y>0;miss=positive&~go
        bins={}
        for label,mask in [('small_0_to_0.1',(y>0)&(y<=.1)),('middle_0.1_to_1',(y>.1)&(y<=1)),('large_over_1',y>1)]:
            bins[label]=dict(states=int(mask.sum()),misses=int((mask&~go).sum()),sampling_mass=float(p[mask].sum()),missed_expected_normalized_gain=float((p[mask&~go]*y[mask&~go]).sum()))
        results[name]=dict(positive_loss_fraction=M[name]['train']['positive_huber_contribution']/M[name]['train']['weighted_huber'],
            weighted_huber=M[name]['train']['weighted_huber'],weighted_one_step_sign_regret=float(p@(np.maximum(y,0)-go*y)),
            missed_positive_expected_net=float((p[miss]*y[miss]).sum()),wasted_negative_expected_net=float(-(p[(y<=0)&go]*y[(y<=0)&go]).sum()),
            missed_big_positive_rows=[dict(meta[i],target=float(y[i]),q_continue=float(q[i,1]),q_stop=float(q[i,0])) for i in np.flatnonzero(miss&(y>1))],bins=bins)
    consensus=read(O/'consensus_failures.json')
    save(O/'decision_loss_crosscheck.json',dict(results=results,consensus_target_summary={s:stats([r['target'] for r in c['rows']]) for s,c in consensus.items()},
        scope='All train states; descriptive fixed bins, not thresholds or new deployment policies. Weighted one-step regret differs from complete search trajectory utility.',
        source_sha256=sha(__import__('pathlib').Path(__file__)),new_fits=0))
    print(__import__('json').dumps({n:{k:results[n][k] for k in ['weighted_huber','weighted_one_step_sign_regret','positive_loss_fraction','bins']} for n in ['DIRECT0','DIRECT1','DIRECT2','JOINT_seed0','JOINT_seed1','JOINT_seed2']},indent=2))
if __name__=='__main__':main()
