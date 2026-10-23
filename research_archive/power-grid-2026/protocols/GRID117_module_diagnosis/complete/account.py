"""Fine-grained cost localization and additive-identifiability proof, saved arrays only."""
from analyze import *
def main():
    assert not (O/'accounting_summary.json').exists();assert read(O/'analysis_receipt.json')['passed'];verify116()
    div=read(O/'first_divergences.json');calls=read(O/'physical_call_contributions.json');a=np.load(O/'component_forward_outputs.npz');D=read(S116/'decision.json');old=read(S116/'payoff_accounting.json')
    rows=[];gauge=[];variant_at_div={}
    for seed in range(3):
        rr=[r for r in div if r['seed']==seed];bad=[r for r in rr if r['joint_continues'] and r['next_batch_net']<=0]
        name=f'BASE_JOINT_seed{seed}';meta=read(S113/'validation_metadata.json');lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)}
        margins=lambda k:a[name+'__validation__'+k][:,1]-a[name+'__validation__'+k][:,0]
        jm=margins('actual');bm=margins('base');dm=margins('delta_actual');idx=[lookup[r['sample_id'],r['position']] for r in bad]
        variant_at_div[str(seed)]={}
        for v in ['factorized_same_weights','uniform_attention','both']:
            vm=margins(v);variant_at_div[str(seed)][v]=dict(unprofitable_joint_extra_prefixes=len(idx),those_changed_to_stop=int((vm[idx]<=0).sum()),
                abs_margin_change_at_those_prefixes=stat(abs(vm[idx]-jm[idx])))
        # q_base+c and delta-c produce exactly the same total; only bookkeeping signs can change.
        for offset in [.25,-.25]:
            base=a[name+'__validation__base'].astype(float);delta=a[name+'__validation__delta_actual'].astype(float);c=np.array([0.,offset])
            transformed=(base+c)+(delta-c);original=base+delta;error=float(np.max(abs(transformed-original)));assert error<1e-12
            assert np.array_equal(transformed[:,1]>transformed[:,0],original[:,1]>original[:,0])
            gauge.append(dict(seed=seed,constant_continue_offset=offset,max_total_score_error=error,total_decisions_unchanged=True,
                bad_extra_prefixes=len(idx),stored_joint_delta_required_count=int((bm[idx]<=0).sum()),
                algebraically_equivalent_delta_required_count=int(((bm[idx].astype(float)+offset)<=0).sum())))
        for r in rr:
            row=dict(r,weighted_gain_difference=None)
            rows.append(row)
    top=[]
    for c in calls[:5]:
        rs=[r for r in div if r['physical_state_group']==c['physical_state_group']]
        top.append(dict(c,steps=sorted({r['step'] for r in rs}),prefix_divergences=len(rs)))
    seed1=D['paired_seeds'][1]['joint_minus_factor']/3;total=D['family_mean_utility']['JOINT']-D['family_mean_utility']['FACTOR']
    summary=dict(seed1_weighted_family_net_contribution=seed1,total_family_delta=total,seed1_share_of_net_gap=seed1/total,
        contribution_sum=float(sum(c['net_contribution'] for c in calls)),negative_calls=sum(c['net_contribution']<0 for c in calls),positive_calls=sum(c['net_contribution']>0 for c in calls),
        most_negative_calls=top,variants_on_first_unprofitable_joint_extras=variant_at_div,
        caveat='Full per-call list retained, prefixes correlated. Gauge transformations are algebraic identities, not parameter updates or new policies. Per-branch attribution is not identified by summed-output loss.')
    assert abs(summary['contribution_sum']-total)<1e-12
    save(O/'accounting_summary.json',summary);save(O/'additive_identifiability_check.json',dict(passed=True,records=gauge,
        proof='For any constant2-vector c, q_total=(q_base+c)+(delta-c). Both networks have final biases; original loss sees only total. No separate loss fixes component meaning.',
        no_model_mutation_or_deployment=True,new_fits=0,source_sha256=sha(W/'account.py')))
    print(json.dumps(summary,indent=2),flush=True);verify116()
if __name__=='__main__':main()
