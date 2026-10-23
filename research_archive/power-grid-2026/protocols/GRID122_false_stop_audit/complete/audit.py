"""Independent direct-distance witnesses and all divergence/label accounting."""
from common122 import *

def independent_public(r):
    active=[i for i in range(17) if r['anchor_action'][34+i]>0 and r['anchor_action'][51+i]>0]
    pairs=[(i,j) for i in active for j in active if i<j]
    signature=lambda a:np.asarray([int(a[i]==a[j]) for i,j in pairs])
    seen=np.stack([signature(a) for a in r['context_action']]);pending=np.stack([signature(a) for a in r['next_action']]);anchor=signature(r['anchor_action'])
    distances=np.asarray([np.mean(p!=s) for p in pending for s in seen]).reshape(len(pending),len(seen));fb=r['context_feedback'];valid=fb[:,2]>0;vals=[float(row[0]) for row in fb if row[2]>0]
    # Preserve float32 mean/std semantics of the existing public feedback descriptors.
    vals=np.asarray(vals,np.float32)
    return np.asarray([len(seen),sum(int(row[3]>0) for row in fb),sum(valid),float(np.mean(vals)) if len(vals) else 0.,float(np.std(vals)) if len(vals) else 0.,
        np.mean([np.mean(p!=anchor) for p in pending]),float(np.mean(np.min(distances,axis=1))),float(np.mean(np.min(distances,axis=1)==0))])

def main():
    assert not (O/'review.json').exists();verify();configure();meta=read(SOURCE/'metadata.json');z=np.load(SOURCE/'train_arrays.npz');raw=z['raw'];y=z['target'];p=z['original_probability'];support=read(O/'support_rows.json');baselines=read(O/'fit_leave_call_out_neighbors.json');div=read(O/'first_divergences.json');public=pickle.loads((S/'train_public.pkl').read_bytes());extra=np.load(O/'visible_feature_descriptions.npz')['public8']
    D=read(SOURCE/'design.json');cases=pickle.loads((SOURCE/'train_cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    for i,m in enumerate(meta):
        c=cm[m['sample_id']];j=m['position'];assert np.array_equal(raw[i],c['states'][j]);assert np.float32((c['gains'][j]-D['cost_per_query']*c['queries'][j])/D['gain_scale'])==y[i]
    topgroups={r['physical_group'] for r in read(SOURCE/'transfer_error_concentration.json')['top_negative'][:3]}
    ids=set(range(0,len(y),151))|{len(y)-1}|{r['index'] for r in div if r['category']=='utility0_huber1_nextpositive'}|{i for i,m in enumerate(meta) if m['physical_state_group'] in topgroups}
    neighbors=0;leavecall=0
    for fold in range(4):
        held_dates=set(D['folds'][fold]['held_dates']);fit=np.asarray([i for i,m in enumerate(meta) if m['date'] not in held_dates]);held=[i for i,m in enumerate(meta) if m['date'] in held_dates];norm=np.load(SOURCE/f'fold{fold}_normalization.npz');x=np.clip((raw-norm['mean'])/norm['scale'],-20,20).astype('f').astype(float)
        bb=baselines[str(fold)];assert [r['index'] for r in bb]==fit.tolist()
        for r in bb:assert all(meta[j]['physical_state_group']!=meta[r['index']]['physical_state_group'] for j in r['nearest_indices'])
        order=np.argsort([r['distance_squared'] for r in bb],kind='stable');weights=p[fit]/p[fit].sum();cum=np.cumsum(weights[order]);q95=bb[int(order[np.searchsorted(cum,.95*cum[-1],side='left')])]['distance_squared']
        assert q95==read(O/'fold_coverage.json')[str(fold)]['fit_lco_q95_squared']
        for i in [j for j in held if j in ids]:
            ss=support[i];assert ss['fold']==fold and ss['fit_lco_q95_squared']==q95
            for kind,ref,record_key,distance_key in [('all',fit,'nearest_indices','distance_squared'),('positive',fit[y[fit]>0],'positive_nearest_indices','positive_distance_squared'),('negative',fit[y[fit]<=0],'negative_nearest_indices','negative_distance_squared')]:
                distances=np.sum((x[ref]-x[i])**2,axis=1);minimum=float(distances.min());nearest=ref[distances==minimum].tolist()
                assert nearest==ss[record_key] and minimum==ss[distance_key],(fold,i,kind,minimum,ss[distance_key]);neighbors+=1
            nn=ss['nearest_indices'];assert ss['nearest_label_positive_fraction']==float(np.mean(y[nn]>0)) and ss['nearest_mean_net']==float(y[nn].mean())
            assert all(meta[j]['date'] not in held_dates for j in nn)
            assert np.allclose(independent_public(public[i]),extra[i],rtol=1e-7,atol=1e-7),i
        for r in bb[::151]+bb[-1:]:
            i=r['index'];refs=np.asarray([j for j in fit if meta[j]['physical_state_group']!=meta[i]['physical_state_group']]);dist=np.sum((x[refs]-x[i])**2,axis=1);assert float(dist.min())==r['distance_squared'] and refs[dist==dist.min()].tolist()==r['nearest_indices'];leavecall+=1
    original=read(SOURCE/'oof_trajectories.json');tr={(r['method'],r['sample_id']):r for r in original};bydiv={(r['seed'],r['sample_id']):r for r in div};counts=collections.Counter();datecount=collections.Counter(c['date'] for c in cases);conserved=0.;Q=np.load(SOURCE/'predictions.npz')
    for seed in range(3):
        for c in cases:
            a=tr[f'UTILITY_BCE_seed{seed}',c['sample_id']];b=tr[f'HUBER_seed{seed}',c['sample_id']];first=next((j for j,(aa,bb) in enumerate(zip(a['actions'],b['actions'])) if aa!=bb),None)
            if first is None:assert (seed,c['sample_id']) not in bydiv;continue
            r=bydiv[seed,c['sample_id']];i=lookup[c['sample_id'],first];assert r['index']==i and r['position']==first and r['target_net']==float(y[i])
            expected=f'utility{a["actions"][first]}_huber{b["actions"][first]}_next'+('positive' if y[i]>0 else 'nonpositive');assert expected==r['category'];counts[expected]+=1
            qa=Q[a['model_name']][i];qb=Q[b['model_name']][i];assert r['utility_margin']==float(qa[1]-qa[0]) and r['huber_margin']==float(qb[1]-qb[0])
            diff=a['net_utility']-b['net_utility'];assert diff==r['sequence_net_difference'];weight=1/3/16/datecount[c['date']];assert abs(weight*diff-r['macro_contribution'])<1e-15;conserved+=weight*diff
    assert abs(conserved-read(SOURCE/'transfer_error_concentration.json')['total_difference'])<1e-12
    sums=read(O/'group_summaries.json')
    for key,co in counts.items():
        rr=[r for r in div if r['category']==key];gg=collections.defaultdict(list)
        for r in rr:gg[r['physical_state_group']].append(r['index'])
        assert sums[key]['records']==co and sums[key]['distinct_calls']==len(gg)
        vectors=np.asarray([np.mean(np.column_stack([raw[ii],extra[ii]]),axis=0) for ii in gg.values()])
        for j,name in enumerate(FEATURES+EXTRA):assert abs(vectors[:,j].mean()-sums[key]['features'][name]['mean'])<1e-10
    factor=read(O/'label_factorization.json');gain=np.array([cm[m['sample_id']]['gains'][m['position']]/D['gain_scale'] for m in meta]);event=gain>0
    assert int(event.sum())==factor['positive_raw_gain_states'];assert abs(float(p@gain)-factor['train_probability_raw_improvement']*factor['conditional_mean_positive_gain'])<1e-12
    verify();sealed();save(O/'review.json',dict(passed=True,all_original_targets_checked=len(y),direct_distance_checks=neighbors,leave_call_out_distance_checks=leavecall,
        independent_public_descriptor_states=len(ids),all_first_divergences_checked=len(div),macro_difference_conserved=True,
        group_association_means_recomputed=True,train_date_references_only=True,training_constant_and_label_identity_recomputed=True,
        all_existing_hashes_unchanged=True,new_fits=0,new_model_forwards=0,optimizer_instances=0,new_physical_calls=0,final_test_materialized=False))
    print('GRID122_READONLY_AUDIT_PASSED',neighbors,len(div),len(ids),flush=True)
if __name__=='__main__':main()
