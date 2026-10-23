"""Conserved OOF divergence/support accounting, no fitted classifier or new rule."""
from common122 import *

def group_summary(rows,raw,extra,support,meta):
    if not rows:return dict(records=0,distinct_calls=0)
    by=collections.defaultdict(list)
    for r in rows:by[r['physical_state_group']].append(r)
    call_vectors=[];near=[];ys=[]
    for group,rr in sorted(by.items()):
        ii=[r['index'] for r in rr]
        call_vectors.append(np.column_stack([raw[ii],extra[ii]]).mean(0))
        near.append([np.mean([support[i]['nearest_label_positive_fraction'] for i in ii]),
            np.mean([support[i]['nearest_mean_net'] for i in ii]),np.mean([support[i]['distance_squared']>support[i]['fit_lco_q95_squared'] for i in ii]),
            np.mean([support[i]['positive_distance_squared']>support[i]['negative_distance_squared'] for i in ii]),np.mean([support[i]['distance_squared'] for i in ii])])
        ys.append(np.mean([r['target_net'] for r in rr]))
    vec=np.asarray(call_vectors);near=np.asarray(near)
    return dict(records=len(rows),unique_states=len({r['index'] for r in rows}),distinct_calls=len(by),
        calls=sorted(by),call_balanced_nearest_positive_fraction=float(near[:,0].mean()),
        call_balanced_nearest_mean_net=float(near[:,1].mean()),call_balanced_beyond_fit_lco_q95=float(near[:,2].mean()),
        call_balanced_positive_farther_than_negative=float(near[:,3].mean()),call_balanced_nearest_distance_squared=float(near[:,4].mean()),
        call_balanced_target_net=float(np.mean(ys)),
        features={name:dict(mean=float(vec[:,j].mean()),median=float(np.median(vec[:,j])),minimum=float(vec[:,j].min()),maximum=float(vec[:,j].max()),std=float(vec[:,j].std())) for j,name in enumerate(FEATURES+EXTRA)})

def main():
    assert not (O/'analysis_receipt.json').exists();verify();configure()
    source_design=read(SOURCE/'design.json');meta=read(SOURCE/'metadata.json');z=np.load(SOURCE/'train_arrays.npz');raw=z['raw'];target=z['target'];prob=z['original_probability'];cases=pickle.loads((SOURCE/'train_cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)}
    public=pickle.loads((S/'train_public.pkl').read_bytes());assert len(public)==len(meta)==3582
    other=read(S/'train_metadata.json');assert [(x['sample_id'],x['position']) for x in meta]==[(x['sample_id'],x['position']) for x in other]
    extras=np.stack([public_descriptors(r) for r in public]);np.savez_compressed(O/'visible_feature_descriptions.npz',raw41=raw,public8=extras)
    groups=[m['physical_state_group'] for m in meta];support={};fit_baselines={};folddetails={}
    for fold in range(4):
        held_dates=set(source_design['folds'][fold]['held_dates']);hold=np.array([m['date'] in held_dates for m in meta]);fit=np.flatnonzero(~hold);held=np.flatnonzero(hold)
        norm=np.load(SOURCE/f'fold{fold}_normalization.npz');x=np.clip((raw-norm['mean'])/norm['scale'],-20,20).astype('f').astype(float)
        baseline=nearest_batch(x[fit],x[fit],fit,fit,groups,True)
        q95=weighted_quantile([r['distance_squared'] for r in baseline],prob[fit]/prob[fit].sum(),.95)
        fit_baselines[str(fold)]=baseline
        nearest=nearest_batch(x[held],x[fit],held,fit)
        positive=fit[target[fit]>0];negative=fit[target[fit]<=0];assert len(positive)>0 and len(negative)>0
        nearpos=nearest_batch(x[held],x[positive],held,positive);nearneg=nearest_batch(x[held],x[negative],held,negative)
        rawmap=collections.defaultdict(list);normmap=collections.defaultdict(list)
        for i in fit:rawmap[raw[i].tobytes()].append(int(i));normmap[x[i].astype('f').tobytes()].append(int(i))
        for r,rp,rn in zip(nearest,nearpos,nearneg):
            i=r['index'];ids=r['nearest_indices'];exactraw=rawmap.get(raw[i].tobytes(),[]);exactnorm=normmap.get(x[i].astype('f').tobytes(),[])
            support[i]=dict(**r,fold=fold,nearest_label_positive_fraction=float(np.mean(target[ids]>0)),nearest_mean_net=float(target[ids].mean()),
                positive_distance_squared=rp['distance_squared'],negative_distance_squared=rn['distance_squared'],
                positive_nearest_indices=rp['nearest_indices'],negative_nearest_indices=rn['nearest_indices'],fit_lco_q95_squared=q95,
                exact_raw_match_indices=exactraw,exact_normalized_match_indices=exactnorm)
        folddetails[str(fold)]=dict(fit_states=len(fit),held_states=len(held),fit_lco_q95_squared=q95,
            held_beyond_q95=int(sum(support[int(i)]['distance_squared']>q95 for i in held)),
            held_exact_raw_matches=sum(bool(support[int(i)]['exact_raw_match_indices']) for i in held),
            held_exact_normalized_matches=sum(bool(support[int(i)]['exact_normalized_match_indices']) for i in held))
    assert len(support)==3582
    Q=np.load(SOURCE/'predictions.npz');tr=read(SOURCE/'oof_trajectories.json');by={(r['method'],r['sample_id']):r for r in tr};div=[];allstates=collections.Counter()
    dates=sorted({m['date'] for m in meta});datecount=collections.Counter(c['date'] for c in cases)
    gains=np.array([cm[m['sample_id']]['gains'][m['position']]/source_design['gain_scale'] for m in meta]);costs=np.array([source_design['cost_per_query']*cm[m['sample_id']]['queries'][m['position']]/source_design['gain_scale'] for m in meta])
    assert np.array_equal((gains-costs).astype('f'),target)
    for seed in range(3):
        for c in cases:
            a=by[f'UTILITY_BCE_seed{seed}',c['sample_id']];b=by[f'HUBER_seed{seed}',c['sample_id']];fold=a['fold'];qa=Q[a['model_name']];qb=Q[b['model_name']]
            for j,(aa,bb) in enumerate(zip(a['actions'],b['actions'])):
                if aa==bb:continue
                i=lookup[c['sample_id'],j];key=f'utility{aa}_huber{bb}_next'+('positive' if target[i]>0 else 'nonpositive')
                div.append(dict(index=i,sample_id=c['sample_id'],position=j,date=c['date'],fold=fold,seed=seed,physical_state_group=c['physical_state_group'],category=key,
                    target_net=float(target[i]),next_gain_normalized=float(gains[i]),known_query_charge=float(costs[i]),
                    utility_margin=float(qa[i,1]-qa[i,0]),huber_margin=float(qb[i,1]-qb[i,0]),
                    sequence_net_difference=a['net_utility']-b['net_utility'],macro_contribution=(a['net_utility']-b['net_utility'])/3/16/datecount[c['date']]))
                break
        for i,m in enumerate(meta):
            fold=support[i]['fold'];qa=Q[f'fold{fold}__UTILITY_BCE_seed{seed}'][i];qb=Q[f'fold{fold}__HUBER_seed{seed}'][i]
            allstates[f'utility{int(qa[1]>qa[0])}_huber{int(qb[1]>qb[0])}_next'+('positive' if target[i]>0 else 'nonpositive')]+=1
    categories=read(O/'plan.json')['groups'];summaries={key:group_summary([r for r in div if r['category']==key],raw,extras,support,meta) for key in categories}
    bad=summaries['utility0_huber1_nextpositive'];saved=summaries['utility0_huber1_nextnonpositive'];associations=[]
    for name in FEATURES+EXTRA:
        aa=bad['features'][name];bb=saved['features'][name];den=(.5*(aa['std']**2+bb['std']**2))**.5
        associations.append(dict(feature=name,bad_mean=aa['mean'],saved_mean=bb['mean'],difference=aa['mean']-bb['mean'],standardized_difference=(aa['mean']-bb['mean'])/den if den>0 else None,
            observed_call_mean_range_overlap=not(aa['minimum']>bb['maximum'] or bb['minimum']>aa['maximum'])))
    # Original fixed task factorization, no fit or relabeling.
    positive_gain=gains>0;rate=float(prob@positive_gain);amount=float((prob[positive_gain]@gains[positive_gain])/prob[positive_gain].sum());assert abs(rate*amount-float(prob@gains))<1e-12
    factor=dict(states=len(target),positive_raw_gain_states=int(positive_gain.sum()),positive_net_states=int((target>0).sum()),
        positive_raw_gain_calls=len({groups[i] for i in np.flatnonzero(positive_gain)}),positive_net_calls=len({groups[i] for i in np.flatnonzero(target>0)}),
        train_probability_raw_improvement=rate,conditional_mean_positive_gain=amount,expected_normalized_gain=float(prob@gains),expected_normalized_charge=float(prob@costs),
        expected_normalized_net=float(prob@(gains-costs)),identity_gain_probability_times_amount=True,
        scope='Label arithmetic only; not evidence a new two-head model predicts either factor well.')
    top=read(SOURCE/'transfer_error_concentration.json')['top_negative'][:3];examples=[]
    for t in top:
        ii=[i for i,m in enumerate(meta) if m['physical_state_group']==t['physical_group']];rr=[r for r in div if r['physical_state_group']==t['physical_group']]
        examples.append(dict(physical_state_group=t['physical_group'],date=t['date'],prior_negative_contribution=t['net_contribution'],
            first_divergence_records=len(rr),profitable_missed_divergences=sum(r['category']=='utility0_huber1_nextpositive' for r in rr),
            group_public_observation_ranges={name:dict(minimum=float(raw[ii,j].min()),maximum=float(raw[ii,j].max())) for j,name in enumerate(FEATURES) if name in ['incumbent_reward','incumbent_rho','last_batch_reward_gain','station_admissible_fraction','local_batches_done','log_next_batch_queries']},
            divergence_rows=rr))
    old=read(SOURCE/'transfer_error_concentration.json');assert abs(sum(r['macro_contribution'] for r in div)-old['total_difference'])<1e-12
    for key,v in old['first_common_prefix_divergences'].items():assert summaries[key]['records']==v['prefix_seed_pairs'] and summaries[key]['distinct_calls']==v['distinct_physical_calls']
    save(O/'support_rows.json',[support[i] for i in range(3582)]);save(O/'fit_leave_call_out_neighbors.json',fit_baselines);save(O/'fold_coverage.json',folddetails)
    save(O/'first_divergences.json',div);save(O/'group_summaries.json',summaries);save(O/'all_state_decision_counts.json',dict(allstates));save(O/'all_feature_associations.json',associations)
    save(O/'label_factorization.json',factor);save(O/'top_call_details.json',examples)
    save(O/'analysis_receipt.json',dict(passed=True,held_states=3582,first_divergences=len(div),all_source_predictions_reused=True,conserved_macro_difference=sum(r['macro_contribution'] for r in div),new_fits=0,new_model_forwards=0,new_physical_calls=0,final_test_materialized=False))
    print(json.dumps(dict(groups={k:{kk:vv for kk,vv in v.items() if kk not in ['features','calls']} for k,v in summaries.items()},factorization=factor,folds=folddetails),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
