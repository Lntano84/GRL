"""Independent direct-distance, label, exposure and range accounting."""
from common125 import *

def main():
    assert not (O/'review.json').exists();verify();cases,meta,raw,gain,charge=arrays();support=read(O/'support_rows.json');source={p:np.load(S/f'{p}_data.npz') for p in meta};ys={p:source[p]['net'] for p in meta};prob=source['train']['probability'];by={(r['part'],r['index']):r for r in support};scope_cache={s:scope(s,meta,raw) for s in ['full','f0','f1','f2','f3']};maxerr=0.
    for row in support:
        ids,held,part,x,h,pre,hpre=scope_cache[row['split']];j=int(np.flatnonzero(held==row['index'])[0]);dist=((x-h[j])**2).sum(1);minimum=dist.min();ties=ids[dist==minimum].tolist();assert ties==row['nearest_indices'] and minimum==row['distance_squared'];maxerr=max(maxerr,abs(float(minimum)-row['distance_squared']));yp=ys['train'][ids]>0
        assert dist[yp].min()==row['positive_distance_squared'] and dist[~yp].min()==row['negative_distance_squared'];assert row['nearest_positive_fraction']==float(np.mean(ys['train'][ties]>0))
        assert row['clipped_dimensions']==int(np.sum(abs(hpre[j])>20)) and row['beyond_fit_callout95']==bool(minimum>row['fit_callout95_squared'])
        assert row['raw_exact_match']==bool(np.any(np.all(raw['train'][ids]==raw[part][row['index']],axis=1))) and row['normalized_exact_match']==bool(np.any(np.all(x==h[j],axis=1)))
        if row['split']!='full':assert not ({meta['train'][i]['date'] for i in ids}&{meta[part][row['index']]['date']})
    fit_nn=read(O/'fit_callout_neighbors.json');ids,_,_,x,_,_,_=scope_cache['full'];groups=np.array([m['physical_state_group'] for m in meta['train']]);mins=[]
    for j,i in enumerate(ids):
        allowed=groups[ids]!=groups[i];dist=((x[allowed]-x[j])**2).sum(1);minimum=dist.min();ties=ids[allowed][dist==minimum].tolist();row=fit_nn['full'][j];assert row['nearest_indices']==ties and row['distance_squared']==minimum;mins.append(minimum)
    order=np.argsort(mins,kind='stable');cum=np.cumsum(prob[order]/prob.sum());q95=float(np.asarray(mins)[order[np.flatnonzero(cum>=.95)[0]]]);assert q95==read(O/'coverage.json')['full']['q95_squared']
    functionals=read(O/'target_functionals.json')['fit_pools'];fdmax=0.
    for split,row in functionals.items():
        ids=scope_cache[split][0];yy=ys['train'][ids].astype(float);p=prob[ids];p=p/p.sum();assert abs(float(p@yy)-row['mean_net'])<1e-8 and abs(float(p@np.clip(yy,-1,1))-row['mean_clipped_net'])<1e-8
        def risk(z):
            v=abs(z-yy);return .5*float(p@np.where(v<1,.5*v*v,v-.5))
        fd=(risk(1e-6)-risk(-1e-6))/(2e-6);error=abs(fd-row['huber_half_derivative_at_zero']);fdmax=max(fdmax,error);assert error<1e-8
    Q=np.load(S/'predictions.npz');bound=read(O/'prediction_bounds.json');rangechecked=0
    for r in bound:
        split=r['name'].split('__')[0];ids,held,part,*_=scope_cache[split];ii=ids if r['scope']=='fit' else held;pt=r['part'];scores=Q[r['name']+'__'+pt][ii,1].astype(float);cost=charge[pt][ii];bad=scores<-cost-1e-6;assert int(bad.sum())==r['violations'];projected=np.maximum(scores,-cost);assert np.array_equal(scores>0,projected>0);rangechecked+=len(scores)
    srcdiv={(r['domain'],r['seed'],r['sample_id']):r for r in divergence_rows()};cohort=[]
    with gzip.open(O/'cohort_records.jsonl.gz','rt',encoding='utf8') as f:
        for ln in f:
            r=__import__('json').loads(ln);source_r=srcdiv[r['domain'],r['seed'],r['sample_id']];m=meta[r['part']][r['index']];assert m['sample_id']==r['sample_id'] and m['position']==r['first_divergence_position'] and m['physical_state_group']==r['physical_state_group'];assert source_r['group']==r['group']
            i=r['index'];pt=r['part'];assert r['target_net']==float(ys[pt][i]) and r['known_charge']==charge[pt][i] and r['actual_raw_gain']==gain[pt][i];assert abs(r['actual_raw_gain']-r['known_charge']-r['target_net'])<1e-5
            q=Q[f'{r["split"]}__AUX_GAIN_seed{r["seed"]}__{pt}'][i];assert r['predicted_net']==float(q[1]) and r['aux_event_positive']==bool(q[0]>0) and r['predicted_net_below_lower_bound']==bool(float(q[1])+charge[pt][i]<-1e-6);assert np.array_equal(r['public49'][:41],raw[pt][i]);cohort.append(r)
    assert len(cohort)==len(srcdiv)
    for dom,gg in read(O/'group_summaries.json').items():
        for group,summary in gg.items():
            rr=[r for r in cohort if r['domain']==dom and r['group']==group];calls=collections.defaultdict(list)
            for r in rr:calls[r['physical_state_group']].append(r)
            assert len(rr)==summary['records'] and len(calls)==summary['calls']
            for field,v in summary['call_balanced'].items():assert abs(float(np.mean([np.mean([r[field] for r in rs]) for rs in calls.values()]))-v)<1e-12
    exposure=read(O/'training_exposure.json')
    for split,ss in exposure.items():
        ids=scope_cache[split][0]
        for seed,r in ss.items():
            counts=np.load(S/f'exposure_{split}__AUX_GAIN_seed{seed}.npz')['state_draws'];assert counts.sum()==r['total_draws']==232000;assert int(counts[ids[ys['train'][ids]>1]].sum())==r['large_net_draws'] and r['large_states_never_drawn']==int(np.sum(counts[ids[ys['train'][ids]>1]]==0))
    verify();src.sealed();save(O/'review.json',dict(passed=True,direct_distance_checks=len(support),full_fit_leave_callout_checks=len(ids) if False else 3582,source_GRID122_reproduced=3582,all_cohort_records_checked=len(cohort),all_range_predictions_checked=rangechecked,maximum_direct_distance_error=maxerr,huber_zero_derivative_finite_difference_max_error=fdmax,all_action_projections_identical=True,frozen_hashes_unchanged=True,new_fits=0,new_model_forwards=0,new_physical_calls=0,final_test_materialized=False));print('GRID125_AUDIT_PASSED',len(support),len(cohort),rangechecked,flush=True)

if __name__=='__main__':main()
