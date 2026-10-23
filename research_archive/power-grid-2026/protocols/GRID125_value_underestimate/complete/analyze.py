"""All read-only numerical accounting. No classifier, network, or optimizer constructed."""
from common125 import *

def target_stats(y,p):
    p=p/p.sum();positive=y>0;high=y>1;total=float(p@np.maximum(y,0))
    return dict(states=len(y),net_positive_states=int(positive.sum()),large_net_states=int(high.sum()),positive_probability=float(p@positive),large_net_probability=float(p@high),mean_net=float(p@y),mean_clipped_net=float(p@np.clip(y,-1,1)),positive_net_mass=total,large_net_positive_mass_fraction=float(p@(y*high)/total) if total else None,huber_half_derivative_at_zero=float(-.5*(p@np.clip(y,-1,1))))

def group_summary(rows):
    if not rows:return dict(records=0,calls=0)
    by=collections.defaultdict(list)
    for r in rows:by[r['physical_state_group']].append(r)
    fields=['nearest_positive_fraction','nearest_mean_net','beyond_fit_callout95','positive_neighbor_farther','raw_exact_match','normalized_exact_match','clipped_dimensions','predicted_net_below_lower_bound','aux_event_positive','target_net','predicted_net','known_charge']
    return dict(records=len(rows),unique_states=len({(r['part'],r['index']) for r in rows}),calls=len(by),call_balanced={k:float(np.mean([np.mean([r[k] for r in rr]) for rr in by.values()])) for k in fields},calls_list=sorted(by))

def main():
    assert not (O/'analysis_receipt.json').exists();verify();src.configure();cases,meta,raw,gain,charge=arrays();Q=np.load(S/'predictions.npz');y={part:np.load(S/f'{part}_data.npz')['net'] for part in meta};prob=np.load(S/'train_data.npz')['probability'];groups=[m['physical_state_group'] for m in meta['train']];names=read(R/'outputs/grid102_search_rl/design.json')['input']['features'];public={part:src.pickle.loads((src.S/f'{part}_public.pkl').read_bytes()) for part in meta};extras={part:np.stack([nn.public_descriptors(v) for v in public[part]]) for part in meta}
    support={};coverage={};targets={};bounds=[];fit_nn={};frozen_old=read(nn.O/'support_rows.json');old_checked=0
    for split in ['full','f0','f1','f2','f3']:
        ids,held,part,x,h,pre,hpre=scope(split,meta,raw);p=prob[ids]/prob[ids].sum();targets[split]=target_stats(y['train'][ids],p)
        assert np.array_equal(x.astype('f'),src.getdata(split)[0][ids])
        if split=='full':assert np.array_equal(h.astype('f'),np.load(S/'validation_data.npz')['X'])
        else:assert np.array_equal(h.astype('f'),src.getdata(split)[0][held])
        baseline=nn.nearest_batch(x,x,ids,ids,groups,True);fit_nn[split]=baseline;q95=nn.weighted_quantile([r['distance_squared'] for r in baseline],p,.95)
        near=nn.nearest_batch(h,x,held,ids);posids=ids[y['train'][ids]>0];negids=ids[y['train'][ids]<=0];where={int(v):i for i,v in enumerate(ids)}
        nearpos=nn.nearest_batch(h,x[[where[int(i)] for i in posids]],held,posids);nearneg=nn.nearest_batch(h,x[[where[int(i)] for i in negids]],held,negids)
        rawmap=collections.defaultdict(list);normmap=collections.defaultdict(list)
        for j,i in enumerate(ids):rawmap[raw['train'][i].tobytes()].append(int(i));normmap[x[j].astype('f').tobytes()].append(int(i))
        for j,(r,rp,rn) in enumerate(zip(near,nearpos,nearneg)):
            i=int(held[j]);ni=r['nearest_indices'];row=dict(split=split,part=part,**r,nearest_positive_fraction=float(np.mean(y['train'][ni]>0)),nearest_mean_net=float(y['train'][ni].mean()),nearest_positive_indices=rp['nearest_indices'],positive_distance_squared=rp['distance_squared'],negative_distance_squared=rn['distance_squared'],positive_neighbor_farther=bool(rp['distance_squared']>rn['distance_squared']),fit_callout95_squared=q95,beyond_fit_callout95=bool(r['distance_squared']>q95),raw_exact_match=bool(rawmap.get(raw[part][i].tobytes(),[])),normalized_exact_match=bool(normmap.get(h[j].astype('f').tobytes(),[])),clipped_dimensions=int(np.sum(abs(hpre[j])>20)),clipped_feature_names=[names[k] for k in np.flatnonzero(abs(hpre[j])>20)])
            support[part,i]=row
            if split!='full':
                old=frozen_old[i];assert old['nearest_indices']==ni and old['distance_squared']==row['distance_squared'] and old['fit_lco_q95_squared']==q95 and old['positive_nearest_indices']==rp['nearest_indices'];old_checked+=1
        coverage[split]=dict(fit_states=len(ids),held_states=len(held),q95_squared=q95,held_beyond_q95=sum(support[part,int(i)]['beyond_fit_callout95'] for i in held),raw_exact_match=sum(support[part,int(i)]['raw_exact_match'] for i in held),normalized_exact_match=sum(support[part,int(i)]['normalized_exact_match'] for i in held),held_states_clipped=sum(support[part,int(i)]['clipped_dimensions']>0 for i in held))
        for arm in ARMS:
            for seed in range(3):
                n=f'{split}__{arm}_seed{seed}'
                for label,idx,pp,pt in [('fit',ids,p,'train'),('held',held,None,part)]:
                    if label=='held':
                        if part=='train':pp=prob[idx]/prob[idx].sum()
                        else:
                            cc=collections.Counter(c['date'] for c in cases if c['partition']=='validation');cm={c['sample_id']:c for c in cases};pp=np.array([1/len(cc)/cc[m['date']]/len(cm[m['sample_id']]['states']) for m in meta['validation']])
                    q=Q[n+'__'+pt][idx,1].astype(float);cost=charge[pt][idx];bad=q+cost < -1e-6;projected=np.maximum(q,-cost);assert np.array_equal(q>0,projected>0)
                    bounds.append(dict(name=n,scope=label,part=pt,states=len(idx),violations=int(bad.sum()),probability_mass=float(pp@bad),minimum_implied_gain=float((q+cost).min()),all_projected_actions_unchanged=True))
    assert old_checked==3582 and len(support)==4438
    lookup={part:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for part,mm in meta.items()};rows=[];exposures={};fit_exposures={}
    for split in ['full','f0','f1','f2','f3']:
        ids,_,_,_,_,_,_=scope(split,meta,raw);p=prob[ids]/prob[ids].sum();fit_exposures[split]={}
        for seed in range(3):
            bp=src.OLD/f'training_exposure_seed{seed}.npz' if split=='full' else src.CV/f'exposure_fold{split[1:]}__HUBER_seed{seed}.npz';base=np.load(bp)['state_draws'];aux=np.load(S/f'exposure_{split}__AUX_GAIN_seed{seed}.npz')['state_draws'];assert np.array_equal(base,aux);exposures[split,seed]=base
            high=ids[y['train'][ids]>1];pos=ids[y['train'][ids]>0];fit_exposures[split][str(seed)]=dict(total_draws=int(base.sum()),positive_draws=int(base[pos].sum()),large_net_draws=int(base[high].sum()),minimum_large_state_draws=int(base[high].min()) if len(high) else None,large_states_never_drawn=int(np.sum(base[high]==0)),negative_draws=int(base[ids[y['train'][ids]<=0]].sum()))
    for r in divergence_rows():
        part='validation' if r['domain']=='validation' else 'train';i=lookup[part][r['sample_id'],r['first_divergence_position']];s=support[part,i];split=s['split'];q=Q[f'{split}__AUX_GAIN_seed{r["seed"]}__{part}'][i];base=Q[f'{split}__NET_ZERO_seed{r["seed"]}__{part}'][i];ni=s['nearest_indices'];nearweights=np.array([exposures[split,r['seed']][j] for j in ni]);row=dict(**r,**{k:v for k,v in s.items() if k!='part'},part=part,target_net=float(y[part][i]),predicted_net=float(q[1]),baseline_net=float(base[1]),known_charge=float(charge[part][i]),actual_raw_gain=float(gain[part][i]),predicted_net_below_lower_bound=bool(q[1]+charge[part][i]<-1e-6),aux_event_positive=bool(q[0]>0),nearest_state_exposure_mean=float(nearweights.mean()),nearest_training_calls=sorted({groups[j] for j in ni}),public49=np.r_[raw[part][i],extras[part][i]].tolist());rows.append(row)
    summaries={dom:{g:group_summary([r for r in rows if r['domain']==dom and r['group']==g]) for g in GROUPS} for dom in ['validation','oof']};bad=[r for r in rows if r['group']==GROUPS[0]];assert len([r for r in bad if r['domain']=='validation'])==13 and len([r for r in bad if r['domain']=='oof'])==64
    by=collections.defaultdict(list)
    for r in bad:by[r['physical_state_group']].append(r)
    call_details=[]
    for call,rr in sorted(by.items()):
        call_details.append(dict(call=call,domain=rr[0]['domain'],records=len(rr),distinct_states=len({r['index'] for r in rr}),net_range=[min(r['target_net'] for r in rr),max(r['target_net'] for r in rr)],predicted_range=[min(r['predicted_net'] for r in rr),max(r['predicted_net'] for r in rr)],fee_range=[min(r['known_charge'] for r in rr),max(r['known_charge'] for r in rr)],lower_bound_violation_records=sum(r['predicted_net_below_lower_bound'] for r in rr),event_recognized_records=sum(r['aux_event_positive'] for r in rr),beyond_coverage_records=sum(r['beyond_fit_callout95'] for r in rr),nearest_positive_records=sum(r['nearest_positive_fraction']>0 for r in rr),mean_public49=np.mean([r['public49'] for r in rr],axis=0).tolist()))
    dates={date:target_stats(y['train'][[m['date']==date for m in meta['train']]],prob[[m['date']==date for m in meta['train']]]) for date in sorted({m['date'] for m in meta['train']})}
    save(O/'support_rows.json',list(support.values()));save(O/'fit_callout_neighbors.json',fit_nn);save(O/'coverage.json',coverage);save(O/'target_functionals.json',dict(fit_pools=targets,train_dates=dates,threshold1_is_original_huber_beta=True));save(O/'training_exposure.json',fit_exposures);save(O/'prediction_bounds.json',bounds);save(O/'group_summaries.json',summaries);save(O/'profitable_miss_calls.json',call_details);save(O/'feature_names.json',names+nn.EXTRA)
    with gzip.open(O/'cohort_records.jsonl.gz','wt',encoding='utf8') as f:
        for r in rows:f.write(__import__('json').dumps(r,ensure_ascii=False)+'\n')
    save(O/'analysis_receipt.json',dict(passed=True,held_support_states=4438,reproduced_GRID122_rows=old_checked,cohort_records=len(rows),profitable_miss_records=len(bad),all_saved_action_projections_unchanged=True,new_fits=0,new_model_forwards=0,new_physical_calls=0,final_test_materialized=False));verify();print('GRID125_ANALYZED',len(rows),len(bad),flush=True)

if __name__=='__main__':main()
