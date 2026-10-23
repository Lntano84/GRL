"""Fixed end points; complete saved-sequence comparison, no fitting."""
import collections,gzip
from common120 import *

def state_stats(y,q,p):
    go=q[:,1]>q[:,0];pos=y>0;regret=np.maximum(y,0)-go*y
    return dict(states=len(y),positive_states=int(pos.sum()),positive_misses=int((pos&~go).sum()),unprofitable_continues=int((~pos&go).sum()),
        weighted_sign_regret=float(p@regret),weighted_realized_net=float(p@(go*y)),weighted_positive_gain_missed=float(p@(np.maximum(y,0)*~go)),
        weighted_negative_net_paid=float(p@(np.maximum(-y,0)*go)),continue_fraction=float(p@go))

def main():
    assert not (O/'evaluation_receipt.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and F['new_fits']==6
    D=read(O/'design.json');fee=D['cost_per_query'];scale=D['gain_scale'];Z=D['objectives']['normalizer'];cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes())
    source={c['sample_id']:c for c in cases};datasets={p:data(p) for p in ['train','validation']};lookups={p:{(m['sample_id'],m['position']):i for i,m in enumerate(ds[2])} for p,ds in datasets.items()}
    Q={};M={};trajectories={p:[] for p in datasets};weights={}
    for part,(x,y,meta) in datasets.items():
        cs=[c for c in cases if c['partition']==part];dc=collections.Counter(c['date'] for c in cs)
        weights[part]=np.asarray([1/len(dc)/dc[m['date']]/len(source[m['sample_id']]['states']) for m in meta]);assert abs(weights[part].sum()-1)<1e-12
        if part=='train':assert np.array_equal(weights[part],np.load(S/'train_targets.npz')['sampling_probability'])
    for rec in F['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";m=build(rec['seed']);cp=torch.load(O/'models'/f'{name}.pt',weights_only=False,map_location='cpu');m.load_state_dict(cp['state_dict']);assert digest(m)==rec['final_hash'];M[name]={}
        for part,(x,y,meta) in datasets.items():
            q=forward(m,x);assert np.isfinite(q).all();Q[name+'__'+part]=q;stats=state_stats(y,q,weights[part])
            # Interpret logits as decision scores only, not numerical action values.
            margin=(q[:,1]-q[:,0]).astype(float);labels=(y>0).astype(float);logloss=np.logaddexp(0.,margin)-labels*margin
            stats['sign_logloss']=float(weights[part]@logloss);stats['utility_weighted_logloss']=float(weights[part]@(logloss*abs(y)/Z));M[name][part]=stats
            for c in cases:
                if c['partition']!=part:continue
                ids=[lookups[part][c['sample_id'],j] for j in range(len(c['states']))]
                trajectories[part].append(trajectory(c,name,q[ids,1]>q[ids,0],fee,scale))
    # Reuse original model outputs; no refit or validation-dependent calibration.
    oldnames=[f'DIRECT{s}' for s in range(3)]+[f'DQN0_{s}' for s in range(3)]
    oldq={(n,p):np.full((len(ds[1]),2),np.nan,np.float32) for n in oldnames for p,ds in datasets.items()}
    with gzip.open(R/'outputs/grid107_direct_value/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);key=r['method'],r['partition']
            if key in oldq:oldq[key][lookups[key[1]][r['sample_id'],r['position']]]=[r['q0'],r['q1']]
    for (name,part),q in oldq.items():
        assert np.isfinite(q).all();Q[name+'__'+part]=q;M.setdefault(name,{})[part]=state_stats(datasets[part][1],q,weights[part])
        if part!='train':continue
        for c in cases:
            if c['partition']!=part:continue
            ids=[lookups[part][c['sample_id'],j] for j in range(len(c['states']))]
            trajectories[part].append(trajectory(c,name,q[ids,1]>q[ids,0],fee,scale))
    old=read(R/'outputs/grid107_direct_value/validation_trajectories.json')
    for r in old:
        c=source[r['sample_id']];t=trajectory(c,r['method'],r['actions']+[0]*len(c['states']),fee,scale)
        for k in ['actions','gain','queries','batches','chosen_pool_id']:assert r[k]==t[k]
        assert abs(r['net_utility']-t['net_utility'])<1e-12;trajectories['validation'].append(t)
    results={p:summaries(tr) for p,tr in trajectories.items()};v=results['validation'];u=lambda n:v[n]['all']['macro']['mean_net_utility']
    formats={'UTILITY':'UTILITY_BCE_seed{}','SIGN':'SIGN_BCE_seed{}','HUBER':'DIRECT{}','DQN':'DQN0_{}'}
    families={k:float(np.mean([u(fmt.format(s)) for s in range(3)])) for k,fmt in formats.items()}
    pairs=[dict(seed=s,utility=u(f'UTILITY_BCE_seed{s}'),sign=u(f'SIGN_BCE_seed{s}'),huber=u(f'DIRECT{s}'),dqn=u(f'DQN0_{s}'),
        utility_minus_sign=u(f'UTILITY_BCE_seed{s}')-u(f'SIGN_BCE_seed{s}'),utility_minus_huber=u(f'UTILITY_BCE_seed{s}')-u(f'DIRECT{s}')) for s in range(3)]
    ds=sorted(v['UTILITY_BCE_seed0']['all']['per_date']);delta={d:float(np.mean([v[f'UTILITY_BCE_seed{s}']['all']['per_date'][d]['mean_net_utility']-v[f'SIGN_BCE_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)])) for d in ds}
    signal=families['UTILITY']>families['SIGN'] and all(p['utility_minus_sign']>0 for p in pairs) and sum(x>0 for x in delta.values())>=3
    anchors=all(families['UTILITY']>x for x in [families['HUBER'],families['DQN'],u('BATCH1')])
    decision=dict(objective_signal=signal,all_anchors_exceeded=anchors,useful_candidate=signal and anchors,
        family_mean_utility=families,batch1_utility=u('BATCH1'),paired_seeds=pairs,date_utility_minus_sign=delta,ready_to_scale=False,
        limits='Four exposed dates, same grid/FULL teacher path; no wholeweek quality, latency, significance or cross-grid claim.')
    np.savez_compressed(O/'state_predictions.npz',**Q)
    for p in datasets:save(O/f'{p}_trajectories.json',trajectories[p]);save(O/f'{p}_results.json',results[p])
    save(O/'state_metrics.json',M);save(O/'decision.json',decision)
    save(O/'evaluation_receipt.json',dict(passed=True,all_six_fixed_fits_before_validation=True,new_validation_trajectories=3540,new_train_trajectories=14268,historical_validation_recosted=len(old),new_fits=0,new_physical_calls=0,final_test_materialized=False))
    save(O/'state.json',dict(stage='evaluated_waiting_audit',fits=6));print(json.dumps(decision,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
