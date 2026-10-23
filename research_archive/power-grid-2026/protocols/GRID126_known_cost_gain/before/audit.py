"""Independent NumPy forward,public-charge labels,training receipts and complete trajectories."""
from common126 import *

def main():
    assert not (O/'review.json').exists();verify();src.configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==30;Q=np.load(O/'predictions.npz');D=read(O/'design.json');maximum=0.;n=0
    for rec in F['records']:
        cp=torch.load(O/'models'/f"{rec['name']}.pt",map_location='cpu',weights_only=False);assert sha(O/'models'/f"{rec['name']}.pt")==rec['checkpoint_sha256'];net=build(rec['seed']);assert digest(net)==rec['initial_hash'];net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash'];assert cp['updates']==7250 and sum(v.numel() for v in net.parameters())==6978
        assert all(int(s['step'])==7250 for s in cp['optimizer']['state'].values());pg=cp['optimizer']['param_groups'][0];assert pg['lr']==.0003 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0.
        for part in (['train','validation'] if rec['split']=='full' else ['train']):
            x,y,g,c,ids,p=getdata(rec['split'],part);z=x.astype(float)
            for k in [0,2,4]:
                z=z@cp['state_dict'][f'q_net.{k}.weight'].numpy().astype(float).T+cp['state_dict'][f'q_net.{k}.bias'].numpy().astype(float)
                if k!=4:z=np.maximum(z,0)
            actual=Q[rec['name']+'__'+part];maximum=max(maximum,float(abs(z-actual).max()));assert np.allclose(z,actual,atol=3e-5,rtol=3e-5);assert np.array_equal(actions(z,c,rec['arm']),actions(actual,c,rec['arm']));n+=len(y)
            if rec['arm']=='COST_POSITIVE':
                independent=z[:,1]+np.sqrt(z[:,1]**2+c**2);assert np.allclose(independent,numpy_gain(z[:,1],c,rec['arm']),atol=1e-10,rtol=1e-10) and (independent>=-1e-12).all();assert np.array_equal(independent>c,z[:,1]>0)
        e=np.load(O/f"exposure_{rec['name']}.npz");expected=np.load(OLD/f"training_exposure_seed{rec['seed']}.npz")['state_draws'] if rec['split']=='full' else np.load(CV/f"fold{rec['split'][1:]}_seed{rec['seed']}_expected_exposure.npz")['state_draws'];assert np.array_equal(e['state_draws'],expected) and e['state_draws'].sum()==232000;assert e['half_main_stop_losses'].shape==(7250,2) and np.isfinite(e['half_main_stop_losses']).all() and np.isfinite(e['gradient_norm_before_clip']).all()
        x,y,g,c,ids,p=getdata(rec['split']);held=np.setdiff1d(np.arange(len(y)),ids);assert e['state_draws'][held].sum()==0
        key=str(rec['seed']) if rec['split']=='full' else f"fold{rec['split'][1:]}_seed{rec['seed']}";expected_hash=read(src.S/'preflight.json')['sampling_streams'][key]['index_sha256'] if rec['split']=='full' else read(CV/'preflight.json')['sampling_streams'][key];assert expected_hash==rec['index_sha256']
    old=np.load(S/'predictions.npz')
    for key in old.files:
        if '__NET_ZERO_' in key:assert np.array_equal(Q[key],old[key])
    cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};meta={p:read(S/f'{p}_metadata.json') for p in ['train','validation']};lookup={p:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for p,mm in meta.items()};labels=0
    for part,mm in meta.items():
        z=np.load(O/f'{part}_targets.npz')
        for i,m in enumerate(mm):
            case=cm[m['sample_id']];j=m['position'];assert m['next_queries']==case['queries'][j] and z['charge'][i]==D['cost_per_query']*m['next_queries']/D['gain_scale'] and z['gain'][i]==case['gains'][j]/D['gain_scale'];assert z['gain'][i]>=0;labels+=1
    results={};checked=0;bydom={}
    for dom,filename,part in [('full_train','full_train_trajectories.json','train'),('validation','full_validation_trajectories.json','validation'),('oof','oof_trajectories.json','train')]:
        rows=read(O/filename);res=read(O/f'{dom}_results.json');bydom[dom]=rows
        for r in rows:
            case=cm[r['sample_id']];ii=[lookup[part][case['sample_id'],j] for j in range(len(case['states']))];q=Q[r['model_name']+'__'+part][ii];arm=r['method'].rsplit('_seed',1)[0];c=np.load(O/f'{part}_targets.npz')['charge'][ii];go=actions(q,c,arm);p=0;seq=[]
            while p<len(go):
                seq.append(int(go[p]))
                if not go[p]:break
                p+=1
            assert seq==r['actions'] and p==r['batches'];gain=float(case['gains'][:p].sum());queries=int(case['queries'][:p].sum());assert r['gain']==gain and r['queries']==queries and r['chosen_pool_id']==case['chosen'][p];assert abs((gain-D['cost_per_query']*queries)/D['gain_scale']-r['net_utility'])<1e-12
            if dom=='oof':assert case['date'] in D['folds'][int(r['model_name'][1])]['held_dates']
            checked+=1
        for name,rr in res.items():
            selected=[r for r in rows if r['method']==name];dates=sorted({r['date'] for r in selected});v=float(np.mean([np.mean([r['net_utility'] for r in selected if r['date']==date]) for date in dates]));assert abs(v-rr['all']['macro']['mean_net_utility'])<1e-12
        results[dom]=res
    dec=read(O/'decision.json');baseline=read(S/'decision.json')['family_utility']
    for dom in ['validation','oof']:
        rr=results[dom];u=lambda arm,s:rr[f'{arm}_seed{s}']['all']['macro']['mean_net_utility'];assert abs(np.mean([u('NET_ZERO',s) for s in range(3)])-baseline[dom]['NET_ZERO'])<1e-12
        for arm in ALL:assert abs(float(np.mean([u(arm,s) for s in range(3)]))-dec['family_utility'][dom][arm])<1e-12
        for key,cmp in dec['comparisons'][dom].items():
            arm,ref=key.split('_minus_');pairs=[u(arm,s)-u(ref,s) for s in range(3)];assert pairs==cmp['paired_seed_differences'];signal=np.mean(pairs)>0 and all(v>0 for v in pairs) and sum(v>0 for v in cmp['date_differences'].values())>=(3 if dom=='validation' else 12);assert bool(signal)==cmp['signal']
        for arm,gg in read(O/'divergence_summary.json')[dom].items():assert abs(sum(v['utility_contribution'] for v in gg.values())-(dec['family_utility'][dom][arm]-dec['family_utility'][dom]['NET_ZERO']))<1e-12
    assert F['wall_s']<1801;verify();src.sealed();save(O/'review.json',dict(passed=True,independent_numpy_forwards=n,maximum_abs_error=maximum,targets_and_public_charges=labels,complete_trajectories_checked=checked,all30_optimizer_steps7250=True,original_capacity_init_sampler=True,zero_holdout_training_draws=True,all_baseline_predictions_exact=True,positive_link_range_and_action_identity=True,all30_endpoints_before_new_validation_inference=True,all_frozen_hashes_unchanged=True,new_physical_calls=0,final_test_materialized=False));print('GRID126_AUDIT_PASSED',n,checked,maximum,flush=True)

if __name__=='__main__':main()
