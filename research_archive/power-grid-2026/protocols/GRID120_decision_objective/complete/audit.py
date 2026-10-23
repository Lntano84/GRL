"""Independent all-state NumPy and utility/decision loss accounting."""
from common120 import *

def main():
    assert not (O/'review.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==6
    D=read(O/'design.json');P=read(O/'preflight.json');Q=np.load(O/'state_predictions.npz');cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());source={c['sample_id']:c for c in cases};datasets={p:data(p) for p in ['train','validation']}
    lookups={p:{(r['sample_id'],r['position']):i for i,r in enumerate(ds[2])} for p,ds in datasets.items()};targets_checked=0;forwards=0;err=0.;sign_disagreements=0
    for part,(x,y,meta) in datasets.items():
        for i,r in enumerate(meta):
            c=source[r['sample_id']];j=r['position'];assert c['partition']==part
            assert np.float32((c['gains'][j]-D['cost_per_query']*c['queries'][j])/D['gain_scale'])==y[i];targets_checked+=1
    for rec in F['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256']
        cp=torch.load(path,weights_only=False,map_location='cpu');m=build(rec['seed']);assert digest(m)==rec['initial_hash']==P['original_initial_hashes'][str(rec['seed'])]
        m.load_state_dict(cp['state_dict']);assert digest(m)==rec['final_hash'];assert cp['updates']==7250
        assert len(cp['optimizer']['state'])==len(cp['state_dict']) and all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
        group=cp['optimizer']['param_groups'][0];assert group['lr']==.0003 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0.
        for part,(x,y,meta) in datasets.items():
            ref=x.astype(float)
            for layer in [0,2,4]:
                ref=ref@cp['state_dict'][f'q_net.{layer}.weight'].numpy().astype(float).T+cp['state_dict'][f'q_net.{layer}.bias'].numpy().astype(float)
                if layer!=4:ref=np.maximum(ref,0)
            actual=Q[name+'__'+part];e=float(np.max(abs(ref-actual)));err=max(err,e);assert np.allclose(ref,actual,atol=3e-5,rtol=3e-5),(name,part,e)
            disagreements=np.sum((ref[:,1]>ref[:,0])!=(actual[:,1]>actual[:,0]));sign_disagreements+=int(disagreements);assert disagreements==0,(name,part,disagreements);forwards+=len(y)
        exposure=np.load(O/f'exposure_{name}.npz');assert len(exposure['minibatch_losses'])==len(exposure['gradient_norm_before_clip'])==7250
        assert np.isfinite(exposure['minibatch_losses']).all() and np.isfinite(exposure['gradient_norm_before_clip']).all()
        assert int(np.sum(exposure['gradient_norm_before_clip']>10))==rec['clipped_updates']
        assert np.array_equal(exposure['state_draws'],np.load(R/f"outputs/grid107_direct_value/training_exposure_seed{rec['seed']}.npz")['state_draws'])
        assert rec['index_sha256']==P['sampling_streams'][str(rec['seed'])]['index_sha256']
    count=0;independent={}
    for part in datasets:
        rr=read(O/f'{part}_trajectories.json');results=read(O/f'{part}_results.json');independent[part]={}
        for r in rr:
            c=source[r['sample_id']];p=r['batches'];key=r['method']+'__'+part
            if key in Q:
                p=0;seq=[]
                while p<len(c['states']):
                    q=Q[key][lookups[part][c['sample_id'],p]];a=int(q[1]>q[0]);seq.append(a)
                    if not a:break
                    p+=1
                assert seq==r['actions'] and p==r['batches']
            assert float(c['gains'][:p].sum())==r['gain'] and int(c['queries'][:p].sum())==r['queries']
            assert c['chosen'][p]==r['chosen_pool_id']
            assert abs((r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale']-r['net_utility'])<1e-12;count+=1
        for name,res in results.items():
            ss=[r for r in rr if r['method']==name];ds=sorted({r['date'] for r in ss});u=float(np.mean([np.mean([r['net_utility'] for r in ss if r['date']==d]) for d in ds]));independent[part][name]=u
            assert abs(u-res['all']['macro']['mean_net_utility'])<1e-12
    # Verify the exact utility-regret reduction for every state under BOTH decisions.
    for part,(_,y,_) in datasets.items():
        for a in [0,1]:assert np.array_equal(np.maximum(y,0)-a*y,abs(y)*(a!=(y>0)))
    dec=read(O/'decision.json');v=independent['validation'];u=lambda n:v[n]
    family={k:float(np.mean([u(fmt.format(s)) for s in range(3)])) for k,fmt in [('UTILITY','UTILITY_BCE_seed{}'),('SIGN','SIGN_BCE_seed{}'),('HUBER','DIRECT{}'),('DQN','DQN0_{}')]};assert family==dec['family_mean_utility']
    for s,p in enumerate(dec['paired_seeds']):assert p['utility_minus_sign']==u(f'UTILITY_BCE_seed{s}')-u(f'SIGN_BCE_seed{s}') and p['utility_minus_huber']==u(f'UTILITY_BCE_seed{s}')-u(f'DIRECT{s}')
    signal=family['UTILITY']>family['SIGN'] and all(p['utility_minus_sign']>0 for p in dec['paired_seeds']) and sum(x>0 for x in dec['date_utility_minus_sign'].values())>=3
    assert signal==dec['objective_signal'];anchors=all(family['UTILITY']>x for x in [family['HUBER'],family['DQN'],u('BATCH1')]);assert anchors==dec['all_anchors_exceeded'] and (signal and anchors)==dec['useful_candidate']
    assert F['wall_s']<1801;verify();sealed()
    save(O/'review.json',dict(passed=True,all_state_numpy_forwards=forwards,maximum_abs_error=err,decision_sign_disagreements=sign_disagreements,targets_checked=targets_checked,complete_trajectories_checked=count,
        all_six_optimizer_steps7250=True,original_initialization_and_sampler_matched=True,regret_reduction_checked_all_states=True,
        train_only_weights_and_validation_after_all_fits=True,original_frozen_sources_unchanged=True,new_physical_calls=0,final_test_materialized=False))
    save(O/'state.json',dict(stage='audited_waiting_publication',fits=6));print('GRID120_AUDIT_PASSED',forwards,count,err,flush=True)
if __name__=='__main__':main()
