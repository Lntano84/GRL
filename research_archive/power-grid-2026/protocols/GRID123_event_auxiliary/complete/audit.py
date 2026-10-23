"""Independent all-state NumPy, exact controls, targets, exposure and sequence accounting."""
from common123 import *

def main():
    assert not (O/'review.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==30;P=read(O/'preflight.json');D=read(O/'design.json');Q=np.load(O/'predictions.npz');n=0;maximum=0.
    for rec in F['records']:
        split,seed,arm=rec['split'],rec['seed'],rec['arm'];name=f'{split}__{arm}_seed{seed}';path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256'];cp=torch.load(path,weights_only=False,map_location='cpu')
        net=build(seed);assert digest(net)==rec['initial_hash']==P['original_initial_hashes'][str(seed)];net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash']
        assert cp['updates']==7250 and len(cp['optimizer']['state'])==len(cp['state_dict']) and all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
        group=cp['optimizer']['param_groups'][0];assert group['lr']==.0003 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0.
        for part in (['train','validation'] if split=='full' else ['train']):
            x,y,event,ids,prob=getdata(split,part);ref=x.astype(float)
            for k in [0,2,4]:
                ref=ref@cp['state_dict'][f'q_net.{k}.weight'].numpy().astype(float).T+cp['state_dict'][f'q_net.{k}.bias'].numpy().astype(float)
                if k!=4:ref=np.maximum(ref,0)
            actual=Q[name+'__'+part];maximum=max(maximum,float(abs(ref-actual).max()));assert np.allclose(ref,actual,atol=3e-5,rtol=3e-5),(name,part,maximum)
            assert np.array_equal(ref[:,1]>0,actual[:,1]>0),(name,part,'main sign');n+=len(y)
        e=np.load(O/f'exposure_{name}.npz');expected=np.load(OLD/f'training_exposure_seed{seed}.npz')['state_draws'] if split=='full' else np.load(CV/f'fold{split[1:]}_seed{seed}_expected_exposure.npz')['state_draws']
        assert np.array_equal(e['state_draws'],expected) and len(e['half_main_aux_losses'])==7250 and e['half_main_aux_losses'].shape[1]==2
        assert np.isfinite(e['half_main_aux_losses']).all() and np.isfinite(e['gradient_norm_before_clip']).all()
        x,y,event,ids,prob=getdata(split);mask=np.ones(len(y),bool);mask[ids]=False;assert e['state_draws'][mask].sum()==0
        dg=read(O/f'gradient_components_{name}.json');assert [g['update'] for g in dg]==[1,7250]
        assert all(np.isfinite(g['main_shared_norm']) and np.isfinite(g['aux_shared_norm']) for g in dg)
        assert dg[0]['main_shared_norm']>0 and dg[0]['aux_shared_norm']>0
        streams=read(S/'preflight.json')['sampling_streams'] if split=='full' else read(CV/'preflight.json')['sampling_streams']
        expected_hash=streams[str(seed)]['index_sha256'] if split=='full' else streams[f'fold{split[1:]}_seed{seed}']
        assert rec['index_sha256']==expected_hash
    # Labels and features derived from the saved original teacher states only.
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={c['sample_id']:c for c in cases};metas={p:read(O/f'{p}_metadata.json') for p in ['train','validation']};lookup={p:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for p,mm in metas.items()};checked=0
    for part,mm in metas.items():
        x,y,event,_,_=getdata('full',part);oldx,oldy,_=data(part);assert np.array_equal(x,oldx) and np.array_equal(y,oldy)
        for i,m in enumerate(mm):
            c=cm[m['sample_id']];j=m['position'];assert c['partition']==part and event[i]==float(c['gains'][j]>0);checked+=1
    oldfold=np.load(CV/'predictions.npz')
    for f in range(4):
        for s in range(3):assert np.array_equal(Q[f'f{f}__NET_ZERO_seed{s}__train'],oldfold[f'fold{f}__HUBER_seed{s}'])
    import gzip
    with gzip.open(OLD/'state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['method'].startswith('DIRECT'):
                s=int(r['method'][-1]);part=r['partition'];i=lookup[part][r['sample_id'],r['position']]
                assert np.array_equal(Q[f'full__NET_ZERO_seed{s}__{part}'][i],np.array([r['q0'],r['q1']],np.float32))
    count=0;independent={}
    for domain,path,part in [('full_train','full_train_trajectories.json','train'),('validation','full_validation_trajectories.json','validation'),('oof','oof_trajectories.json','train')]:
        rows=read(O/path);results=read(O/f'{domain}_results.json');independent[domain]={}
        if domain=='oof':assert len(rows)==9*2378 and all(v==1 for v in collections.Counter((r['method'],r['sample_id']) for r in rows).values())
        for r in rows:
            c=cm[r['sample_id']];p=r['batches']
            if 'model_name' in r:
                key=r['model_name']+'__'+part;q=Q[key];p=0;seq=[]
                if domain=='oof':assert c['date'] in D['folds'][int(r['model_name'][1])]['held_dates']
                while p<len(c['states']):
                    score=q[lookup[part][c['sample_id'],p]];go=int(score[1]>0);seq.append(go)
                    if not go:break
                    p+=1
                assert seq==r['actions'] and p==r['batches']
            assert float(c['gains'][:p].sum())==r['gain'] and int(c['queries'][:p].sum())==r['queries'] and c['chosen'][p]==r['chosen_pool_id']
            assert abs((r['gain']-D['cost_per_query']*r['queries'])/D['gain_scale']-r['net_utility'])<1e-12;count+=1
        for name,res in results.items():
            rr=[r for r in rows if r['method']==name];ds=sorted({r['date'] for r in rr});u=float(np.mean([np.mean([r['net_utility'] for r in rr if r['date']==d]) for d in ds]));assert abs(u-res['all']['macro']['mean_net_utility'])<1e-12;independent[domain][name]=u
    dec=read(O/'decision.json')
    for domain in ['validation','oof']:
        rr=read(O/f'{domain}_results.json');u=independent[domain]
        for a in ['NET_ZERO']+ARMS:assert float(np.mean([u[f'{a}_seed{s}'] for s in range(3)]))==dec['family_utility'][domain][a]
        for ref in ['NET_ZERO','AUX_NETSIGN']:
            pairs=[u[f'AUX_GAIN_seed{s}']-u[f'{ref}_seed{s}'] for s in range(3)];dd={d:float(np.mean([rr[f'AUX_GAIN_seed{s}']['all']['per_date'][d]['mean_net_utility']-rr[f'{ref}_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)])) for d in rr['AUX_GAIN_seed0']['all']['per_date']};actual=dec['comparisons'][domain][ref]
            assert pairs==actual['paired_seed_differences'] and dd==actual['date_differences']
            signal=np.mean(pairs)>0 and all(v>0 for v in pairs) and sum(v>0 for v in dd.values())>=(3 if domain=='validation' else 12);assert bool(signal)==actual['signal']
    assert F['wall_s']<1801;verify();sealed();save(O/'review.json',dict(passed=True,independent_numpy_forwards=n,maximum_abs_error=maximum,all_targets_checked=checked,complete_trajectories_checked=count,
        all30_optimizer_steps7250=True,main_output_only_used_for_decisions=True,main_loss_half_coefficient_preserved=True,original_capacity_init_sampling_and_fold_preprocessing=True,
        no_holdout_gradient_samples=True,all_frozen_hashes_unchanged=True,all30_endpoints_before_evaluation=True,new_physical_calls=0,final_test_materialized=False));print('GRID123_AUDIT_PASSED',n,count,maximum,flush=True)
if __name__=='__main__':main()
