"""Independent NumPy predictions, optimizer scope, fixed base, and saved-sequence accounting."""
import importlib.util
from common119 import *
spec=importlib.util.spec_from_file_location('audit113_numpy',R/'work/grid113_relational_learning/audit.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def numpy_total(w,row,arm):
    x=row['global41'].astype(np.float64)
    for i in [0,2,4]:
        x=x@w[f'base.q_net.{i}.weight'].T+w[f'base.q_net.{i}.bias']
        if i!=4:x=np.maximum(x,0)
    if arm=='BASE_EXTRA':return x
    delta={k.removeprefix('delta.'):v for k,v in w.items() if k.startswith('delta.')}
    return x+old.numpy_forward(delta,row,arm.removeprefix('FIXED_'))

def main():
    assert not (O/'review.json').exists();verify();configure();F=read(O/'fits.json');assert F['passed'] and len(F['records'])==9
    D=read(O/'design.json');P=read(O/'preflight.json');fee=D['unchanged']['cost_per_query'];scale=D['unchanged']['gain_scale'];Q=np.load(O/'state_predictions.npz');SQ=np.load(O/'shifted_predictions.npz')
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};meta={};rows={};checked=0;num=0;errmax=0.;weights={}
    exposures=np.load(PLAN/'training_target_and_exposure_check.npz')
    for part in ['train','validation']:
        meta[part]=read(S/f'{part}_metadata.json');rows[part]=pickle.loads((S/f'{part}_public.pkl').read_bytes());target=np.load(S/f'{part}_targets.npz')['target']
        for i,m in enumerate(meta[part]):
            c,j=cm[m['sample_id'],m['position']];assert c['partition']==part
            assert np.float32((c['gains'][j]-fee*c['queries'][j])/scale)==target[i];checked+=1
    for rec in F['records']:
        arm,seed=rec['arm'],rec['seed'];name=f'{arm}_seed{seed}';path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256']
        cp=torch.load(path,weights_only=False,map_location='cpu');m=build(seed,arm);m.load_state_dict(cp['state_dict']);assert digest(m)==rec['final_hash']
        names=[n for n,p in m.named_parameters() if p.requires_grad];assert names==cp['optimizer_parameter_names']
        assert cp['updates']==7250 and len(cp['optimizer']['state'])==len(names)
        assert all(int(s['step'])==7250 for s in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['param_groups'])==1
        group=cp['optimizer']['param_groups'][0]
        assert group['lr']==.0003 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0.
        if arm!='BASE_EXTRA':
            assert all(n.startswith('delta.') for n in names)
            assert digest(m.base)==P['frozen_base_hashes'][str(seed)]==rec['base_final_hash']
            assert sum(p.numel() for p in trainable(m))==5938
        else:assert sum(p.numel() for p in trainable(m))==6978
        assert rec['initial_hash']==P['initial_hashes'][name]
        w={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};weights[name]=w
        for part in ['train','validation']:
            y=np.load(S/f'{part}_targets.npz')['target'];ids=sorted(set([0,len(y)-1]+list(range(0,len(y),151))+list(np.flatnonzero(y>0))))
            for i in ids:
                ref=numpy_total(w,rows[part][i],arm);actual=Q[name+'__'+part][i];error=float(np.max(abs(ref-actual)));errmax=max(errmax,error)
                assert np.allclose(ref,actual,atol=3e-5,rtol=3e-5),(name,part,i,error)
                assert bool(ref[1]>ref[0])==bool(actual[1]>actual[0]);num+=1
            if arm=='BASE_EXTRA':assert np.array_equal(Q[name+'__'+part],SQ[name+'__'+part])
        e=np.load(O/f'exposure_{name}.npz');assert len(e['minibatch_losses'])==7250 and np.isfinite(e['minibatch_losses']).all()
        assert np.array_equal(e['state_draws'],exposures[f'extra_stage_exposure_seed{seed}'])
        assert rec['draws']==232000 and rec['scalar_targets']==464000
        assert rec['index_sha256']==P['sampling_streams'][str(seed)]['proposed_extra_stage']['index_sha256']
        flow=read(O/f'first_steps_{name}.json');assert [s['update'] for s in flow]==[1,2]
        if arm!='BASE_EXTRA':
            assert all(v==0 for k,v in flow[0]['gradient_norms_before_clip'].items() if k!='delta.head')
            assert all(v>0 for k,v in flow[1]['gradient_norms_before_clip'].items() if k!='delta.head')
    bycase={c['sample_id']:c for c in cases};count=0
    for part in ['train','validation']:
        lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta[part])};tr=read(O/f'{part}_trajectories.json');results=read(O/f'{part}_results.json')
        for r in tr:
            c=bycase[r['sample_id']];p=r['batches']
            if r['method'].startswith(('BASE_EXTRA_','FIXED_JOINT_','FIXED_FACTOR_')):
                p=0;seq=[]
                while p<len(c['states']):
                    q=Q[r['method']+'__'+part][lookup[c['sample_id'],p]];go=int(q[1]>q[0]);seq.append(go)
                    if not go:break
                    p+=1
                assert seq==r['actions'] and p==r['batches']
            assert float(c['gains'][:p].sum())==r['gain'] and int(c['queries'][:p].sum())==r['queries']
            assert c['chosen'][p]==r['chosen_pool_id']
            assert abs((r['gain']-fee*r['queries'])/scale-r['net_utility'])<1e-12;count+=1
        for name,res in results.items():
            rr=[r for r in tr if r['method']==name];ds=sorted({r['date'] for r in rr})
            u=float(np.mean([np.mean([r['net_utility'] for r in rr if r['date']==d]) for d in ds]))
            assert abs(u-res['all']['macro']['mean_net_utility'])<1e-12
    val=read(O/'validation_results.json');dec=read(O/'decision.json');u=lambda n:val[n]['all']['macro']['mean_net_utility']
    families={k:float(np.mean([u(fmt.format(s)) for s in range(3)])) for k,fmt in [('JOINT','FIXED_JOINT_seed{}'),('FACTOR','FIXED_FACTOR_seed{}'),('EXTRA','BASE_EXTRA_seed{}'),('BASE','DIRECT{}'),('DQN','DQN0_{}')]}
    assert families==dec['family_mean_utility']
    for s in range(3):
        p=dec['paired_seeds'][s]
        assert p['joint_minus_factor']==u(f'FIXED_JOINT_seed{s}')-u(f'FIXED_FACTOR_seed{s}')
        assert p['joint_minus_extra']==u(f'FIXED_JOINT_seed{s}')-u(f'BASE_EXTRA_seed{s}')
    dates=sorted(val['FIXED_JOINT_seed0']['all']['per_date'])
    dd={d:float(np.mean([val[f'FIXED_JOINT_seed{s}']['all']['per_date'][d]['mean_net_utility']-val[f'FIXED_FACTOR_seed{s}']['all']['per_date'][d]['mean_net_utility'] for s in range(3)])) for d in dates}
    assert dd==dec['date_joint_minus_factor']
    signal=families['JOINT']>families['FACTOR'] and all(x['joint_minus_factor']>0 for x in dec['paired_seeds']) and sum(x>0 for x in dd.values())>=3
    anchors=all(families['JOINT']>x for x in [families['BASE'],families['EXTRA'],families['DQN'],u('BATCH1')])
    assert signal==dec['mechanism_signal'] and anchors==dec['all_anchors_exceeded'] and (signal and anchors)==dec['competitive_candidate']
    probes=read(O/'critical_probes.json');sources={r['context']:r for r in read(R/'outputs/grid111_action_feedback/critical_details.json')};probechecks=0
    for p in probes:
        src=sources[p['context']];call=read(R/src['source']);inp=pack(src['public_input'],np.asarray(call['learned_decisions'][0]['normalized_features'],np.float32),fee,scale)
        for rec in F['records']:
            name=f"{rec['arm']}_seed{rec['seed']}";q=numpy_total(weights[name],inp,rec['arm']);saved=p['scores'][name]
            assert np.allclose(q,[saved['q0'],saved['q1']],atol=3e-5,rtol=3e-5);assert bool(q[1]>q[0])==bool(saved['action']);probechecks+=1
    assert F['wall_s']<1801;verify();sealed()
    save(O/'review.json',dict(passed=True,original_targets_checked=checked,independent_numpy_forwards=num,maximum_abs_error=errmax,
        independent_critical_forwards=probechecks,trajectories_checked=count,all_nine_steps7250=True,
        frozen_bases_exact=True,optimizer_parameter_scope_and_fresh_steps_checked=True,continuation_sampling_and_counts_checked=True,
        first_two_steps_in_budget_and_gradient_flow_checked=True,all_frozen_hashes_unchanged=True,
        validation_after_all_nine_fits=True,new_physical_calls=0,final_test_materialized=False,
        scope='Saved teacher-path offline results, not independent AC or whole-week quality/latency.'))
    save(O/'state.json',dict(stage='audited_waiting_publication',fits=9));print('GRID119_AUDIT_PASSED',num,count,errmax,flush=True)
if __name__=='__main__':main()
